"""
Preprocessing: BraTS NIfTI volumes -> normalized 2D axial slices + manifests.
=============================================================================

Pipeline
--------
1. Discover BraTS patient folders (each holds <id>-t1n.nii.gz, -t1c, -t2w, -t2f,
   and -seg.nii.gz).
2. Load, robust-normalize each modality, resize axial slices to target_size.
3. Save each slice as a (C, H, W) float16 .npy; save lesion masks for slices
   that contain segmentation labels.
4. Build manifests:
       healthy.txt        lesion-free + buffer slices from TRAIN patients
       val_healthy.txt    lesion-free slices from VAL patients (calibration)
       test_anom.txt      lesion slices from TEST patients (evaluation)
   and a stats.json summary.

Patient-level splits are read from ``splits/{train,val,test}.txt`` to guarantee
no patient leakage across sets.

Ref for the healthy-buffer idea (avoid mass-effect contamination near lesions):
Behrendt et al., 2023 (MIDL); Baur et al., 2021, "Autoencoders for Unsupervised
Anomaly Segmentation in Brain MRI: A Comparative Study" (Medical Image Analysis).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import CONFIG
from ..utils.exceptions import PreprocessingError
from ..utils.io import write_json
from ..utils.logging_utils import get_logger
from .normalization import robust_normalize_volume

log = get_logger("pdm.preprocess")

try:  # nibabel is required only for preprocessing.
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None


def _require_nibabel() -> None:
    if nib is None:
        raise PreprocessingError(
            "nibabel is required for preprocessing. Install with `pip install nibabel`."
        )


def _resize_slice(slice_2d: np.ndarray, size: int) -> np.ndarray:
    """Center-crop/pad a 2D slice to (size, size)."""
    h, w = slice_2d.shape
    out = np.full((size, size), -1.0, dtype=np.float32)  # -1 == background
    # Crop source region.
    sh, sw = min(h, size), min(w, size)
    src = slice_2d[
        (h - sh) // 2 : (h - sh) // 2 + sh, (w - sw) // 2 : (w - sw) // 2 + sw
    ]
    out[(size - sh) // 2 : (size - sh) // 2 + sh,
        (size - sw) // 2 : (size - sw) // 2 + sw] = src
    return out


def _load_patient(patient_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load and normalize all modalities + segmentation for one patient."""
    pid = patient_dir.name
    mods: dict[str, np.ndarray] = {}
    for m in CONFIG.data.modalities:
        f = patient_dir / f"{pid}-{m}.nii.gz"
        if not f.exists():
            raise PreprocessingError(f"Missing modality {m} for {pid}: {f}")
        vol = np.asarray(nib.load(str(f)).dataobj, dtype=np.float32)
        mods[m] = robust_normalize_volume(vol)

    seg_f = patient_dir / f"{pid}-seg.nii.gz"
    seg = (
        np.asarray(nib.load(str(seg_f)).dataobj, dtype=np.uint8)
        if seg_f.exists()
        else np.zeros_like(next(iter(mods.values())), dtype=np.uint8)
    )
    return mods, seg


def _lesion_slice_flags(seg: np.ndarray) -> np.ndarray:
    """Boolean array over axial index z: True if slice z has any lesion voxel."""
    return (seg > 0).any(axis=(0, 1))


def _healthy_slice_flags(lesion_flags: np.ndarray, buffer: int) -> np.ndarray:
    """Slices that are lesion-free AND >= `buffer` away from any lesion slice."""
    healthy = ~lesion_flags
    if lesion_flags.any():
        lesion_idx = np.where(lesion_flags)[0]
        for z in np.where(healthy)[0]:
            if np.min(np.abs(lesion_idx - z)) < buffer:
                healthy[z] = False
    return healthy


def preprocess(splits_dir: Path, limit_patients: int | None = None) -> dict:
    """Run preprocessing for all split patients. Returns a stats dict."""
    _require_nibabel()
    paths = CONFIG.paths
    paths.slices_dir.mkdir(parents=True, exist_ok=True)
    paths.masks_dir.mkdir(parents=True, exist_ok=True)
    paths.manifests_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        name: [
            ln.strip()
            for ln in (splits_dir / f"{name}.txt").read_text().splitlines()
            if ln.strip()
        ]
        for name in ("train", "val", "test")
    }

    manifests: dict[str, list[str]] = {"healthy": [], "val_healthy": [], "test_anom": []}
    n_lesion = n_healthy = 0
    size = CONFIG.data.target_size

    for split_name, pids in splits.items():
        if limit_patients:
            pids = pids[:limit_patients]
        for pid in pids:
            patient_dir = paths.data_root / pid
            if not patient_dir.exists():
                log.warning("Patient dir missing, skipping: %s", patient_dir)
                continue
            mods, seg = _load_patient(patient_dir)
            lesion_flags = _lesion_slice_flags(seg)
            healthy_flags = _healthy_slice_flags(lesion_flags, CONFIG.data.healthy_buffer)
            n_z = seg.shape[2]

            for z in range(n_z):
                stack = np.stack(
                    [_resize_slice(mods[m][:, :, z], size) for m in CONFIG.data.modalities]
                ).astype(np.float16)
                if (stack > -1.0).mean() < CONFIG.data.min_foreground_frac:
                    continue  # near-empty slice
                stem = f"{pid}_z{z:03d}"

                if lesion_flags[z]:
                    n_lesion += 1
                    np.save(paths.slices_dir / f"{stem}.npy", stack)
                    mask = _resize_slice((seg[:, :, z] > 0).astype(np.float32), size)
                    np.save(paths.masks_dir / f"{stem}.npy", (mask > 0.5).astype(np.uint8))
                    if split_name == "test":
                        manifests["test_anom"].append(stem)
                elif healthy_flags[z]:
                    n_healthy += 1
                    np.save(paths.slices_dir / f"{stem}.npy", stack)
                    if split_name == "train":
                        manifests["healthy"].append(stem)
                    elif split_name == "val":
                        manifests["val_healthy"].append(stem)
            log.info("Processed %s (%s)", pid, split_name)

    for name, items in manifests.items():
        (paths.manifests_dir / f"{name}.txt").write_text("\n".join(items) + "\n")

    stats = {
        "n_train_patients": len(splits["train"][:limit_patients] if limit_patients else splits["train"]),
        "n_val_patients": len(splits["val"][:limit_patients] if limit_patients else splits["val"]),
        "n_test_patients": len(splits["test"][:limit_patients] if limit_patients else splits["test"]),
        "n_lesion_slices": n_lesion,
        "n_healthy_slices": n_healthy,
        "manifest_counts": {k: len(v) for k, v in manifests.items()},
        "modalities": list(CONFIG.data.modalities),
        "target_size": size,
        "healthy_buffer": CONFIG.data.healthy_buffer,
    }
    write_json(paths.processed_root / "stats.json", stats)
    log.info("Preprocessing complete: %s", stats["manifest_counts"])
    return stats
