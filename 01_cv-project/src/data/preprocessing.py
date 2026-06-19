"""
BraTS-PEDs NIfTI → 2D slice extraction pipeline.

Extracts axial 2D slices from 3D NIfTI volumes, applies volume-level
z-score normalization, pads to ``TARGET_SIZE × TARGET_SIZE``, and saves
into healthy / anomalous directories for downstream training and evaluation.

Slices are saved as z-score-normalised float32 arrays.  The mapping onto
the VAE's expected [-1, 1] range happens later (``pipeline.diffusion.normalize_for_vae``).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

# pyrefly: ignore [missing-import]
import nibabel as nib

from core import constants as C

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Normalization helpers
# ─────────────────────────────────────────────
def zscore_normalize_volume(volume_3d: np.ndarray) -> np.ndarray:
    """Z-score normalize a 3D volume using statistics pooled over ALL non-zero
    (brain) voxels in the volume; background stays exactly 0.

    Volume-level (rather than per-slice) normalization is the BraTS standard and
    preserves the relative intensity gradients across the Z-axis.
    """
    out = np.zeros_like(volume_3d, dtype=np.float32)
    brain_mask = volume_3d > 0

    if not np.any(brain_mask):
        return out

    brain_voxels = volume_3d[brain_mask].astype(np.float32)
    mean = brain_voxels.mean()
    std = brain_voxels.std()

    if std < 1e-8:
        return out

    out[brain_mask] = (brain_voxels - mean) / std
    return out


def symmetric_pad(arr: np.ndarray, target: int = C.TARGET_SIZE) -> np.ndarray:
    """Symmetrically zero-pad a 2D/3D array to ``(target, target)`` on the last 2 dims."""
    h, w = arr.shape[-2], arr.shape[-1]
    pad_h = target - h
    pad_w = target - w

    if pad_h < 0 or pad_w < 0:
        raise ValueError(
            f"Input spatial dims ({h}×{w}) exceed target ({target}×{target}). "
            "Center-cropping is not implemented; adjust TARGET_SIZE."
        )

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if arr.ndim == 3:
        return np.pad(arr, ((0, 0), (top, bottom), (left, right)), mode="constant")
    return np.pad(arr, ((top, bottom), (left, right)), mode="constant")


# ─────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────
def load_nifti(filepath: Path) -> np.ndarray:
    """Load a NIfTI file and return the data as a numpy array."""
    return nib.load(str(filepath)).get_fdata()


def _resolve(patient_dir: Path, patient_id: str, suffix: str) -> Path | None:
    """Resolve a NIfTI file path with either .nii.gz or .nii extension."""
    for ext in (".nii.gz", ".nii"):
        p = patient_dir / f"{patient_id}-{suffix}{ext}"
        if p.exists():
            return p
    return None


def load_split_file(split_file: Path) -> set[str]:
    """Return the set of patient IDs listed in a split text file (one per line)."""
    with open(split_file) as f:
        return {line.strip() for line in f if line.strip()}


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def preprocess(
    raw_dir: Path = C.RAW_DATA_DIR,
    out_dir: Path = C.PROCESSED_DIR,
    max_healthy: int = 50,
    max_anomalous: int = 20,
    split_file: Path | None = None,
    healthy_subdir: str = "train_healthy",
    max_healthy_per_patient: int | None = None,
) -> None:
    """Convert NIfTI volumes to 2D ``.npy`` slices.

    Parameters
    ----------
    split_file : optional
        Text file listing patient IDs (one per line) to include.
        When provided, only those patients are processed, enabling
        separate preprocessing runs per train/val/test split.
    healthy_subdir : str
        Subdirectory name under ``out_dir`` for healthy slices
        (e.g. ``"train_healthy"`` or ``"val_healthy"``).
    max_healthy_per_patient : optional
        Cap healthy slices per patient to ensure diversity across patients.
    """
    dir_healthy = out_dir / healthy_subdir
    dir_anomalous = out_dir / "test_anomalous"
    dir_masks = out_dir / "test_masks"
    for d in (dir_healthy, dir_anomalous, dir_masks):
        d.mkdir(parents=True, exist_ok=True)

    allowed_patients: set[str] | None = None
    if split_file is not None:
        allowed_patients = load_split_file(split_file)
        log.info("Split file '%s': %d patient IDs loaded", split_file, len(allowed_patients))

    patient_dirs = sorted(
        p for p in raw_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if allowed_patients is not None:
        patient_dirs = [p for p in patient_dirs if p.name in allowed_patients]

    random.seed(42)
    random.shuffle(patient_dirs)

    log.info("Processing %d patient folders from %s", len(patient_dirs), raw_dir)

    healthy_count = 0
    anomalous_count = 0

    for patient_dir in tqdm(patient_dirs, desc="Patients", unit="pat"):
        if healthy_count >= max_healthy and anomalous_count >= max_anomalous:
            log.info(
                "Reached quotas (%d healthy, %d anomalous). Stopping.",
                max_healthy, max_anomalous,
            )
            break

        patient_id = patient_dir.name

        volumes = []
        skip = False
        for mod in C.MODALITIES:
            nii_path = _resolve(patient_dir, patient_id, mod)
            if nii_path is None:
                log.warning("Missing %s for %s — skipping patient.", mod, patient_id)
                skip = True
                break
            volumes.append(load_nifti(nii_path))
        if skip:
            continue

        seg_path = _resolve(patient_dir, patient_id, C.SEG_SUFFIX)
        if seg_path is None:
            log.warning("Missing seg for %s — skipping patient.", patient_id)
            continue
        seg_vol = load_nifti(seg_path)

        normed_volumes = [zscore_normalize_volume(vol) for vol in volumes]

        depth = volumes[0].shape[2]
        z_indices = list(range(depth))
        random.shuffle(z_indices)
        pat_healthy_count = 0

        for z in z_indices:
            if healthy_count >= max_healthy and anomalous_count >= max_anomalous:
                break

            mod_slices = [vol[:, :, z] for vol in volumes]
            seg_slice = seg_vol[:, :, z]

            brain_any = np.maximum.reduce([s > 0 for s in mod_slices])
            total_area = mod_slices[0].shape[0] * mod_slices[0].shape[1]
            if brain_any.sum() / total_area < C.MIN_BRAIN_FRAC:
                continue

            normed = [nv[:, :, z] for nv in normed_volumes]
            stacked = np.stack(normed, axis=0)
            stacked_padded = symmetric_pad(stacked, C.TARGET_SIZE)
            seg_padded = symmetric_pad(seg_slice, C.TARGET_SIZE)

            has_lesion = seg_padded.sum() > 0

            if has_lesion and anomalous_count < max_anomalous:
                fname = f"{patient_id}_z{z:03d}.npy"
                np.save(dir_anomalous / fname, stacked_padded)
                np.save(dir_masks / fname, seg_padded)
                anomalous_count += 1
            elif not has_lesion and healthy_count < max_healthy:
                if max_healthy_per_patient is not None and pat_healthy_count >= max_healthy_per_patient:
                    continue
                fname = f"{patient_id}_z{z:03d}.npy"
                np.save(dir_healthy / fname, stacked_padded)
                healthy_count += 1
                pat_healthy_count += 1

        log.info(
            "%s  →  healthy: %d/%d  |  anomalous: %d/%d",
            patient_id, healthy_count, max_healthy,
            anomalous_count, max_anomalous,
        )

    log.info("=" * 55)
    log.info("DONE  —  Saved %d healthy + %d anomalous slices", healthy_count, anomalous_count)
    log.info("Output directory: %s", out_dir)
    log.info("  Healthy dir       : %s", dir_healthy)
    log.info("  Anomalous dir     : %s", dir_anomalous)
    log.info("  Slice shape       : (3, %d, %d)", C.TARGET_SIZE, C.TARGET_SIZE)
    log.info("=" * 55)
