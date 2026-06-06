"""
Phase 1 — BraTS-PEDs 2D Slice Extraction Pipeline
===================================================
Extracts axial 2D slices from 3D NIfTI volumes (BraTS-PEDs dataset),
applies z-score normalization, pads to 256×256, and splits into
healthy / anomalous directories for downstream anomaly-detection
or diffusion-model training.

Slices are saved as z-score-normalised float32 arrays.  The mapping onto
the VAE's expected [-1, 1] range happens later (utils.normalize_for_vae),
so the .npy files stay interpretable and the transform lives in one place.

Usage
-----
    python src/preprocess_to_2d.py                        # defaults
    python src/preprocess_to_2d.py --max-healthy 5000 \
                                   --max-anomalous 2000   # full run
"""

import argparse
import logging
from pathlib import Path

# pyrefly: ignore [missing-import]
import nibabel as nib
import numpy as np
from tqdm import tqdm

import config as C

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def zscore_normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    """
    Z-score normalizes a single 2D slice on its non-zero (brain) voxels.
    Background (zero) stays zero.
    """
    out = np.zeros_like(slice_2d, dtype=np.float32)
    brain_mask = slice_2d > 0

    if not np.any(brain_mask):
        return out

    brain_voxels = slice_2d[brain_mask].astype(np.float32)
    mean = brain_voxels.mean()
    std  = brain_voxels.std()

    if std < 1e-8:                      # constant region → leave as zero
        return out

    out[brain_mask] = (brain_voxels - mean) / std
    return out


def symmetric_pad(arr: np.ndarray, target: int = C.TARGET_SIZE) -> np.ndarray:
    """Symmetrically zero-pad a 2D/3D array to (target, target) on the last 2 dims."""
    h, w = arr.shape[-2], arr.shape[-1]
    pad_h = target - h
    pad_w = target - w

    if pad_h < 0 or pad_w < 0:
        raise ValueError(
            f"Input spatial dims ({h}×{w}) exceed target ({target}×{target}). "
            "Center-cropping is not implemented; adjust TARGET_SIZE."
        )

    top    = pad_h // 2
    bottom = pad_h - top
    left   = pad_w // 2
    right  = pad_w - left

    if arr.ndim == 3:                   # (C, H, W)
        return np.pad(arr, ((0, 0), (top, bottom), (left, right)), mode="constant")
    return np.pad(arr, ((top, bottom), (left, right)), mode="constant")


def load_nifti(filepath: Path) -> np.ndarray:
    """Load a NIfTI file and return the raw data array."""
    return nib.load(str(filepath)).get_fdata()


def _resolve(patient_dir: Path, patient_id: str, suffix: str) -> Path | None:
    """Find ``{id}-{suffix}.nii.gz`` (or .nii); return None if missing."""
    for ext in (".nii.gz", ".nii"):
        p = patient_dir / f"{patient_id}-{suffix}{ext}"
        if p.exists():
            return p
    return None


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def preprocess(
    raw_dir: Path       = C.RAW_DATA_DIR,
    out_dir: Path       = C.PROCESSED_DIR,
    max_healthy: int    = 50,
    max_anomalous: int  = 20,
) -> None:
    """
    NIfTI volumes → 2D .npy slices.

    Output tree
    -----------
        data/processed/
        ├── train_healthy/       # (3, 256, 256) .npy — no tumour
        ├── test_anomalous/      # (3, 256, 256) .npy — has tumour
        └── test_masks/          # (256, 256)    .npy — binary mask
    """
    dir_healthy   = out_dir / "train_healthy"
    dir_anomalous = out_dir / "test_anomalous"
    dir_masks     = out_dir / "test_masks"
    for d in (dir_healthy, dir_anomalous, dir_masks):
        d.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted(
        p for p in raw_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    log.info("Found %d patient folders in %s", len(patient_dirs), raw_dir)

    healthy_count = 0
    anomalous_count = 0

    for patient_dir in tqdm(patient_dirs, desc="Patients", unit="pat"):
        if healthy_count >= max_healthy and anomalous_count >= max_anomalous:
            log.info("Reached quotas (%d healthy, %d anomalous). Stopping.",
                     max_healthy, max_anomalous)
            break

        patient_id = patient_dir.name

        # ── load 3 modalities ────────────────────────────────────
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

        # ── load segmentation mask ───────────────────────────────
        seg_path = _resolve(patient_dir, patient_id, C.SEG_SUFFIX)
        if seg_path is None:
            log.warning("Missing seg for %s — skipping patient.", patient_id)
            continue
        seg_vol = load_nifti(seg_path)

        depth = volumes[0].shape[2]     # volumes are (H, W, D)

        for z in range(depth):
            if healthy_count >= max_healthy and anomalous_count >= max_anomalous:
                break

            mod_slices = [vol[:, :, z] for vol in volumes]
            seg_slice  = seg_vol[:, :, z]

            # Skip near-empty slices — brain presence across ALL modalities
            brain_any  = np.maximum.reduce([s > 0 for s in mod_slices])
            total_area = mod_slices[0].shape[0] * mod_slices[0].shape[1]
            if brain_any.sum() / total_area < C.MIN_BRAIN_FRAC:
                continue

            normed = [zscore_normalize_slice(s) for s in mod_slices]
            stacked = np.stack(normed, axis=0)                      # (3, 240, 240)

            stacked_padded = symmetric_pad(stacked, C.TARGET_SIZE)  # (3, 256, 256)
            seg_padded     = symmetric_pad(seg_slice, C.TARGET_SIZE)  # (256, 256)

            has_lesion = seg_padded.sum() > 0

            if has_lesion and anomalous_count < max_anomalous:
                fname = f"{patient_id}_z{z:03d}.npy"
                np.save(dir_anomalous / fname, stacked_padded)
                np.save(dir_masks     / fname, seg_padded)
                anomalous_count += 1
            elif not has_lesion and healthy_count < max_healthy:
                fname = f"{patient_id}_z{z:03d}.npy"
                np.save(dir_healthy / fname, stacked_padded)
                healthy_count += 1

        log.info("%s  →  healthy: %d/%d  |  anomalous: %d/%d",
                 patient_id, healthy_count, max_healthy,
                 anomalous_count, max_anomalous)

    log.info("=" * 55)
    log.info("DONE  —  Saved %d healthy + %d anomalous slices", healthy_count, anomalous_count)
    log.info("Output directory: %s", out_dir)
    log.info("  Slice shape       : (3, %d, %d)", C.TARGET_SIZE, C.TARGET_SIZE)
    log.info("=" * 55)


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1 — BraTS-PEDs 2D slice extraction pipeline",
    )
    parser.add_argument("--raw-dir", type=Path, default=C.RAW_DATA_DIR,
                        help="Path to the raw BraTS-PEDs Training folder.")
    parser.add_argument("--out-dir", type=Path, default=C.PROCESSED_DIR,
                        help="Root output directory for processed slices.")
    parser.add_argument("--max-healthy", type=int, default=50,
                        help="Stop after saving this many healthy slices (default: 50).")
    parser.add_argument("--max-anomalous", "--max-anomaly", type=int, default=20,
                        help="Stop after saving this many anomalous slices (default: 20).")
    args = parser.parse_args()

    preprocess(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        max_healthy=args.max_healthy,
        max_anomalous=args.max_anomalous,
    )
