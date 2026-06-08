"""
Phase 1 — BraTS-PEDs 2D Slice Extraction Pipeline
===================================================
Extracts axial 2D slices from 3D NIfTI volumes (BraTS-PEDs dataset),
applies z-score normalization, pads to 256×256, and saves into
healthy / anomalous directories for downstream training and evaluation.

Slices are saved as z-score-normalised float32 arrays.  The mapping onto
the VAE's expected [-1, 1] range happens later (utils.normalize_for_vae).

Output tree (with patient-level splits)
----------------------------------------
    data/processed/
    ├── train_healthy/    # healthy slices from train patients
    ├── val_healthy/      # healthy slices from val patients
    ├── test_anomalous/   # lesion slices from test patients
    └── test_masks/       # corresponding binary masks

Usage
-----
    # Full run (no splits):
    python src/preprocess_to_2d.py --max-healthy 5000 --max-anomalous 2000

    # Train split only:
    python src/preprocess_to_2d.py --split-file splits/train.txt \\
        --healthy-subdir train_healthy --max-healthy 800

    # Val split (healthy only):
    python src/preprocess_to_2d.py --split-file splits/val.txt \\
        --healthy-subdir val_healthy --max-healthy 200

    # Test split (anomalous + masks):
    python src/preprocess_to_2d.py --split-file splits/test.txt \\
        --max-healthy 0 --max-anomalous 300
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
    """Z-score normalizes a 2D slice on non-zero (brain) voxels; background stays 0."""
    out = np.zeros_like(slice_2d, dtype=np.float32)
    brain_mask = slice_2d > 0

    if not np.any(brain_mask):
        return out

    brain_voxels = slice_2d[brain_mask].astype(np.float32)
    mean = brain_voxels.mean()
    std  = brain_voxels.std()

    if std < 1e-8:
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

    if arr.ndim == 3:
        return np.pad(arr, ((0, 0), (top, bottom), (left, right)), mode="constant")
    return np.pad(arr, ((top, bottom), (left, right)), mode="constant")


def load_nifti(filepath: Path) -> np.ndarray:
    return nib.load(str(filepath)).get_fdata()


def _resolve(patient_dir: Path, patient_id: str, suffix: str) -> Path | None:
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
    raw_dir: Path       = C.RAW_DATA_DIR,
    out_dir: Path       = C.PROCESSED_DIR,
    max_healthy: int    = 50,
    max_anomalous: int  = 20,
    split_file: Path | None = None,
    healthy_subdir: str = "train_healthy",
) -> None:
    """
    NIfTI volumes → 2D .npy slices.

    Parameters
    ----------
    split_file : optional
        Text file listing patient IDs (one per line) to include.
        When provided, only those patients are processed, enabling
        separate preprocessing runs per train/val/test split.
    healthy_subdir : str
        Subdirectory name under out_dir for healthy slices
        (e.g. "train_healthy" or "val_healthy").
    """
    dir_healthy   = out_dir / healthy_subdir
    dir_anomalous = out_dir / "test_anomalous"
    dir_masks     = out_dir / "test_masks"
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
    log.info("Processing %d patient folders from %s", len(patient_dirs), raw_dir)

    healthy_count   = 0
    anomalous_count = 0

    for patient_dir in tqdm(patient_dirs, desc="Patients", unit="pat"):
        if healthy_count >= max_healthy and anomalous_count >= max_anomalous:
            log.info("Reached quotas (%d healthy, %d anomalous). Stopping.",
                     max_healthy, max_anomalous)
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

        depth = volumes[0].shape[2]

        for z in range(depth):
            if healthy_count >= max_healthy and anomalous_count >= max_anomalous:
                break

            mod_slices = [vol[:, :, z] for vol in volumes]
            seg_slice  = seg_vol[:, :, z]

            brain_any  = np.maximum.reduce([s > 0 for s in mod_slices])
            total_area = mod_slices[0].shape[0] * mod_slices[0].shape[1]
            if brain_any.sum() / total_area < C.MIN_BRAIN_FRAC:
                continue

            normed = [zscore_normalize_slice(s) for s in mod_slices]
            stacked = np.stack(normed, axis=0)
            stacked_padded = symmetric_pad(stacked, C.TARGET_SIZE)
            seg_padded     = symmetric_pad(seg_slice, C.TARGET_SIZE)

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
    log.info("  Healthy dir       : %s", dir_healthy)
    log.info("  Anomalous dir     : %s", dir_anomalous)
    log.info("  Slice shape       : (3, %d, %d)", C.TARGET_SIZE, C.TARGET_SIZE)
    log.info("=" * 55)


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1 — BraTS-PEDs 2D slice extraction pipeline",
    )
    parser.add_argument("--raw-dir",        type=Path, default=C.RAW_DATA_DIR,
                        help="Path to the raw BraTS-PEDs Training folder.")
    parser.add_argument("--out-dir",        type=Path, default=C.PROCESSED_DIR,
                        help="Root output directory for processed slices.")
    parser.add_argument("--max-healthy",    type=int,  default=50,
                        help="Stop after saving this many healthy slices (default: 50).")
    parser.add_argument("--max-anomalous",  "--max-anomaly", type=int, default=20,
                        help="Stop after saving this many anomalous slices (default: 20).")
    parser.add_argument("--split-file",     type=Path, default=None,
                        help="Text file with one patient ID per line; only those patients "
                             "are processed (enables per-split preprocessing).")
    parser.add_argument("--healthy-subdir", type=str,  default="train_healthy",
                        help="Subdirectory under --out-dir for healthy slices "
                             "(default: train_healthy; use val_healthy for val split).")
    args = parser.parse_args()

    preprocess(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        max_healthy=args.max_healthy,
        max_anomalous=args.max_anomalous,
        split_file=args.split_file,
        healthy_subdir=args.healthy_subdir,
    )
