"""
VAE dataset preparation  (faithful-codec training set)
======================================================
The diffusion stage learns the *healthy* manifold, so it trains on healthy
slices only.  The VAE is different: it is a **codec** that must reconstruct
*everything* — tumour included — or lesion signal is destroyed in the latent
before the UNet ever sees it (Rombach et al. 2022; Pinaya et al. 2022).  So the
VAE training set is **every brain-bearing slice**, healthy *and* lesion, from
the train/val patients.  Test patients are never touched here (no leakage).

This script reuses the volume-level z-score + symmetric-pad helpers from
``preprocess_to_2d`` so the VAE sees pixels on exactly the same scale the rest
of the pipeline uses.

Output tree
-----------
    data/processed/
    ├── vae_train/        # all slices from splits/train.txt patients
    ├── vae_val/          # all slices from splits/val.txt patients
    └── vae_val_masks/    # binary lesion masks for val slices that have lesion
                          #   (drives the tumour-vs-healthy fidelity metric)

Usage
-----
    # Build train + val from the patient-level split files:
    python src/prepare_vae_dataset.py --max-per-patient 40 --max-total 6000

    # Smoke-sized set for a quick training dry-run:
    python src/prepare_vae_dataset.py --max-per-patient 4 --max-total 60
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

import config as C
from preprocess_to_2d import (
    zscore_normalize_volume,
    symmetric_pad,
    load_nifti,
    load_split_file,
    _resolve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _extract_split(
    raw_dir: Path,
    split_file: Path,
    out_dir: Path,
    mask_dir: Path | None,
    max_per_patient: int,
    max_total: int,
) -> int:
    """Write every brain-bearing slice (healthy + lesion) for the patients listed
    in ``split_file``.  When ``mask_dir`` is given, lesion slices also get their
    binary mask saved (used only for the val fidelity metric)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if mask_dir is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)

    allowed = load_split_file(split_file)
    patient_dirs = sorted(
        p for p in raw_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name in allowed
    )
    log.info("Split '%s': %d/%d patients present in %s",
             split_file.name, len(patient_dirs), len(allowed), raw_dir)

    saved = 0
    for patient_dir in tqdm(patient_dirs, desc=out_dir.name, unit="pat"):
        if saved >= max_total:
            break
        patient_id = patient_dir.name

        # Load the modality volumes (skip the patient if any is missing).
        volumes, skip = [], False
        for mod in C.MODALITIES:
            nii = _resolve(patient_dir, patient_id, mod)
            if nii is None:
                log.warning("Missing %s for %s — skipping.", mod, patient_id)
                skip = True
                break
            volumes.append(load_nifti(nii))
        if skip:
            continue

        seg_path = _resolve(patient_dir, patient_id, C.SEG_SUFFIX)
        seg_vol = load_nifti(seg_path) if seg_path is not None else None

        # Volume-level z-score (computed once per modality over the whole volume).
        normed_volumes = [zscore_normalize_volume(v) for v in volumes]
        depth = volumes[0].shape[2]

        per_patient = 0
        for z in range(depth):
            if saved >= max_total or per_patient >= max_per_patient:
                break

            # Brain footprint from RAW intensities (z-scored values go negative).
            mod_slices = [v[:, :, z] for v in volumes]
            brain_any = np.maximum.reduce([s > 0 for s in mod_slices])
            total = mod_slices[0].shape[0] * mod_slices[0].shape[1]
            if brain_any.sum() / total < C.MIN_BRAIN_FRAC:
                continue

            stacked = np.stack([nv[:, :, z] for nv in normed_volumes], axis=0)
            stacked = symmetric_pad(stacked, C.TARGET_SIZE)

            fname = f"{patient_id}_z{z:03d}.npy"
            np.save(out_dir / fname, stacked.astype(np.float32))

            if mask_dir is not None and seg_vol is not None:
                seg_slice = seg_vol[:, :, z]
                if seg_slice.sum() > 0:
                    seg_padded = symmetric_pad(seg_slice, C.TARGET_SIZE)
                    np.save(mask_dir / fname, (seg_padded > 0).astype(np.float32))

            saved += 1
            per_patient += 1

    log.info("  → %d slices written to %s", saved, out_dir)
    return saved


def main(args):
    n_train = _extract_split(
        args.raw_dir, args.train_split, C.VAE_TRAIN_DIR,
        mask_dir=None,
        max_per_patient=args.max_per_patient, max_total=args.max_total,
    )
    n_val = _extract_split(
        args.raw_dir, args.val_split, C.VAE_VAL_DIR,
        mask_dir=C.VAE_VAL_MASKS_DIR,
        max_per_patient=args.max_per_patient,
        max_total=max(1, args.max_total // 5),
    )

    log.info("=" * 55)
    log.info("VAE dataset ready")
    log.info("  train slices : %d  → %s", n_train, C.VAE_TRAIN_DIR)
    log.info("  val   slices : %d  → %s", n_val,   C.VAE_VAL_DIR)
    log.info("  val   masks  :        %s", C.VAE_VAL_MASKS_DIR)
    log.info("  slice shape  : (3, %d, %d)", C.TARGET_SIZE, C.TARGET_SIZE)
    log.info("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the VAE fine-tuning dataset (all slices, healthy+lesion)")
    parser.add_argument("--raw-dir",     type=Path, default=C.RAW_DATA_DIR,
                        help="BraTS-PEDs Training folder.")
    parser.add_argument("--train-split", type=Path, default=C.SPLITS_DIR / "train.txt",
                        help="Patient-ID list for the VAE train set.")
    parser.add_argument("--val-split",   type=Path, default=C.SPLITS_DIR / "val.txt",
                        help="Patient-ID list for the VAE val set.")
    parser.add_argument("--max-per-patient", type=int, default=40,
                        help="Cap slices taken per patient (spreads coverage; default 40).")
    parser.add_argument("--max-total", type=int, default=6000,
                        help="Cap total train slices (val is capped at max_total/5).")
    args = parser.parse_args()
    main(args)
