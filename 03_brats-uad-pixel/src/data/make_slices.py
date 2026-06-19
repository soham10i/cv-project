"""
Stage 0 — BraTS-PEDs NIfTI → 2D slice extraction.
==================================================
Extracts axial slices from every patient's 4 co-registered modalities
[t1n, t1c, t2w, t2f] + segmentation, applies robust per-volume normalisation,
symmetric-pads to 256×256, and writes each slice ONCE to ``slices/`` (float16)
with its lesion mask (if any) to ``masks/`` (uint8).

Dataset *views* are then defined by manifest files (lists of slice stems),
honouring the patient-level train/val/test splits and avoiding any data leakage:

  * vae_train   — all slices, train+val patients      (codec sees lesions too)
  * vae_val     — all slices, val patients            (recon monitoring)
  * healthy     — lesion-free + ≥BUFFER from any lesion, TRAIN patients
  * val_healthy — lesion-free, VAL patients           (calibration)
  * test_anom   — lesion slices, TEST patients         (evaluation)

Usage
-----
    python src/data/make_slices.py                      # full extraction
    python src/data/make_slices.py --limit-patients 4   # quick local smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# pyrefly: ignore [missing-import]
import nibabel as nib

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config as C
from data.normalization import normalize_volume

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_split(name: str) -> list[str]:
    """Read a split file (one patient ID per line). Returns [] if absent."""
    p = C.SPLITS_DIR / f"{name}.txt"
    if not p.exists():
        return []
    with open(p) as f:
        return [ln.strip() for ln in f if ln.strip()]


def resolve(patient_dir: Path, pid: str, suffix: str) -> Path | None:
    for ext in (".nii.gz", ".nii"):
        cand = patient_dir / f"{pid}-{suffix}{ext}"
        if cand.exists():
            return cand
    return None


def symmetric_pad(arr: np.ndarray, target: int = C.TARGET_SIZE) -> np.ndarray:
    """Symmetric zero-pad the last two dims to (target, target)."""
    h, w = arr.shape[-2], arr.shape[-1]
    ph, pw = target - h, target - w
    if ph < 0 or pw < 0:
        raise ValueError(f"Slice {h}×{w} exceeds target {target}; raise TARGET_SIZE.")
    top, left = ph // 2, pw // 2
    pad = ((top, ph - top), (left, pw - left))
    if arr.ndim == 3:
        pad = ((0, 0), *pad)
    return np.pad(arr, pad, mode="constant")


# ─────────────────────────────────────────────
# Per-patient extraction
# ─────────────────────────────────────────────
def process_patient(pid: str, slices_dir: Path, masks_dir: Path,
                    min_brain_frac: float) -> list[dict]:
    """Extract all qualifying slices for one patient.

    Returns a list of per-slice records: {stem, z, has_lesion, dist_to_lesion}.
    """
    patient_dir = C.DATA_ROOT / pid
    if not patient_dir.is_dir():
        log.warning("Patient dir missing: %s", patient_dir)
        return []

    vols = []
    for mod in C.MODALITIES:
        p = resolve(patient_dir, pid, mod)
        if p is None:
            log.warning("Missing %s for %s — skipping patient.", mod, pid)
            return []
        vols.append(nib.load(str(p)).get_fdata())
    seg_path = resolve(patient_dir, pid, C.SEG_SUFFIX)
    if seg_path is None:
        log.warning("Missing seg for %s — skipping patient.", pid)
        return []
    seg = nib.load(str(seg_path)).get_fdata()

    normed = [normalize_volume(v) for v in vols]
    depth = vols[0].shape[2]
    total_area = vols[0].shape[0] * vols[0].shape[1]

    # First pass: which z-slices carry a lesion (for buffer distance).
    lesion_z = np.array([seg[:, :, z].sum() > 0 for z in range(depth)], dtype=bool)
    lesion_idx = np.where(lesion_z)[0]

    def dist_to_lesion(z: int) -> int:
        if lesion_idx.size == 0:
            return 10_000
        return int(np.min(np.abs(lesion_idx - z)))

    records = []
    for z in range(depth):
        brain_any = np.maximum.reduce([(v[:, :, z] > 0) for v in vols])
        if brain_any.sum() / total_area < min_brain_frac:
            continue

        stacked = np.stack([nv[:, :, z] for nv in normed], axis=0).astype(np.float32)
        stacked = symmetric_pad(stacked, C.TARGET_SIZE).astype(np.float16)
        seg_slice = symmetric_pad(seg[:, :, z], C.TARGET_SIZE)
        has_lesion = bool(seg_slice.sum() > 0)

        stem = f"{pid}_z{z:03d}"
        np.save(slices_dir / f"{stem}.npy", stacked)
        if has_lesion:
            np.save(masks_dir / f"{stem}.npy", (seg_slice > 0).astype(np.uint8))

        records.append({"stem": stem, "z": z, "has_lesion": has_lesion,
                        "dist": dist_to_lesion(z)})
    return records


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args):
    for d in (C.SLICES_DIR, C.MASKS_DIR, C.MANIFEST_DIR):
        d.mkdir(parents=True, exist_ok=True)

    train_ids = load_split("train")
    val_ids   = load_split("val")
    test_ids  = load_split("test")
    if not (train_ids and val_ids and test_ids):
        log.warning("Split files incomplete in %s (train=%d val=%d test=%d). "
                    "Run with proper splits for a leak-free experiment.",
                    C.SPLITS_DIR, len(train_ids), len(val_ids), len(test_ids))

    if args.limit_patients:
        train_ids = train_ids[:args.limit_patients]
        val_ids   = val_ids[:max(1, args.limit_patients // 2)]
        test_ids  = test_ids[:max(1, args.limit_patients // 2)]
        log.info("SMOKE: limiting to %d train / %d val / %d test patients",
                 len(train_ids), len(val_ids), len(test_ids))

    split_of = {pid: "train" for pid in train_ids}
    split_of.update({pid: "val" for pid in val_ids})
    split_of.update({pid: "test" for pid in test_ids})

    manifests: dict[str, list[str]] = {
        "vae_train": [], "vae_val": [], "healthy": [],
        "val_healthy": [], "test_anom": [],
    }
    n_lesion = n_healthy = 0

    all_pids = train_ids + val_ids + test_ids
    for pid in tqdm(all_pids, desc="Patients", unit="pat"):
        split = split_of[pid]
        recs = process_patient(pid, C.SLICES_DIR, C.MASKS_DIR, args.min_brain_frac)
        for r in recs:
            stem, has_les, dist = r["stem"], r["has_lesion"], r["dist"]
            n_lesion += int(has_les)
            n_healthy += int(not has_les)

            # VAE codec trains on every train+val slice (lesions included).
            if split in ("train", "val"):
                manifests["vae_train"].append(stem)
            if split == "val":
                manifests["vae_val"].append(stem)

            # Diffusion healthy manifold: lesion-free + buffer-clean, train only.
            if split == "train" and (not has_les) and dist >= C.HEALTHY_BUFFER:
                manifests["healthy"].append(stem)
            # Calibration: lesion-free val slices.
            if split == "val" and (not has_les):
                manifests["val_healthy"].append(stem)
            # Evaluation: lesion slices from test patients.
            if split == "test" and has_les:
                manifests["test_anom"].append(stem)

    paths = {
        "vae_train": C.MANIFEST_VAE_TRAIN, "vae_val": C.MANIFEST_VAE_VAL,
        "healthy": C.MANIFEST_HEALTHY, "val_healthy": C.MANIFEST_VAL_HEALTHY,
        "test_anom": C.MANIFEST_TEST_ANOM,
    }
    for name, stems in manifests.items():
        with open(paths[name], "w") as f:
            f.write("\n".join(sorted(stems)) + ("\n" if stems else ""))

    stats = {
        "n_patients": len(all_pids),
        "n_train_patients": len(train_ids),
        "n_val_patients": len(val_ids),
        "n_test_patients": len(test_ids),
        "n_slices_total": n_lesion + n_healthy,
        "n_lesion_slices": n_lesion,
        "n_healthy_slices": n_healthy,
        "manifest_counts": {k: len(v) for k, v in manifests.items()},
        "channels": C.MODALITIES,
        "target_size": C.TARGET_SIZE,
        "healthy_buffer": C.HEALTHY_BUFFER,
        "norm_pct": [C.NORM_PCT_LOW, C.NORM_PCT_HIGH],
    }
    with open(C.STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    log.info("=" * 60)
    log.info("DONE — %d slices (%d lesion / %d healthy) from %d patients",
             stats["n_slices_total"], n_lesion, n_healthy, len(all_pids))
    for k, v in stats["manifest_counts"].items():
        log.info("  view %-12s : %d slices", k, v)
    log.info("  slices  → %s", C.SLICES_DIR)
    log.info("  masks   → %s", C.MASKS_DIR)
    log.info("  stats   → %s", C.STATS_PATH)
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        p = argparse.ArgumentParser(description="Stage 0 — BraTS NIfTI → 2D slices")
        p.add_argument("--limit-patients", type=int, default=0,
                       help="Process only the first N patients (local smoke test).")
        p.add_argument("--min-brain-frac", type=float, default=C.MIN_BRAIN_FRAC,
                       help="Skip slices with brain area below this fraction.")
        main(p.parse_args())
    except Exception as e:
        logging.exception("Fatal error in make_slices: %s", e)
        sys.exit(1)
