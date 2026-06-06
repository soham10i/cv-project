"""
MedSegDiff Phase 1 — Supervised Segmentation Dataset
=====================================================
Builds paired (image, binary-tumour-mask) 2D slices from ALL raw BraTS-PEDs
patients, with a **patient-level** train/val/test split (so slices from one
patient never span splits — no leakage).

For each patient:
  * every brain-containing axial slice with tumour (seg>0)  → POSITIVE sample
  * a capped random subset of healthy slices (empty mask)    → NEGATIVE sample
    (teaches the model to output empty masks on healthy tissue)

Output tree
-----------
    data/seg/
    ├── split.json                  # patient lists + counts
    ├── train/{images,masks}/*.npy
    ├── val/{images,masks}/*.npy
    └── test/{images,masks}/*.npy
Images are (3, S, S) float16 z-scored slices; masks are (S, S) uint8 {0,1}.

Usage
-----
    python src/preprocess_seg.py                       # all patients
    python src/preprocess_seg.py --max-patients 6      # smoke subset
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

import config as C
from preprocess_to_2d import zscore_normalize_slice, symmetric_pad, load_nifti, _resolve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def split_patients(patient_ids, fracs, seed):
    """Deterministic patient-level split into train/val/test."""
    pats = list(patient_ids)
    random.Random(seed).shuffle(pats)
    n = len(pats)
    n_tr = int(n * fracs[0])
    n_va = int(n * fracs[1])
    return {
        "train": sorted(pats[:n_tr]),
        "val":   sorted(pats[n_tr:n_tr + n_va]),
        "test":  sorted(pats[n_tr + n_va:]),
    }


def preprocess_seg(raw_dir: Path, out_dir: Path, neg_ratio: float,
                   max_patients: int | None, seed: int) -> None:
    patient_dirs = sorted(
        p for p in raw_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if max_patients:
        patient_dirs = patient_dirs[:max_patients]
    log.info("Found %d patient folders in %s", len(patient_dirs), raw_dir)

    split = split_patients([p.name for p in patient_dirs], C.SEG_SPLIT_FRACS, seed)
    pat2split = {pid: s for s, pids in split.items() for pid in pids}
    log.info("Patient split — train %d / val %d / test %d",
             len(split["train"]), len(split["val"]), len(split["test"]))

    for s in ("train", "val", "test"):
        (out_dir / s / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / s / "masks").mkdir(parents=True, exist_ok=True)

    counts = {s: {"pos": 0, "neg": 0, "patients": 0} for s in ("train", "val", "test")}
    rng = random.Random(seed)

    for patient_dir in tqdm(patient_dirs, desc="Patients", unit="pat"):
        pid = patient_dir.name
        s = pat2split[pid]
        counts[s]["patients"] += 1

        # ── load 3 modalities + seg ───────────────────────────────
        vols, skip = [], False
        for mod in C.MODALITIES:
            pth = _resolve(patient_dir, pid, mod)
            if pth is None:
                log.warning("Missing %s for %s — skipping.", mod, pid)
                skip = True
                break
            vols.append(load_nifti(pth))
        if skip:
            continue
        seg_path = _resolve(patient_dir, pid, C.SEG_SUFFIX)
        if seg_path is None:
            log.warning("Missing seg for %s — skipping.", pid)
            continue
        seg_vol = load_nifti(seg_path)

        depth = vols[0].shape[2]
        pos_idx, neg_idx, cache = [], [], {}

        for z in range(depth):
            mod_slices = [v[:, :, z] for v in vols]
            brain_any = np.maximum.reduce([m > 0 for m in mod_slices])
            if brain_any.sum() / mod_slices[0].size < C.MIN_BRAIN_FRAC:
                continue

            normed = np.stack([zscore_normalize_slice(m) for m in mod_slices], axis=0)
            stacked = symmetric_pad(normed, C.SEG_IMG_SIZE).astype(np.float16)
            binmask = symmetric_pad(
                (seg_vol[:, :, z] > 0).astype(np.uint8), C.SEG_IMG_SIZE)

            cache[z] = (stacked, binmask)
            (pos_idx if binmask.sum() > 0 else neg_idx).append(z)

        # ── subsample negatives ───────────────────────────────────
        keep_neg = set()
        if neg_ratio > 0 and neg_idx:
            target = int(round((len(pos_idx) or 6) * neg_ratio))
            k = min(len(neg_idx), target)
            if k > 0:
                keep_neg = set(rng.sample(neg_idx, k))

        for z in pos_idx + sorted(keep_neg):
            stacked, binmask = cache[z]
            name = f"{pid}_z{z:03d}.npy"
            np.save(out_dir / s / "images" / name, stacked)
            np.save(out_dir / s / "masks" / name, binmask)
            counts[s]["pos" if binmask.sum() > 0 else "neg"] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "split.json", "w") as f:
        json.dump({"fracs": C.SEG_SPLIT_FRACS, "seed": seed,
                   "img_size": C.SEG_IMG_SIZE, "neg_ratio": neg_ratio,
                   "counts": counts, "patients": split}, f, indent=2)

    log.info("=" * 60)
    for s in ("train", "val", "test"):
        c = counts[s]
        log.info("%-5s : %4d pos + %4d neg = %4d slices  (%d patients)",
                 s, c["pos"], c["neg"], c["pos"] + c["neg"], c["patients"])
    log.info("Output: %s  |  split.json written", out_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedSegDiff Phase 1 — paired seg dataset")
    parser.add_argument("--raw-dir", type=Path, default=C.RAW_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=C.SEG_DATA_DIR)
    parser.add_argument("--neg-ratio", type=float, default=C.SEG_NEG_RATIO,
                        help="Healthy (empty-mask) slices kept per tumour slice.")
    parser.add_argument("--max-patients", type=int, default=None,
                        help="Cap number of patients (for smoke tests).")
    parser.add_argument("--seed", type=int, default=C.SEED)
    args = parser.parse_args()

    preprocess_seg(args.raw_dir, args.out_dir, args.neg_ratio,
                   args.max_patients, args.seed)
