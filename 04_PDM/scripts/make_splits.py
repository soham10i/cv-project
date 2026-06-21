#!/usr/bin/env python
"""
Make reproducible patient-level train/val/test splits.
======================================================

Scans the raw dataset for valid patient folders (those containing all required
modalities + segmentation), shuffles with a fixed seed, and writes
splits/{train,val,test}.txt. Patient-level (never slice-level) so no patient
appears in two sets — this is what keeps the reported metrics honest.

Note: BraTS-PED public labels exist only for the "Training" release, so we carve
our own held-out val/test from it. State this in the report.

Usage
-----
    python scripts/make_splits.py --dry-run                 # just print counts
    python scripts/make_splits.py --ratio 70 15 15          # write the files
    python scripts/make_splits.py --ratio 70 15 15 --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from _bootstrap import init

from src.config import CONFIG


def _is_valid_patient(d: Path) -> bool:
    """A patient folder is valid iff it has every modality + a segmentation."""
    if not d.is_dir():
        return False
    pid = d.name
    needed = [f"{pid}-{m}.nii.gz" for m in CONFIG.data.modalities]
    needed.append(f"{pid}-seg.nii.gz")
    return all((d / f).exists() for f in needed)


def main() -> int:
    p = argparse.ArgumentParser(description="Generate patient-level splits")
    p.add_argument("--ratio", type=int, nargs=3, default=[70, 15, 15],
                   metavar=("TRAIN", "VAL", "TEST"), help="percentages, sum≈100")
    p.add_argument("--seed", type=int, default=CONFIG.train.seed)
    p.add_argument("--out", type=Path, default=Path("splits"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    log, _ = init("make_splits")
    root = CONFIG.paths.data_root
    if not root.exists():
        log.error("Raw data root not found: %s (set PDM_DATA_ROOT)", root)
        return 1

    patients = sorted(d.name for d in root.iterdir() if _is_valid_patient(d))
    n_total = len(patients)
    if n_total == 0:
        log.error("No valid patient folders under %s", root)
        return 1
    log.info("Found %d valid patients under %s", n_total, root)

    random.Random(args.seed).shuffle(patients)
    tr, va, _ = args.ratio
    n_tr = round(n_total * tr / 100)
    n_va = round(n_total * va / 100)
    splits = {
        "train": patients[:n_tr],
        "val": patients[n_tr : n_tr + n_va],
        "test": patients[n_tr + n_va :],
    }
    for name, ids in splits.items():
        log.info("  %-5s : %d patients", name, len(ids))

    if args.dry_run:
        log.info("Dry run — no files written.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (args.out / f"{name}.txt").write_text("\n".join(ids) + "\n")
    log.info("Wrote splits to %s/ (seed=%d). Next: scripts/00_preprocess.py", args.out, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
