#!/usr/bin/env python
"""
Stage 0 — Preprocess BraTS NIfTI volumes into normalized 2D slices + manifests.
===============================================================================

Usage
-----
    python scripts/00_preprocess.py --splits splits
    python scripts/00_preprocess.py --splits splits --limit-patients 4   # smoke

Reads patient-level splits from ``<splits>/{train,val,test}.txt``. Writes slices,
masks, manifests, and stats.json under PDM_PROCESSED_ROOT.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import init

from src.data.preprocessing import preprocess
from src.utils.exceptions import PDMError


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 0 — preprocessing")
    p.add_argument("--splits", type=Path, default=Path("splits"),
                   help="Directory with train.txt/val.txt/test.txt")
    p.add_argument("--limit-patients", type=int, default=None,
                   help="Process only the first N patients per split (smoke test)")
    args = p.parse_args()

    log, _ = init("preprocess")
    try:
        stats = preprocess(args.splits, limit_patients=args.limit_patients)
        log.info("Done. Healthy=%d  Lesion=%d",
                 stats["n_healthy_slices"], stats["n_lesion_slices"])
        return 0
    except PDMError as exc:
        log.error("Preprocessing failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
