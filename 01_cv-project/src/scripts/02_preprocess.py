"""
Phase 2 — Preprocess BraTS volumes into 2D diffusion dataset
=============================================================
Converts 3D NIfTI files into 2D ``.npy`` slices suitable for training.
Applies volume-level z-score normalisation, filters empty/background slices,
and pads to the network's target size.

Usage
-----
    python src/preprocess.py  # Uses config.py defaults

For a split-specific run (e.g. creating val_healthy):
    python src/preprocess.py --split-file splits/val_patients.txt \\
                             --healthy-subdir val_healthy \\
                             --max-healthy 500 \\
                             --max-anomalous 0
"""

import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import logging
from core import constants as C
from data.preprocessing import preprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Preprocess BraTS volumes to 2D slices.")
        parser.add_argument("--raw-dir", type=Path, default=C.RAW_DATA_DIR)
        parser.add_argument("--out-dir", type=Path, default=C.PROCESSED_DIR)
        parser.add_argument("--split-file", type=Path, default=None,
                            help="Optional txt file with allowed patient IDs.")
        parser.add_argument("--healthy-subdir", type=str, default="train_healthy",
                            help="Subdir name for healthy slices.")
        parser.add_argument("--max-healthy", type=int, default=15000)
        parser.add_argument("--max-anomalous", type=int, default=1000)
        parser.add_argument("--max-healthy-per-pat", type=int, default=100)
        args = parser.parse_args()

        preprocess(
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            max_healthy=args.max_healthy,
            max_anomalous=args.max_anomalous,
            split_file=args.split_file,
            healthy_subdir=args.healthy_subdir,
            max_healthy_per_patient=args.max_healthy_per_pat,
        )
    except Exception as e:
        logging.exception(f"Fatal error in preprocess: {e}")
        sys.exit(1)
