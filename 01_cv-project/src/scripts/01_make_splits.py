"""
Patient-level train / val / test split generation
==================================================
Scans a BraTS raw directory for patient folders and writes three text files
(one patient ID per line) so that preprocessing, training, and evaluation
operate on non-overlapping patient sets — preventing data leakage.

Output
------
    splits/train.txt
    splits/val.txt
    splits/test.txt

Usage
-----
    python src/make_splits.py --raw-dir data/BraTS-PEDs-v1/Training \\
        --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2

    # Or with a fixed seed for reproducibility:
    python src/make_splits.py --seed 42
"""

import argparse
import logging
import random
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from core import constants as C

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def make_splits(
    raw_dir: Path,
    out_dir: Path,
    train_ratio: float = 0.6,
    val_ratio: float   = 0.2,
    test_ratio: float  = 0.2,
    seed: int          = C.SEED,
) -> dict[str, list[str]]:
    """
    Split patient folders into train/val/test by ratio, write text files.

    Returns a dict mapping split name → list of patient IDs.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio}+{val_ratio}+{test_ratio}"
            f"={train_ratio + val_ratio + test_ratio:.4f}"
        )

    patient_ids = sorted(
        p.name for p in raw_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if not patient_ids:
        raise FileNotFoundError(f"No patient folders found in {raw_dir}")

    log.info("Found %d patients in %s", len(patient_ids), raw_dir)

    rng = random.Random(seed)
    shuffled = patient_ids[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val   = int(round(n * val_ratio))
    n_test  = n - n_train - n_val

    splits = {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train : n_train + n_val],
        "test":  shuffled[n_train + n_val :],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        path = out_dir / f"{name}.txt"
        with open(path, "w") as f:
            f.write("\n".join(ids) + "\n")
        log.info("  %-6s: %3d patients → %s", name, len(ids), path)

    log.info("Splits written to %s", out_dir)
    return splits


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Generate patient-level train/val/test split files",
        )
        parser.add_argument("--raw-dir",      type=Path,  default=C.RAW_DATA_DIR,
                            help="Path to BraTS-PEDs Training folder.")
        parser.add_argument("--out-dir",      type=Path,  default=C.SPLITS_DIR,
                            help="Output directory for split txt files (default: splits/).")
        parser.add_argument("--train-ratio",  type=float, default=0.6,
                            help="Fraction of patients for training (default: 0.6).")
        parser.add_argument("--val-ratio",    type=float, default=0.2,
                            help="Fraction for validation (default: 0.2).")
        parser.add_argument("--test-ratio",   type=float, default=0.2,
                            help="Fraction for test (default: 0.2).")
        parser.add_argument("--seed",         type=int,   default=C.SEED,
                            help=f"Random seed (default: {C.SEED}).")
        args = parser.parse_args()

        make_splits(
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    except Exception as e:
        log.exception(f"Fatal error in make_splits: {e}")
        sys.exit(1)

