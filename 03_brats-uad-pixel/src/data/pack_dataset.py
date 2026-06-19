"""
Pack the processed dataset into ONE archive for Colab/Drive upload.
===================================================================
Bundles only the files actually needed for train / val / test:

  * every slice referenced by any manifest (drops test-patient *healthy*
    slices that no view uses — pure upload savings)
  * the masks for those slices (lesion slices only)
  * all manifest files + stats.json

Uploading one big tar (instead of ~40 000 tiny .npy files) is the difference
between a working Colab session and one that spends its whole runtime on Drive
I/O. In Colab you copy this tar to local disk and extract there — fast reads.

Usage
-----
    python src/data/pack_dataset.py                      # → <processed>/../brats_processed.tar
    python src/data/pack_dataset.py --out /tmp/ds.tar
"""

from __future__ import annotations

import argparse
import logging
import sys
import tarfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config as C

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def referenced_stems() -> set[str]:
    manifests = [C.MANIFEST_VAE_TRAIN, C.MANIFEST_VAE_VAL, C.MANIFEST_HEALTHY,
                 C.MANIFEST_VAL_HEALTHY, C.MANIFEST_TEST_ANOM]
    stems: set[str] = set()
    for m in manifests:
        if m.exists():
            with open(m) as f:
                stems.update(ln.strip() for ln in f if ln.strip())
    return stems


def main(args):
    out = Path(args.out) if args.out else (C.PROCESSED_DIR.parent / "brats_processed.tar")
    out.parent.mkdir(parents=True, exist_ok=True)

    stems = referenced_stems()
    if not stems:
        log.error("No manifests in %s — run make_slices.py first.", C.MANIFEST_DIR)
        sys.exit(1)

    n_slices = n_masks = 0
    missing = 0
    arcroot = "processed"          # extract → <dest>/processed/{slices,masks,manifests}
    log.info("Packing %d referenced slices → %s", len(stems), out)
    with tarfile.open(out, "w") as tar:
        for stem in sorted(stems):
            sp = C.SLICES_DIR / f"{stem}.npy"
            if sp.exists():
                tar.add(sp, arcname=f"{arcroot}/slices/{stem}.npy")
                n_slices += 1
            else:
                missing += 1
            mp = C.MASKS_DIR / f"{stem}.npy"
            if mp.exists():
                tar.add(mp, arcname=f"{arcroot}/masks/{stem}.npy")
                n_masks += 1
        # manifests + stats
        for m in C.MANIFEST_DIR.glob("*.txt"):
            tar.add(m, arcname=f"{arcroot}/manifests/{m.name}")
        if C.STATS_PATH.exists():
            tar.add(C.STATS_PATH, arcname=f"{arcroot}/stats.json")

    size_gb = out.stat().st_size / 1e9
    log.info("=" * 60)
    log.info("DONE — %s", out)
    log.info("  slices %d | masks %d | missing %d | size %.2f GB",
             n_slices, n_masks, missing, size_gb)
    log.info("  Upload this single file to Drive, then in Colab:")
    log.info("    tar -xf %s -C /content/   →  /content/processed/", out.name)
    log.info("    export BUAD_PROCESSED_DIR=/content/processed")
    log.info("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pack processed dataset into one tar")
    p.add_argument("--out", type=str, default=None, help="Output .tar path.")
    main(p.parse_args())
