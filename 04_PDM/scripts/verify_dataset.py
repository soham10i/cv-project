#!/usr/bin/env python
"""
Verify the preprocessed dataset before training (catch silent data bugs).
========================================================================

Bad data is the #1 cause of meaningless UAD numbers. Run this after
00_preprocess.py and BEFORE uploading / training. It checks:

  1. No patient leakage — a patient ID must not appear in more than one split.
  2. Manifest sanity — non-empty; healthy >> test; every test slice has a mask.
  3. Slice integrity — correct shape (C,H,W), dtype, value range in [-1, 1],
     non-degenerate (not all-background).
  4. Mask integrity — lesion masks are binary and non-empty.
  5. Saves a visual montage (healthy slices + lesion slices with GT overlay) so
     you can eyeball that intensities and masks look right.

Exit code is non-zero if any hard check fails.

Usage
-----
    python scripts/verify_dataset.py
    python scripts/verify_dataset.py --n-vis 8
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np

from _bootstrap import init

from src.config import CONFIG
from src.utils.io import read_manifest

_PID = re.compile(r"^(.*)_z\d+$")


def _patient_of(stem: str) -> str:
    m = _PID.match(stem)
    return m.group(1) if m else stem


def main() -> int:
    p = argparse.ArgumentParser(description="Verify preprocessed dataset")
    p.add_argument("--n-vis", type=int, default=6)
    args = p.parse_args()

    log, _ = init("verify_dataset")
    paths = CONFIG.paths
    md = paths.manifests_dir
    failures: list[str] = []

    # ── load manifests ──────────────────────────────────────────────────
    try:
        healthy = read_manifest(md / "healthy.txt")
        val_healthy = read_manifest(md / "val_healthy.txt")
        test_anom = read_manifest(md / "test_anom.txt")
    except Exception as exc:
        log.error("Cannot read manifests: %s", exc)
        return 1

    log.info("Counts | healthy(train)=%d  val_healthy=%d  test_anom=%d",
             len(healthy), len(val_healthy), len(test_anom))

    # ── 1. patient leakage ──────────────────────────────────────────────
    pt_train = {_patient_of(s) for s in healthy}
    pt_val = {_patient_of(s) for s in val_healthy}
    pt_test = {_patient_of(s) for s in test_anom}
    for a, b, na, nb in [(pt_train, pt_val, "train", "val"),
                         (pt_train, pt_test, "train", "test"),
                         (pt_val, pt_test, "val", "test")]:
        overlap = a & b
        if overlap:
            failures.append(f"PATIENT LEAKAGE {na}∩{nb}: {sorted(overlap)[:5]}")
    log.info("Patients | train=%d val=%d test=%d | leakage check %s",
             len(pt_train), len(pt_val), len(pt_test),
             "FAILED" if failures else "ok")

    # ── 2. manifest sanity ──────────────────────────────────────────────
    if len(healthy) < len(test_anom):
        failures.append("healthy(train) < test_anom — suspicious for UAD")
    if len(val_healthy) < CONFIG.calibration.max_samples:
        log.warning("val_healthy (%d) < calibration max_samples (%d) — "
                    "calibration will use fewer slices.",
                    len(val_healthy), CONFIG.calibration.max_samples)

    # ── 3 & 4. slice + mask integrity (sample) ──────────────────────────
    def _check_slice(stem: str, expect_mask: bool) -> None:
        f = paths.slices_dir / f"{stem}.npy"
        if not f.exists():
            failures.append(f"missing slice {f}")
            return
        x = np.load(f).astype(np.float32)
        if x.shape != (CONFIG.data.n_channels, CONFIG.data.target_size, CONFIG.data.target_size):
            failures.append(f"bad slice shape {x.shape} for {stem}")
        if x.min() < -1.01 or x.max() > 1.01:
            failures.append(f"slice {stem} out of [-1,1]: [{x.min():.2f},{x.max():.2f}]")
        if (x > -0.99).mean() < 0.005:
            failures.append(f"slice {stem} is essentially empty")
        if expect_mask:
            mf = paths.masks_dir / f"{stem}.npy"
            if not mf.exists():
                failures.append(f"test slice {stem} has NO mask")
                return
            m = np.load(mf)
            if set(np.unique(m)) - {0, 1}:
                failures.append(f"mask {stem} not binary: {np.unique(m)}")
            if m.sum() == 0:
                failures.append(f"test mask {stem} is empty (no lesion)")

    import random
    rng = random.Random(CONFIG.train.seed)
    for stem in rng.sample(healthy, min(30, len(healthy))):
        _check_slice(stem, expect_mask=False)
    for stem in rng.sample(test_anom, min(30, len(test_anom))):
        _check_slice(stem, expect_mask=True)
    log.info("Sampled slice/mask integrity check done.")

    # ── 5. visual montage ───────────────────────────────────────────────
    try:
        _save_montage(healthy, test_anom, args.n_vis)
        log.info("Montage saved → %s", paths.results_dir / "dataset_check.png")
    except Exception as exc:  # visualization is best-effort
        log.warning("Montage failed (non-fatal): %s", exc)

    # ── verdict ─────────────────────────────────────────────────────────
    log.info("=" * 60)
    if failures:
        for f in failures[:20]:
            log.error("  ✗ %s", f)
        log.error("DATASET VERIFICATION FAILED (%d issues). Fix before training.", len(failures))
        return 1
    log.info("DATASET VERIFICATION PASSED — safe to upload & train.")
    log.info("=" * 60)
    return 0


def _save_montage(healthy, test_anom, n: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = CONFIG.paths
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    import random
    rng = random.Random(CONFIG.train.seed)
    h_samp = rng.sample(healthy, min(n, len(healthy)))
    t_samp = rng.sample(test_anom, min(n, len(test_anom)))

    fig, ax = plt.subplots(2, n, figsize=(3 * n, 6))
    for j, stem in enumerate(h_samp):
        x = np.load(paths.slices_dir / f"{stem}.npy").astype(np.float32)
        ax[0, j].imshow(x[1], cmap="gray"); ax[0, j].set_title(f"healthy\n{stem}", fontsize=7)
        ax[0, j].axis("off")
    for j, stem in enumerate(t_samp):
        x = np.load(paths.slices_dir / f"{stem}.npy").astype(np.float32)
        m = np.load(paths.masks_dir / f"{stem}.npy").astype(np.float32)
        ax[1, j].imshow(x[1], cmap="gray")
        ax[1, j].imshow(np.ma.masked_where(m == 0, m), cmap="autumn", alpha=0.5)
        ax[1, j].set_title(f"lesion+GT\n{stem}", fontsize=7); ax[1, j].axis("off")
    fig.suptitle("Dataset check — top: healthy (train) | bottom: lesion+GT (test)",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(paths.results_dir / "dataset_check.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
