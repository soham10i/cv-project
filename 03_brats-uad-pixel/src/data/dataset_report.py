"""
Dataset audit — what feeds the VAE, the diffusion model, and evaluation.
=========================================================================
Reads the processed manifests + masks and reports, per dataset view:

  * slice counts and how many distinct patients contribute
  * lesion prevalence (per-voxel) where masks exist
  * a hard **leakage check** — the patient sets behind the diffusion-train,
    calibration, and test views must be mutually disjoint
  * per-patient slice-count distribution (spot data imbalance)

Run after make_slices.py. Exits non-zero if a leakage check fails.

Usage
-----
    python src/data/dataset_report.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config as C

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def stems(manifest: Path) -> list[str]:
    if not manifest.exists():
        return []
    with open(manifest) as f:
        return [ln.strip() for ln in f if ln.strip()]


def patients(stem_list: list[str]) -> set[str]:
    return {s.rsplit("_z", 1)[0] for s in stem_list}


def lesion_prevalence(stem_list: list[str], sample: int = 200) -> float:
    """Mean per-voxel lesion fraction over brain area (sampled)."""
    fracs = []
    for s in stem_list[:sample]:
        mp = C.MASKS_DIR / f"{s}.npy"
        if mp.exists():
            m = np.load(mp)
            fracs.append(float(m.sum()) / m.size)
    return float(np.mean(fracs)) if fracs else 0.0


def main():
    views = {
        "vae_train":   C.MANIFEST_VAE_TRAIN,
        "vae_val":     C.MANIFEST_VAE_VAL,
        "healthy":     C.MANIFEST_HEALTHY,
        "val_healthy": C.MANIFEST_VAL_HEALTHY,
        "test_anom":   C.MANIFEST_TEST_ANOM,
    }
    purpose = {
        "vae_train":   "VAE training (codec; all slices, train+val patients)",
        "vae_val":     "VAE recon monitoring (val patients)",
        "healthy":     "Diffusion training (lesion-free + buffer, TRAIN patients)",
        "val_healthy": "Calibration (lesion-free, VAL patients)",
        "test_anom":   "Evaluation (lesion slices, TEST patients)",
    }

    data = {k: stems(v) for k, v in views.items()}
    if not any(data.values()):
        log.error("No manifests found in %s — run make_slices.py first.", C.MANIFEST_DIR)
        sys.exit(1)

    log.info("=" * 78)
    log.info("DATASET REPORT  (%s)", C.PROCESSED_DIR)
    log.info("=" * 78)
    log.info("%-12s %8s %9s  %s", "view", "slices", "patients", "purpose")
    log.info("-" * 78)
    for k in views:
        log.info("%-12s %8d %9d  %s", k, len(data[k]), len(patients(data[k])), purpose[k])

    # Lesion prevalence on the test set (the evaluation target).
    prev = lesion_prevalence(data["test_anom"])
    log.info("-" * 78)
    log.info("Lesion prevalence (test, per-voxel of full slice): %.4f  "
             "(AUPRC above this = real signal)", prev)

    # Per-patient slice-count distribution for the diffusion training pool.
    h_pat = {}
    for s in data["healthy"]:
        h_pat[s.rsplit("_z", 1)[0]] = h_pat.get(s.rsplit("_z", 1)[0], 0) + 1
    if h_pat:
        counts = np.array(list(h_pat.values()))
        log.info("Healthy slices/patient: min %d | median %d | max %d | mean %.1f",
                 counts.min(), int(np.median(counts)), counts.max(), counts.mean())

    # ── Leakage checks (the important part) ──────────────────────────
    log.info("=" * 78)
    log.info("LEAKAGE CHECKS (patient-level disjointness)")
    train_p = patients(data["healthy"])
    val_p   = patients(data["val_healthy"])
    test_p  = patients(data["test_anom"])
    checks = {
        "train ∩ test":  train_p & test_p,
        "train ∩ val":   train_p & val_p,
        "val ∩ test":    val_p & test_p,
    }
    ok = True
    for name, overlap in checks.items():
        status = "PASS ✅" if not overlap else f"FAIL ❌ ({len(overlap)} shared)"
        log.info("  %-14s : %s", name, status)
        ok = ok and not overlap

    if (C.STATS_PATH).exists():
        with open(C.STATS_PATH) as f:
            st = json.load(f)
        log.info("-" * 78)
        log.info("From stats.json: %d patients | %d slices (%d lesion / %d healthy)",
                 st.get("n_patients", 0), st.get("n_slices_total", 0),
                 st.get("n_lesion_slices", 0), st.get("n_healthy_slices", 0))

    log.info("=" * 78)
    if not ok:
        log.error("LEAKAGE DETECTED — splits overlap. Fix splits before training.")
        sys.exit(2)
    log.info("All leakage checks passed — splits are patient-disjoint. ✅")


if __name__ == "__main__":
    main()
