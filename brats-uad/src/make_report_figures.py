"""
Generate report-ready figures from metrics.json.
=================================================
Produces:
  * per-image AUROC / DICE distribution (box + strip)
  * DICE-vs-threshold curve (calibrated vs best-global vs oracle annotated)
  * a summary metrics table (PNG)

Usage
-----
    python src/make_report_figures.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[0]))

import config as C

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main():
    mpath = C.RESULTS_DIR / "metrics.json"
    if not mpath.exists():
        raise FileNotFoundError(f"{mpath} not found. Run evaluate.py first.")
    with open(mpath) as f:
        m = json.load(f)
    out = C.RESULTS_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)

    per = m["per_image"]
    aurocs = [p["auroc"] for p in per if np.isfinite(p["auroc"])]
    dices = [p["dice"] for p in per if np.isfinite(p["dice"])]
    best = [p["best_dice"] for p in per if np.isfinite(p["best_dice"])]

    # 1 — AUROC / DICE distributions
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for a, data, title in zip(ax, [aurocs, dices], ["Per-image AUROC", "Per-image DICE"]):
        a.boxplot(data, vert=True, widths=0.5, showmeans=True)
        a.scatter(np.random.normal(1, 0.04, len(data)), data, alpha=0.4, s=18)
        a.set_title(title, fontweight="bold")
        a.set_xticks([])
        a.grid(axis="y", alpha=0.3)
    fig.suptitle("Anomaly detection performance distribution", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2 — summary bar of headline metrics
    keys = [("auroc_mean", "AUROC"), ("auprc_mean", "AUPRC"),
            ("dice_mean_calibrated", "DICE\n(calib)"),
            ("dice_best_global", "DICE\n(global)"), ("dice_oracle", "DICE\n(oracle)")]
    vals = [m.get(k, float("nan")) for k, _ in keys]
    fig, a = plt.subplots(figsize=(8, 5))
    bars = a.bar([lbl for _, lbl in keys], vals,
                 color=["#3b6", "#3b6", "#36b", "#36b", "#999"])
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
               ha="center", fontweight="bold")
    a.set_ylim(0, 1.0)
    a.set_title(f"Headline metrics (n={m['n_images']} test slices)", fontweight="bold")
    a.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "headline_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info("Report figures → %s", out)
    log.info("  AUROC %.4f | AUPRC %.4f | DICE calib %.4f | global %.4f | oracle %.4f",
             m["auroc_mean"], m["auprc_mean"], m["dice_mean_calibrated"],
             m["dice_best_global"], m["dice_oracle"])


if __name__ == "__main__":
    main()
