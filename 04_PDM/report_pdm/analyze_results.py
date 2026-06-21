#!/usr/bin/env python
"""
Turn PDM training/inference outputs into report figures and numbers.
====================================================================

Run this AFTER train -> calibrate -> evaluate (-> explain) have produced outputs.
It is read-only w.r.t. the model; it only parses logs + metrics.json and writes
PNGs into report_pdm/figures/.

Produces:
  * fig_training_curve.png   — train vs. val noise-prediction MSE + LR
  * fig_metric_distribution.png — per-slice AUROC / Dice spread
  * fig_operating_points.png — Dice at calibrated / best-global / oracle thresholds
  * prints a copy-paste results table (means ± std).

Usage
-----
    python report_pdm/analyze_results.py \
        --outputs /content/drive/MyDrive/pdm/outputs        # Colab
    python report_pdm/analyze_results.py                    # auto-detect locally
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# Matches: "Epoch 12/250 | train 0.09432 | val 0.25559 | lr 5.60e-05 | ..."
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s*\|\s*train\s+([\d.]+)\s*\|\s*val\s+([\d.]+)\s*\|\s*lr\s+([\d.eE+-]+)"
)


def _autodetect_outputs(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    for cand in (
        os.environ.get("PDM_OUTPUT_ROOT"),
        "/content/drive/MyDrive/pdm/outputs",
        str(HERE.parent / "outputs"),
    ):
        if cand and Path(cand).exists():
            return Path(cand)
    raise SystemExit("No outputs dir found. Pass --outputs /path/to/outputs")


def training_curve(out: Path) -> None:
    logs = sorted(glob.glob(str(out / "logs" / "train_*.log")), key=os.path.getmtime)
    if not logs:
        print("[skip] no train_*.log under", out / "logs")
        return
    ep, tr, va, lr = [], [], [], []
    for line in Path(logs[-1]).read_text(errors="ignore").splitlines():
        m = EPOCH_RE.search(line)
        if m:
            ep.append(int(m[1])); tr.append(float(m[2])); va.append(float(m[3])); lr.append(float(m[4]))
    if not ep:
        print("[skip] no epoch lines parsed from", logs[-1]); return
    best = int(np.argmin(va))
    fig, ax1 = plt.subplots(figsize=(7.2, 4.3))
    ax1.plot(ep, tr, "o-", color="#1f4e79", lw=2, ms=4, label=r"Train MSE ($\epsilon$-loss)")
    ax1.plot(ep, va, "s-", color="#c0392b", lw=2, ms=4, label="Val denoise MSE (EMA)")
    ax1.axvline(ep[best], color="grey", ls="--", lw=1.1)
    ax1.annotate(f"best val={va[best]:.4f}\n(epoch {ep[best]})",
                 xy=(ep[best], va[best]), xytext=(ep[best] + 0.4, va[best] + 0.04),
                 fontsize=9, arrowprops=dict(arrowstyle="->", color="grey"))
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Noise-prediction MSE"); ax1.grid(alpha=0.3)
    ax2 = ax1.twinx(); ax2.plot(ep, lr, ":", color="#27ae60", lw=1.6, label="Learning rate")
    ax2.set_ylabel("Learning rate", color="#27ae60"); ax2.tick_params(axis="y", labelcolor="#27ae60")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "fig_training_curve.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] fig_training_curve.png  ({len(ep)} epochs, best val {va[best]:.4f} @ ep {ep[best]})")


def metrics_figures(out: Path) -> None:
    mpath = out / "results" / "metrics.json"
    if not mpath.exists():
        print("[skip] no metrics.json at", mpath); return
    s = json.loads(mpath.read_text())
    per = s.get("per_image", [])
    aurocs = [p["auroc"] for p in per if p.get("auroc") == p.get("auroc")]  # drop NaN
    dices = [p["dice"] for p in per if p.get("dice") == p.get("dice")]

    # ---- results table (copy-paste) ----
    print("\n================ PDM RESULTS (paste into report) ================")
    print(f"  n_images              : {s.get('n_images')}")
    print(f"  AUROC  (mean ± std)   : {s.get('auroc_mean'):.4f} ± {s.get('auroc_std'):.4f}")
    print(f"  AUPRC  (mean)         : {s.get('auprc_mean'):.4f}")
    print(f"  Dice @ calibrated thr : {s.get('dice_calibrated'):.4f}")
    print(f"  Dice @ best-global    : {s.get('dice_best_global'):.4f}  (thr={s.get('best_global_threshold'):.4f})")
    print(f"  Dice oracle (ceiling) : {s.get('dice_oracle'):.4f}")
    print("=================================================================\n")

    # ---- per-slice metric distribution ----
    if aurocs and dices:
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        for a, data, name, col in ((ax[0], aurocs, "AUROC", "#1f4e79"),
                                   (ax[1], dices, "Dice (calibrated)", "#c0392b")):
            a.hist(data, bins=15, color=col, alpha=0.55, edgecolor="black", lw=0.4)
            a.axvline(float(np.mean(data)), color="black", ls="--",
                      label=f"mean={np.mean(data):.3f}")
            a.set_title(f"Per-slice {name}  (n={len(data)})"); a.set_xlabel(name)
            a.set_ylabel("slices"); a.legend(fontsize=9); a.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(FIGS / "fig_metric_distribution.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        print("[ok] fig_metric_distribution.png")

    # ---- operating-point bar ----
    pts = [("Calibrated\n(95th pct)", s.get("dice_calibrated")),
           ("Best-global", s.get("dice_best_global")),
           ("Oracle\n(ceiling)", s.get("dice_oracle"))]
    pts = [(k, v) for k, v in pts if v is not None and v == v]
    if pts:
        fig, ax = plt.subplots(figsize=(5.2, 4))
        labels, vals = zip(*pts)
        bars = ax.bar(labels, vals, color=["#c0392b", "#d97706", "#27ae60"][: len(pts)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
        ax.set_ylabel("Dice"); ax.set_title("Dice by operating point")
        ax.set_ylim(0, max(vals) * 1.25 + 1e-3); ax.grid(axis="y", alpha=0.25)
        fig.tight_layout(); fig.savefig(FIGS / "fig_operating_points.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        print("[ok] fig_operating_points.png")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build PDM report figures from outputs")
    ap.add_argument("--outputs", default=None, help="path to pdm outputs dir")
    args = ap.parse_args()
    out = _autodetect_outputs(args.outputs)
    print("Reading outputs from:", out)
    training_curve(out)
    metrics_figures(out)
    print("\nFigures written to:", FIGS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
