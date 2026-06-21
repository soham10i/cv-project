#!/usr/bin/env python
"""
Faithful XAI figures for the PDM report (offline, from saved arrays).
====================================================================

The Grad-CAM map is uninformative for a generative denoiser (no class logit; a
dense-regression target has no localizing gradient). The faithful explainers for
diffusion UAD are intrinsic to the decision:

  * Counterfactual generation  -- the healthy reconstruction \hat{x} IS the
    quantity the model thresholds; |x - \hat{x}| is the explanation, and \hat{x}
    is the projection onto the healthy manifold/prototype.
  * Residual attribution       -- the score is an EXACT additive sum over
    modalities (incl. CE); attribution is therefore exact, not approximate.

This script regenerates both from the .npz arrays dumped by fig_failure_slices.py
(keys: orig[4], recon[4], residual[5], score, gt, mask) -- no model/GPU needed.

Writes:
  r-counterfactual-success.png / r-counterfactual-failure.png
  r-residual-attribution.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CONFIG                       # noqa: E402
from src.scoring.residual import _channel_weights   # noqa: E402

FIGS = Path(__file__).resolve().parent / "figures"
ARR = FIGS / "arrays"
MODS = list(CONFIG.data.modalities) + (["CE"] if CONFIG.scoring.use_ce_channel else [])

SUCCESS = "BraTS-PED-00002-000_z094"
FAILURE = "BraTS-PED-00216-000_z037"


def _load(stem: str):
    d = np.load(ARR / f"{stem}.npz")
    return {k: d[k] for k in d.files}


def counterfactual_panel(stem: str, tag: str) -> None:
    """[input T1c | healthy counterfactual (prototype) | |x-xhat| | GT contour]."""
    a = _load(stem)
    orig, recon, gt = a["orig"], a["recon"], (a["gt"] > 0).astype(float)
    t1c_o, t1c_r = orig[1], recon[1]
    diff = np.abs(orig - recon).mean(0) * (a["mask"] > 0)
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
    ax[0].imshow(t1c_o, cmap="gray"); ax[0].set_title("Input $x$ (T1c)")
    ax[1].imshow(t1c_r, cmap="gray"); ax[1].set_title(r"Healthy counterfactual $\hat{x}$ (prototype)")
    im = ax[2].imshow(diff, cmap="hot"); ax[2].set_title(r"Counterfactual diff $|x-\hat{x}|$")
    plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    ax[3].imshow(t1c_o, cmap="gray")
    if gt.sum() > 0:
        ax[3].contour(gt, levels=[0.5], colors="#39ff14", linewidths=1.4)
    ax[3].set_title("Input + GT lesion (green)")
    for x in ax:
        x.axis("off")
    fig.suptitle(f"[{tag}] {stem} | counterfactual explanation (faithful: $\\hat{{x}}$ is the decision variable)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / f"r-counterfactual-{tag.lower()}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] r-counterfactual-{tag.lower()}.png")


def residual_attribution() -> None:
    """Per-modality weighted-residual contribution (sum over brain): success vs failure."""
    w = _channel_weights()                                  # (K,) normalized
    out = {}
    for stem, tag in ((SUCCESS, "success"), (FAILURE, "failure")):
        a = _load(stem)
        stack, mask = a["residual"], (a["mask"] > 0).astype(np.float32)
        contrib = np.array([(w[i] * stack[i] * mask).sum() for i in range(len(MODS))])
        out[tag] = contrib / (contrib.sum() + 1e-8)         # fraction of total score
    x = np.arange(len(MODS)); bw = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.bar(x - bw / 2, out["success"], bw, color="#27ae60", label=f"success ({SUCCESS[-4:]})")
    ax.bar(x + bw / 2, out["failure"], bw, color="#c0392b", label=f"failure ({FAILURE[-4:]})")
    ax.set_xticks(x); ax.set_xticklabels(MODS)
    ax.set_ylabel("fraction of total anomaly score")
    ax.set_title("Residual attribution by modality (exact additive decomposition)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    dom_s = MODS[int(np.argmax(out["success"]))]; dom_f = MODS[int(np.argmax(out["failure"]))]
    ax.text(0.02, 0.95, f"dominant: success={dom_s}, failure={dom_f}",
            transform=ax.transAxes, fontsize=9, va="top")
    fig.tight_layout()
    fig.savefig(FIGS / "r-residual-attribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] r-residual-attribution.png | dominant success={dom_s} failure={dom_f}")
    for tag in ("success", "failure"):
        print(f"     {tag}: " + ", ".join(f"{m}={v:.2f}" for m, v in zip(MODS, out[tag])))


def main() -> int:
    if not ARR.exists():
        raise SystemExit(f"No arrays dir at {ARR}; run fig_failure_slices.py --save-npy first.")
    counterfactual_panel(SUCCESS, "SUCCESS")
    counterfactual_panel(FAILURE, "FAILURE")
    residual_attribution()
    print("Figures ->", FIGS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
