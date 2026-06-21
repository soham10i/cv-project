"""
XAI visualization: assemble explanation panels.
===============================================

Two figures per explained case:
  * a counterfactual + attribution panel (original, healthy counterfactual,
    counterfactual difference, per-modality attribution bars, per-scale maps),
  * a healing-trajectory strip (DDIM frames in-painting the lesion).

These make the model's decision auditable: a clinician sees *what* was changed,
*on which modality*, and *at what spatial scale*.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import CONFIG
from ..scoring.multiscale import ScoreResult
from .attribution import dominant_modality, modality_attribution, scale_attribution
from .counterfactual import CounterfactualResult


def save_explanation_panel(
    name: str,
    result: ScoreResult,
    gt: np.ndarray,
    out_dir: Path,
) -> str:
    """Counterfactual + attribution figure for one slice. Returns dominant modality."""
    out_dir.mkdir(parents=True, exist_ok=True)
    attrib = modality_attribution(result.orig, result.recon)
    scales = scale_attribution(result)
    top_mod, totals = dominant_modality(result.orig, result.recon)

    n_scale = len(scales)
    fig, ax = plt.subplots(2, max(4, n_scale + 1), figsize=(4 * max(4, n_scale + 1), 8))

    ax[0, 0].imshow(result.orig[1], cmap="gray"); ax[0, 0].set_title("Original (T1c)")
    ax[0, 1].imshow(result.recon[1], cmap="gray"); ax[0, 1].set_title("Counterfactual (healthy)")
    diff = np.abs(result.orig[1] - result.recon[1])
    ax[0, 2].imshow(diff, cmap="hot"); ax[0, 2].set_title("CF difference (T1c)")
    ax[0, 3].imshow(gt, cmap="gray"); ax[0, 3].set_title("Ground truth")
    for j in range(4, ax.shape[1]):
        ax[0, j].axis("off")

    # Bottom row: per-noise-scale attribution maps.
    for j, (t, m) in enumerate(scales.items()):
        ax[1, j].imshow(m, cmap="hot"); ax[1, j].set_title(f"Scale T={t}")
    # Last bottom cell: modality contribution bar chart.
    bar_ax = ax[1, ax.shape[1] - 1]
    names = list(totals.keys())
    vals = [totals[k] for k in names]
    bar_ax.barh(names, vals, color="#d97706")
    bar_ax.set_title(f"Modality attribution\n(dominant: {top_mod})")

    for a in ax[0, :4]:
        a.axis("off")
    for j in range(len(scales)):
        ax[1, j].axis("off")

    fig.suptitle(f"{name} | Counterfactual explanation | dominant modality: {top_mod}",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / f"xai_{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return top_mod


def save_gradcam_panel(
    name: str,
    result: ScoreResult,
    gradcam: np.ndarray,
    gt: np.ndarray,
    out_dir: Path,
) -> None:
    """Grad-CAM 'model-focus' figure: where the network looked vs. what it scored.

    Columns: original T1c, fused anomaly score, Grad-CAM model-focus map, the
    Grad-CAM overlaid on the anatomy, and the ground-truth lesion mask.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.5))
    t1c = result.orig[1]
    ax[0].imshow(t1c, cmap="gray"); ax[0].set_title("Original (T1c)")
    im1 = ax[1].imshow(result.score, cmap="hot"); ax[1].set_title("Anomaly score (fused)")
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    im2 = ax[2].imshow(gradcam, cmap="jet"); ax[2].set_title("Grad-CAM (model focus)")
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    ax[3].imshow(t1c, cmap="gray")
    ax[3].imshow(gradcam, cmap="jet", alpha=0.45); ax[3].set_title("Grad-CAM overlay")
    ax[4].imshow(gt, cmap="gray"); ax[4].set_title("Ground truth")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{name} | Grad-CAM model-focus at inference", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / f"gradcam_{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_trajectory_strip(name: str, cf: CounterfactualResult, out_dir: Path) -> None:
    """Healing-trajectory strip: DDIM frames in-painting the anomaly."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = cf.trajectory or [cf.counterfactual]
    n = len(frames) + 2
    fig, ax = plt.subplots(1, n, figsize=(3 * n, 3.2))
    ax[0].imshow(cf.original[1], cmap="gray"); ax[0].set_title("Input")
    for i, fr in enumerate(frames):
        ax[i + 1].imshow(fr[1], cmap="gray"); ax[i + 1].set_title(f"DDIM {i+1}")
    ax[-1].imshow(np.abs(cf.difference[1]), cmap="hot"); ax[-1].set_title("Healed Δ")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{name} | counterfactual healing trajectory", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / f"trajectory_{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
