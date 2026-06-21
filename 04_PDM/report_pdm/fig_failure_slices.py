#!/usr/bin/env python
"""
Per-slice failure diagnostics for the PDM report (run on Colab/GPU).
===================================================================

For a curated set of success + failure test slices, this re-scores each slice
with the trained model and produces the figures that explain *why* the calibrated
Dice is low, at the voxel/distribution level:

  fig_hist_<stem>.png        (1) healthy vs lesion score histogram + tau line
  fig_modres_<stem>.png      (3) per-modality (+CE) reconstruction residual maps
  fig_maskedge_<stem>.png    (4) brain-mask boundary vs GT (the edge artefact)
  fig_perscale_<stem>.png    (7) per-noise-scale score maps + per-scale Dice
  fig_roc_pr_<stem>.png      (8) ROC + PR curves for the slice
  fig_dice_vs_threshold.png  (2) mean Dice-vs-threshold over the processed slices,
                                 with calibrated tau and best-global tau* marked
Optionally dumps .npy arrays (--save-npy) for offline re-plotting.

Usage
-----
    python report_pdm/fig_failure_slices.py \
        --slices BraTS-PED-00002-000_z094,BraTS-PED-00002-000_z093,\
BraTS-PED-00216-000_z037,BraTS-PED-00076-000_z033 --save-npy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

# Make `src` importable whether run from repo root or report_pdm/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CONFIG
from src.data.dataset import AnomalousSliceDataset
from src.evaluation.metrics import dice_coefficient, dice_over_grid, pixel_auroc, pixel_auprc
from src.models.diffusion import DiffusionProcess
from src.models.unet import load_unet
from src.noise.factory import build_noise_strategy
from src.scoring.multiscale import MultiScaleScorer
from src.scoring.residual import residual_stack
from src.utils.device import get_device, set_seed

FIGS = Path(__file__).resolve().parent / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
MODS = list(CONFIG.data.modalities) + (["CE"] if CONFIG.scoring.use_ce_channel else [])

# Sensible defaults: 2 strong + 2 weak slices (edit to taste / availability).
DEFAULT_SLICES = [
    "BraTS-PED-00002-000_z094",   # success
    "BraTS-PED-00002-000_z093",   # success
    "BraTS-PED-00216-000_z037",   # failure (edge-rim)
    "BraTS-PED-00076-000_z033",   # failure (low Dice)
]


def load_threshold(cli_tau: float | None) -> float:
    if cli_tau is not None:
        return cli_tau
    p = CONFIG.paths.calibration_path
    if Path(p).exists():
        return float(json.loads(Path(p).read_text())["threshold"])
    print("[warn] no calibration file; defaulting tau=0.128")
    return 0.128


def fig_hist(stem, score, gt, mask, tau):
    brain = mask > 0
    healthy = score[brain & (gt == 0)]
    lesion = score[brain & (gt > 0)]
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, max(float(score[brain].max()), tau * 1.5), 60)
    ax.hist(healthy, bins=bins, density=True, alpha=0.6, color="#1f77b4", label=f"healthy ({healthy.size})")
    if lesion.size:
        ax.hist(lesion, bins=bins, density=True, alpha=0.6, color="#c0392b", label=f"lesion ({lesion.size})")
    ax.axvline(tau, color="black", ls="--", lw=1.5, label=f"calibrated $\\tau$={tau:.3f}")
    ax.set_yscale("log"); ax.set_xlabel("anomaly score"); ax.set_ylabel("density (log)")
    ax.set_title(f"{stem}  score distribution"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / f"fig_hist_{stem}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_modres(stem, orig, recon, score):
    stack = residual_stack(orig, recon)                 # (K,H,W)
    n = len(MODS)
    fig, ax = plt.subplots(1, n + 2, figsize=(3.1 * (n + 2), 3.2))
    ax[0].imshow(orig[1], cmap="gray"); ax[0].set_title("Original (T1c)")
    ax[1].imshow(recon[1], cmap="gray"); ax[1].set_title("Healthy recon (T1c)")
    for i, name in enumerate(MODS):
        a = ax[i + 2]; im = a.imshow(stack[i], cmap="hot"); a.set_title(f"|res| {name}")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{stem}  per-modality reconstruction residual", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGS / f"fig_modres_{stem}.png", dpi=140, bbox_inches="tight"); plt.close(fig)


def fig_maskedge(stem, orig, mask, gt, score):
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.4))
    ax[0].imshow(orig[1], cmap="gray")
    ax[0].contour(mask, levels=[0.5], colors="#27ae60", linewidths=1.2)
    if gt.sum() > 0:
        ax[0].contour(gt, levels=[0.5], colors="#c0392b", linewidths=1.2)
    ax[0].set_title("brain mask (green) vs GT (red)")
    im = ax[1].imshow(score, cmap="hot"); ax[1].contour(mask, levels=[0.5], colors="cyan", linewidths=0.8)
    ax[1].set_title("anomaly score + mask boundary")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{stem}  edge-artefact check", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGS / f"fig_maskedge_{stem}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_perscale(stem, res, gt, tau):
    ts = list(res.per_scale.keys())
    fig, ax = plt.subplots(1, len(ts) + 1, figsize=(3.4 * (len(ts) + 1), 3.6))
    for i, t in enumerate(ts):
        m = res.per_scale[t]
        d = dice_coefficient((m > tau).astype(np.float32), gt)
        ax[i].imshow(m, cmap="hot"); ax[i].set_title(f"T={t}  Dice@$\\tau$={d:.3f}")
    df = dice_coefficient((res.score > tau).astype(np.float32), gt)
    ax[-1].imshow(res.score, cmap="hot"); ax[-1].set_title(f"fused  Dice@$\\tau$={df:.3f}")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{stem}  per-noise-scale contribution", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGS / f"fig_perscale_{stem}.png", dpi=140, bbox_inches="tight"); plt.close(fig)


def fig_roc_pr(stem, score, gt, mask):
    brain = mask.flatten() > 0
    y = gt.flatten()[brain]; s = score.flatten()[brain]
    if not (0 < y.sum() < y.size):
        return
    fpr, tpr, _ = roc_curve(y, s); prec, rec, _ = precision_recall_curve(y, s)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].plot(fpr, tpr, color="#1f4e79"); ax[0].plot([0, 1], [0, 1], "k--", lw=0.7)
    ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].set_title(f"ROC (AUROC={pixel_auroc(score,gt,mask):.3f})")
    ax[1].plot(rec, prec, color="#c0392b")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title(f"PR (AUPRC={pixel_auprc(score,gt,mask):.3f})")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(stem, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGS / f"fig_roc_pr_{stem}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_dice_vs_threshold(scored, tau):
    """scored: list of (score, gt). Mean Dice over slices at each grid threshold."""
    smax = max(float(s.max()) for s, _ in scored)
    grid = np.linspace(max(1e-4, smax * 0.02), smax, 80).astype(np.float32)
    curves = [dice_over_grid(s, g, grid) for s, g in scored if g.sum() > 0]
    if not curves:
        return
    mean = np.mean(curves, axis=0)
    bi = int(np.argmax(mean))
    d_at_tau = float(np.interp(tau, grid, mean))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(grid, mean, color="#6d28d9", lw=2)
    ax.axvline(tau, color="black", ls="--", lw=1.4, label=f"calibrated $\\tau$={tau:.3f} (Dice={d_at_tau:.3f})")
    ax.axvline(grid[bi], color="#27ae60", ls=":", lw=1.6, label=f"best $\\tau^*$={grid[bi]:.3f} (Dice={mean[bi]:.3f})")
    ax.set_xlabel("threshold"); ax.set_ylabel(f"mean Dice over {len(curves)} slices")
    ax.set_title("Dice vs. threshold (threshold headroom)"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "fig_dice_vs_threshold.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"[ok] fig_dice_vs_threshold.png | Dice@tau={d_at_tau:.3f}  best={mean[bi]:.3f} @ {grid[bi]:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-slice PDM failure diagnostics")
    ap.add_argument("--slices", default=",".join(DEFAULT_SLICES),
                    help="comma-separated test-slice stems")
    ap.add_argument("--tau", type=float, default=None, help="override calibrated threshold")
    ap.add_argument("--save-npy", action="store_true", help="dump raw arrays for offline plotting")
    args = ap.parse_args()

    set_seed(CONFIG.train.seed)
    device = get_device()
    tau = load_threshold(args.tau)
    print(f"Device {device} | tau={tau:.4f}")

    unet = load_unet(device)
    scorer = MultiScaleScorer(unet, DiffusionProcess(build_noise_strategy()), device)

    ds = AnomalousSliceDataset(CONFIG.paths.manifests_dir / "test_anom.txt")
    index = {stem: i for i, stem in enumerate(ds.stems)}

    wanted = [s.strip() for s in args.slices.split(",") if s.strip()]
    scored = []
    npy_dir = FIGS / "arrays"
    if args.save_npy:
        npy_dir.mkdir(exist_ok=True)
    for stem in wanted:
        if stem not in index:
            print(f"[skip] {stem} not in test_anom manifest"); continue
        img_t, gt_t, _ = ds[index[stem]]
        img = img_t.numpy(); gt = (gt_t.numpy() > 0).astype(np.float32)
        res = scorer.score_slice(img)
        score, mask, recon, orig = res.score, res.mask, res.recon, res.orig
        au = pixel_auroc(score, gt, mask); dc = dice_coefficient((score > tau).astype(np.float32), gt)
        print(f"  {stem} | AUROC {au:.3f} | Dice@tau {dc:.3f} | lesion px {int(gt.sum())}")

        fig_hist(stem, score, gt, mask, tau)
        fig_modres(stem, orig, recon, score)
        fig_maskedge(stem, orig, mask, gt, score)
        fig_perscale(stem, res, gt, tau)
        fig_roc_pr(stem, score, gt, mask)
        scored.append((score, gt))
        if args.save_npy:
            np.savez(npy_dir / f"{stem}.npz", score=score, gt=gt, mask=mask,
                     recon=recon, orig=orig, residual=residual_stack(orig, recon))

    if scored:
        fig_dice_vs_threshold(scored, tau)
    print("Figures ->", FIGS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
