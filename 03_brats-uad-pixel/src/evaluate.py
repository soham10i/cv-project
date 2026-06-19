"""
Stage 4 — Evaluate anomaly detection on test lesion slices (pixel-space).
==========================================================================
Runs DDIM partial-noise reconstruction on TEST-patient lesion slices, computes
the calibrated modality-weighted anomaly map (single- or multi-T), thresholds
it, and reports per-image + aggregate AUROC / AUPRC / DICE.

No VAE involved — the UNet operates directly on (4, 256, 256) pixel images.

Usage
-----
    python src/evaluate.py                  # all test slices
    python src/evaluate.py --n-images 30 --max-plots 12
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(str(Path(__file__).resolve().parents[0]))

import config as C
import utils
from data.datasets import AnomalousSliceDataset
from data.normalization import normalize_for_vae
from models.unet import load_unet
from pipeline.diffusion import (make_ddim_scheduler, inference_timesteps,
                                reconstruct_healthy)
from pipeline.scoring import brain_mask_2d, pixel_residual_2d, compute_dice
from pipeline.calibration import (load_single_t, load_multi_t, score_image_multi_t)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(ch)
log.propagate = False


def save_panel(name, orig, recon, score, pred, gt, auroc, dice, out_dir):
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    ax[0].imshow(orig[1], cmap="gray"); ax[0].set_title("Original (T1ce)")
    ax[1].imshow(recon[1], cmap="gray"); ax[1].set_title("Healthy recon (T1ce)")
    im = ax[2].imshow(score, cmap="hot"); ax[2].set_title("Anomaly map")
    plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    ax[3].imshow(pred, cmap="gray"); ax[3].set_title("Prediction")
    ax[4].imshow(gt, cmap="gray"); ax[4].set_title("Ground truth")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{name} | AUROC {auroc:.3f} | DICE {dice:.3f}", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / f"eval_{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate(args):
    print("=> START evaluate.py (pixel-space)")
    utils.set_seed()
    device = utils.get_device()
    log.info("Device: %s", device)
    gen = torch.Generator(device=device).manual_seed(C.SEED)

    # Load only the UNet — no VAE needed!
    unet = load_unet(device)

    use_multi = C.USE_MULTI_T and args.t_int is None
    baselines, mcalib = (load_multi_t() if use_multi else (None, None))
    if use_multi and mcalib is None:
        log.warning("Multi-T requested but no multi-T calibration — using single-T.")
        use_multi = False
    m_baseline, calib = load_single_t()
    if calib is None and not use_multi:
        raise FileNotFoundError("No calibration found. Run calibrate.py first.")

    ddim = make_ddim_scheduler()
    t_int = args.t_int or (calib["t_int"] if calib else C.T_INT)
    timesteps = inference_timesteps(ddim, t_int, C.DDIM_STEPS)
    op_thr = mcalib["threshold"] if use_multi else calib["threshold"]
    use_fusion = False  # Never use fusion in pixel-space mode.
    log.info("Mode: %s | threshold %.4f | fusion=%s | PIXEL-SPACE",
             "multi-T" if use_multi else f"single-T@{t_int}", op_thr, use_fusion)

    ds = AnomalousSliceDataset(C.MANIFEST_TEST_ANOM,
                               limit=args.n_images if args.n_images else None)
    out_dir = C.RESULTS_DIR / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Threshold grid for oracle / best-global DICE.
    seg_grid = np.linspace(max(op_thr * 0.05, 1e-4), op_thr * 4.0, 60).astype(np.float32)
    grid_sum = np.zeros(len(seg_grid)); n_grid = 0

    aurocs, auprcs, dices, best_dices, per_image = [], [], [], [], []
    for i in range(len(ds)):
        image, gt_mask, name = ds[i]
        image = image.unsqueeze(0).to(device)
        gt = (gt_mask.numpy() > 0).astype(np.float32)

        if use_multi:
            score, bmask, recon = score_image_multi_t(
                unet, ddim, image, baselines, mcalib, gen)
            orig = normalize_for_vae(image)[0].cpu().numpy()
        else:
            orig_t, recon_t = reconstruct_healthy(
                unet, ddim, image, timesteps, t_int, gen)
            orig = orig_t[0].cpu().numpy(); recon = recon_t[0].cpu().numpy()
            bmask = brain_mask_2d(orig)
            pm = pixel_residual_2d(orig, recon, m_baseline, bmask)
            score = pm / (calib["pixel_scale"] + 1e-8)

        brain = bmask.flatten() > 0
        gt_f = gt.flatten()[brain]; sc_f = score.flatten()[brain]

        auroc = float("nan")
        if 0 < gt_f.sum() < gt_f.size and sc_f.max() > sc_f.min():
            auroc = float(roc_auc_score(gt_f, sc_f))
        pred = ((score > op_thr).astype(np.float32)) * bmask
        dice = compute_dice(pred, gt)

        auprc = best_dice = float("nan")
        if gt.sum() > 0:
            if gt_f.sum() > 0 and sc_f.max() > sc_f.min():
                auprc = float(average_precision_score(gt_f, sc_f))
            preds = score[None] > seg_grid[:, None, None]
            inter = (preds * gt[None]).sum(axis=(1, 2))
            psum = preds.reshape(len(seg_grid), -1).sum(axis=1)
            dgrid = 2.0 * inter / (psum + gt.sum() + 1e-8)
            best_dice = float(dgrid.max())
            grid_sum += dgrid; n_grid += 1
            if not np.isnan(auprc):
                auprcs.append(auprc)
            best_dices.append(best_dice)

        if not np.isnan(auroc):
            aurocs.append(auroc)
        dices.append(dice)
        per_image.append({"name": name, "auroc": auroc, "auprc": auprc,
                          "dice": dice, "best_dice": best_dice})
        log.info("[%d/%d] %s | AUROC %.3f | AUPRC %.3f | DICE %.3f | best %.3f",
                 i + 1, len(ds), name, auroc, auprc, dice, best_dice)
        if i < args.max_plots:
            save_panel(name, orig, recon, score, pred, gt, auroc, dice, out_dir)
        utils.clear_cache()

    gbest_thr = gbest_dice = float("nan")
    if n_grid:
        mean_by_thr = grid_sum / n_grid
        bi = int(np.argmax(mean_by_thr))
        gbest_thr, gbest_dice = float(seg_grid[bi]), float(mean_by_thr[bi])

    summary = {
        "n_images": len(per_image),
        "auroc_mean": float(np.mean(aurocs)) if aurocs else float("nan"),
        "auroc_std": float(np.std(aurocs)) if aurocs else float("nan"),
        "auprc_mean": float(np.mean(auprcs)) if auprcs else float("nan"),
        "dice_mean_calibrated": float(np.mean(dices)) if dices else float("nan"),
        "dice_best_global": gbest_dice, "best_global_threshold": gbest_thr,
        "dice_oracle": float(np.mean(best_dices)) if best_dices else float("nan"),
        "calibrated_threshold": float(op_thr), "multi_t": use_multi,
        "t_int": t_int, "per_image": per_image,
    }
    with open(C.RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info("AGGREGATE over %d images", summary["n_images"])
    log.info("  AUROC                 : %.4f ± %.4f", summary["auroc_mean"], summary["auroc_std"])
    log.info("  AUPRC                 : %.4f", summary["auprc_mean"])
    log.info("  DICE @ calibrated thr : %.4f", summary["dice_mean_calibrated"])
    log.info("  DICE @ best global thr: %.4f  (thr=%.4f)", gbest_dice, gbest_thr)
    log.info("  DICE oracle (ceiling) : %.4f", summary["dice_oracle"])
    log.info("  → %s", C.RESULTS_DIR / "metrics.json")
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        p = argparse.ArgumentParser(description="Stage 4 — anomaly detection evaluation (pixel-space)")
        p.add_argument("--n-images", type=int, default=None)
        p.add_argument("--max-plots", type=int, default=12)
        p.add_argument("--t-int", type=int, default=None)
        evaluate(p.parse_args())
    except Exception as e:
        logging.exception("Fatal error in evaluate: %s", e)
        sys.exit(1)
