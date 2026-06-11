"""
VAE-Only Anomaly Detection Baseline
=====================================
Quantifies how much the diffusion model improves over using the VAE alone.

Pipeline
--------
1. Load the frozen SD VAE (no UNet).
2. Calibrate on val-healthy slices:
   - Encode deterministically (posterior mean) → decode immediately.
   - Compute raw pixel residual, build M_baseline and percentile threshold.
3. Evaluate on test-anomalous slices:
   - Encode → decode.
   - Compute brain-masked calibrated pixel residual.
   - Report per-image AUROC / DICE and aggregate metrics.
4. Save results to RESULTS_DIR/metrics_vae.json.
   Save up to --max-plots visualisation panels.

Usage
-----
    python src/evaluate_vae_baseline.py
    python src/evaluate_vae_baseline.py --n-images 50 --max-plots 12
"""

import argparse
import json
import logging

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score

# pyrefly: ignore [missing-import]
from diffusers import AutoencoderKL

import config as C
import utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# VAE encode → decode (no diffusion)
# ─────────────────────────────────────────────
@torch.no_grad()
def vae_reconstruct(vae, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode deterministically (posterior mean) then decode immediately.

    Returns
    -------
    orig_norm : (B, 3, H, W) numpy  — normalised input
    recon     : (B, 3, H, W) numpy  — VAE reconstruction
    """
    x_norm = utils.normalize_for_vae(images)
    z = vae.encode(x_norm).latent_dist.mean * C.SCALING_FACTOR
    recon = vae.decode(z / C.SCALING_FACTOR).sample
    return x_norm.cpu().numpy(), recon.cpu().numpy()


# ─────────────────────────────────────────────
# Calibration on val-healthy slices
# ─────────────────────────────────────────────
@torch.no_grad()
def calibrate_vae_baseline(vae, device,
                            percentile: float = C.THRESHOLD_PERCENTILE,
                            max_samples: int  = C.MAX_CAL_SAMPLES):
    """
    Run VAE encode→decode on val-healthy slices to build:
      * m_baseline  — mean |orig − recon| (3, H, W)
      * threshold   — percentile-th of max healthy residual
    """
    val_dir = C.VAL_HEALTHY_DIR if (C.VAL_HEALTHY_DIR.exists() and
                                     any(C.VAL_HEALTHY_DIR.glob("*.npy"))) \
              else C.HEALTHY_DIR
    log.info("VAE calibration from %s (up to %d slices)", val_dir, max_samples)

    val_files = sorted(val_dir.glob("*.npy"))[:max_samples]
    if not val_files:
        raise FileNotFoundError(f"No .npy files in {val_dir}")

    residual_sum  = None
    n_samples     = 0
    raw_diffs     = []
    brain_masks   = []

    for f in val_files:
        img = torch.from_numpy(np.load(f)).float().unsqueeze(0).to(device)
        orig_np, recon_np = vae_reconstruct(vae, img)
        raw_diff = np.abs(orig_np[0] - recon_np[0])            # (3,H,W)
        residual_sum = raw_diff.copy() if residual_sum is None else residual_sum + raw_diff
        bmask = utils.brain_mask_2d(orig_np[0])
        raw_diffs.append(raw_diff)
        brain_masks.append(bmask)
        n_samples += 1

    m_baseline = (residual_sum / n_samples).astype(np.float32)

    max_residuals = []
    for rd, bm in zip(raw_diffs, brain_masks):
        diff = np.clip(rd - m_baseline, 0, None).mean(axis=0) * bm
        max_residuals.append(diff.max())

    threshold = float(np.percentile(max_residuals, percentile))
    log.info("VAE baseline calibrated: M_baseline mean=%.4f | thr=%.4f (p%.1f) | n=%d",
             m_baseline.mean(), threshold, percentile, n_samples)

    return m_baseline, threshold, n_samples


# ─────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(args):
    utils.set_seed()
    device = utils.get_device()
    log.info("Device: %s", device)

    log.info("Loading VAE from '%s' …", C.VAE_CKPT)
    vae = AutoencoderKL.from_pretrained(C.VAE_CKPT).to(device)
    vae.eval()
    vae.requires_grad_(False)

    m_baseline, threshold, n_cal = calibrate_vae_baseline(vae, device)

    # ── Test set ─────────────────────────────────────────────────
    img_files = sorted(C.ANOMALOUS_DIR.glob("*.npy"))
    img_files = [f for f in img_files if (C.MASKS_DIR / f.name).exists()]
    if args.n_images is not None:
        img_files = img_files[:args.n_images]
    if not img_files:
        raise FileNotFoundError(f"No anomalous/mask pairs in {C.ANOMALOUS_DIR}")

    log.info("Evaluating %d test slices (saving up to %d plots) …",
             len(img_files), args.max_plots)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    per_image = []
    aurocs, dices, auprcs = [], [], []

    for i, img_path in enumerate(img_files):
        img  = torch.from_numpy(np.load(img_path)).float().unsqueeze(0).to(device)
        mask = np.load(C.MASKS_DIR / img_path.name)
        gt_bin = (mask > 0).astype(np.float32)

        orig_np, recon_np = vae_reconstruct(vae, img)
        bmask   = utils.brain_mask_2d(orig_np[0])
        m_pixel = utils.pixel_residual_2d(orig_np[0], recon_np[0], m_baseline, bmask)

        brain_flat = bmask.flatten() > 0
        gt_flat    = gt_bin.flatten()[brain_flat]
        score_flat = m_pixel.flatten()[brain_flat]

        auroc = float("nan")
        if gt_flat.sum() > 0 and gt_flat.sum() < gt_flat.size and score_flat.max() > score_flat.min():
            try:
                auroc = float(roc_auc_score(gt_flat, score_flat))
            except ValueError:
                pass

        auprc = float("nan")
        if gt_flat.sum() > 0 and score_flat.max() > score_flat.min():
            try:
                auprc = float(average_precision_score(gt_flat, score_flat))
            except ValueError:
                pass

        pred_bin = ((m_pixel > threshold).astype(np.float32)) * bmask
        dice = utils.compute_dice(pred_bin, gt_bin)

        if not np.isnan(auroc):
            aurocs.append(auroc)
        if not np.isnan(auprc):
            auprcs.append(auprc)
        dices.append(dice)
        per_image.append({"name": img_path.stem, "auroc": auroc, "auprc": auprc, "dice": dice})

        log.info("[%d/%d] %s  AUROC %.4f | AUPRC %.4f | DICE %.4f",
                 i + 1, len(img_files), img_path.stem, auroc, auprc, dice)

        if i < args.max_plots:
            _save_panel(img_path.stem, orig_np[0], recon_np[0],
                        m_pixel, gt_bin, pred_bin, auroc, dice)

        utils.clear_cache()

    summary = {
        "model": "vae_baseline",
        "n_images": len(per_image),
        "n_cal_samples": n_cal,
        "threshold_percentile": C.THRESHOLD_PERCENTILE,
        "threshold": float(threshold),
        "auroc_mean": float(np.mean(aurocs)) if aurocs else float("nan"),
        "auroc_std":  float(np.std(aurocs))  if aurocs else float("nan"),
        "auprc_mean": float(np.mean(auprcs)) if auprcs else float("nan"),
        "dice_mean":  float(np.mean(dices))  if dices  else float("nan"),
        "dice_std":   float(np.std(dices))   if dices  else float("nan"),
        "per_image": per_image,
    }
    out_path = C.RESULTS_DIR / "metrics_vae.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info("VAE BASELINE — %d images:", summary["n_images"])
    log.info("  AUROC : %.4f ± %.4f", summary["auroc_mean"], summary["auroc_std"])
    log.info("  AUPRC : %.4f",         summary["auprc_mean"])
    log.info("  DICE  : %.4f ± %.4f", summary["dice_mean"],  summary["dice_std"])
    log.info("  Saved → %s", out_path)
    log.info("=" * 60)


def _save_panel(name, orig_np, recon_np, score, gt_bin, pred_bin, auroc, dice):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    axes[0].imshow(orig_np[1], cmap="gray")
    axes[0].set_title("Original (T2w)", fontweight="bold"); axes[0].axis("off")

    axes[1].imshow(recon_np[1], cmap="gray")
    axes[1].set_title("VAE Recon (T2w)", fontweight="bold"); axes[1].axis("off")

    im2 = axes[2].imshow(score, cmap="hot")
    axes[2].set_title("Pixel Residual", fontweight="bold"); axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(pred_bin, cmap="gray")
    axes[3].set_title("Prediction", fontweight="bold"); axes[3].axis("off")

    axes[4].imshow(gt_bin, cmap="gray")
    axes[4].set_title("Ground Truth", fontweight="bold"); axes[4].axis("off")

    fig.suptitle(f"{name}  |  AUROC: {auroc:.4f}  |  DICE: {dice:.4f}",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(C.RESULTS_DIR / f"vae_eval_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VAE-only anomaly detection baseline (no diffusion)",
    )
    parser.add_argument("--n-images",  type=int, default=None,
                        help="Number of test images (default: all).")
    parser.add_argument("--max-plots", type=int, default=12,
                        help="Max per-image figures to save (default: 12).")
    args = parser.parse_args()

    evaluate(args)
