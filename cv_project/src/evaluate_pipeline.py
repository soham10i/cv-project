"""
Phase 5 — Evaluate Anomaly Detection Pipeline
==============================================
Runs DDIM partial-noise reconstruction on anomalous test slices, computes a
brain-masked pixel residual calibrated against the healthy baseline, extracts
self-attention attribution maps (SAAM), thresholds with the calibrated
percentile threshold, and reports per-image + aggregate AUROC / DICE.

Improvements over the original:
  * EMA UNet weights are loaded preferentially.
  * VAE-space normalisation + deterministic encoding (utils, matches training).
  * SAAM halo fix (MIN_SEQ_LEN) so only high-res attention is aggregated.
  * Brain-masked scoring + AUROC computed over brain voxels (no background bias).
  * Operational threshold read from calibration.json (replaces per-image Otsu,
    which always "finds" a lesion even in healthy tissue).
  * Optional dual-space fusion (config.USE_LATENT_FUSION).
  * Aggregate metrics (mean ± std) + metrics.json.

Usage
-----
    python src/evaluate_pipeline.py                  # all test slices
    python src/evaluate_pipeline.py --n-images 20    # first 20 only
"""

import argparse
import json
import logging

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from skimage.filters import threshold_otsu

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
# Dataset
# ─────────────────────────────────────────────
class AnomalousSliceDataset(torch.utils.data.Dataset):
    """Loads anomalous slices + matching ground-truth masks (mask must exist)."""

    def __init__(self, img_dir, mask_dir):
        self.mask_dir = mask_dir
        all_imgs = sorted(img_dir.glob("*.npy"))
        # IMPROVE-04: only keep slices whose ground-truth mask exists.
        self.img_files = [p for p in all_imgs if (mask_dir / p.name).exists()]
        n_missing = len(all_imgs) - len(self.img_files)
        if n_missing:
            log.warning("Skipping %d slice(s) with no matching mask.", n_missing)
        if len(self.img_files) == 0:
            raise FileNotFoundError(f"No usable .npy/mask pairs in {img_dir}")
        log.info("Test set: %d anomalous slices", len(self.img_files))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img  = np.load(self.img_files[idx])
        mask = np.load(self.mask_dir / self.img_files[idx].name)
        return (
            torch.from_numpy(img).float(),       # (3, 256, 256)
            torch.from_numpy(mask).float(),      # (256, 256)
            self.img_files[idx].stem,
        )


# ─────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(args):
    utils.set_seed()
    device = utils.get_device()
    log.info("Device: %s", device)
    utils.clear_cache()
    generator = utils.make_generator(device)

    # ── Models ───────────────────────────────────────────────────
    log.info("Loading VAE from '%s' …", C.VAE_CKPT)
    vae = AutoencoderKL.from_pretrained(C.VAE_CKPT).to(device)
    vae.eval()
    vae.requires_grad_(False)

    unet = utils.load_unet(device)

    ddim = utils.make_ddim_scheduler()
    timesteps = utils.inference_timesteps(ddim, args.t_int, args.ddim_steps)
    log.info("DDIM: %d steps total → %d steps from T_int=%d to 0",
             args.ddim_steps, len(timesteps), args.t_int)

    # ── Calibration (baseline + threshold + scales) ──────────────
    m_baseline, calib = utils.load_calibration()
    log.info("M_baseline loaded: shape %s, mean %.4f", m_baseline.shape, m_baseline.mean())

    use_fusion = C.USE_LATENT_FUSION
    if calib is not None:
        thr_pixel    = calib["threshold_pixel"]
        thr_fused    = calib["threshold_fused"]
        pixel_scale  = calib["pixel_scale"]
        latent_scale = calib["latent_scale"]
        alpha        = calib["alpha"]
        log.info("Calibration: thr_pixel %.4f | thr_fused %.4f | fusion=%s",
                 thr_pixel, thr_fused, use_fusion)
    else:
        log.warning("No calibration.json — falling back to per-image Otsu threshold.")
        thr_pixel = thr_fused = None
        pixel_scale = latent_scale = 1.0
        alpha = C.LATENT_FUSION_ALPHA

    # ── Data ─────────────────────────────────────────────────────
    dataset = AnomalousSliceDataset(C.ANOMALOUS_DIR, C.MASKS_DIR)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    n_total = len(dataset) if args.n_images is None else min(args.n_images, len(dataset))

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Evaluating %d images  (saving up to %d plots)", n_total, args.max_plots)
    log.info("=" * 60)

    per_image = []
    aurocs, dices = [], []

    for i, (image, gt_mask, name) in enumerate(loader):
        if args.n_images is not None and i >= args.n_images:
            break

        image    = image.to(device)
        gt_bin   = (gt_mask[0].numpy() > 0).astype(np.float32)
        name_str = name[0]

        # SAAM is only needed for the (capped) saved figures, and the manual
        # attention path is ~3× slower — so only hook attention when plotting.
        want_plot = i < args.max_plots
        attn_stores = utils.install_attn_hooks(unet) if want_plot else None

        # 1. Encode (deterministic) → noise @ T_int → DDIM denoise ──────────
        z0 = utils.encode_to_latents(vae, image, sample=False)
        noise = torch.randn(z0.shape, device=device, generator=generator)
        t_tensor = torch.tensor([args.t_int], device=device, dtype=torch.long)
        z_noisy = ddim.add_noise(z0, noise, t_tensor)

        attn_accum = np.zeros((C.TARGET_SIZE, C.TARGET_SIZE), dtype=np.float32)
        n_steps = 0
        for t in timesteps:
            noise_pred = unet(z_noisy, t).sample
            z_noisy = ddim.step(noise_pred, t, z_noisy).prev_sample
            if want_plot:
                attn_accum += utils.aggregate_step_attention(attn_stores)
                n_steps += 1
        a_total = attn_accum / max(n_steps, 1)
        z_den = z_noisy
        if want_plot:
            utils.restore_default_processors(unet)

        # 2. Decode + residual scoring ───────────────────────────────
        recon = vae.decode(z_den / C.SCALING_FACTOR).sample
        orig_np  = utils.normalize_for_vae(image)[0].cpu().numpy()   # (3,256,256)
        recon_np = recon[0].cpu().numpy()

        bmask   = utils.brain_mask_2d(orig_np)
        m_pixel = utils.pixel_residual_2d(orig_np, recon_np, m_baseline, bmask)

        if use_fusion:
            m_latent = utils.latent_residual_2d(z0, z_den) * bmask
            score = utils.fuse_maps(m_pixel, m_latent, pixel_scale, latent_scale, alpha)
            thr = thr_fused
        else:
            score = m_pixel
            thr = thr_pixel

        # 3. Metrics (over brain voxels) ─────────────────────────────
        brain_flat = bmask.flatten() > 0
        gt_flat    = gt_bin.flatten()[brain_flat]
        score_flat = score.flatten()[brain_flat]

        auroc = float("nan")
        if gt_flat.sum() > 0 and gt_flat.sum() < gt_flat.size and score_flat.max() > score_flat.min():
            try:
                auroc = float(roc_auc_score(gt_flat, score_flat))
            except ValueError:
                pass

        # Threshold: calibrated percentile (preferred) or Otsu fallback
        if thr is None:
            try:
                thr = float(threshold_otsu(score[brain_flat.reshape(score.shape)]))
            except ValueError:
                thr = float(score.max())
        pred_bin = ((score > thr).astype(np.float32)) * bmask
        dice = utils.compute_dice(pred_bin, gt_bin)

        if not np.isnan(auroc):
            aurocs.append(auroc)
        dices.append(dice)
        per_image.append({"name": name_str, "auroc": auroc, "dice": dice})
        log.info("[%d/%d] %s  AUROC %.4f | DICE %.4f", i + 1, n_total, name_str, auroc, dice)

        # 4. Visualisation (capped) ──────────────────────────────────
        if i < args.max_plots:
            _save_panel(name_str, orig_np, recon_np, score, a_total, gt_bin,
                        pred_bin, auroc, dice, use_fusion)

        del z0, z_noisy, z_den, noise, recon
        utils.clear_cache()

    # ── Aggregate ────────────────────────────────────────────────
    summary = {
        "n_images": len(per_image),
        "auroc_mean": float(np.mean(aurocs)) if aurocs else float("nan"),
        "auroc_std":  float(np.std(aurocs))  if aurocs else float("nan"),
        "dice_mean":  float(np.mean(dices))  if dices  else float("nan"),
        "dice_std":   float(np.std(dices))   if dices  else float("nan"),
        "fusion": use_fusion,
        "t_int": args.t_int,
        "per_image": per_image,
    }
    with open(C.RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info("AGGREGATE over %d images:", summary["n_images"])
    log.info("  AUROC: %.4f ± %.4f", summary["auroc_mean"], summary["auroc_std"])
    log.info("  DICE : %.4f ± %.4f", summary["dice_mean"], summary["dice_std"])
    log.info("  Metrics → %s", C.RESULTS_DIR / "metrics.json")
    log.info("=" * 60)


def _save_panel(name, orig_np, recon_np, score, a_total, gt_bin, pred_bin,
                auroc, dice, use_fusion):
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))

    axes[0].imshow(orig_np[1], cmap="gray")
    axes[0].set_title("Original (T2w)", fontweight="bold"); axes[0].axis("off")

    axes[1].imshow(recon_np[1], cmap="gray")
    axes[1].set_title("Healthy Recon (T2w)", fontweight="bold"); axes[1].axis("off")

    score_title = "Fused Residual" if use_fusion else r"Pixel Residual ($M_{pixel}$)"
    im2 = axes[2].imshow(score, cmap="hot")
    axes[2].set_title(score_title, fontweight="bold"); axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(a_total, cmap="inferno")
    axes[3].set_title("Attention Heatmap (SAAM)", fontweight="bold"); axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    axes[4].imshow(pred_bin, cmap="gray")
    axes[4].set_title("Prediction (thresholded)", fontweight="bold"); axes[4].axis("off")

    axes[5].imshow(gt_bin, cmap="gray")
    axes[5].set_title("Ground Truth Mask", fontweight="bold"); axes[5].axis("off")

    fig.suptitle(f"{name}   |   AUROC: {auroc:.4f}   |   DICE: {dice:.4f}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = C.RESULTS_DIR / f"eval_{name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 5 — Anomaly detection evaluation",
    )
    parser.add_argument("--n-images",   type=int, default=None, help="Number of test images (default: all)")
    parser.add_argument("--max-plots",  type=int, default=12,   help="Max per-image figures to save (default: 12)")
    parser.add_argument("--ddim-steps", type=int, default=C.DDIM_STEPS, help=f"Total DDIM steps (default: {C.DDIM_STEPS})")
    parser.add_argument("--t-int",      type=int, default=C.T_INT, help=f"Intermediate noise timestep (default: {C.T_INT})")
    args = parser.parse_args()

    evaluate(args)
