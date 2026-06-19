"""
Phase 5 — Evaluate Anomaly Detection Pipeline
==============================================
Runs DDIM partial-noise reconstruction on anomalous test slices, computes a
brain-masked pixel residual calibrated against the healthy baseline, extracts
self-attention attribution maps (SAAM), thresholds with the calibrated
percentile threshold, and reports per-image + aggregate AUROC / DICE.

Usage
-----
    python src/evaluate.py                  # all test slices
    python src/evaluate.py --n-images 20    # first 20 only
"""

import argparse
import json
import logging

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score
from skimage.filters import threshold_otsu

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from core import constants as C

from data.datasets import AnomalousSliceDataset
from models.factory import build_vae, load_unet
from models.attention import install_attn_hooks, aggregate_step_attention, restore_default_processors
from pipeline.diffusion import make_ddim_scheduler, inference_timesteps, encode_to_latents, normalize_for_vae
from pipeline.scoring import brain_mask_2d, pixel_residual_2d, latent_residual_2d, fuse_maps, compute_dice
from pipeline.calibration import load_calibration, load_calibration_multi_t, score_image_multi_t


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def make_seg_grid(score_max: float, n: int = 80) -> np.ndarray:
    lo = max(score_max * 0.001, 1e-4)
    hi = score_max * 1.05
    split = hi * 0.3
    g = np.concatenate([
        np.linspace(lo, split, n // 2),
        np.linspace(split, hi, n - n // 2),
    ])
    return g.astype(np.float32)


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


@torch.no_grad()
def evaluate(args):
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    generator = torch.Generator(device=device).manual_seed(C.SEED)

    vae = build_vae(device)
    unet = load_unet(device)

    use_multi_t = C.USE_MULTI_T and args.t_int is None
    multi_baselines = multi_calib = None
    if use_multi_t:
        multi_baselines, multi_calib = load_calibration_multi_t()
        if multi_calib is None:
            log.warning("USE_MULTI_T is on but no multi-T calibration found. Falling back to single-T.")
            use_multi_t = False
        else:
            log.info("MULTI-T eval | T_list=%s | agg=%s | threshold=%.4f | fusion=%s",
                     multi_calib["t_list"], multi_calib["agg"],
                     multi_calib["threshold"], multi_calib["use_fusion"])

    try:
        m_baseline, calib = load_calibration()
        log.info("M_baseline loaded: shape %s, mean %.4f", m_baseline.shape, m_baseline.mean())
    except FileNotFoundError:
        if not use_multi_t:
            raise
        m_baseline, calib = None, None

    if args.t_int is None:
        args.t_int = calib["t_int"] if calib else C.T_INT
    if args.ddim_steps is None:
        args.ddim_steps = calib["ddim_steps"] if calib else C.DDIM_STEPS
    if calib is not None and (args.t_int != calib["t_int"] or args.ddim_steps != calib["ddim_steps"]):
        log.warning("EVAL T_int=%d/steps=%d ≠ CALIBRATION T_int=%d/steps=%d",
                    args.t_int, args.ddim_steps, calib["t_int"], calib["ddim_steps"])

    ddim = make_ddim_scheduler()
    timesteps = inference_timesteps(ddim, args.t_int, args.ddim_steps)
    log.info("DDIM: %d denoising steps from T_int=%d to 0", len(timesteps), args.t_int)

    use_fusion = C.USE_LATENT_FUSION
    if calib is not None:
        thr_pixel    = calib["threshold_pixel"]
        thr_fused    = calib["threshold_fused"]
        pixel_scale  = calib["pixel_scale"]
        latent_scale = calib["latent_scale"]
        alpha        = calib["alpha"]
    else:
        thr_pixel = thr_fused = None
        pixel_scale = latent_scale = 1.0
        alpha = C.LATENT_FUSION_ALPHA

    dataset = AnomalousSliceDataset(C.ANOMALOUS_DIR, C.MASKS_DIR)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    n_total = len(dataset) if args.n_images is None else min(args.n_images, len(dataset))

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Evaluating %d images  (saving up to %d plots)", n_total, args.max_plots)
    log.info("=" * 60)

    op_thr = multi_calib["threshold"] if use_multi_t else (thr_fused if use_fusion else thr_pixel)
    if op_thr is None:
        op_thr = 0.5
    seg_grid = make_seg_grid(max(op_thr * 40, 1.0))

    per_image = []
    aurocs, dices = [], []
    auprcs, best_dices = [], []
    seg_grid_sum = np.zeros(len(seg_grid), dtype=np.float64)
    n_seg = 0

    for i, (image, gt_mask, name) in enumerate(loader):
        if args.n_images is not None and i >= args.n_images:
            break

        image    = image.to(device)
        gt_bin   = (gt_mask[0].numpy() > 0).astype(np.float32)
        name_str = name[0]

        want_plot = i < args.max_plots

        if use_multi_t:
            score, bmask, recon_np = score_image_multi_t(
                vae, unet, ddim, image, multi_baselines, multi_calib, generator)
            orig_np = normalize_for_vae(image)[0].cpu().numpy()
            thr = multi_calib["threshold"]
            a_total = np.zeros((C.TARGET_SIZE, C.TARGET_SIZE), dtype=np.float32)
        else:
            attn_stores = install_attn_hooks(unet) if want_plot else None

            z0 = encode_to_latents(vae, image, sample=False)
            noise = torch.randn(z0.shape, device=device, generator=generator)
            t_tensor = torch.tensor([args.t_int], device=device, dtype=torch.long)
            z_noisy = ddim.add_noise(z0, noise, t_tensor)

            attn_accum = np.zeros((C.TARGET_SIZE, C.TARGET_SIZE), dtype=np.float32)
            n_steps = 0
            for t in timesteps:
                noise_pred = unet(z_noisy, t).sample
                z_noisy = ddim.step(noise_pred, t, z_noisy).prev_sample
                if want_plot:
                    attn_accum += aggregate_step_attention(attn_stores)
                    n_steps += 1
            a_total = attn_accum / max(n_steps, 1)
            z_den = z_noisy
            if want_plot:
                restore_default_processors(unet)

            recon = vae.decode(z_den / C.SCALING_FACTOR).sample
            orig_np  = normalize_for_vae(image)[0].cpu().numpy()
            recon_np = recon[0].cpu().numpy()

            bmask   = brain_mask_2d(orig_np)
            m_pixel = pixel_residual_2d(orig_np, recon_np, m_baseline, bmask)

            if use_fusion:
                m_latent = latent_residual_2d(z0, z_den) * bmask
                score = fuse_maps(m_pixel, m_latent, pixel_scale, latent_scale, alpha)
                thr = thr_fused
            else:
                score = m_pixel
                thr = thr_pixel

        brain_flat = bmask.flatten() > 0
        gt_flat    = gt_bin.flatten()[brain_flat]
        score_flat = score.flatten()[brain_flat]

        auroc = float("nan")
        if gt_flat.sum() > 0 and gt_flat.sum() < gt_flat.size and score_flat.max() > score_flat.min():
            try:
                auroc = float(roc_auc_score(gt_flat, score_flat))
            except ValueError:
                pass

        if thr is None:
            try:
                thr = float(threshold_otsu(score[brain_flat.reshape(score.shape)]))
            except ValueError:
                thr = float(score.max())
        pred_bin = ((score > thr).astype(np.float32)) * bmask
        dice = compute_dice(pred_bin, gt_bin)

        auprc = float("nan")
        best_dice = float("nan")
        gt_sum = float(gt_bin.sum())
        if gt_sum > 0:
            if gt_flat.sum() > 0 and score_flat.max() > score_flat.min():
                try:
                    auprc = float(average_precision_score(gt_flat, score_flat))
                except ValueError:
                    pass
            preds = score[None, :, :] > seg_grid[:, None, None]
            inter = (preds * gt_bin[None, :, :]).sum(axis=(1, 2))
            psum  = preds.reshape(len(seg_grid), -1).sum(axis=1)
            dice_grid = 2.0 * inter / (psum + gt_sum + 1e-8)
            best_dice = float(dice_grid.max())
            seg_grid_sum += dice_grid
            n_seg += 1
            if not np.isnan(auprc):
                auprcs.append(auprc)
            best_dices.append(best_dice)

        if not np.isnan(auroc):
            aurocs.append(auroc)
        dices.append(dice)
        per_image.append({"name": name_str, "auroc": auroc, "dice": dice,
                          "auprc": auprc, "best_dice": best_dice})
        log.info("[%d/%d] %s  AUROC %.4f | AUPRC %.4f | DICE %.4f | bestDICE %.4f",
                 i + 1, n_total, name_str, auroc, auprc, dice, best_dice)

        if want_plot:
            _save_panel(name_str, orig_np, recon_np, score, a_total, gt_bin,
                        pred_bin, auroc, dice, use_fusion)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    global_best_thr = float("nan")
    global_best_dice = float("nan")
    if n_seg > 0:
        mean_dice_by_thr = seg_grid_sum / n_seg
        best_idx = int(np.argmax(mean_dice_by_thr))
        global_best_thr  = float(seg_grid[best_idx])
        global_best_dice = float(mean_dice_by_thr[best_idx])

    summary = {
        "n_images": len(per_image),
        "auroc_mean": float(np.mean(aurocs)) if aurocs else float("nan"),
        "auroc_std":  float(np.std(aurocs))  if aurocs else float("nan"),
        "auprc_mean": float(np.mean(auprcs)) if auprcs else float("nan"),
        "dice_mean":  float(np.mean(dices))  if dices  else float("nan"),
        "dice_std":   float(np.std(dices))   if dices  else float("nan"),
        "best_dice_mean_oracle": float(np.mean(best_dices)) if best_dices else float("nan"),
        "global_best_threshold": global_best_thr,
        "global_best_dice_mean": global_best_dice,
        "calibrated_thr_pixel": thr_pixel,
        "fusion": use_fusion,
        "multi_t": use_multi_t,
        "multi_t_list": multi_calib["t_list"] if use_multi_t else None,
        "multi_t_agg": multi_calib["agg"] if use_multi_t else None,
        "multi_t_threshold": multi_calib["threshold"] if use_multi_t else None,
        "t_int": args.t_int,
        "per_image": per_image,
    }
    with open(C.RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info("AGGREGATE over %d images:", summary["n_images"])
    log.info("  AUROC               : %.4f ± %.4f", summary["auroc_mean"], summary["auroc_std"])
    log.info("  AUPRC               : %.4f", summary["auprc_mean"])
    log.info("  DICE @ calib thr=%.3f: %.4f ± %.4f  (detection threshold)",
             thr_pixel if thr_pixel else float("nan"), summary["dice_mean"], summary["dice_std"])
    log.info("  DICE @ best global thr=%.3f: %.4f  (realistic single cutoff)",
             global_best_thr, global_best_dice)
    log.info("  DICE oracle (per-slice best): %.4f  (ceiling)", summary["best_dice_mean_oracle"])
    log.info("  Metrics → %s", C.RESULTS_DIR / "metrics.json")
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Phase 5 — Anomaly detection evaluation",
        )
        parser.add_argument("--n-images",   type=int, default=None, help="Number of test images (default: all)")
        parser.add_argument("--max-plots",  type=int, default=12,   help="Max per-image figures to save (default: 12)")
        parser.add_argument("--ddim-steps", type=int, default=None, help="Denoising steps in [0, T_int] (default: value from calibration.json)")
        parser.add_argument("--t-int",      type=int, default=None, help="Intermediate noise timestep (default: value from calibration.json)")
        args = parser.parse_args()

        evaluate(args)
    except Exception as e:
        log.exception(f"Fatal error in evaluate: {e}")
        sys.exit(1)
