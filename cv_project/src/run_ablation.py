"""
Ablation sweep — diffusion UAD hyperparameters
================================================
Sweeps T_INT × DDIM_STEPS × THRESHOLD_PERCENTILE × USE_LATENT_FUSION and
records AUROC / AUPRC / DICE for every configuration.

Efficient design
----------------
Diffusion reconstruction is the only expensive operation, and it depends ONLY
on (t_int, ddim_steps).  ``percentile`` and ``fusion`` are cheap post-hoc
derivations from the cached residual maps.  So the driver:

    load VAE + EMA UNet  ……………… once
    for (t_int, ddim_steps):
        calibrate on val-healthy  …… reconstruct once  → M_baseline, scales,
                                      per-slice healthy max scores
        reconstruct test set  ……… once  → cache per-slice pixel + fused
                                      brain-masked scores, GT
        for (percentile, fusion):   …… pure numpy, no model calls
            threshold = percentile-th of healthy max scores
            AUROC/AUPRC (threshold-free) + DICE@threshold over test
            append CSV row

This makes the full grid practical: one model load and one reconstruction pass
per (t_int, ddim_steps) cell instead of per full configuration.

Usage
-----
    python src/run_ablation.py                       # full grid, 200 test slices
    python src/run_ablation.py --n-images 100
    python src/run_ablation.py --t-int 250 300 --ddim-steps 50 \\
        --percentiles 98 99 --fusion off
"""

import argparse
import csv
import logging
import time

import numpy as np
import torch

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
# Reconstruction helpers (model calls)
# ─────────────────────────────────────────────
@torch.no_grad()
def calibrate_cell(vae, unet, healthy_files, device, t_int, ddim_steps,
                   alpha, generator, max_samples):
    """
    Reconstruct healthy slices for one (t_int, ddim_steps) cell.

    Returns
    -------
    m_baseline       : (3, H, W)
    pixel_scale      : float
    latent_scale     : float
    healthy_px_voxels: np.ndarray    pooled healthy brain-voxel pixel scores
    healthy_fz_voxels: np.ndarray    pooled healthy brain-voxel fused scores

    Thresholds are derived as percentiles of the pooled BRAIN-VOXEL score
    distribution (per-voxel FPR), matching utils.calibrate_on_healthy.
    """
    ddim = utils.make_ddim_scheduler()
    timesteps = utils.inference_timesteps(ddim, t_int, ddim_steps)

    # Pass 1: M_baseline + cache raw maps
    residual_sum = None
    cache = []   # (raw_diff(3,H,W), latent_2d(H,W), bmask(H,W))
    n = 0
    for f in healthy_files[:max_samples]:
        img = torch.from_numpy(np.load(f)).float().unsqueeze(0).to(device)
        orig_norm, recon, z0, z_den = utils.reconstruct_healthy(
            vae, unet, ddim, img, timesteps, t_int, generator)
        orig_np  = orig_norm[0].cpu().numpy()
        recon_np = recon[0].cpu().numpy()
        raw_diff = np.abs(orig_np - recon_np)
        residual_sum = raw_diff.copy() if residual_sum is None else residual_sum + raw_diff
        cache.append((
            raw_diff,
            utils.latent_residual_2d(z0, z_den),
            utils.brain_mask_2d(orig_np),
        ))
        n += 1
        utils.clear_cache()

    if n == 0:
        raise RuntimeError("Healthy calibration set is empty.")
    m_baseline = (residual_sum / n).astype(np.float32)

    # Pass 2: scales, then pooled brain-voxel score distributions
    pixel_maps, latent_maps, masks = [], [], []
    for raw_diff, lat_2d, bmask in cache:
        pixel_maps.append(np.clip(raw_diff - m_baseline, 0, None).mean(axis=0) * bmask)
        latent_maps.append(lat_2d * bmask)
        masks.append(bmask > 0)

    pixel_scale  = float(np.mean([m[m > 0].mean() if np.any(m > 0) else 0.0 for m in pixel_maps])) or 1.0
    latent_scale = float(np.mean([m[m > 0].mean() if np.any(m > 0) else 0.0 for m in latent_maps])) or 1.0

    healthy_px_voxels = np.concatenate([mp[m] for mp, m in zip(pixel_maps, masks)])
    healthy_fz_voxels = np.concatenate([
        utils.fuse_maps(mp, ml, pixel_scale, latent_scale, alpha)[m]
        for mp, ml, m in zip(pixel_maps, latent_maps, masks)
    ])
    return m_baseline, pixel_scale, latent_scale, healthy_px_voxels, healthy_fz_voxels


@torch.no_grad()
def reconstruct_test_cell(vae, unet, test_pairs, device, t_int, ddim_steps,
                          m_baseline, pixel_scale, latent_scale, alpha, generator):
    """
    Reconstruct test slices for one cell and cache brain-masked scores + GT.

    Returns a list of dicts with flattened brain-voxel arrays + precomputed
    threshold-free metrics (AUROC/AUPRC for pixel and fused score spaces).
    """
    ddim = utils.make_ddim_scheduler()
    timesteps = utils.inference_timesteps(ddim, t_int, ddim_steps)

    cached = []
    for img_path, mask_path in test_pairs:
        img  = torch.from_numpy(np.load(img_path)).float().unsqueeze(0).to(device)
        gt_bin = (np.load(mask_path) > 0).astype(np.float32)

        orig_norm, recon, z0, z_den = utils.reconstruct_healthy(
            vae, unet, ddim, img, timesteps, t_int, generator)
        orig_np  = orig_norm[0].cpu().numpy()
        recon_np = recon[0].cpu().numpy()

        bmask   = utils.brain_mask_2d(orig_np)
        m_pixel = utils.pixel_residual_2d(orig_np, recon_np, m_baseline, bmask)
        m_latent = utils.latent_residual_2d(z0, z_den) * bmask
        m_fused = utils.fuse_maps(m_pixel, m_latent, pixel_scale, latent_scale, alpha)

        brain = bmask.flatten() > 0
        gt_brain    = gt_bin.flatten()[brain]
        pixel_brain = m_pixel.flatten()[brain]
        fused_brain = m_fused.flatten()[brain]
        gt_full_sum = float(gt_bin.sum())

        auroc_p = _safe_auroc(gt_brain, pixel_brain)
        auroc_f = _safe_auroc(gt_brain, fused_brain)
        auprc_p = _safe_auprc(gt_brain, pixel_brain)
        auprc_f = _safe_auprc(gt_brain, fused_brain)

        cached.append({
            "gt_brain": gt_brain, "gt_sum": gt_full_sum,
            "pixel": pixel_brain, "fused": fused_brain,
            "auroc_p": auroc_p, "auroc_f": auroc_f,
            "auprc_p": auprc_p, "auprc_f": auprc_f,
        })
        utils.clear_cache()
    return cached


# ─────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────
def _safe_auroc(gt, score):
    if gt.sum() > 0 and gt.sum() < gt.size and score.max() > score.min():
        try:
            return float(roc_auc_score(gt, score))
        except ValueError:
            return float("nan")
    return float("nan")


def _safe_auprc(gt, score):
    if gt.sum() > 0 and score.max() > score.min():
        try:
            return float(average_precision_score(gt, score))
        except ValueError:
            return float("nan")
    return float("nan")


def _dice_at(score_brain, gt_brain, gt_sum, thr):
    pred = (score_brain > thr).astype(np.float32)
    inter = float((pred * gt_brain).sum())
    return 2.0 * inter / (float(pred.sum()) + gt_sum + 1e-8)


def _nanmean(xs):
    xs = [x for x in xs if not np.isnan(x)]
    return float(np.mean(xs)) if xs else float("nan")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args):
    utils.set_seed()
    device = utils.get_device()
    log.info("Device: %s", device)
    generator = utils.make_generator(device)

    # ── Models (loaded once) ─────────────────────────────────────
    log.info("Loading VAE from '%s' …", C.VAE_CKPT)
    vae = AutoencoderKL.from_pretrained(C.VAE_CKPT).to(device)
    vae.eval(); vae.requires_grad_(False)
    unet = utils.load_unet(device)

    # ── Data ─────────────────────────────────────────────────────
    healthy_dir = (C.VAL_HEALTHY_DIR if (C.VAL_HEALTHY_DIR.exists() and
                   any(C.VAL_HEALTHY_DIR.glob("*.npy"))) else C.HEALTHY_DIR)
    healthy_files = sorted(healthy_dir.glob("*.npy"))
    if not healthy_files:
        raise FileNotFoundError(f"No healthy slices in {healthy_dir}")
    log.info("Calibration (healthy): %d slices from %s", len(healthy_files), healthy_dir)

    test_pairs = [
        (p, C.MASKS_DIR / p.name)
        for p in sorted(C.ANOMALOUS_DIR.glob("*.npy"))
        if (C.MASKS_DIR / p.name).exists()
    ]
    if args.n_images is not None:
        test_pairs = test_pairs[:args.n_images]
    if not test_pairs:
        raise FileNotFoundError(f"No anomalous/mask pairs in {C.ANOMALOUS_DIR}")
    log.info("Test set: %d anomalous slices", len(test_pairs))

    alpha = C.LATENT_FUSION_ALPHA
    fusion_modes = {"off": [False], "on": [True], "both": [False, True]}[args.fusion]

    # ── CSV output ───────────────────────────────────────────────
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = C.RESULTS_DIR / "ablation_results.csv"
    header = ["t_int", "ddim_steps", "threshold_percentile", "use_latent_fusion",
              "threshold", "auroc_mean", "auroc_std", "auprc_mean",
              "dice_mean", "dice_std", "n_test"]
    write_header = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(header)

    n_cells = len(args.t_int) * len(args.ddim_steps)
    n_total = n_cells * len(args.percentiles) * len(fusion_modes)
    log.info("=" * 60)
    log.info("Ablation grid: %d configs (%d reconstruction cells)", n_total, n_cells)
    log.info("  T_INT       : %s", args.t_int)
    log.info("  DDIM_STEPS  : %s", args.ddim_steps)
    log.info("  PERCENTILES : %s", args.percentiles)
    log.info("  FUSION      : %s", fusion_modes)
    log.info("=" * 60)

    row_n = 0
    for t_int in args.t_int:
        for ddim_steps in args.ddim_steps:
            t0 = time.time()
            log.info("── CELL  t_int=%d  ddim_steps=%d  — reconstructing …", t_int, ddim_steps)

            m_baseline, p_scale, l_scale, h_vox_p, h_vox_f = calibrate_cell(
                vae, unet, healthy_files, device, t_int, ddim_steps, alpha,
                generator, args.max_cal_samples)

            test_cache = reconstruct_test_cell(
                vae, unet, test_pairs, device, t_int, ddim_steps,
                m_baseline, p_scale, l_scale, alpha, generator)

            # Threshold-free metrics are fixed per score space → compute once
            auroc_pixel = _nanmean([c["auroc_p"] for c in test_cache])
            auroc_fused = _nanmean([c["auroc_f"] for c in test_cache])
            auprc_pixel = _nanmean([c["auprc_p"] for c in test_cache])
            auprc_fused = _nanmean([c["auprc_f"] for c in test_cache])

            log.info("   reconstructed in %.1fs — deriving %d threshold configs",
                     time.time() - t0, len(args.percentiles) * len(fusion_modes))

            for fusion in fusion_modes:
                key       = "fused" if fusion else "pixel"
                h_vox     = h_vox_f if fusion else h_vox_p
                auroc_m   = auroc_fused if fusion else auroc_pixel
                auprc_m   = auprc_fused if fusion else auprc_pixel

                for pct in args.percentiles:
                    thr = float(np.percentile(h_vox, pct))
                    dices = [
                        _dice_at(c[key], c["gt_brain"], c["gt_sum"], thr)
                        for c in test_cache
                    ]
                    auroc_std = float(np.std([
                        c["auroc_f" if fusion else "auroc_p"] for c in test_cache
                        if not np.isnan(c["auroc_f" if fusion else "auroc_p"])
                    ])) if test_cache else float("nan")

                    writer.writerow([
                        t_int, ddim_steps, pct, fusion, f"{thr:.6f}",
                        f"{auroc_m:.6f}", f"{auroc_std:.6f}", f"{auprc_m:.6f}",
                        f"{float(np.mean(dices)):.6f}", f"{float(np.std(dices)):.6f}",
                        len(test_cache),
                    ])
                    csv_file.flush()
                    row_n += 1
                    log.info("   [%d/%d] pct=%-4s fusion=%-5s thr=%.4f → "
                             "AUROC %.4f | AUPRC %.4f | DICE %.4f",
                             row_n, n_total, pct, fusion, thr,
                             auroc_m, auprc_m, float(np.mean(dices)))

    csv_file.close()
    log.info("=" * 60)
    log.info("Ablation complete — %d rows → %s", row_n, csv_path)
    log.info("=" * 60)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion UAD hyperparameter ablation")
    parser.add_argument("--t-int",       type=int,   nargs="+", default=[150, 250, 300, 350, 450],
                        help="T_INT values to sweep.")
    parser.add_argument("--ddim-steps",  type=int,   nargs="+", default=[25, 50, 100],
                        help="DDIM step counts to sweep (actual steps in [0, t_int]).")
    parser.add_argument("--percentiles", type=float, nargs="+", default=[90, 93, 95, 97, 98, 99],
                        help="Threshold percentiles to sweep (voxel-level FPR).")
    parser.add_argument("--fusion",      choices=["off", "on", "both"], default="both",
                        help="Latent-fusion modes to evaluate (default: both).")
    parser.add_argument("--n-images",    type=int,   default=200,
                        help="Test slices to use (default: 200; None-like via 0 = all).")
    parser.add_argument("--max-cal-samples", type=int, default=C.MAX_CAL_SAMPLES,
                        help=f"Healthy slices for calibration (default: {C.MAX_CAL_SAMPLES}).")
    args = parser.parse_args()

    if args.n_images is not None and args.n_images <= 0:
        args.n_images = None

    main(args)
