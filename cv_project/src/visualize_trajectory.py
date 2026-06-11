"""
Trajectory Visualisation — Forward & Reverse Diffusion
=======================================================
Finds a slice with a large tumour, dumps the forward diffusion (destruction)
and reverse DDIM (reconstruction) trajectories as individual PNGs, runs the
SAAM XAI engine (halo-fixed, shared with the evaluation pipeline), and saves a
final evaluation figure.

All reconstruction / scoring uses the shared ``utils`` so it is consistent
with training-time calibration and the evaluation script (same VAE
normalisation, same T_INT, same brain-masked calibrated residual + threshold).

Usage
-----
    python src/visualize_trajectory.py
"""

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

# Forward snapshot schedule: 0 → T_INT in steps of 50 (always includes T_INT).
FORWARD_STEPS = sorted(set(list(range(0, C.T_INT + 1, 50)) + [C.T_INT]))
REVERSE_SAVE_EVERY = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Smart data loading — find a big tumour
# ─────────────────────────────────────────────
def find_large_tumour(masks_dir, img_dir, min_pixels: int = 1000):
    """Return the first (image, mask, name) whose tumour area > min_pixels."""
    mask_files = sorted(masks_dir.glob("*.npy"))
    log.info("Scanning %d masks for tumour size > %d px …", len(mask_files), min_pixels)
    for mf in mask_files:
        mask = np.load(mf)
        if (mask > 0).sum() > min_pixels:
            img_path = img_dir / mf.name
            if not img_path.exists():
                continue
            img = np.load(img_path)
            log.info("Selected: %s  (tumour area = %d px)", mf.name, int((mask > 0).sum()))
            return (torch.from_numpy(img).float(),
                    torch.from_numpy(mask).float(), mf.stem)
    raise RuntimeError(f"No mask with > {min_pixels} anomalous pixels in {masks_dir}")


def save_t2_png(tensor_3ch: np.ndarray, path, title: str = ""):
    """Save channel 1 (T2w) of a (3, 256, 256) array as a clean PNG."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(tensor_3ch[1], cmap="gray")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
@torch.no_grad()
def main():
    utils.set_seed()
    device = utils.get_device()
    log.info("Device: %s", device)
    utils.clear_cache()
    generator = utils.make_generator(device)

    C.TRAJ_DIR.mkdir(parents=True, exist_ok=True)

    # ── Models + calibration ─────────────────────────────────────
    log.info("Loading VAE …")
    vae = AutoencoderKL.from_pretrained(C.VAE_CKPT).to(device)
    vae.eval(); vae.requires_grad_(False)

    unet = utils.load_unet(device)
    ddim = utils.make_ddim_scheduler()

    m_baseline, calib = utils.load_calibration()
    thr_pixel = calib["threshold_pixel"] if calib else None
    log.info("M_baseline loaded (mean %.4f) | thr_pixel=%s", m_baseline.mean(), thr_pixel)

    # ── Find a big tumour + encode z0 (deterministic) ────────────
    image, gt_mask, name = find_large_tumour(C.MASKS_DIR, C.ANOMALOUS_DIR, min_pixels=1000)
    image  = image.unsqueeze(0).to(device)
    gt_bin = (gt_mask.numpy() > 0).astype(np.float32)

    z0 = utils.encode_to_latents(vae, image, sample=False)        # (1, 4, 32, 32)
    noise = torch.randn(z0.shape, device=device, generator=generator)  # fixed noise

    # ── Forward trajectory (destruction) ─────────────────────────
    log.info("─── Forward trajectory: %s ───", FORWARD_STEPS)
    for t_val in FORWARD_STEPS:
        if t_val == 0:
            z_t = z0
        else:
            t_tensor = torch.tensor([t_val], device=device, dtype=torch.long)
            z_t = ddim.add_noise(z0, noise, t_tensor)
        decoded_np = vae.decode(z_t / C.SCALING_FACTOR).sample[0].cpu().numpy()
        save_t2_png(decoded_np, C.TRAJ_DIR / f"forward_t{t_val:03d}.png",
                    title=f"Forward  t = {t_val}")
    utils.clear_cache()

    # ── Reverse trajectory (reconstruction) + SAAM ───────────────
    timesteps = utils.inference_timesteps(ddim, C.T_INT, C.DDIM_STEPS)
    log.info("─── Reverse trajectory: %d DDIM steps from T_int=%d ───",
             len(timesteps), C.T_INT)

    t_start = torch.tensor([C.T_INT], device=device, dtype=torch.long)
    z_noisy = ddim.add_noise(z0, noise, t_start)

    attn_stores = utils.install_attn_hooks(unet)
    attn_accum = np.zeros((C.TARGET_SIZE, C.TARGET_SIZE), dtype=np.float32)
    n_attn = 0
    for step_idx, t in enumerate(timesteps):
        noise_pred = unet(z_noisy, t).sample
        z_noisy = ddim.step(noise_pred, t, z_noisy).prev_sample
        attn_accum += utils.aggregate_step_attention(attn_stores)
        n_attn += 1
        if step_idx % REVERSE_SAVE_EVERY == 0:
            snap = vae.decode(z_noisy / C.SCALING_FACTOR).sample[0].cpu().numpy()
            save_t2_png(snap, C.TRAJ_DIR / f"reverse_step{step_idx:03d}.png",
                        title=f"Reverse  step {step_idx}  (t={int(t)})")
    a_total = attn_accum / max(n_attn, 1)
    utils.restore_default_processors(unet)

    # ── Final decode + brain-masked calibrated residual ──────────
    z_den = z_noisy
    recon = vae.decode(z_den / C.SCALING_FACTOR).sample
    orig_np  = utils.normalize_for_vae(image)[0].cpu().numpy()
    recon_np = recon[0].cpu().numpy()

    bmask   = utils.brain_mask_2d(orig_np)
    m_pixel = utils.pixel_residual_2d(orig_np, recon_np, m_baseline, bmask)

    # Metrics over brain voxels (consistent with evaluate_pipeline)
    brain_flat = bmask.flatten() > 0
    gt_flat    = gt_bin.flatten()[brain_flat]
    score_flat = m_pixel.flatten()[brain_flat]

    auroc = float("nan")
    if gt_flat.sum() > 0 and gt_flat.sum() < gt_flat.size and score_flat.max() > score_flat.min():
        try:
            auroc = float(roc_auc_score(gt_flat, score_flat))
        except ValueError:
            pass

    thr = thr_pixel
    if thr is None:
        try:
            thr = float(threshold_otsu(m_pixel[bmask > 0]))
        except ValueError:
            thr = float(m_pixel.max())
    pred_bin = ((m_pixel > thr).astype(np.float32)) * bmask
    dice = utils.compute_dice(pred_bin, gt_bin)
    log.info("AUROC: %.4f  |  DICE: %.4f", auroc, dice)

    # ── Final 6-panel figure ─────────────────────────────────────
    fig, axes = plt.subplots(1, 6, figsize=(32, 5.5))
    axes[0].imshow(orig_np[1], cmap="gray")
    axes[0].set_title("Original (T2w)", fontsize=13, fontweight="bold"); axes[0].axis("off")
    axes[1].imshow(recon_np[1], cmap="gray")
    axes[1].set_title("Healthy Reconstruction", fontsize=13, fontweight="bold"); axes[1].axis("off")
    im2 = axes[2].imshow(m_pixel, cmap="hot")
    axes[2].set_title(r"Pixel Residual ($M_{pixel}$)", fontsize=13, fontweight="bold"); axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    im3 = axes[3].imshow(a_total, cmap="inferno")
    axes[3].set_title("SAAM Heatmap (halo-fixed)", fontsize=13, fontweight="bold"); axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    axes[4].imshow(pred_bin, cmap="gray")
    axes[4].set_title("Prediction (thresholded)", fontsize=13, fontweight="bold"); axes[4].axis("off")
    axes[5].imshow(gt_bin, cmap="gray")
    axes[5].set_title("Ground Truth Mask", fontsize=13, fontweight="bold"); axes[5].axis("off")

    fig.suptitle(f"{name}   |   AUROC: {auroc:.4f}   |   DICE: {dice:.4f}",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    final_path = C.TRAJ_DIR / "final_eval.png"
    fig.savefig(final_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Final eval saved → %s", final_path)

    utils.clear_cache()
    log.info("=" * 60)
    log.info("DONE — all outputs in %s", C.TRAJ_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
