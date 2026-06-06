"""
VAE Reconstruction Sanity Check
================================
Loads the StabilityAI SD-VAE-ft-MSE, encodes a random healthy brain slice
into the latent space, decodes it back, and plots original vs. reconstruction
side-by-side for visual verification.

The slice is mapped onto the VAE's expected [-1, 1] range with the SAME
transform (``utils.normalize_for_vae``) used everywhere else in the pipeline,
so this check reflects exactly what the encoder sees during training and
evaluation.

Usage
-----
    python src/test_vae_reconstruction.py
"""

import logging
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

# pyrefly: ignore [missing-import]
from diffusers import AutoencoderKL

import config as C
import utils

OUTPUT_PLOT = C.PROCESSED_DIR / "vae_reconstruction_check.png"
LATENT_PLOT = C.PROCESSED_DIR / "vae_latent_channels.png"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main() -> None:
    utils.set_seed()
    device = utils.get_device()
    log.info("Using device: %s", device)

    # ── 1. Load & freeze VAE ─────────────────────────────────────
    log.info("Loading VAE from '%s' …", C.VAE_CKPT)
    vae = AutoencoderKL.from_pretrained(C.VAE_CKPT).to(device)
    vae.eval()
    vae.requires_grad_(False)
    log.info("VAE loaded and frozen  (%s parameters)",
             f"{sum(p.numel() for p in vae.parameters()):,}")

    # ── 2. Pick a random healthy slice ───────────────────────────
    npy_files = sorted(C.HEALTHY_DIR.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {C.HEALTHY_DIR}")

    chosen = random.choice(npy_files)
    log.info("Selected slice: %s", chosen.name)

    arr = np.load(chosen)                          # (3, 256, 256), z-scored
    tensor = torch.from_numpy(arr).float().unsqueeze(0).to(device)  # (1,3,256,256)
    log.info("Input tensor shape : %s  dtype: %s", tensor.shape, tensor.dtype)

    # ── 3. Encode (deterministic mean) through the real transform ─
    with torch.no_grad():
        scaled_latents = utils.encode_to_latents(vae, tensor, sample=False)

    assert scaled_latents.shape == (1, 4, 32, 32), (
        f"Expected (1, 4, 32, 32), got {scaled_latents.shape}"
    )

    sl_cpu = scaled_latents.cpu()
    latent_mean = sl_cpu.mean().item()
    latent_var  = sl_cpu.var().item()
    log.info("Scaled latent shape: %s  ✓", scaled_latents.shape)
    log.info("Scaled latent mean : %.4f", latent_mean)
    log.info("Scaled latent var  : %.4f  (target ≈ 1.0)", latent_var)

    # ── 4. Visualize latent channels (UNet input) ────────────────
    latent_np = sl_cpu[0].numpy()                   # (4, 32, 32)
    fig_lat, axes_lat = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        ch = latent_np[i]
        im = axes_lat[i].imshow(ch, cmap="inferno")
        axes_lat[i].set_title(
            f"Latent Ch {i}\nμ={ch.mean():.3f}  σ²={ch.var():.3f}",
            fontsize=11, fontweight="bold",
        )
        axes_lat[i].axis("off")
        plt.colorbar(im, ax=axes_lat[i], fraction=0.046, pad=0.04)

    fig_lat.suptitle(
        f"VAE Latent Space  —  {chosen.name}\n"
        f"Shape: {tuple(scaled_latents.shape)}  →  This is what the UNet sees during diffusion",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig_lat.savefig(LATENT_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig_lat)
    log.info("Latent channels plot saved → %s", LATENT_PLOT)

    # ── 5. Decode ────────────────────────────────────────────────
    with torch.no_grad():
        decoded = vae.decode(scaled_latents / C.SCALING_FACTOR).sample  # (1,3,256,256)
    log.info("Decoded output shape: %s", decoded.shape)

    # ── 6. Visualize (T2w channel = index 1) in normalised space ──
    # Compare against normalize_for_vae(input) — the same space the decoder
    # outputs into — so the difference map reflects true reconstruction error.
    original = utils.normalize_for_vae(tensor)[0, 1].cpu().numpy()  # (256, 256)
    reconstr = decoded[0, 1].cpu().numpy()                          # (256, 256)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original  (T2w, VAE space)", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(reconstr, cmap="gray")
    axes[1].set_title("VAE Reconstruction  (T2w)", fontsize=14, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    diff = np.abs(original - reconstr)
    im2 = axes[2].imshow(diff, cmap="hot")
    axes[2].set_title("|Original − Reconstruction|", fontsize=14, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"VAE Sanity Check  —  {chosen.name}\n"
        f"Latent shape: {tuple(scaled_latents.shape)}   "
        f"mean: {latent_mean:.4f}   var: {latent_var:.4f}",
        fontsize=12, y=0.02,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Plot saved → %s", OUTPUT_PLOT)


if __name__ == "__main__":
    main()
