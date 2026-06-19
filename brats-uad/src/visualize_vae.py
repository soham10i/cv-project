"""
Visualize VAE reconstruction quality.
=====================================
This script loads the trained VAE from `local_models/vae`, passes several
healthy/anomalous slices through the encode-decode bottleneck, and plots the
Original vs Reconstructed images.

This generates a graphic perfectly suited for the "Case Study: The Segmentation Bottleneck"
section of the report, demonstrating how the VAE compresses high-frequency detail
(like skull boundaries) and causes them to blur upon reconstruction.
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[0]))

import config as C
import utils
from data.datasets import SliceDataset
from data.normalization import normalize_for_vae
from models.kl_vae import KLVAE

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("visualize_vae")

def main():
    utils.set_seed(42)
    device = utils.get_device()
    log.info(f"Using device: {device}")

    # Set MODEL_DIR to local_models so it loads the trained VAE
    local_models_dir = C.PKG_ROOT / "local_models"
    vae_dir = local_models_dir / "vae"
    
    if not vae_dir.exists():
        log.error(f"Could not find trained VAE at {vae_dir}")
        sys.exit(1)

    log.info(f"Loading VAE from {vae_dir}...")
    vae = KLVAE.from_pretrained(vae_dir, map_location=device).to(device).eval()
    vae.requires_grad_(False)

    # Load a few slices directly from the local cv-project folder
    log.info("Loading sample slices...")
    
    sample_files = [
        "01_cv-project/data/processed_export/train_healthy/BraTS-PED-00161-000_z062.npy",
        "01_cv-project/data/processed_export/train_healthy/BraTS-PED-00070-000_z084.npy",
        "01_cv-project/data/processed_export/train_healthy/BraTS-PED-00006-000_z044.npy",
        "01_cv-project/data/processed_export/train_healthy/BraTS-PED-00041-000_z131.npy",
        "01_cv-project/data/processed_export/train_healthy/BraTS-PED-00140-000_z126.npy"
    ]
    
    n_samples = len(sample_files)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    
    # We will plot the T1ce channel (index 1) as it usually has the highest contrast
    channel_idx = 1 
    channel_name = "T1ce"

    for row_idx, file_path in enumerate(sample_files):
        # Load the raw slice: (C, 256, 256)
        slice_path = C.PROJECT_ROOT / file_path
        image_np = np.load(slice_path).astype(np.float32)
        
        # If the local dataset has 3 channels, pad to 4 channels so VAE accepts it
        if image_np.shape[0] == 3:
            padded = np.zeros((4, 256, 256), dtype=np.float32)
            padded[1:] = image_np # assuming T1ce, T2w, T2f
            image_np = padded
            
        image = torch.from_numpy(image_np)
        
        # Original pixel data (numpy)
        orig_np = image[channel_idx].numpy()
        
        # Prepare for VAE (add batch dim, normalize, push to device)
        image_t = image.unsqueeze(0).to(device)
        norm_t = normalize_for_vae(image_t)
        
        # Encode -> Decode
        with torch.no_grad():
            latents = vae.encode(norm_t).mean
            recon_t = vae.decode(latents)
        
        # Reconstructed pixel data (numpy)
        recon_np = recon_t[0, channel_idx].cpu().numpy()
        
        # Calculate Absolute Error (Residual)
        error_np = np.abs(orig_np - recon_np)
        
        # Plotting
        ax_orig = axes[row_idx, 0]
        ax_recon = axes[row_idx, 1]
        ax_error = axes[row_idx, 2]
        
        ax_orig.imshow(orig_np, cmap="gray")
        ax_orig.set_title(f"Original ({channel_name})", fontsize=12)
        ax_orig.axis("off")
        
        ax_recon.imshow(recon_np, cmap="gray")
        ax_recon.set_title(f"VAE Reconstruction (Blurry Edges)", fontsize=12)
        ax_recon.axis("off")
        
        im_err = ax_error.imshow(error_np, cmap="hot")
        ax_error.set_title(f"Absolute Error (|Orig - Recon|)", fontsize=12)
        ax_error.axis("off")
        plt.colorbar(im_err, ax=ax_error, fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    results_dir = C.PKG_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "vae_reconstruction_analysis.png"
    
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info(f"Successfully generated graphic: {out_path}")

if __name__ == "__main__":
    main()
