"""
Standalone recalibration — no retraining needed
=================================================
Loads the saved EMA UNet + VAE, runs the full DDIM calibration on the
healthy validation set using the current config settings (T_INT, DDIM_STEPS,
THRESHOLD_PERCENTILE, USE_LATENT_FUSION), and overwrites M_baseline.npy and
calibration.json.

Run this whenever config settings change (percentile, T_INT, fusion) without
needing to retrain the model from scratch.

Usage
-----
    python src/recalibrate.py
"""

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

# pyrefly: ignore [missing-import]
from diffusers import AutoencoderKL

import config as C
import utils
from train_healthy_manifold import HealthySliceDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@torch.no_grad()
def main():
    utils.set_seed()
    device = utils.get_device()
    log.info("Device: %s", device)
    generator = utils.make_generator(device)

    log.info("Loading VAE …")
    vae = AutoencoderKL.from_pretrained(C.VAE_CKPT).to(device)
    vae.eval(); vae.requires_grad_(False)

    unet = utils.load_unet(device)

    cal_dir = C.VAL_HEALTHY_DIR if (C.VAL_HEALTHY_DIR.exists() and
              any(C.VAL_HEALTHY_DIR.glob("*.npy"))) else C.HEALTHY_DIR
    log.info("Calibration data: %s", cal_dir)

    dataset = HealthySliceDataset(cal_dir)
    loader  = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    log.info("Calibrating with T_INT=%d | DDIM_STEPS=%d | p%.0f | fusion=%s",
             C.T_INT, C.DDIM_STEPS, C.THRESHOLD_PERCENTILE, C.USE_LATENT_FUSION)

    m_baseline, calib = utils.calibrate_on_healthy(
        vae, unet, loader, device,
        t_int=C.T_INT,
        ddim_steps=C.DDIM_STEPS,
        max_samples=C.MAX_CAL_SAMPLES,
        percentile=C.THRESHOLD_PERCENTILE,
        generator=generator,
    )
    utils.save_calibration(m_baseline, calib)

    log.info("=" * 55)
    log.info("Calibration saved:")
    log.info("  M_baseline mean   : %.6f", m_baseline.mean())
    log.info("  threshold_pixel   : %.4f  (p%.0f voxel-level)",
             calib["threshold_pixel"], C.THRESHOLD_PERCENTILE)
    log.info("  threshold_fused   : %.4f", calib["threshold_fused"])
    log.info("  threshold_pixel_slicemax: %.4f  (slice-level detection)",
             calib["threshold_pixel_slicemax"])
    log.info("  n_samples         : %d", calib["n_samples"])
    log.info("  Baseline  → %s", C.BASELINE_PATH)
    log.info("  Calib     → %s", C.CALIBRATION_PATH)
    log.info("=" * 55)


if __name__ == "__main__":
    main()
