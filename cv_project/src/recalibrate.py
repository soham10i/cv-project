"""
Recalibrate (no retraining)
===========================
Regenerate ``M_baseline.npy`` + ``calibration.json`` from the *already trained*
EMA UNet, using held-out healthy slices.  Use this after changing the
calibration logic (e.g. the per-voxel segmentation threshold fix) so you can
re-evaluate without paying for a full training run.

    python src/recalibrate.py                 # 100 healthy slices (default)
    python src/recalibrate.py --max 200

Then re-run:  python src/evaluate_pipeline.py
"""

import argparse
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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


class HealthySliceDataset(Dataset):
    """Loads preprocessed (3, 256, 256) healthy .npy slices as float32 tensors."""

    def __init__(self, data_dir):
        self.files = sorted(data_dir.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"No .npy files in {data_dir}")
        log.info("Healthy calibration pool: %d slices from %s", len(self.files), data_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.from_numpy(np.load(self.files[idx])).float()


def main(args):
    utils.set_seed()
    device = utils.get_device()
    log.info("Device: %s", device)
    generator = utils.make_generator(device)

    log.info("Loading VAE '%s' …", C.VAE_CKPT)
    vae = AutoencoderKL.from_pretrained(C.VAE_CKPT).to(device)
    vae.eval(); vae.requires_grad_(False)

    unet = utils.load_unet(device)          # prefers EMA weights

    ds = HealthySliceDataset(C.HEALTHY_DIR)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False)

    m_baseline, calib = utils.calibrate_on_healthy(
        vae, unet, loader, device,
        t_int=args.t_int, max_samples=args.max, generator=generator,
    )
    utils.save_calibration(m_baseline, calib)

    log.info("=" * 60)
    log.info("Recalibrated on %d healthy slices (t_int=%d)", calib["n_samples"], calib["t_int"])
    log.info("  M_baseline mean        : %.6f", float(m_baseline.mean()))
    log.info("  threshold_pixel (seg)  : %.6f   ← per-voxel %.0fth pct (the fix)",
             calib["threshold_pixel"], calib["percentile"])
    log.info("  threshold_detect (img) : %.6f   ← per-slice-max (detection only)",
             calib["threshold_detect_imagelevel"])
    log.info("  healthy mu / sigma     : %.6f / %.6f", calib["healthy_mu"], calib["healthy_sigma"])
    log.info("  Saved → %s , %s", C.BASELINE_PATH, C.CALIBRATION_PATH)
    log.info("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Recalibrate baseline + thresholds (no retraining)")
    p.add_argument("--max",   type=int, default=C.MAX_CAL_SAMPLES, help="Healthy slices to use")
    p.add_argument("--bs",    type=int, default=8, help="Batch size")
    p.add_argument("--t-int", type=int, default=C.T_INT, help="Intermediate noise timestep")
    main(p.parse_args())
