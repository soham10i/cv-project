"""
Stage 3 — Calibrate anomaly thresholds on healthy val slices.
==============================================================
Loads the frozen VAE + trained UNet, runs DDIM partial-noise reconstruction on
lesion-free VAL slices, and derives the baseline residual + operating threshold
for both single-T and (if enabled) multi-T scoring.

Usage
-----
    python src/calibrate.py
    python src/calibrate.py --smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[0]))

import config as C
import utils
from data.datasets import SliceDataset
from models.kl_vae import KLVAE
from models.unet import load_unet
from pipeline.calibration import (calibrate_single_t, calibrate_multi_t,
                                  save_single_t, save_multi_t)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def main(args):
    print("=> START calibrate.py")
    utils.set_seed()
    device = utils.get_device()
    log.info("Device: %s", device)
    gen = torch.Generator(device=device).manual_seed(C.SEED)

    vae = KLVAE.from_pretrained(C.VAE_DIR, map_location=device).to(device).eval()
    vae.requires_grad_(False)
    unet = load_unet(device)
    sf = C.load_scaling_factor()
    log.info("scaling_factor=%.5f", sf)

    limit = 16 if args.smoke else None
    max_samples = 16 if args.smoke else C.MAX_CAL_SAMPLES
    ds = SliceDataset(C.MANIFEST_VAL_HEALTHY, limit=limit)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=0)

    log.info("Single-T calibration @ T_int=%d on ≤%d slices …", C.T_INT, max_samples)
    m_base, calib = calibrate_single_t(vae, unet, loader, device, t_int=C.T_INT,
                                       max_samples=max_samples, generator=gen,
                                       scaling_factor=sf)
    save_single_t(m_base, calib)
    log.info("  M_baseline %s mean %.5f | threshold %.4f | fusion=%s → %s",
             m_base.shape, m_base.mean(), calib["threshold"], calib["use_fusion"],
             C.CALIBRATION_PATH)

    if C.USE_MULTI_T:
        log.info("Multi-T calibration | T_list=%s agg=%s …", C.MULTI_T_LIST, C.MULTI_T_AGG)
        baselines, mcalib = calibrate_multi_t(vae, unet, loader, device,
                                              max_samples=max_samples, generator=gen,
                                              scaling_factor=sf)
        save_multi_t(baselines, mcalib)
        log.info("  multi-T threshold %.4f | per-T baselines %s → %s",
                 mcalib["threshold"], list(baselines.keys()), C.MULTI_CALIBRATION_PATH)

    log.info("=" * 60)
    log.info("DONE — calibration saved. Next: python src/evaluate.py")
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        p = argparse.ArgumentParser(description="Stage 3 — healthy calibration")
        p.add_argument("--bs", type=int, default=8)
        p.add_argument("--smoke", action="store_true")
        main(p.parse_args())
    except Exception as e:
        logging.exception("Fatal error in calibrate: %s", e)
        sys.exit(1)
