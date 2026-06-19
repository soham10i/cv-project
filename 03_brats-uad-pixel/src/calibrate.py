"""
Stage 3 — Calibrate anomaly threshold on healthy validation set (pixel-space).
===============================================================================
Runs the full DDIM partial-noise reconstruction on held-out HEALTHY slices.
No VAE involved — works directly on (4, 256, 256) pixel images.
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
from models.unet import load_unet
from pipeline.calibration import (calibrate_single_t, calibrate_multi_t,
                                  save_single_t, save_multi_t)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(ch)
log.propagate = False


def main(args):
    print("=> START calibrate.py (pixel-space)")
    utils.set_seed()
    device = utils.get_device()
    log.info("Device: %s", device)

    max_samples = 16 if args.smoke else C.MAX_CAL_SAMPLES
    bs = min(args.bs, max_samples)
    gen = torch.Generator(device=device).manual_seed(C.SEED)

    val_ds = SliceDataset(C.MANIFEST_VAL_HEALTHY,
                          limit=(16 if args.smoke else None))
    loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                        num_workers=0, pin_memory=False)

    # Load UNet only — no VAE needed!
    unet = load_unet(device)

    log.info("Single-T calibration @ T_int=%d on ≤%d slices …", C.T_INT, max_samples)
    m_base, calib = calibrate_single_t(unet, loader, device,
                                       max_samples=max_samples, generator=gen)
    save_single_t(m_base, calib)
    log.info("  M_baseline %s mean %.5f | threshold %.4f | fusion=%s → %s",
             m_base.shape, m_base.mean(), calib["threshold"], calib["use_fusion"],
             C.CALIBRATION_PATH)

    if C.USE_MULTI_T:
        log.info("Multi-T calibration | T_list=%s agg=%s …", C.MULTI_T_LIST, C.MULTI_T_AGG)
        baselines, mcalib = calibrate_multi_t(unet, loader, device,
                                              max_samples=max_samples, generator=gen)
        save_multi_t(baselines, mcalib)
        log.info("  multi-T threshold %.4f | per-T baselines %s → %s",
                 mcalib["threshold"], list(baselines.keys()), C.MULTI_CALIBRATION_PATH)

    log.info("=" * 60)
    log.info("DONE — calibration saved. Next: python src/evaluate.py")
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        p = argparse.ArgumentParser(description="Stage 3 — healthy calibration (pixel-space)")
        p.add_argument("--bs", type=int, default=8)
        p.add_argument("--smoke", action="store_true")
        main(p.parse_args())
    except Exception as e:
        logging.exception("Fatal error in calibrate: %s", e)
        sys.exit(1)
