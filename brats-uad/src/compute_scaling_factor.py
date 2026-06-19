"""
Stage 1b — Compute the empirical latent scaling factor.
========================================================
Stable Diffusion hardcodes ``0.18215`` because that is ``1/std`` of *its*
latents.  Our from-scratch medical VAE has a different latent scale, so reusing
0.18215 would mis-calibrate the diffusion noise schedule.  Here we encode a
sample of training slices, measure the latent std, and persist
``scaling_factor = 1 / std`` — so that ``scaling_factor · z`` has unit variance,
matching the unit-variance Gaussian the diffusion model is trained against.

Usage
-----
    python src/compute_scaling_factor.py
    python src/compute_scaling_factor.py --n 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[0]))

import config as C
import utils
from data.datasets import SliceDataset
from data.normalization import normalize_for_vae
from models.kl_vae import KLVAE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


@torch.no_grad()
def main(args):
    utils.set_seed()
    device = utils.get_device()
    model = KLVAE.from_pretrained(C.VAE_DIR, map_location=device).to(device).eval()

    ds = SliceDataset(C.MANIFEST_VAE_TRAIN, limit=args.n)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=2)

    # Welford-style accumulation of mean/var over all latent elements.
    total = 0
    s = s2 = 0.0
    for x in loader:
        z = model.encode(normalize_for_vae(x.to(device))).mean    # deterministic
        z = z.flatten().cpu().double()      # CPU: MPS has no float64
        total += z.numel()
        s += z.sum().item()
        s2 += (z * z).sum().item()
    mean = s / total
    var = s2 / total - mean ** 2
    std = float(np.sqrt(max(var, 1e-12)))
    scaling = 1.0 / std

    C.SCALING_FACTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(C.SCALING_FACTOR_PATH, "w") as f:
        json.dump({"scaling_factor": scaling, "latent_std": std,
                   "latent_mean": mean, "n_elements": total,
                   "n_slices": len(ds)}, f, indent=2)

    log.info("=" * 60)
    log.info("Latent mean %.5f | std %.5f  (over %d slices)", mean, std, len(ds))
    log.info("scaling_factor = 1/std = %.5f  →  %s", scaling, C.SCALING_FACTOR_PATH)
    log.info("  (SD's 0.18215 is NOT used — this is medical-VAE specific.)")
    log.info("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute empirical latent scaling factor")
    p.add_argument("--n", type=int, default=2000, help="Slices to sample (default 2000).")
    p.add_argument("--bs", type=int, default=16)
    main(p.parse_args())
