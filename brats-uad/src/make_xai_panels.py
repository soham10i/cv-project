"""
XAI explainability panels — the counterfactual story, per modality.
====================================================================
For a handful of test lesion slices, renders a 3×(M+1) grid:

  Row 1  Originals          : each modality + GT overlaid on T1ce
  Row 2  Healthy recon      : the diffusion counterfactual ("what healthy looks like")
  Row 3  Calibrated residual: per-modality |orig−recon|−baseline + the FUSED anomaly map

This is the model-intrinsic explanation: you see *what the brain should look like*
and *exactly where it deviates*, modality by modality, plus the CE channel — the
signal the old 3-channel pipeline could not show at all.

Usage
-----
    python src/make_xai_panels.py --n 6
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[0]))

import config as C
import utils
from data.datasets import AnomalousSliceDataset
from models.kl_vae import KLVAE
from models.unet import load_unet
from pipeline.diffusion import (make_ddim_scheduler, inference_timesteps,
                                reconstruct_healthy)
from pipeline.scoring import (brain_mask_2d, residual_stack, pixel_residual_2d,
                              latent_residual_2d, fuse_maps)
from pipeline.calibration import load_single_t

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


@torch.no_grad()
def main(args):
    utils.set_seed()
    device = utils.get_device()
    gen = torch.Generator(device=device).manual_seed(C.SEED)
    log.info("Device: %s", device)

    vae = KLVAE.from_pretrained(C.VAE_DIR, map_location=device).to(device).eval()
    unet = load_unet(device)
    sf = C.load_scaling_factor()
    m_baseline, calib = load_single_t()
    if calib is None:
        raise FileNotFoundError("Run calibrate.py first (need single-T calibration).")

    ddim = make_ddim_scheduler()
    t_int = calib["t_int"]
    timesteps = inference_timesteps(ddim, t_int, calib["ddim_steps"])

    ds = AnomalousSliceDataset(C.MANIFEST_TEST_ANOM, limit=args.n)
    out = C.RESULTS_DIR / "xai"
    out.mkdir(parents=True, exist_ok=True)

    mods = C.MODALITIES
    ncol = len(mods) + 1
    ce_idx = len(mods)                    # CE channel sits last in the residual stack

    for k in range(len(ds)):
        image, gt_mask, name = ds[k]
        image = image.unsqueeze(0).to(device)
        gt = (gt_mask.numpy() > 0).astype(np.float32)

        orig_t, recon_t, z0, zden = reconstruct_healthy(
            vae, unet, ddim, image, timesteps, t_int, gen, sf)
        orig = orig_t[0].cpu().numpy()
        recon = recon_t[0].cpu().numpy()
        bmask = brain_mask_2d(orig)

        stack = residual_stack(orig, recon)                 # (M+1, H, W)
        resid = np.clip(stack - m_baseline, 0, None) * bmask[None]

        # Fused anomaly map (what the detector actually thresholds).
        pm = pixel_residual_2d(orig, recon, m_baseline, bmask)
        if calib["use_fusion"]:
            lm = latent_residual_2d(z0, zden) * bmask
            score = fuse_maps(pm, lm, calib["pixel_scale"], calib["latent_scale"],
                              calib["alpha"])
        else:
            score = pm / (calib["pixel_scale"] + 1e-8)

        fig, ax = plt.subplots(3, ncol, figsize=(4 * ncol, 12))
        # Row 1 — originals
        for c, m in enumerate(mods):
            ax[0, c].imshow(orig[c], cmap="gray"); ax[0, c].set_title(f"Orig {m}")
        ax[0, ce_idx].imshow(orig[1], cmap="gray")
        ax[0, ce_idx].imshow(np.ma.masked_where(gt == 0, gt), cmap="autumn", alpha=0.6)
        ax[0, ce_idx].set_title("GT overlay (T1ce)")
        # Row 2 — healthy counterfactual recon
        for c, m in enumerate(mods):
            ax[1, c].imshow(recon[c], cmap="gray"); ax[1, c].set_title(f"Healthy {m}")
        ce_resid = resid[ce_idx]
        ax[1, ce_idx].imshow(ce_resid, cmap="hot")
        ax[1, ce_idx].set_title("CE residual (t1c−t1n)")
        # Row 3 — per-modality residual + fused map
        for c, m in enumerate(mods):
            ax[2, c].imshow(resid[c], cmap="hot"); ax[2, c].set_title(f"Residual {m}")
        im = ax[2, ce_idx].imshow(score, cmap="hot")
        ax[2, ce_idx].set_title("FUSED anomaly map")
        plt.colorbar(im, ax=ax[2, ce_idx], fraction=0.046, pad=0.04)

        for a in ax.ravel():
            a.axis("off")
        fig.suptitle(f"XAI — {name}  |  counterfactual healthy reconstruction + residuals",
                     fontsize=15, fontweight="bold")
        plt.tight_layout()
        fig.savefig(out / f"xai_{name}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        log.info("[%d/%d] %s → panel saved", k + 1, len(ds), name)

    log.info("XAI panels → %s", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Render counterfactual XAI panels")
    p.add_argument("--n", type=int, default=6, help="Number of test slices to render.")
    main(p.parse_args())
