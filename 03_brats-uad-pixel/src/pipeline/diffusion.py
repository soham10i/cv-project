"""
Core diffusion operations — pixel-space (no VAE).
===================================================
Owns the full inference path:  normalise → add noise @ T_int → DDIM denoise.
No encoding/decoding through a VAE — the UNet works directly on pixel images.
"""

from __future__ import annotations

import logging

import torch

# pyrefly: ignore [missing-import]
from diffusers import DDIMScheduler, DDPMScheduler

import config as C
from data.normalization import normalize_for_vae

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Schedulers
# ─────────────────────────────────────────────
def make_ddpm_scheduler() -> DDPMScheduler:
    return DDPMScheduler(num_train_timesteps=C.NUM_TRAIN_TIMESTEPS,
                         beta_schedule=C.BETA_SCHEDULE, beta_start=C.BETA_START,
                         beta_end=C.BETA_END, prediction_type=C.PREDICTION_TYPE)


def make_ddim_scheduler() -> DDIMScheduler:
    return DDIMScheduler(num_train_timesteps=C.NUM_TRAIN_TIMESTEPS,
                         beta_schedule=C.BETA_SCHEDULE, beta_start=C.BETA_START,
                         beta_end=C.BETA_END, prediction_type=C.PREDICTION_TYPE)


def inference_timesteps(ddim: DDIMScheduler, t_int: int = C.T_INT,
                        ddim_steps: int = C.DDIM_STEPS) -> torch.Tensor:
    """DDIM timesteps from t_int → 0 (≈ddim_steps steps inside [0, t_int])."""
    full = max(ddim_steps, round(ddim_steps * C.NUM_TRAIN_TIMESTEPS / max(t_int, 1)))
    ddim.set_timesteps(full)
    ts = ddim.timesteps
    return ts[ts <= t_int]


# ─────────────────────────────────────────────
# Denoise / reconstruct (pixel-space)
# ─────────────────────────────────────────────
@torch.no_grad()
def ddim_denoise(unet, ddim, x_noisy, timesteps):
    """Run DDIM denoising loop directly on pixel images."""
    for t in timesteps:
        noise_pred = unet(x_noisy, t).sample
        x_noisy = ddim.step(noise_pred, t, x_noisy).prev_sample
    return x_noisy


@torch.no_grad()
def reconstruct_healthy(unet, ddim, images, timesteps,
                        t_int=C.T_INT, generator=None):
    """normalise → noise @ t_int → DDIM denoise.

    Returns (orig_norm, recon).
    No VAE involved — the UNet operates directly on the (B, 4, 256, 256) pixel image.
    """
    orig_norm = normalize_for_vae(images)
    noise = torch.randn(orig_norm.shape, device=orig_norm.device, generator=generator)
    t_tensor = torch.tensor([t_int], device=orig_norm.device, dtype=torch.long)
    x_noisy = ddim.add_noise(orig_norm, noise, t_tensor)
    recon = ddim_denoise(unet, ddim, x_noisy, timesteps)
    return orig_norm, recon
