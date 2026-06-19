"""
Core diffusion operations: VAE encode, schedulers, DDIM partial-noise recon.
=============================================================================
Owns the full inference path:  encode → add noise @ T_int → DDIM denoise → decode.

The latent scaling factor is the EMPIRICAL value from compute_scaling_factor.py
(1/std of this VAE's latents), loaded at call time — never SD's 0.18215.
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
# Encode / denoise / reconstruct
# ─────────────────────────────────────────────
@torch.no_grad()
def encode_to_latents(vae, images: torch.Tensor, sample: bool = True,
                      generator=None, scaling_factor: float | None = None) -> torch.Tensor:
    """Encode (B, C, 256, 256) images → scaled (B, LATENT_CH, 32, 32) latents."""
    sf = C.load_scaling_factor() if scaling_factor is None else scaling_factor
    post = vae.encode(normalize_for_vae(images))
    z = post.sample(generator) if sample else post.mean
    return z * sf


@torch.no_grad()
def ddim_denoise(unet, ddim, z_noisy, timesteps):
    for t in timesteps:
        noise_pred = unet(z_noisy, t).sample
        z_noisy = ddim.step(noise_pred, t, z_noisy).prev_sample
    return z_noisy


@torch.no_grad()
def reconstruct_healthy(vae, unet, ddim, images, timesteps,
                        t_int=C.T_INT, generator=None, scaling_factor=None):
    """encode → noise @ t_int → DDIM denoise → decode.

    Returns (orig_norm, recon, z0, z_denoised).
    """
    sf = C.load_scaling_factor() if scaling_factor is None else scaling_factor
    z0 = encode_to_latents(vae, images, sample=False, scaling_factor=sf)
    noise = torch.randn(z0.shape, device=z0.device, generator=generator)
    t_tensor = torch.tensor([t_int], device=z0.device, dtype=torch.long)
    z_noisy = ddim.add_noise(z0, noise, t_tensor)
    z_denoised = ddim_denoise(unet, ddim, z_noisy, timesteps)
    recon = vae.decode(z_denoised / sf)
    orig_norm = normalize_for_vae(images)
    return orig_norm, recon, z0, z_denoised
