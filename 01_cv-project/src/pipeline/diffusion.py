"""
Core diffusion operations: VAE encoding, noise scheduling, and DDIM reconstruction.

This module owns the full inference path:
  encode → add noise @ T_int → DDIM denoise → decode

All functions are deterministic when possible (``sample=False`` at inference)
to ensure calibration and evaluation are distribution-matched.
"""

from __future__ import annotations

import logging

import torch
import numpy as np

# pyrefly: ignore [missing-import]
from diffusers import DDIMScheduler, DDPMScheduler

from core import constants as C

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# VAE input normalisation
# ─────────────────────────────────────────────
def normalize_for_vae(x: torch.Tensor, clip: float = C.VAE_CLIP) -> torch.Tensor:
    """Map z-score-normalised slices onto the VAE's expected ~[-1, 1] range.

    Clip to ±clip sigma, then linearly scale that range onto [-1, 1].
    Applied immediately before every ``vae.encode`` so the encoder never
    sees out-of-distribution inputs.
    """
    return torch.clamp(x, -clip, clip) / clip


# ─────────────────────────────────────────────
# Scheduler factories
# ─────────────────────────────────────────────
def make_ddpm_scheduler() -> DDPMScheduler:
    """Create a DDPM scheduler with project-wide diffusion hyperparameters."""
    return DDPMScheduler(
        num_train_timesteps=C.NUM_TRAIN_TIMESTEPS,
        beta_schedule=C.BETA_SCHEDULE,
        beta_start=C.BETA_START,
        beta_end=C.BETA_END,
        prediction_type=C.PREDICTION_TYPE,
    )


def make_ddim_scheduler() -> DDIMScheduler:
    """Create a DDIM scheduler for deterministic inference."""
    return DDIMScheduler(
        num_train_timesteps=C.NUM_TRAIN_TIMESTEPS,
        beta_schedule=C.BETA_SCHEDULE,
        beta_start=C.BETA_START,
        beta_end=C.BETA_END,
        prediction_type=C.PREDICTION_TYPE,
    )


def inference_timesteps(
    ddim: DDIMScheduler,
    t_int: int = C.T_INT,
    ddim_steps: int = C.DDIM_STEPS,
) -> torch.Tensor:
    """Compute DDIM timesteps from ``t_int`` → 0 for partial-noise reconstruction.

    Scales ``num_inference_steps`` by ``1000 / t_int`` so that approximately
    ``ddim_steps`` denoising steps land within the ``[0, t_int]`` window.
    """
    full_steps = max(
        ddim_steps,
        round(ddim_steps * C.NUM_TRAIN_TIMESTEPS / max(t_int, 1)),
    )
    ddim.set_timesteps(full_steps)
    all_ts = ddim.timesteps
    return all_ts[all_ts <= t_int]


# ─────────────────────────────────────────────
# Encode / denoise / reconstruct
# ─────────────────────────────────────────────
@torch.no_grad()
def encode_to_latents(
    vae, images: torch.Tensor,
    sample: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Encode images to scaled VAE latents.

    Parameters
    ----------
    images : (B, 3, 256, 256) tensor
    sample : bool
        ``True``  → stochastic latent (training).
        ``False`` → deterministic posterior mean (calibration / inference).

    Returns
    -------
    (B, 4, 32, 32) scaled latent tensor.
    """
    dist = vae.encode(normalize_for_vae(images)).latent_dist
    latents = dist.sample(generator=generator) if sample else dist.mean
    return latents * C.SCALING_FACTOR


@torch.no_grad()
def ddim_denoise(
    unet, ddim: DDIMScheduler,
    z_noisy: torch.Tensor, timesteps: torch.Tensor,
) -> torch.Tensor:
    """Run the DDIM reverse loop over ``timesteps`` and return the final latent."""
    for t in timesteps:
        noise_pred = unet(z_noisy, t).sample
        z_noisy = ddim.step(noise_pred, t, z_noisy).prev_sample
    return z_noisy


@torch.no_grad()
def reconstruct_healthy(
    vae, unet, ddim, images, timesteps,
    t_int=C.T_INT, generator=None,
):
    """Full inference pipeline: encode → noise @ T_int → DDIM denoise → decode.

    Returns
    -------
    orig_norm  : (B, 3, 256, 256)  normalised input (comparison reference)
    recon      : (B, 3, 256, 256)  decoded healthy reconstruction
    z0         : (B, 4, 32, 32)    clean latent
    z_denoised : (B, 4, 32, 32)    denoised latent
    """
    z0 = encode_to_latents(vae, images, sample=False)  # deterministic
    noise = torch.randn(z0.shape, device=z0.device, generator=generator)
    t_tensor = torch.tensor([t_int], device=z0.device, dtype=torch.long)
    z_noisy = ddim.add_noise(z0, noise, t_tensor)
    z_denoised = ddim_denoise(unet, ddim, z_noisy, timesteps)
    recon = vae.decode(z_denoised / C.SCALING_FACTOR).sample
    orig_norm = normalize_for_vae(images)
    return orig_norm, recon, z0, z_denoised
