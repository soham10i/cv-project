"""Diffusion pipeline: encoding, denoising, scoring, and calibration."""

from pipeline.diffusion import (
    normalize_for_vae,
    make_ddpm_scheduler,
    make_ddim_scheduler,
    inference_timesteps,
    encode_to_latents,
    ddim_denoise,
    reconstruct_healthy,
)
from pipeline.scoring import (
    brain_mask_2d,
    latent_residual_2d,
    pixel_residual_2d,
    fuse_maps,
    compute_dice,
)
from pipeline.calibration import (
    calibrate_on_healthy,
    save_calibration,
    load_calibration,
    calibrate_on_healthy_multi_t,
    save_calibration_multi_t,
    load_calibration_multi_t,
    aggregate_t_scores,
    reconstruct_and_score_t,
    score_image_multi_t,
)
from pipeline.metrics import ssim_2d, compute_recon_metrics

__all__ = [
    # diffusion
    "normalize_for_vae", "make_ddpm_scheduler", "make_ddim_scheduler",
    "inference_timesteps", "encode_to_latents", "ddim_denoise",
    "reconstruct_healthy",
    # scoring
    "brain_mask_2d", "latent_residual_2d", "pixel_residual_2d",
    "fuse_maps", "compute_dice",
    # calibration
    "calibrate_on_healthy", "save_calibration", "load_calibration",
    "calibrate_on_healthy_multi_t", "save_calibration_multi_t",
    "load_calibration_multi_t", "aggregate_t_scores",
    "reconstruct_and_score_t", "score_image_multi_t",
    # metrics
    "ssim_2d", "compute_recon_metrics",
]
