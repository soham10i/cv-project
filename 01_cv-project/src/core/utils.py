"""
Backward-compatible utils.py.

All core logic has been refactored into the `data`, `models`, and `pipeline`
packages. This file merely re-exports the most commonly used components
so older scripts (like `train_vae.py`) continue to function without changes.
"""

import sys

# Data layer
from data.preprocessing import zscore_normalize_volume, symmetric_pad, load_nifti

# Models layer
from models.factory import build_vae, build_unet, load_unet
from models.ema import EMA
from models.attention import AttnMapStore, install_attn_hooks, restore_default_processors, aggregate_step_attention

# Pipeline layer
from pipeline.diffusion import (
    normalize_for_vae, make_ddpm_scheduler, make_ddim_scheduler,
    inference_timesteps, encode_to_latents, ddim_denoise, reconstruct_healthy
)
from pipeline.scoring import brain_mask_2d, latent_residual_2d, pixel_residual_2d, fuse_maps, compute_dice
from pipeline.calibration import (
    calibrate_on_healthy, save_calibration, load_calibration,
    calibrate_on_healthy_multi_t, save_calibration_multi_t, load_calibration_multi_t,
    aggregate_t_scores, reconstruct_and_score_t, score_image_multi_t
)
from pipeline.metrics import ssim_2d, compute_recon_metrics

# Re-export some common things that might be used
import torch
import numpy as np
import random
from core import constants as C

def set_seed(seed: int = C.SEED) -> None:
    """Set global RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Return the best available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def clear_cache() -> None:
    """Clear PyTorch CUDA/MPS cache to free up VRAM."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def make_generator(device: torch.device, seed: int = C.SEED) -> torch.Generator:
    """Create a seeded torch.Generator for the given device."""
    return torch.Generator(device=device).manual_seed(seed)
