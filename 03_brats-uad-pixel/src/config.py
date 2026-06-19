"""
Central configuration — Pixel-Space Diffusion (no VAE).
========================================================
Every script imports paths and hyper-parameters from here so the data,
training, calibration, and evaluation stages stay perfectly consistent.

This config is for PIXEL-SPACE diffusion: the UNet operates directly on
(4, 256, 256) normalised MRI slices — no VAE compression, no latent space.
Sharp edges are perfectly preserved, eliminating the boundary-artifact problem
that plagued the latent-diffusion pipeline.

All paths can be redirected with environment variables so the same code runs
unchanged on a laptop (smoke tests) and on a GPU box (real training):

    BUAD_DATA_ROOT      raw BraTS-PEDs root (contains <patient>/<patient>-t1n.nii.gz …)
    BUAD_SPLITS_DIR     dir with train.txt / val.txt / test.txt (patient IDs)
    BUAD_PROCESSED_DIR  output dir for extracted .npy slices
    BUAD_MODEL_DIR      where trained models are saved
    BUAD_RESULTS_DIR    where figures / metrics are saved
"""

from __future__ import annotations

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Roots
# ─────────────────────────────────────────────
PKG_ROOT     = Path(__file__).resolve().parents[1]          # …/brats-uad-pixel
PROJECT_ROOT = PKG_ROOT.parent                              # …/deep_vision


def _path(env: str, default: Path) -> Path:
    val = os.environ.get(env)
    return Path(val).expanduser().resolve() if val else default


# Raw data + splits default to the existing local copy; override on the GPU box.
DATA_ROOT   = _path("BUAD_DATA_ROOT",
                    PROJECT_ROOT / "cv-project" / "data" / "BraTS-PEDs-v1" / "Training")
SPLITS_DIR  = _path("BUAD_SPLITS_DIR", PKG_ROOT / "splits")

PROCESSED_DIR = _path("BUAD_PROCESSED_DIR", PKG_ROOT / "data" / "processed")
MODEL_DIR     = _path("BUAD_MODEL_DIR",     PKG_ROOT / "models")
RESULTS_DIR   = _path("BUAD_RESULTS_DIR",   PKG_ROOT / "results")
LOG_DIR       = _path("BUAD_LOG_DIR",       PKG_ROOT / "logs")

# Processed layout (populated by make_slices.py).
SLICES_DIR   = PROCESSED_DIR / "slices"
MASKS_DIR    = PROCESSED_DIR / "masks"
MANIFEST_DIR = PROCESSED_DIR / "manifests"

# Dataset views (manifest files, one slice stem per line):
MANIFEST_VAE_TRAIN   = MANIFEST_DIR / "vae_train.txt"
MANIFEST_VAE_VAL     = MANIFEST_DIR / "vae_val.txt"
MANIFEST_HEALTHY     = MANIFEST_DIR / "healthy.txt"
MANIFEST_VAL_HEALTHY = MANIFEST_DIR / "val_healthy.txt"
MANIFEST_TEST_ANOM   = MANIFEST_DIR / "test_anom.txt"
STATS_PATH           = PROCESSED_DIR / "stats.json"

# Model output dirs — NO VAE directories needed for pixel-space!
UNET_DIR      = MODEL_DIR / "unet"
UNET_EMA_DIR  = MODEL_DIR / "unet_ema"
UNET_CKPT_DIR = MODEL_DIR / "unet_ckpt"

# Calibration / scoring artefacts
BASELINE_PATH          = PROCESSED_DIR / "M_baseline.npy"
CALIBRATION_PATH       = PROCESSED_DIR / "calibration.json"
MULTI_BASELINE_PATH    = PROCESSED_DIR / "M_baseline_multi.npz"
MULTI_CALIBRATION_PATH = PROCESSED_DIR / "calibration_multi.json"

# ─────────────────────────────────────────────
# Data / preprocessing
# ─────────────────────────────────────────────
MODALITIES   = ["t1n", "t1c", "t2w", "t2f"]
N_CHANNELS   = len(MODALITIES)
SEG_SUFFIX   = "seg"
TARGET_SIZE  = 256

# Pixel-space: NO latent space, UNet works directly on pixel images.
# These are kept only for backward-compat with shared utility code.
PIXEL_SPACE  = True                 # flag: pixel-space mode

MIN_BRAIN_FRAC = 0.05
VAE_INPUT_CLIP = 5.0
NORM_PCT_LOW   = 0.5
NORM_PCT_HIGH  = 99.5
HEALTHY_BUFFER = 3

# ─────────────────────────────────────────────
# Diffusion
# ─────────────────────────────────────────────
NUM_TRAIN_TIMESTEPS = 1000
BETA_SCHEDULE       = "squaredcos_cap_v2"
BETA_START          = 0.0001
BETA_END            = 0.02
PREDICTION_TYPE     = "epsilon"

DDIM_STEPS = 50
T_INT      = 300                   # intermediate noise level for partial-noise recon

# Multi-timestep residual aggregation (eval-time only; no UNet retraining).
USE_MULTI_T  = True
MULTI_T_LIST = [100, 250, 400]
MULTI_T_AGG  = "mean"

# ─────────────────────────────────────────────
# Scoring — modality-weighted residual
# ─────────────────────────────────────────────
MODALITY_WEIGHTS = [0.5, 1.0, 0.75, 1.5]
USE_CE_CHANNEL   = True
CE_WEIGHT        = 1.0

# No latent fusion in pixel-space mode (there is no latent space).
USE_LATENT_FUSION   = False
LATENT_FUSION_ALPHA = 0.0

# ─────────────────────────────────────────────
# Pixel-space UNet architecture
# ─────────────────────────────────────────────
# 5-stage UNet: 256→128→64→32→16→8
# Much deeper than the latent UNet (which only saw 32x32),
# giving it enough receptive field to cover the entire brain.
UNET_BASE_CHANNELS = (64, 128, 256, 512, 512)

# ─────────────────────────────────────────────
# Diffusion training
# ─────────────────────────────────────────────
DIFF_LR        = 1e-4
DIFF_EPOCHS    = 50
DIFF_BATCH     = 16                # larger images need smaller batch (adjustable on H200)
DIFF_PATIENCE  = 15
EMA_DECAY      = 0.999
GRAD_CLIP_NORM = 1.0
LR_ETA_MIN     = 1e-6

# ─────────────────────────────────────────────
# Calibration / checkpointing / reproducibility
# ─────────────────────────────────────────────
MAX_CAL_SAMPLES      = 200
THRESHOLD_PERCENTILE = 95
CKPT_EVERY           = 5
CKPT_KEEP_LAST       = 3
SEED                 = 42


def snapshot() -> dict:
    """Serialisable dict of public config constants — written into each run dir."""
    out = {}
    for k, v in sorted(globals().items()):
        if k.isupper() and not k.startswith("_"):
            out[k] = str(v) if isinstance(v, Path) else v
    return out
