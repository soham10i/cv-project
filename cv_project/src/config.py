"""
Central configuration — single source of truth
================================================
Every script imports paths and hyperparameters from here so that the
training, calibration, evaluation and visualisation stages stay perfectly
consistent.

    from config import T_INT, SCALING_FACTOR, VAE_CKPT, BASELINE_PATH, ...
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
# The data / model / results roots can be redirected with environment
# variables (CV_RAW_DIR, CV_PROCESSED_DIR, CV_MODEL_DIR, CV_RESULTS_DIR).
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path(env: str, default: Path) -> Path:
    val = os.environ.get(env)
    return Path(val).expanduser().resolve() if val else default


RAW_DATA_DIR    = _path("CV_RAW_DIR",       PROJECT_ROOT / "data" / "BraTS-PEDs-v1" / "Training")
PROCESSED_DIR   = _path("CV_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed")
HEALTHY_DIR     = PROCESSED_DIR / "train_healthy"
VAL_HEALTHY_DIR = PROCESSED_DIR / "val_healthy"
ANOMALOUS_DIR   = PROCESSED_DIR / "test_anomalous"
MASKS_DIR       = PROCESSED_DIR / "test_masks"

BASELINE_PATH    = PROCESSED_DIR / "M_baseline.npy"
CALIBRATION_PATH = PROCESSED_DIR / "calibration.json"

SPLITS_DIR = PROJECT_ROOT / "splits"

MODEL_DIR    = _path("CV_MODEL_DIR",   PROJECT_ROOT / "model")
UNET_DIR     = MODEL_DIR / "unet"
UNET_EMA_DIR = MODEL_DIR / "unet_ema"

RESULTS_DIR   = _path("CV_RESULTS_DIR", PROJECT_ROOT / "results")
TRAJ_DIR      = RESULTS_DIR / "trajectory"
TRAIN_LOG_DIR = RESULTS_DIR / "train_logs"

# ─────────────────────────────────────────────
# Data / preprocessing
# ─────────────────────────────────────────────
MODALITIES     = ["t1c", "t2w", "t2f"]
SEG_SUFFIX     = "seg"
TARGET_SIZE    = 256
MIN_BRAIN_FRAC = 0.05

# ─────────────────────────────────────────────
# VAE
# ─────────────────────────────────────────────
VAE_CKPT       = "stabilityai/sd-vae-ft-mse"
SCALING_FACTOR = 0.18215

# SD VAE expects inputs in ~[-1, 1]; z-scored slices are clipped to ±VAE_CLIP
# and linearly mapped onto that range before encoding.
VAE_CLIP = 3.0

# ─────────────────────────────────────────────
# Diffusion
# ─────────────────────────────────────────────
NUM_TRAIN_TIMESTEPS = 1000
BETA_SCHEDULE       = "linear"
BETA_START          = 0.0001
BETA_END            = 0.02
PREDICTION_TYPE     = "epsilon"

DDIM_STEPS = 50
# Intermediate noise level for partial-noise reconstruction — shared by
# training-calibration, evaluation, and visualisation.
T_INT = 300

# ─────────────────────────────────────────────
# SAAM (Self-Attention Attribution Maps)
# ─────────────────────────────────────────────
# Only aggregate layers whose sequence length ≥ this value to avoid
# blurry low-resolution halo artefacts.
MIN_SEQ_LEN = 256

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
EMA_DECAY      = 0.999
GRAD_CLIP_NORM = 1.0
LR_ETA_MIN     = 1e-6

# ─────────────────────────────────────────────
# Calibration / scoring
# ─────────────────────────────────────────────
MAX_CAL_SAMPLES      = 100
# Percentile of the POOLED healthy brain-voxel score distribution used as the
# operating threshold → a per-voxel false-positive rate of (100-percentile)%.
# 95 (5% voxel FPR) is a reasonable segmentation default; the ablation sweeps
# this to find the DICE-optimal point for a given trained model.
THRESHOLD_PERCENTILE = 90

# Dual-space fusion (pixel + latent residual). OFF by default.
USE_LATENT_FUSION   = True
LATENT_FUSION_ALPHA = 0.5

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
SEED = 42
