"""
Central configuration — single source of truth
================================================
Every script imports paths and hyperparameters from here so that the
training, calibration, evaluation and visualisation stages stay perfectly
consistent (this is what fixes the old ``T_INT = 250`` vs ``350`` bug).

    from config import T_INT, SCALING_FACTOR, VAE_CKPT, BASELINE_PATH, ...
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
# The data / model / results roots can be redirected with environment
# variables (CV_RAW_DIR, CV_PROCESSED_DIR, CV_MODEL_DIR, CV_RESULTS_DIR).
# This lets a smoke test run in a sandbox without clobbering real artifacts;
# unset, everything resolves under the project tree as before.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path(env: str, default: Path) -> Path:
    val = os.environ.get(env)
    return Path(val).expanduser().resolve() if val else default


RAW_DATA_DIR    = _path("CV_RAW_DIR",       PROJECT_ROOT / "data" / "BraTS-PEDs-v1" / "Training")
PROCESSED_DIR   = _path("CV_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed")
HEALTHY_DIR     = PROCESSED_DIR / "train_healthy"
ANOMALOUS_DIR   = PROCESSED_DIR / "test_anomalous"
MASKS_DIR       = PROCESSED_DIR / "test_masks"

BASELINE_PATH   = PROCESSED_DIR / "M_baseline.npy"
CALIBRATION_PATH = PROCESSED_DIR / "calibration.json"   # threshold + scales + t_int

MODEL_DIR       = _path("CV_MODEL_DIR",   PROJECT_ROOT / "model")
UNET_DIR        = MODEL_DIR / "unet"          # raw (last/best) weights
UNET_EMA_DIR    = MODEL_DIR / "unet_ema"      # EMA weights (used for inference)

RESULTS_DIR     = _path("CV_RESULTS_DIR", PROJECT_ROOT / "results")
TRAJ_DIR        = RESULTS_DIR / "trajectory"

# ─────────────────────────────────────────────
# Data / preprocessing
# ─────────────────────────────────────────────
MODALITIES   = ["t1c", "t2w", "t2f"]   # 3-channel stack (t1n discarded)
SEG_SUFFIX   = "seg"
TARGET_SIZE  = 256                      # pad 240×240 → 256×256
MIN_BRAIN_FRAC = 0.05                   # skip near-empty slices

# ─────────────────────────────────────────────
# VAE
# ─────────────────────────────────────────────
VAE_CKPT       = "stabilityai/sd-vae-ft-mse"
SCALING_FACTOR = 0.18215

# The SD VAE expects inputs in ~[-1, 1].  Our slices are z-score normalised
# (unbounded floats), so we clip to ±VAE_CLIP sigma and linearly map that
# range onto [-1, 1] before encoding.  Background (0) stays 0.
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
# Intermediate noise level for partial-noise reconstruction.  ONE value,
# shared by training-calibration, evaluation and visualisation.
T_INT = 300

# ─────────────────────────────────────────────
# SAAM (Self-Attention Attribution Maps)
# ─────────────────────────────────────────────
# Only aggregate attention layers whose sequence length ≥ this value.  The
# UNet's highest-resolution attention runs at 16×16 (seq_len=256); lower
# resolutions (8×8=64, 4×4=16) upsample into blurry "halo" artefacts.
MIN_SEQ_LEN = 256

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
EMA_DECAY      = 0.999     # window ≈ 1/(1-decay) steps; suits these dataset sizes
GRAD_CLIP_NORM = 1.0
LR_ETA_MIN     = 1e-6      # cosine-annealing floor

# ─────────────────────────────────────────────
# Calibration / scoring
# ─────────────────────────────────────────────
MAX_CAL_SAMPLES      = 100   # held-out healthy slices used for baseline+threshold
THRESHOLD_PERCENTILE = 99    # operational threshold from healthy residual dist.

# Dual-space fusion (M_pixel + M_latent).  OFF by default — the pixel-only
# path is the validated default.  When ON, maps are standardised by healthy
# scales (saved at calibration) so a single global threshold stays valid.
USE_LATENT_FUSION   = False
LATENT_FUSION_ALPHA = 0.5

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
SEED = 42
