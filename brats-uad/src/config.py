"""
Central configuration — single source of truth.
=================================================
Every script imports paths and hyper-parameters from here so the data,
training, calibration, and evaluation stages stay perfectly consistent.

All paths can be redirected with environment variables so the same code runs
unchanged on a laptop (smoke tests) and on a RunPod GPU (real training):

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
PKG_ROOT     = Path(__file__).resolve().parents[1]          # …/brats-uad
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

# Processed layout (populated by make_slices.py).  Each slice is stored exactly
# ONCE under slices/ (and its mask, if any, under masks/); the different dataset
# *views* are defined by manifest files listing slice stems — this avoids
# duplicating ~25K slices (25 GB+) across multiple directories.
#   slices/    <patient>_z###.npy   (N_CHANNELS, 256, 256) float16
#   masks/     <patient>_z###.npy   (256, 256) uint8   (lesion slices only)
#   manifests/ *.txt                 one slice stem per line
SLICES_DIR   = PROCESSED_DIR / "slices"
MASKS_DIR    = PROCESSED_DIR / "masks"
MANIFEST_DIR = PROCESSED_DIR / "manifests"

# Dataset views (manifest files, one slice stem per line):
#   vae_train   — every slice, train+val patients  → VAE codec training
#   vae_val     — every slice, val patients         → VAE recon monitoring
#   healthy     — lesion-free + buffer-clean, train patients → diffusion training
#   val_healthy — lesion-free, val patients         → calibration
#   test_anom   — lesion slices, test patients       → evaluation (mask in masks/)
MANIFEST_VAE_TRAIN   = MANIFEST_DIR / "vae_train.txt"
MANIFEST_VAE_VAL     = MANIFEST_DIR / "vae_val.txt"
MANIFEST_HEALTHY     = MANIFEST_DIR / "healthy.txt"
MANIFEST_VAL_HEALTHY = MANIFEST_DIR / "val_healthy.txt"
MANIFEST_TEST_ANOM   = MANIFEST_DIR / "test_anom.txt"
STATS_PATH           = PROCESSED_DIR / "stats.json"

# Model output dirs
VAE_DIR       = MODEL_DIR / "vae"
VAE_CKPT_DIR  = MODEL_DIR / "vae_ckpt"
UNET_DIR      = MODEL_DIR / "unet"
UNET_EMA_DIR  = MODEL_DIR / "unet_ema"
UNET_CKPT_DIR = MODEL_DIR / "unet_ckpt"

# Calibration / scoring artefacts
SCALING_FACTOR_PATH    = MODEL_DIR / "scaling_factor.json"
BASELINE_PATH          = PROCESSED_DIR / "M_baseline.npy"
CALIBRATION_PATH       = PROCESSED_DIR / "calibration.json"
MULTI_BASELINE_PATH    = PROCESSED_DIR / "M_baseline_multi.npz"
MULTI_CALIBRATION_PATH = PROCESSED_DIR / "calibration_multi.json"

# ─────────────────────────────────────────────
# Data / preprocessing
# ─────────────────────────────────────────────
# 4 modalities — T1-native is INCLUDED (the old pipeline dropped it, losing the
# T1ce−T1 contrast-enhancement signal, the strongest active-tumour marker).
MODALITIES   = ["t1n", "t1c", "t2w", "t2f"]
N_CHANNELS   = len(MODALITIES)
SEG_SUFFIX   = "seg"
TARGET_SIZE  = 256
LATENT_SIZE  = 32                  # 256 / 8  (×8 spatial downsample)
LATENT_CH    = 4

MIN_BRAIN_FRAC = 0.05              # skip near-empty slices
# Slices are saved z-scored+percentile-clipped; the VAE sees them divided by this
# clip (in σ units) → bounded ~[-1, 1]. MUST match across every vae.encode call.
VAE_INPUT_CLIP = 5.0
# Robust intensity normalisation: per-volume z-score over brain voxels, then clip
# to these percentiles (NOT a hard ±3σ, which truncated the FLAIR/edema
# hyperintensity that the anomaly score depends on).
NORM_PCT_LOW   = 0.5
NORM_PCT_HIGH  = 99.5
# A lesion-free slice counts as "healthy" only if it is ≥ BUFFER slices away from
# any lesion-bearing slice, reducing mass-effect contamination of the manifold.
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
MULTI_T_AGG  = "mean"              # "mean" (smoother) | "max" (most sensitive)

# ─────────────────────────────────────────────
# Scoring — modality-weighted residual
# ─────────────────────────────────────────────
# FLAIR (t2f, edema) and the CE map (t1c−t1n, enhancement) carry the most lesion
# signal, so they dominate the per-voxel anomaly score instead of a flat mean.
# Order matches MODALITIES = [t1n, t1c, t2w, t2f].
MODALITY_WEIGHTS = [0.5, 1.0, 0.75, 1.5]
USE_CE_CHANNEL   = True            # add |(t1c−t1n)_orig − (t1c−t1n)_recon| to the score
CE_WEIGHT        = 1.0

USE_LATENT_FUSION   = False
LATENT_FUSION_ALPHA = 0.5

# ─────────────────────────────────────────────
# VAE training
# ─────────────────────────────────────────────
VAE_BASE_CH    = 64
VAE_CH_MULT    = (1, 2, 4, 4)      # 4 levels → ×8 downsample (256→32)
VAE_NUM_RES    = 2
VAE_KL_WEIGHT  = 1e-6
VAE_MSSSIM_W   = 0.5
VAE_LR         = 2e-4
VAE_EPOCHS     = 60
VAE_BATCH      = 16
VAE_PATIENCE   = 12         # early-stop after N epochs w/o val improvement (0=off)

# ─────────────────────────────────────────────
# Diffusion training
# ─────────────────────────────────────────────
UNET_BASE_CHANNELS = (128, 256, 384, 512)
DIFF_LR        = 1e-4
DIFF_EPOCHS    = 40
DIFF_BATCH     = 32
DIFF_PATIENCE  = 10         # early-stop after N epochs w/o val improvement (0=off)
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

# Fallback used ONLY until train_vae.py computes the empirical value. Never the
# SD constant 0.18215 — that was calibrated for natural-image latents.
DEFAULT_SCALING_FACTOR = 1.0


def load_scaling_factor() -> float:
    """Empirical 1/std of VAE latents, written by compute_scaling_factor.py."""
    import json
    if SCALING_FACTOR_PATH.exists():
        with open(SCALING_FACTOR_PATH) as f:
            return float(json.load(f)["scaling_factor"])
    return DEFAULT_SCALING_FACTOR


def snapshot() -> dict:
    """Serialisable dict of public config constants — written into each run dir."""
    out = {}
    for k, v in sorted(globals().items()):
        if k.isupper() and not k.startswith("_"):
            out[k] = str(v) if isinstance(v, Path) else v
    return out
