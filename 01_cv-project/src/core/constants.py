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
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(env: str, default: Path) -> Path:
    val = os.environ.get(env)
    return Path(val).expanduser().resolve() if val else default


RAW_DATA_DIR    = _path("CV_RAW_DIR",       PROJECT_ROOT / "data" / "BraTS-PEDs-v1" / "Training")
PROCESSED_DIR   = _path("CV_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed")
HEALTHY_DIR     = _path("CV_HEALTHY_DIR", PROCESSED_DIR / "train_healthy")
PILOT_HEALTHY_DIR = PROCESSED_DIR / "train_healthy_pilot"
VAL_HEALTHY_DIR = PROCESSED_DIR / "val_healthy"
ANOMALOUS_DIR   = PROCESSED_DIR / "test_anomalous"
MASKS_DIR       = PROCESSED_DIR / "test_masks"

BASELINE_PATH    = PROCESSED_DIR / "M_baseline.npy"
CALIBRATION_PATH = PROCESSED_DIR / "calibration.json"

# Multi-timestep calibration artefacts (one M_baseline per T_int level).
MULTI_BASELINE_PATH    = PROCESSED_DIR / "M_baseline_multi.npz"
MULTI_CALIBRATION_PATH = PROCESSED_DIR / "calibration_multi.json"

# VAE fine-tuning datasets — ALL slices (healthy + lesion), patient-disjoint.
# The VAE is a *codec*: it must reconstruct lesions faithfully, so unlike the
# diffusion stage it is trained on every slice, not healthy-only.
VAE_TRAIN_DIR     = PROCESSED_DIR / "vae_train"
VAE_VAL_DIR       = PROCESSED_DIR / "vae_val"
VAE_VAL_MASKS_DIR = PROCESSED_DIR / "vae_val_masks"   # lesion masks for val fidelity

SPLITS_DIR = PROJECT_ROOT / "splits"

MODEL_DIR    = _path("CV_MODEL_DIR",   PROJECT_ROOT / "models")
UNET_DIR     = MODEL_DIR / "unet"
UNET_EMA_DIR = MODEL_DIR / "unet_ema"
VAE_FT_DIR   = MODEL_DIR / "vae_ft"      # fine-tuned medical VAE (train_vae.py output)

RESULTS_DIR   = _path("CV_RESULTS_DIR", PROJECT_ROOT / "results")
TRAJ_DIR      = RESULTS_DIR / "xai_trajectories"
EVAL_DIR      = RESULTS_DIR / "evaluation"

# Unified per-run logs: logs/<stage>_<timestamp>/  (run.log, metrics.csv/jsonl,
# config_snapshot.json, tensorboard/).  See logkit.RunLogger.
LOG_DIR = _path("CV_LOG_DIR", PROJECT_ROOT / "logs")
TRAIN_LOG_DIR = LOG_DIR

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

# ── VAE fine-tuning (train_vae.py) ───────────────────────────────────
# A natural-image VAE blurs lesion texture; fine-tuning the pretrained codec on
# BraTS (ALL slices) keeps it a *faithful* reconstructor while preserving the
# 4ch / 32×32 latent geometry, so the diffusion UNet stays unchanged.  Loss is
# L1 + λ_lpips·LPIPS (Zhang et al. 2018) + λ_kl·KL (Rombach et al. 2022, LDM).
USE_FINETUNED_VAE = os.environ.get("CV_USE_FINETUNED_VAE", "0") == "1"
# ↑ Set CV_USE_FINETUNED_VAE=1 (or edit to True) once train_vae.py has populated
#   VAE_FT_DIR, so the diffusion / calibration / evaluation stages load the
#   fine-tuned medical codec via resolve_vae_source().
VAE_FT_LR         = 1e-5    # low LR — we are adapting, not retraining from scratch
VAE_FT_EPOCHS     = 30
VAE_FT_BATCH      = 4
VAE_KL_WEIGHT     = 1e-7    # gentle regulariser — at 1e-6 the raw SD-VAE KL (~6e4)
                           # dominated the loss; 1e-7 lets L1+LPIPS drive reconstruction
VAE_LPIPS_WEIGHT  = 0.1
VAE_FT_VAL_EVERY  = 1       # run validation every N epochs
VAE_FT_XAI_EVERY  = 5       # save reconstruction/explainability panels every N epochs


def resolve_vae_source() -> str:
    """Return the VAE checkpoint the downstream (diffusion) stage should load:
    the locally fine-tuned medical codec when requested and present, otherwise
    the pretrained Hugging Face codec.  Keeps a single switch (USE_FINETUNED_VAE)
    in charge of which VAE the whole pipeline runs on."""
    if USE_FINETUNED_VAE and (VAE_FT_DIR / "config.json").exists():
        return str(VAE_FT_DIR)
    return VAE_CKPT

# ─────────────────────────────────────────────
# Diffusion
# ─────────────────────────────────────────────
NUM_TRAIN_TIMESTEPS = 1000
BETA_SCHEDULE       = "squaredcos_cap_v2"
BETA_START          = 0.0001
BETA_END            = 0.02
PREDICTION_TYPE     = "epsilon"

DDIM_STEPS = 50
# Intermediate noise level for partial-noise reconstruction — shared by
# training-calibration, evaluation, and visualisation.
T_INT = 300

# ─────────────────────────────────────────────
# Multi-timestep residual aggregation (evaluation-time; no UNet retraining)
# ─────────────────────────────────────────────
# A subtle, low-contrast lesion is effectively "noised out" at a low T (the
# model happily reconstructs healthy tissue in its place → large residual),
# whereas a massive, high-contrast lesion still retains structural remnants at
# a high T (the model reconstructs the lesion → suppressed residual).  A single
# fixed T_INT therefore cannot expose both regimes.  When USE_MULTI_T is on,
# the anomaly score is computed at each T in MULTI_T_LIST and aggregated, so the
# map captures both fine texture anomalies and large structural deformities.
#
# This is calibration + inference only (run src/recalibrate.py) — the UNet is
# NOT retrained.
USE_MULTI_T  = True
MULTI_T_LIST = [100, 250, 400]
# How per-T (healthy-scale-standardised) score maps are combined:
#   "mean" → smoother, fewer false positives (default)
#   "max"  → most sensitive; flags a voxel anomalous at ANY noise level
MULTI_T_AGG  = "mean"

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

# ── Checkpointing (ckptkit) — resumable bundles for Colab/preemption safety ──
# Point CV_MODEL_DIR at Google Drive so these survive a disconnect, then pass
# --resume to continue.  Periodic epoch snapshots are pruned to CKPT_KEEP_LAST.
CKPT_EVERY     = 5
CKPT_KEEP_LAST = 3
VAE_CKPT_DIR   = MODEL_DIR / "vae_ft_ckpt"     # train_vae.py resume bundles
UNET_CKPT_DIR  = MODEL_DIR / "unet_ckpt"       # train_healthy_manifold.py resume bundles

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


def snapshot() -> dict:
    """Serialisable dict of the public config constants — written into every run
    directory (config_snapshot.json) so any logged result is fully reproducible."""
    out = {}
    for k, v in sorted(globals().items()):
        if k.isupper() and not k.startswith("_"):
            out[k] = str(v) if isinstance(v, Path) else v
    return out
