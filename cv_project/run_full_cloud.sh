#!/usr/bin/env bash
# ============================================================================
# FULL-SCALE RUN  (cloud CUDA GPU)
# ----------------------------------------------------------------------------
# Run this AFTER `bash cv_project/smoke_test.sh` is all-green locally.
# utils.get_device() auto-selects CUDA, so no device flag is needed.
#
# Stages:
#   splits → preprocess → VAE dataset → Stage 1 (VAE fine-tune)
#         → Stage 2 (diffusion train + calibrate + evaluate)
#
# Usage:
#   bash cv_project/run_full_cloud.sh                       # full defaults
#   DIFF_EPOCHS=500 VAE_EPOCHS=60 bash cv_project/run_full_cloud.sh
#   nohup bash cv_project/run_full_cloud.sh > full_run.out 2>&1 &   # detached
#
# Env knobs (defaults are full-scale; override as needed):
#   MAX_HEALTHY MAX_ANOMALOUS VAL_HEALTHY VAE_TOTAL VAE_PER_PATIENT
#   VAE_EPOCHS VAE_BS DIFF_EPOCHS DIFF_BS LR CAL_EVERY  PYTHON  CV_RAW_DIR
#
# NOTE on the fine-tuned VAE: Stage 2 currently loads the *pretrained* VAE.
#   To run diffusion on the fine-tuned codec produced by Stage 1, the diffusion/
#   calibrate/evaluate scripts must call config.resolve_vae_source() and you must
#   set USE_FINETUNED_VAE = True  (that is the pending "Stage 2 wiring").
# NOTE on precision: the project is fp32 (no AMP) for portability/correctness.
#   Mixed precision can be added later for throughput once results are validated.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # → cv_project/
SRC="$HERE/src"
PY="${PYTHON:-python}"
[[ -x "$HERE/../myenv/bin/python" ]] && PY="$HERE/../myenv/bin/python"

# ── full-scale defaults ─────────────────────────────────────────────────────
# Defaults are tuned for a single 15 GB GPU (Colab T4, fp32). Bump for bigger cards.
MAX_HEALTHY="${MAX_HEALTHY:-5000}"
MAX_ANOMALOUS="${MAX_ANOMALOUS:-2000}"
VAL_HEALTHY="${VAL_HEALTHY:-1000}"
VAE_TOTAL="${VAE_TOTAL:-8000}"
VAE_PER_PATIENT="${VAE_PER_PATIENT:-40}"
VAE_EPOCHS="${VAE_EPOCHS:-40}"
VAE_BS="${VAE_BS:-4}"           # 256² activations are heavy; 4 is safe on T4 (try 8)
VAE_ACCUM="${VAE_ACCUM:-4}"     # effective VAE batch = VAE_BS × VAE_ACCUM = 16
DIFF_EPOCHS="${DIFF_EPOCHS:-300}"
DIFF_BS="${DIFF_BS:-16}"        # latent 32² is light; 16 safe on T4 (try 32)
DIFF_ACCUM="${DIFF_ACCUM:-1}"
LR="${LR:-1e-4}"
CAL_EVERY="${CAL_EVERY:-20}"
NUM_WORKERS="${NUM_WORKERS:-2}" # T4 Colab ≈ 2 vCPUs
RESUME="${RESUME:-1}"           # re-running after a Colab disconnect continues
SPLITS_DIR="${CV_SPLITS_DIR:-$HERE/splits}"
RESUME_FLAG=""; [[ "$RESUME" == "1" ]] && RESUME_FLAG="--resume"

cd "$SRC"

# Fail fast if no GPU is visible — this script is meant for cloud CUDA.
"$PY" - <<'PY'
import torch
assert torch.cuda.is_available(), "No CUDA GPU visible — run smoke_test.sh on CPU/MPS, or fix the GPU env."
print("GPU:", torch.cuda.get_device_name(0))
PY

echo ">>> [1/7] splits"
"$PY" make_splits.py --out-dir "$SPLITS_DIR"

echo ">>> [2a/7] preprocess train → train_healthy/"
"$PY" preprocess_to_2d.py --split-file "$SPLITS_DIR/train.txt" \
      --healthy-subdir train_healthy --max-healthy "$MAX_HEALTHY" --max-anomalous 0

echo ">>> [2b/7] preprocess val → val_healthy/"
"$PY" preprocess_to_2d.py --split-file "$SPLITS_DIR/val.txt" \
      --healthy-subdir val_healthy   --max-healthy "$VAL_HEALTHY" --max-anomalous 0

echo ">>> [2c/7] preprocess test → test_anomalous/ + test_masks/"
"$PY" preprocess_to_2d.py --split-file "$SPLITS_DIR/test.txt" \
      --healthy-subdir train_healthy --max-healthy 0 --max-anomalous "$MAX_ANOMALOUS"

echo ">>> [3/7] VAE dataset (all slices: healthy + lesion)"
"$PY" prepare_vae_dataset.py --max-per-patient "$VAE_PER_PATIENT" --max-total "$VAE_TOTAL"

echo ">>> [4/7] Stage 1 — VAE fine-tune (L1 + LPIPS + KL)"
"$PY" train_vae.py --epochs "$VAE_EPOCHS" --bs "$VAE_BS" --grad-accum "$VAE_ACCUM" \
      --num-workers "$NUM_WORKERS" $RESUME_FLAG

echo ">>> [5/7] Stage 2 — diffusion train + calibrate"
"$PY" train_healthy_manifold.py --epochs "$DIFF_EPOCHS" --bs "$DIFF_BS" \
      --grad-accum "$DIFF_ACCUM" --num-workers "$NUM_WORKERS" --lr "$LR" \
      --cal-every "$CAL_EVERY" $RESUME_FLAG

echo ">>> [6/7] recalibrate (baselines + thresholds)"
"$PY" recalibrate.py

echo ">>> [7/7] evaluate (DICE / AUPRC / oracle)"
"$PY" evaluate_pipeline.py

echo "=================================================================="
echo " DONE."
echo "  Metrics : $HERE/results/metrics.json"
echo "  Runs    : $HERE/logs/<stage>_*/   (run.log, metrics.csv/jsonl, tensorboard/)"
echo "  VAE ckpt: $HERE/model/vae_ft/"
echo "=================================================================="
