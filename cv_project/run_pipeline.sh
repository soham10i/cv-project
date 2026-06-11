#!/usr/bin/env bash
# ============================================================================
# End-to-end pipeline: splits → preprocess → train+calibrate → evaluate
# ============================================================================
# Usage:
#   ./run_pipeline.sh                  # full run
#   ./run_pipeline.sh --smoke          # tiny smoke test (fast, CPU/GPU)
#
# Env knobs (override on the command line before ./run_pipeline.sh):
#   MAX_HEALTHY, MAX_ANOMALOUS, EPOCHS, BS, LR, CAL_EVERY
#   CV_RAW_DIR, CV_PROCESSED_DIR, CV_MODEL_DIR, CV_RESULTS_DIR
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"

if [[ -x "$HERE/../myenv/bin/python" ]]; then
    PY="$HERE/../myenv/bin/python"
else
    PY="${PYTHON:-python}"
fi

# ── Defaults (full run) ──────────────────────────────────────────────────────
MAX_HEALTHY="${MAX_HEALTHY:-5000}"
MAX_ANOMALOUS="${MAX_ANOMALOUS:-2000}"
EPOCHS="${EPOCHS:-30}"
BS="${BS:-8}"
LR="${LR:-1e-4}"
CAL_EVERY="${CAL_EVERY:-5}"

TRAIN_RATIO="${TRAIN_RATIO:-0.6}"
VAL_RATIO="${VAL_RATIO:-0.2}"
TEST_RATIO="${TEST_RATIO:-0.2}"

SPLITS_DIR="${CV_SPLITS_DIR:-$HERE/splits}"

# ── Smoke-test overrides ─────────────────────────────────────────────────────
if [[ "${1:-}" == "--smoke" ]]; then
    echo ">>> SMOKE TEST MODE (tiny subset, 1 epoch)"
    MAX_HEALTHY=40
    MAX_ANOMALOUS=12
    EPOCHS=1
    BS=4
    CAL_EVERY=1
fi

echo "=================================================================="
echo " Python      : $PY"
echo " Healthy/Anom: $MAX_HEALTHY / $MAX_ANOMALOUS"
echo " Train       : epochs=$EPOCHS  bs=$BS  lr=$LR  cal_every=$CAL_EVERY"
echo " Splits dir  : $SPLITS_DIR"
echo "=================================================================="

cd "$SRC"

# ── Step 1: Patient-level splits ─────────────────────────────────────────────
echo
echo ">>> [1/6] Generate patient-level train/val/test splits"
"$PY" make_splits.py \
    --train-ratio "$TRAIN_RATIO" \
    --val-ratio   "$VAL_RATIO" \
    --test-ratio  "$TEST_RATIO" \
    --out-dir     "$SPLITS_DIR"

# ── Step 2: Preprocess per split ─────────────────────────────────────────────
echo
echo ">>> [2a/6] Preprocess train split → train_healthy/"
"$PY" preprocess_to_2d.py \
    --split-file     "$SPLITS_DIR/train.txt" \
    --healthy-subdir train_healthy \
    --max-healthy    "$MAX_HEALTHY" \
    --max-anomalous  0

echo
echo ">>> [2b/6] Preprocess val split → val_healthy/"
# Use a fraction of max_healthy for validation (≈20% of training quota)
VAL_MAX_HEALTHY=$(( MAX_HEALTHY / 4 ))
[[ "$VAL_MAX_HEALTHY" -lt 1 ]] && VAL_MAX_HEALTHY=1
"$PY" preprocess_to_2d.py \
    --split-file     "$SPLITS_DIR/val.txt" \
    --healthy-subdir val_healthy \
    --max-healthy    "$VAL_MAX_HEALTHY" \
    --max-anomalous  0

echo
echo ">>> [2c/6] Preprocess test split → test_anomalous/ + test_masks/"
"$PY" preprocess_to_2d.py \
    --split-file     "$SPLITS_DIR/test.txt" \
    --healthy-subdir train_healthy \
    --max-healthy    0 \
    --max-anomalous  "$MAX_ANOMALOUS"

# ── Step 3: Train latent DDPM on healthy manifold ────────────────────────────
echo
echo ">>> [3/6] Train healthy manifold + calibrate (baseline + threshold)"
"$PY" train_healthy_manifold.py \
    --epochs    "$EPOCHS" \
    --bs        "$BS" \
    --lr        "$LR" \
    --cal-every "$CAL_EVERY"

# ── Step 4: Evaluate diffusion UAD pipeline ───────────────────────────────────
echo
echo ">>> [4/6] Evaluate diffusion anomaly detection pipeline"
"$PY" evaluate_pipeline.py

# ── Step 5: VAE-only baseline ─────────────────────────────────────────────────
echo
echo ">>> [5/6] Evaluate VAE-only baseline"
"$PY" evaluate_vae_baseline.py || echo "    (VAE baseline failed; continuing)"

# ── Step 6: Trajectory visualisation ──────────────────────────────────────────
echo
echo ">>> [6/6] Trajectory visualisation"
"$PY" visualize_trajectory.py || echo "    (trajectory viz is optional; skipped on error)"

echo
echo "=================================================================="
echo " DONE."
echo "  Diffusion metrics : ${CV_RESULTS_DIR:-$HERE/results}/metrics.json"
echo "  VAE baseline      : ${CV_RESULTS_DIR:-$HERE/results}/metrics_vae.json"
echo "  XAI trajectory    : ${CV_RESULTS_DIR:-$HERE/results}/trajectory/"
echo "  Train logs        : ${CV_RESULTS_DIR:-$HERE/results}/train_logs/"
echo "=================================================================="
