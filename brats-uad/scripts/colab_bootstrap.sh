#!/usr/bin/env bash
# One-shot Colab setup — collapses install + dataset-extract + Drive dirs + audit.
# Run AFTER cloning the repo and `cd`-ing into it.
#
#   bash scripts/colab_bootstrap.sh [TAR_NAME] [DRIVE_DIR] [LOCAL_DS]
#
# Defaults:
#   TAR_NAME  = brats_dev.tar
#   DRIVE_DIR = /content/drive/MyDrive/brats-uad
#   LOCAL_DS  = /content/processed
#
# It extracts the dataset to fast LOCAL disk, points all OUTPUTS at Drive, runs
# the leakage audit, and writes /content/brats_env.sh — every training/eval cell
# then just does:  !source /content/brats_env.sh && python src/<script>.py ...
set -euo pipefail
cd "$(dirname "$0")/.."

TAR_NAME="${1:-brats_dev.tar}"
DRIVE="${2:-/content/drive/MyDrive/brats-uad}"
LOCAL_DS="${3:-/content/processed}"
# Resolves to /content/brats_env.sh under the Colab default, portable otherwise.
ENV_FILE="$(dirname "$LOCAL_DS")/brats_env.sh"

echo "==> [1/4] Installing requirements"
pip install -q -r requirements.txt

echo "==> [2/4] Dataset → local SSD ($LOCAL_DS)"
if [ -d "$LOCAL_DS/slices" ]; then
  echo "    already extracted — skipping (delete $LOCAL_DS to force re-extract)"
else
  if [ ! -f "$DRIVE/$TAR_NAME" ]; then
    echo "ERROR: $DRIVE/$TAR_NAME not found. Upload the packed tar to Drive first."
    exit 1
  fi
  # Stage beside LOCAL_DS (same filesystem → fast move). Read tar straight from
  # Drive (one sequential read; the small-file slowness only hits random access).
  STAGE="$(dirname "$LOCAL_DS")/.brats_stage_$$"
  mkdir -p "$STAGE"
  tar -xf "$DRIVE/$TAR_NAME" -C "$STAGE/"      # tar contains a top-level processed/
  rm -rf "$LOCAL_DS"
  mv "$STAGE/processed" "$LOCAL_DS"
  rm -rf "$STAGE"
  echo "    extracted."
fi

echo "==> [3/4] Creating Drive output dirs (persist across disconnects)"
mkdir -p "$DRIVE/models" "$DRIVE/logs" "$DRIVE/results"

# Env file sourced by every subsequent cell.
cat > "$ENV_FILE" <<EOF
export BUAD_PROCESSED_DIR="$LOCAL_DS"
export BUAD_MODEL_DIR="$DRIVE/models"
export BUAD_LOG_DIR="$DRIVE/logs"
export BUAD_RESULTS_DIR="$DRIVE/results"
EOF

echo "==> [4/4] Dataset audit (must show leakage PASS)"
# shellcheck disable=SC1090
source "$ENV_FILE"
python src/data/dataset_report.py

echo
echo "================================================================"
echo " Setup complete. In every training / eval cell, prefix with:"
echo "   !source $ENV_FILE && python src/train_vae.py --epochs 25 --bs 16 --resume"
echo " Outputs autosave to: $DRIVE  (models / logs / results)"
echo "================================================================"
