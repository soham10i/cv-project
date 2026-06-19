#!/usr/bin/env bash
# Create the processed datasets and audit them.
#   bash scripts/make_dataset.sh smoke   # 4 patients   — plumbing test (fast, tiny)
#   bash scripts/make_dataset.sh dev      # 30 patients  — 8GB-laptop pilot
#   bash scripts/make_dataset.sh full     # all patients — actual fine-tuning (~20 GB)
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-full}"
case "$MODE" in
  smoke) LIMIT="--limit-patients 4";  DIR="$PWD/data/processed_smoke" ;;
  dev)   LIMIT="--limit-patients 30"; DIR="$PWD/data/processed_dev" ;;
  full)  LIMIT="";                    DIR="${BUAD_PROCESSED_DIR:-$PWD/data/processed}" ;;
  *) echo "usage: make_dataset.sh [smoke|dev|full]"; exit 1 ;;
esac

export BUAD_PROCESSED_DIR="$DIR"
echo "==> Building '$MODE' dataset → $DIR"
python3 src/data/make_slices.py $LIMIT
echo "==> Auditing"
python3 src/data/dataset_report.py
echo "==> Done. Use it with:  export BUAD_PROCESSED_DIR=$DIR"
