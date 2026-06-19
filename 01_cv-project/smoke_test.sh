#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo "    Deep Vision Pipeline Smoke Test     "
echo "========================================"
echo "Running on MacOS with minimal parameters to ensure no crashes."

export PYTHONPATH=src

echo -e "\n[1/3] Preprocessing (20 healthy, 5 anomalous)..."
python src/preprocess.py \
    --healthy-subdir train_healthy_pilot \
    --max-healthy 20 \
    --max-anomalous 5 \
    --max-healthy-per-pat 2

echo -e "\n[2/3] Training Diffusion (2 epochs)..."
python src/train_diffusion.py \
    --data-dir data/processed/train_healthy_pilot \
    --epochs 2 \
    --bs 2 \
    --cal-every 1 \
    --val-batches 2 \
    --no-tensorboard

echo -e "\n[3/3] Evaluating Pipeline..."
python src/evaluate.py --n-images 2 --max-plots 2

echo -e "\n========================================"
echo " ✅ Smoke Test Passed! Code is Colab-ready."
echo "========================================"
