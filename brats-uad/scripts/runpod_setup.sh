#!/usr/bin/env bash
# RunPod (or any fresh GPU box) one-time setup.
# Assumes a PyTorch base image; installs the remaining deps and sets paths.
set -euo pipefail

PROJ="${1:-/workspace/brats-uad}"
cd "$PROJ"

echo "==> Installing requirements"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "==> Persistent paths (point these at the RunPod volume so they survive a stop)"
cat > .env <<'EOF'
# Raw BraTS-PEDs Training dir (upload here, or mount a volume)
export BUAD_DATA_ROOT=/workspace/data/BraTS-PEDs-v1/Training
export BUAD_SPLITS_DIR=/workspace/brats-uad/splits
# Outputs on the persistent volume
export BUAD_PROCESSED_DIR=/workspace/data/processed
export BUAD_MODEL_DIR=/workspace/models
export BUAD_RESULTS_DIR=/workspace/results
export BUAD_LOG_DIR=/workspace/logs
EOF
echo "    wrote $PROJ/.env  — run:  source .env"

echo "==> GPU check"
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "==> Done. Next:  source .env && bash scripts/run_all.sh"
