#!/bin/bash
set -e

# Define colours for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}        Deep Vision Medical Anomaly Detection Pipeline          ${NC}"
echo -e "${BLUE}================================================================${NC}"

# Add src to python path so scripts can be run sequentially
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo -e "\n${GREEN}[1/6] Generating Patient Splits...${NC}"
python src/scripts/01_make_splits.py

echo -e "\n${GREEN}[2/6] Preprocessing 3D Scans into 2D Slices...${NC}"
python src/scripts/02_preprocess.py

echo -e "\n${GREEN}[3/6] Fine-tuning Medical VAE (Codec)...${NC}"
python src/scripts/03_train_vae.py

echo -e "\n${GREEN}[4/6] Training Diffusion UNet on Healthy Manifold...${NC}"
python src/scripts/04_train_diffusion.py

echo -e "\n${GREEN}[5/6] Recalibrating and Computing Residual Thresholds...${NC}"
python src/scripts/05_recalibrate.py

echo -e "\n${GREEN}[6/6] Evaluating on Test Set and Generating xAI Panels...${NC}"
python src/scripts/06_evaluate.py

echo -e "\n${GREEN}Pipeline Complete!${NC}"
echo "Results are stored in the following directories:"
echo "- Models and Checkpoints: models/"
echo "- XAI Visualisations: results/xai_trajectories/"
echo "- Evaluation Metrics: results/evaluation/"
echo "- Training Logs: logs/"
