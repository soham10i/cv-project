#!/usr/bin/env bash
# Full pipeline, end to end. Each stage is resumable (--resume) so a stop/restart
# continues where it left off. Run `source .env` first on RunPod.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=python3

echo "==================================================================="
echo " STAGE 0 — extract slices  (CPU; ~6 min for 259 patients) + audit"
echo "==================================================================="
$PY src/data/make_slices.py
$PY src/data/dataset_report.py

echo "==================================================================="
echo " STAGE 1 — train medical KL-VAE  (GPU; ~15-25h depending on epochs)"
echo "==================================================================="
$PY src/train_vae.py --resume
$PY src/compute_scaling_factor.py

echo "==================================================================="
echo " STAGE 2 — train healthy-manifold latent DDPM  (GPU; ~20-30h)"
echo "==================================================================="
$PY src/train_diffusion.py --resume

echo "==================================================================="
echo " STAGE 3 — calibrate thresholds on healthy val slices"
echo "==================================================================="
$PY src/calibrate.py

echo "==================================================================="
echo " STAGE 4 — evaluate on test lesion slices + report figures"
echo "==================================================================="
$PY src/evaluate.py
$PY src/make_report_figures.py
$PY src/make_xai_panels.py --n 8

echo "==================================================================="
echo " DONE. Metrics:  \$BUAD_RESULTS_DIR/metrics.json"
echo "       Figures:  \$BUAD_RESULTS_DIR/figures/  +  evaluation/ panels"
echo "==================================================================="
