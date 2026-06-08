#!/usr/bin/env bash
# ============================================================================
# Ablation sweep — thin wrapper around src/run_ablation.py
# ============================================================================
# The actual sweep is a Python driver so the VAE + EMA UNet load ONCE and each
# (t_int, ddim_steps) cell is reconstructed ONCE; percentile/fusion are derived
# cheaply from cached residual maps.  See src/run_ablation.py for details.
#
# Usage:
#   ./run_ablation.sh                       # full grid, 200 test slices
#   N_IMAGES=100 ./run_ablation.sh          # fewer test slices
#   ./run_ablation.sh --t-int 250 300 --ddim-steps 50 --fusion off
#
# Any extra args are forwarded to run_ablation.py.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"

if [[ -x "$HERE/../myenv/bin/python" ]]; then
    PY="$HERE/../myenv/bin/python"
else
    PY="${PYTHON:-python}"
fi

EXTRA=()
if [[ -n "${N_IMAGES:-}" ]]; then
    EXTRA+=(--n-images "$N_IMAGES")
fi

cd "$SRC"
exec "$PY" run_ablation.py "${EXTRA[@]}" "$@"
