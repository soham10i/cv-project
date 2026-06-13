#!/usr/bin/env bash
# ============================================================================
# LOCAL SMOKE TEST  (macOS/MPS or CPU)
# ----------------------------------------------------------------------------
# Proves every pipeline stage RUNS end-to-end on a tiny subset:
#   Stage 1 (VAE fine-tune)  +  Stage 2 (diffusion → calibrate → evaluate)
#
# Metric *quality* is irrelevant here (1-epoch runs are garbage) — this is a
# GO / NO-GO check for committing to a full cloud-GPU run.  It never aborts on
# the first error: it records PASS/FAIL per stage and prints a summary table.
#
# Usage (from anywhere):
#   bash cv_project/smoke_test.sh
#
# Env knobs:  PYTHON, CV_RAW_DIR (if data isn't at the default path),
#             TRAIN_HEALTHY, VAL_HEALTHY, TEST_ANOM, VAE_TOTAL, VAE_PER_PATIENT
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # → cv_project/
SRC="$HERE/src"

if   [[ -x "$HERE/../myenv/bin/python" ]]; then PY="$HERE/../myenv/bin/python"
elif [[ -x "$HERE/.venv/bin/python"   ]]; then PY="$HERE/.venv/bin/python"
else PY="${PYTHON:-python}"; fi

export PYTORCH_ENABLE_MPS_FALLBACK=1     # CPU-fallback for any op without an MPS kernel

# ── tiny caps (override via env) ────────────────────────────────────────────
SPLITS_DIR="${CV_SPLITS_DIR:-$HERE/splits}"
TRAIN_HEALTHY="${TRAIN_HEALTHY:-80}"
VAL_HEALTHY="${VAL_HEALTHY:-40}"
TEST_ANOM="${TEST_ANOM:-30}"
VAE_PER_PATIENT="${VAE_PER_PATIENT:-8}"
VAE_TOTAL="${VAE_TOTAL:-120}"

declare -a STEP RESULT
mark() { STEP+=("$1"); RESULT+=("$2"); }
run()  { local name="$1"; shift; echo; echo ">>> $name";
         if "$@"; then mark "$name" "PASS"; else mark "$name" "FAIL"; echo "    !! FAILED: $name"; fi; }

cd "$SRC"
echo "=================================================================="
echo " LOCAL SMOKE TEST"
echo " Python : $PY"
"$PY" - <<'PY' || true
import torch
print(" torch  :", torch.__version__,
      "| MPS", torch.backends.mps.is_available(),
      "| CUDA", torch.cuda.is_available())
PY
echo "=================================================================="

run "1  splits"            "$PY" make_splits.py --out-dir "$SPLITS_DIR"
run "2a preprocess train"  "$PY" preprocess_to_2d.py --split-file "$SPLITS_DIR/train.txt" \
                                 --healthy-subdir train_healthy --max-healthy "$TRAIN_HEALTHY" --max-anomalous 0
run "2b preprocess val"    "$PY" preprocess_to_2d.py --split-file "$SPLITS_DIR/val.txt" \
                                 --healthy-subdir val_healthy   --max-healthy "$VAL_HEALTHY"   --max-anomalous 0
run "2c preprocess test"   "$PY" preprocess_to_2d.py --split-file "$SPLITS_DIR/test.txt" \
                                 --healthy-subdir train_healthy  --max-healthy 0 --max-anomalous "$TEST_ANOM"
run "3  vae dataset"       "$PY" prepare_vae_dataset.py --max-per-patient "$VAE_PER_PATIENT" --max-total "$VAE_TOTAL"
run "4  VAE smoke"         "$PY" train_vae.py --smoke

# parse the VAE smoke verdict it writes to logs/vae_smoke/smoke_result.json
VERDICT="$("$PY" - <<'PY'
import json, glob, os
d = sorted(glob.glob("../logs/vae_smoke*"), key=os.path.getmtime)
if not d: print("NOFILE")
else:
    p = os.path.join(d[-1], "smoke_result.json")
    try:    print("PASS" if json.load(open(p)).get("passed") else "FAIL")
    except Exception: print("NOFILE")
PY
)"
mark "4b VAE verdict" "$VERDICT"

run "5  diffusion 1ep"     "$PY" train_healthy_manifold.py --epochs 1 --bs 4 --cal-every 1
run "6  recalibrate"       "$PY" recalibrate.py
run "7  evaluate"          "$PY" evaluate_pipeline.py --n-images 10 --max-plots 2
run "7b metrics.json"      test -f "${CV_RESULTS_DIR:-$HERE/results}/metrics.json"

# ── summary ────────────────────────────────────────────────────────────────
echo; echo "==================== SMOKE SUMMARY ===================="
fail=0
for i in "${!STEP[@]}"; do
    printf "  %-22s %s\n" "${STEP[$i]}" "${RESULT[$i]}"
    [[ "${RESULT[$i]}" == "PASS" ]] || fail=1
done
echo "======================================================="
echo " Logs    : $HERE/logs/<stage>_*/   (run.log, metrics.csv/jsonl, xai/, tensorboard/)"
echo " Results : ${CV_RESULTS_DIR:-$HERE/results}/metrics.json"
if [[ $fail -eq 0 ]]; then
    echo "✅ ALL GREEN — pipeline runs end-to-end. Cleared for full cloud-GPU run."
    exit 0
else
    echo "❌ Some steps failed — fix before scaling (check the run.log printed above)."
    exit 1
fi
