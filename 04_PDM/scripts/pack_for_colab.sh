#!/usr/bin/env bash
# =============================================================================
# pack_for_colab.sh — build the two zips you upload to Google Drive.
# =============================================================================
# Workflow:
#   1. (locally, once) preprocess the raw NIfTI data:
#        export PDM_DATA_ROOT=/path/to/BraTS-PEDs-v1/Training
#        export PDM_PROCESSED_ROOT="$PWD/data/processed"
#        python scripts/00_preprocess.py --splits splits
#   2. run this script to produce:
#        dist/pdm_code.zip       — the code (no data/outputs/caches)
#        dist/pdm_processed.zip  — the preprocessed slices/masks/manifests
#   3. upload both zips to your mounted Google Drive on this Mac.
#   4. in Colab, copy them into /content, unzip, and run (see notebooks/).
#
# Both zips get a sha256 printed so you can verify integrity after upload.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

PROCESSED_ROOT="${PDM_PROCESSED_ROOT:-$HERE/data/processed}"
DIST="$HERE/data"
mkdir -p "$DIST"

# ── 1. Code zip ─────────────────────────────────────────────────────────────
echo "==> Packing code → data/pdm_code.zip"
rm -f "$DIST/pdm_code.zip"
zip -rq "$DIST/pdm_code.zip" \
    src scripts notebooks docs tests report_pdm requirements.txt README.md splits \
    -x "*/__pycache__/*" "*.pyc" "*/.DS_Store" "splits/*.txt.bak" \
       "report_pdm/figures/*" "report_pdm/figures/arrays/*"

# ── 2. Processed-data zip ───────────────────────────────────────────────────
if [ ! -d "$PROCESSED_ROOT" ]; then
  echo "ERROR: processed data not found at $PROCESSED_ROOT" >&2
  echo "Run scripts/00_preprocess.py locally first." >&2
  exit 1
fi
echo "==> Packing processed data ($PROCESSED_ROOT) → data/pdm_processed.zip"
echo "    (this can take a few minutes for tens of thousands of .npy files)"
rm -f "$DIST/pdm_processed.zip"
# -0 (store, no compression): float16 .npy barely compress, and store is much
# faster to zip AND to unzip on Colab. Switch to -1 if you need a smaller file.
( cd "$(dirname "$PROCESSED_ROOT")" && \
  zip -rq -0 "$DIST/pdm_processed.zip" "$(basename "$PROCESSED_ROOT")" \
      -x "*/.DS_Store" )

# ── 3. Report ───────────────────────────────────────────────────────────────
sha() { if command -v shasum >/dev/null; then shasum -a 256 "$1" | cut -d' ' -f1; \
        else sha256sum "$1" | cut -d' ' -f1; fi; }

echo ""
echo "==================== DONE ===================="
for f in pdm_code.zip pdm_processed.zip; do
  p="$DIST/$f"
  printf "%-22s %8s  sha256=%s\n" "$f" "$(du -h "$p" | cut -f1)" "$(sha "$p")"
done
echo "=============================================="
echo "Next: upload both files in data/ to your Google Drive, then open"
echo "      notebooks/run_pdm_colab.ipynb in Colab."
