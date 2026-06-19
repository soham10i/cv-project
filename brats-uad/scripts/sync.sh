#!/usr/bin/env bash
# Optional: mirror models + results to a backup location (e.g. mounted volume or
# rclone remote) so a pod termination never loses a trained model.
#   bash scripts/sync.sh /workspace/backup
#   bash scripts/sync.sh remote:brats-uad    # if rclone is configured
set -euo pipefail
DEST="${1:?usage: sync.sh <dest dir or rclone remote>}"
: "${BUAD_MODEL_DIR:=/workspace/models}"
: "${BUAD_RESULTS_DIR:=/workspace/results}"

if command -v rclone >/dev/null && [[ "$DEST" == *:* ]]; then
  rclone copy "$BUAD_MODEL_DIR"   "$DEST/models"  --progress
  rclone copy "$BUAD_RESULTS_DIR" "$DEST/results" --progress
else
  mkdir -p "$DEST"
  rsync -a "$BUAD_MODEL_DIR/"   "$DEST/models/"
  rsync -a "$BUAD_RESULTS_DIR/" "$DEST/results/"
fi
echo "Synced models + results → $DEST"
