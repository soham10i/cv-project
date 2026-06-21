# Running PDM on Google Colab (A100)

This is the credit-aware runbook for training/evaluating PDM on Colab using
**compute credits**. Companion notebook: [`notebooks/run_pdm_colab.ipynb`](../notebooks/run_pdm_colab.ipynb).

> **GPU note.** Colab does *not* offer H100. Top tier is **A100 40GB**. Your model
> is small (25M params, 96×96 patches), so an **L4** is actually the best
> credit-value and an A100 is only worth it for faster wall-clock. This runbook
> targets A100 as requested; swap the runtime to L4 to stretch credits ~3×.

---

## The data strategy (why two zips → /content)

Your raw dataset is 33 GB. We avoid moving that to Colab by:

1. **Preprocessing locally** (on your Mac) — turns NIfTI into 2D slices + manifests.
2. **Zipping** the processed output + the code into two files.
3. **Uploading** the zips to Google Drive (small, one-time).
4. In Colab, **copying the zips into `/content`** (fast local SSD) and unzipping
   there. Training reads from `/content`, **not** from the Drive FUSE mount —
   reading tens of thousands of `.npy` over Drive is 10–50× slower and will
   bottleneck the GPU.
5. **Outputs/checkpoints are written to Drive** so a disconnect never loses
   progress.

---

## Step A — Local (on your Mac), once

```bash
cd 04_PDM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# point at your raw data + choose where processed output goes
export PDM_DATA_ROOT="/Users/sohampatel/OTH/Study Material/semester 4/deep_vision/BraTS-PEDs-v1/Training"
export PDM_PROCESSED_ROOT="$PWD/data/processed"

# 1) reproducible patient-level splits (no leakage). 257 patients → 180/39/38.
python scripts/make_splits.py --ratio 70 15 15 --seed 42

# 2) extract slices + build manifests (CPU-bound; ~1–1.5 h for all 257 patients)
python scripts/00_preprocess.py --splits splits

# 3) VERIFY before spending any credits — checks leakage, masks, intensities
python scripts/verify_dataset.py            # must print "VERIFICATION PASSED"

# 4) build the two zips (dist/pdm_code.zip, dist/pdm_processed.zip)
bash scripts/pack_for_colab.sh
```

The script prints each zip's size + sha256. **Note the processed-data size**:
- full 257 patients ≈ ~20 GB (needs Google One 100 GB on Drive),
- a 60-patient subset ≈ ~5 GB (fits free Drive). To make a subset, just put
  fewer IDs in `splits/*.txt` before preprocessing.

Upload both `dist/*.zip` to a Drive folder, e.g. `MyDrive/pdm/`.

---

## Step B — Colab

Open `notebooks/run_pdm_colab.ipynb` in Colab, set runtime to **A100 GPU**, set
`DRIVE_DIR` to your upload folder, and run the cells top to bottom. They:
mount Drive → copy+unzip the zips into `/content` → install deps → set env vars
→ train (`--resume`) → calibrate → evaluate → explain.

---

## Surviving disconnects (critical on A100)

Colab sessions drop (idle timeout + ~12 h cap). The trainer checkpoints a full
state (model + EMA + optimizer + scheduler + epoch) to Drive every epoch, so:

- **If the session dies mid-training, just re-run the training cell.** `--resume`
  reloads `outputs/checkpoints/last_state.pt` from Drive and continues from the
  next epoch. You lose at most one epoch.
- The same `--resume` is safe on a fresh run (starts from scratch if no state).

---

## Credit budget (A100)

A100 on Colab burns roughly **~12–13 compute units/hour** (rates drift — check
your session). With **50 credits ≈ ~4 hours** of A100:

| Stage | A100 time | Notes |
|------|-----------|-------|
| Copy + unzip data | ~3–8 min | one-time per session, off-GPU mostly |
| Train (300 ep, bs 128, bf16) | ~2.5 h | the main spend |
| Calibrate (200 slices) | ~10 min | |
| Evaluate (full test set) | ~30–45 min | scales with #slices × 3 noise scales |
| Explain (12 cases) | ~5 min | |

**This fits in ~50 credits for one full run.** To leave room for an ablation
(e.g. simplex-vs-gaussian) either: run on **L4** instead (~3× cheaper), reduce
`--epochs` to ~150 (often enough — watch the val-loss plateau), or evaluate a
subset with `--n-images 100` first.

**Credit-saving tips**
- Do the `--smoke` checks on a **free T4** session, not the A100.
- Stop/close the A100 runtime the moment the run finishes — units accrue while
  the runtime is allocated, even if idle.
- Lower `--epochs` if `train_metrics.csv` shows the val loss has plateaued.

---

## What to take to your report

From `MyDrive/pdm/outputs/`:
- `logs/train_metrics.csv` → train/val denoise-loss curve (proves the model learned).
- `results/metrics.json` → AUROC / AUPRC / DICE (calibrated, best-global, oracle).
- `results/eval_*.png` → qualitative segmentation panels.
- `xai/*.png` → counterfactual + attribution explanations.
