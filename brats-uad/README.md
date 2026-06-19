# brats-uad — Unsupervised Lesion Anomaly Detection on Brain MRI

Latent-diffusion unsupervised anomaly detection (UAD) for pediatric brain tumors
(**BraTS-PEDs**). A medical KL-VAE encodes 4-modality MRI to a compact latent;
a DDPM learns the **healthy** latent manifold; at test time a slice is partially
noised and denoised back onto that manifold, and the **residual** localizes the
lesion — no lesion labels used for training.

```
encode → add noise @ T_int → DDIM denoise (healthy manifold) → decode → residual → calibrate → threshold
```

## What this fixes vs. the previous pipeline

| Problem (old) | Fix (here) |
|---|---|
| SD natural-image VAE (RGB bias) forced onto MRI | **From-scratch 4-ch medical KL-VAE** (`src/models/kl_vae.py`) |
| `scaling_factor` hardcoded to SD's `0.18215` | **Empirical** `1/std` of *this* VAE's latents (`compute_scaling_factor.py`) |
| T1-native modality dropped | **All 4 modalities** `[t1n, t1c, t2w, t2f]` + explicit **CE = t1c−t1n** residual channel |
| ±3σ clip truncated FLAIR/edema hyperintensity | **Robust percentile clip** `[0.5, 99.5]` keeps the lesion signal |
| Diffusion trained on ≤50 slices | Trained on the **full healthy slice pool** (thousands), patient-level split |
| "Healthy" slices from tumor patients contaminated the manifold | Lesion-free **+ ≥3-slice buffer** from any lesion |
| Flat modality averaging in the score | **Modality-weighted** residual (FLAIR + CE dominate) |

## Pipeline

| Stage | Script | Output |
|---|---|---|
| 0 — extract slices | `src/data/make_slices.py` | `slices/`, `masks/`, `manifests/` |
| 0b — audit dataset | `src/data/dataset_report.py` | counts + lesion prevalence + leakage checks |
| 1 — train VAE | `src/train_vae.py` → `src/compute_scaling_factor.py` | `models/vae/`, `scaling_factor.json` |
| 2 — train diffusion | `src/train_diffusion.py` | `models/unet/`, `models/unet_ema/` |
| 3 — calibrate | `src/calibrate.py` | `M_baseline.npy`, `calibration*.json` |
| 4 — evaluate | `src/evaluate.py` → `make_report_figures.py` / `make_xai_panels.py` | `metrics.json`, `results/figures/`, `results/xai/` |

### Datasets (`scripts/make_dataset.sh`)
| Mode | Patients | Use |
|---|---|---|
| `smoke` | 4 | plumbing test (seconds) |
| `dev` | 30 | 8 GB-laptop pilot (see [docs/PILOT_8GB.md](docs/PILOT_8GB.md)) |
| `full` | all 259 | actual fine-tuning |

Each build runs the leakage audit automatically.

### Monitoring & overfitting
Both training scripts write a per-run folder `logs/<stage>_<timestamp>/`
(`run.log`, `metrics.csv`, `metrics.jsonl`, `config_snapshot.json`,
`tensorboard/`). **Early stopping**, **weight decay**, and **val/train overfit-gap
warnings** are on by default (`--patience`, `--weight-decay`). View curves with
`tensorboard --logdir logs`.

### Docs
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the KL-VAE and UNet explained, why each beats the old pipeline, with links to official diffusers docs.
- [docs/PILOT_8GB.md](docs/PILOT_8GB.md) — laptop-GPU pilot before the full RunPod run.
- [docs/COLAB.md](docs/COLAB.md) + [notebooks/colab_train.ipynb](notebooks/colab_train.ipynb) — train on Colab with Drive persistence. Pack the dataset with `python src/data/pack_dataset.py` (one tar → upload to Drive → extract to local SSD; outputs autosave to Drive; `--resume` survives disconnects).

## Quick start (local smoke test — CPU/MPS, minutes)

```bash
cd brats-uad
export BUAD_PROCESSED_DIR=$PWD/data/processed_smoke \
       BUAD_MODEL_DIR=$PWD/models_smoke \
       BUAD_RESULTS_DIR=$PWD/results_smoke
python3 src/data/make_slices.py --limit-patients 4
python3 src/train_vae.py --smoke
python3 src/compute_scaling_factor.py --n 32
python3 src/train_diffusion.py --smoke
python3 src/calibrate.py --smoke
python3 src/evaluate.py --n-images 4 --max-plots 2
```

## Full run on RunPod (the real training)

**Why RunPod On-Demand:** a dedicated pod runs uninterrupted until *you* stop it
— no Colab 12h cap / disconnects. An RTX 4090 (24 GB, ~€0.34–0.44/hr) is plenty.

1. Create an **On-Demand** RTX 4090 pod (PyTorch template) with a **persistent
   volume** at `/workspace`.
2. Upload this `brats-uad/` folder and the raw `BraTS-PEDs-v1/Training/` to the
   volume (raw NIfTI ≈ 6–13 GB, smaller than processed slices).
3. ```bash
   cd /workspace/brats-uad
   bash scripts/runpod_setup.sh        # installs deps, writes .env
   source .env                          # sets BUAD_* paths to the volume
   bash scripts/run_all.sh              # stages 0→4, all resumable
   ```
4. Optional safety net: `bash scripts/sync.sh /workspace/backup` after each stage.

Stages are `--resume`-safe (atomic `last.pt` every epoch), so a restart continues.

### Budget (≈ €50 cap)
| Stage | GPU-hr | ~€ |
|---|---|---|
| VAE (`VAE_EPOCHS=60`) | 15–25 | 6–11 |
| Diffusion (`DIFF_EPOCHS=40`) | 20–30 | 8–13 |
| Calib + eval + reruns | 5–10 | 2–4 |
| **Total** | **40–65** | **≈ 16–28** |

Do smoke tests on Kaggle/Colab free tiers; spend real budget only on the two
training jobs. Lower `--epochs` first if you want a fast full dry run.

## Config

All paths and hyper-parameters live in [`src/config.py`](src/config.py) (single
source of truth), overridable via `BUAD_*` env vars. Notable knobs:
`MODALITY_WEIGHTS`, `USE_CE_CHANNEL`, `T_INT`, `MULTI_T_LIST`, `HEALTHY_BUFFER`,
`THRESHOLD_PERCENTILE`, `VAE_EPOCHS`, `DIFF_EPOCHS`.

## Metrics reported

`evaluate.py` writes per-image and aggregate **AUROC**, **AUPRC**, and **DICE**
at three operating points: the *calibrated* healthy-percentile threshold
(deployable), the *best-global* single threshold (realistic ceiling), and the
*oracle* per-slice best (upper bound). Use these for the report's results table.

## For the report
- **Method narrative:** the table above (each fix is a defensible design choice).
- **Figures:** `results/figures/headline_metrics.png`, `distribution.png`, and the
  qualitative `results/evaluation/eval_*.png` panels.
- **Ablations (Day 6):** flip one knob at a time and re-run `evaluate.py` —
  e.g. `USE_CE_CHANNEL=False`, flat `MODALITY_WEIGHTS`, single-T vs multi-T,
  `T_INT ∈ {150,300,450}` — to show each fix's contribution.

## Layout
```
brats-uad/
  src/config.py                 # single source of truth
  src/data/                     # normalization, make_slices, datasets
  src/models/                   # kl_vae, unet, ema, losses
  src/pipeline/                 # diffusion, scoring, calibration
  src/train_vae.py  compute_scaling_factor.py
  src/train_diffusion.py  calibrate.py  evaluate.py  make_report_figures.py
  scripts/                      # runpod_setup.sh, run_all.sh, sync.sh
  splits/                       # train/val/test patient IDs (disjoint)
  requirements.txt
```
