# CLAUDE.md — Project memory for cv-project

> Auto-loaded by Claude Code at the start of every session in this repo. Keep it
> accurate and concise. It is the durable "memory" that lets a fresh chat resume
> with full context.

## Project

**BraTS-PED pediatric brain-tumour unsupervised anomaly detection** via a
**Latent Diffusion Model (LDM)**. Pipeline: encode an MRI slice with a VAE →
add noise to an intermediate timestep `T_int` → DDIM-denoise on the *healthy*
manifold → the **residual** between input and healthy reconstruction localises
the tumour. All code is in `cv_project/src/`, driven by a single `config.py`.

Data: BraTS-PEDs, 3 MRI modalities per slice `[t1c, t2w, t2f]`, 2D axial slices
at 256×256, saved as `(3, 256, 256)` float32 `.npy`.

## Core design invariants — DO NOT break these

- **Asymmetric data split (the key idea):**
  - **VAE** trains on **ALL slices (healthy + lesion)** — it is a *codec* and
    must reconstruct tumour faithfully, or lesion signal is destroyed in the
    latent before the diffusion UNet ever sees it.
  - **Diffusion UNet** trains on **healthy slices only** — so lesions denoise
    *away* and show up in the residual.
  - Grounding: Rombach et al. 2022 (LDM); Pinaya et al. 2022 (MICCAI).
- **Latent geometry is 4ch / 32×32** (8× downsample of 256px). Keep it so the
  existing `UNet2DModel` (`sample_size=32, in_channels=4`) is reused unchanged.
  This is why we *fine-tune* `stabilityai/sd-vae-ft-mse` rather than train a VAE
  from scratch.
- **Volume-level z-score normalization**, never per-slice — see
  `preprocess_to_2d.zscore_normalize_volume`. Per-slice stats let a hyper-intense
  lesion distort surrounding healthy tissue.
- **Calibration contract:** the operating threshold is a percentile of the
  **pooled healthy brain-voxel** score distribution (not per-slice max). Multi-T
  scoring keeps **one baseline + scale per T** so the percentile stays valid.
- **`config.py` is the single source of truth** — every script imports paths and
  hyperparameters from it. A `config.resolve_vae_source()` switch
  (`USE_FINETUNED_VAE`) selects the fine-tuned VAE vs the pretrained one.
- **All runs log through `logkit.RunLogger`** → `logs/<stage>_<timestamp>/`
  (`run.log`, `metrics.csv`, `metrics.jsonl`, `config_snapshot.json`,
  `tensorboard/`, `xai/`). `logs/` is git-ignored.

## Pipeline stages & run order

```bash
python src/make_splits.py                              # patient-level train/val/test
python src/preprocess_to_2d.py ...                     # healthy/anomalous 2D slices (diffusion)
python src/prepare_vae_dataset.py --max-total 6000     # ALL slices for the VAE
python src/train_vae.py --smoke                         # Stage 1 plumbing test
python src/train_vae.py                                 # Stage 1 full fine-tune
# → set USE_FINETUNED_VAE = True in config.py
python src/train_healthy_manifold.py                    # Stage 2 diffusion (healthy-only)
python src/recalibrate.py                               # build baselines + thresholds
python src/evaluate_pipeline.py                         # metrics (DICE, AUPRC, oracle)
tensorboard --logdir logs/
```

**One-command entry points** (in `cv_project/`):
- `bash smoke_test.sh` — local macOS/MPS go/no-go: runs Stage 1 + Stage 2 on tiny
  caps, prints a PASS/FAIL table. Use before committing to cloud GPU.
- `bash run_full_cloud.sh` — full-scale CUDA run (auto-selects GPU). Env-knob caps/epochs.
- `bash run_pipeline.sh` — legacy diffusion-only path (no VAE stage).

## Key files (`cv_project/src/`)

| File | Role |
|------|------|
| `config.py` | Single source of truth: paths, hyperparameters, `resolve_vae_source()`, `snapshot()`. |
| `utils.py` | Shared: `normalize_for_vae`, `encode_to_latents`, schedulers, `brain_mask_2d`, residuals, calibration, multi-T scoring, `EMA`, SAAM attention hooks. |
| `logkit.py` | `RunLogger` — unified per-run logging (log/CSV/JSONL/TensorBoard). |
| `preprocess_to_2d.py` | NIfTI → 2D slices; `zscore_normalize_volume` (volume-level). |
| `prepare_vae_dataset.py` | VAE dataset = ALL slices (healthy+lesion), patient-disjoint; saves val lesion masks. |
| `train_vae.py` | **Stage 1**: fine-tune SD-VAE with L1+LPIPS+KL; tracks train/val loss, MAE/PSNR/SSIM, tumour-vs-healthy fidelity; XAI panels; `--smoke`. |
| `train_healthy_manifold.py` | **Stage 2**: latent DDPM on healthy slices; EMA; SAAM XAI trajectory; calibration. |
| `recalibrate.py` | Recompute baselines/thresholds (single- or multi-T) without retraining. |
| `evaluate_pipeline.py` | DICE / AUPRC / oracle-DICE; single- or multi-T scoring. |
| `vae_fidelity_diagnostic.py` | Pure encode→decode tumour-vs-healthy recon-error ratio (decides "swap VAE?"). |

## Current state

- Active branch: **`version-3`** (draft PR #7 → `main`).
- **Stage 1 (VAE fine-tuning) complete & validated** (smoke-passed on T4).
- **Stage 2 wiring complete**: diffusion / recalibrate / evaluate load the VAE via
  `resolve_vae_source()` (switch with `CV_USE_FINETUNED_VAE=1` or `USE_FINETUNED_VAE`);
  diffusion trainer now has a validation denoising loss, before/after (epoch-0 vs
  final) SAAM XAI, and unified `logkit` logging. Pending: the full cloud training
  runs + the pretrained-vs-fine-tuned ablation.

## Known issues / honesty notes

- **Oracle DICE ≈ 0.107** (per-slice best-threshold ceiling) → the residual maps
  weakly localise tumour *regardless of threshold*. This is a **signal** problem,
  not a thresholding one — the motivation for the medical-VAE work.
- The original `vae_fidelity` tumour/healthy ratio of **1.76 is intensity-
  confounded** (tumour voxels are simply brighter). `train_vae.py` validation now
  also reports an **intensity-normalised "relative" ratio** that isolates true
  structural fidelity loss.

## Conventions

- Python, fp32 (no AMP — MPS-friendly). Device via `utils.get_device()`
  (cuda → mps → cpu). Seed via `utils.set_seed()`.
- Optional deps (`lpips`, `tensorboard`) **degrade gracefully** if absent.
- Training runs on the user's Mac (MPS); the web/cloud container has no ML stack,
  so changes are validated by `py_compile` + static checks and smoke-tested locally.
- **Checkpoint/resume (`ckptkit.py`)**: both trainers write `last.pt` every epoch
  (model+optim+sched+EMA+RNG) plus periodic `ckpt_ep###.pt` (pruned to
  `CKPT_KEEP_LAST`). Pass `--resume` to continue. CLI knobs: `--grad-accum`,
  `--num-workers`, `--save-every`, `--keep-last`.
- **Colab T4 (15 GB)**: VAE bs≈4 (×accum), diffusion bs≈16, fp32. Mount Drive and
  set `CV_MODEL_DIR` / `CV_PROCESSED_DIR` / `CV_LOG_DIR` there so checkpoints/data
  survive disconnects; re-running `run_full_cloud.sh` (RESUME=1) continues.

## Next steps (post Stage-2-wiring)

1. **Full cloud runs**: VAE fine-tune → set `CV_USE_FINETUNED_VAE=1` → retrain the
   diffusion UNet on the new latents (the UNet **must** be retrained — switching
   the VAE changes the latent distribution) → recalibrate → evaluate.
2. **The decisive ablation**: pretrained-VAE vs fine-tuned-VAE pipeline, same UNet
   recipe — this is what justifies the Stage-1 effort. Report DICE / AUPRC / oracle-DICE.
3. **Contingency levers** if the residual still under-localizes: simplex noise
   (AnoDDPM), T_int sweep, 2.5D input, residual post-processing.
