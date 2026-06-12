# Session handoff — 2026-06-11/12

A snapshot of the Claude Code (web) session that built **Stage 1 (medical VAE
fine-tuning)**, so a local chat can pick up exactly where it left off. Durable
project facts live in `CLAUDE.md` (auto-loaded); this file is the narrative +
next actions.

- **Branch:** `version-3`  ·  **PR:** #7 (draft → `main`) — https://github.com/soham10i/cv-project/pull/7
- **Stage 1 commit:** `b290cfb` "Stage 1: medical VAE fine-tuning with unified logging"

## How this work reaches your Mac

The session ran in an **ephemeral cloud container** with its own clone — it
cannot write to your local disk. Delivery is via git only:

```bash
git fetch origin version-3
git checkout version-3
git pull origin version-3
```

To continue locally with full context, run `claude` inside the repo — it
auto-loads `CLAUDE.md`. For a truly local (no-GitHub) agent, install Claude Code
locally (`npm install -g @anthropic-ai/claude-code`) or the IDE extension and run
it against your folder.

## Problem framing & the expert correction

The pipeline localises tumours via diffusion residuals but underperformed
(DICE ≈ 0.173, **oracle DICE ≈ 0.107** → weak residual signal regardless of
threshold). The decision was to build a **medical VAE**.

**Correction baked into the design:** the VAE is a **codec**, not the detector.
It must reconstruct lesions faithfully, so it trains on **ALL slices (healthy +
lesion)**; only the **diffusion UNet** trains on **healthy-only**. (Rombach 2022;
Pinaya 2022.)

## Decisions locked in (via AskUserQuestion)

1. **Fine-tune `stabilityai/sd-vae-ft-mse`** (keep 4ch/32×32 latent → UNet unchanged).
2. **Loss = L1 + LPIPS + KL** (no PatchGAN; stable on small pediatric data).
3. **Logging = file logs + CSV/JSONL + TensorBoard** (self-contained).
4. **Sequence = VAE stage first**, then diffusion.

## What was built in Stage 1

- `cv_project/src/logkit.py` — `RunLogger`: one timestamped folder per run with
  `run.log`, `metrics.csv`, `metrics.jsonl`, `config_snapshot.json`,
  `tensorboard/`. TensorBoard/torch optional → degrades to CSV/JSONL.
- `cv_project/src/prepare_vae_dataset.py` — builds the VAE set: every brain-
  bearing slice (healthy + lesion) from patient-disjoint train/val splits,
  reusing `zscore_normalize_volume`; saves val lesion masks for the fidelity metric.
- `cv_project/src/train_vae.py` — fine-tunes the VAE with `L1 + λ_lpips·LPIPS +
  λ_kl·KL`. Tracks per-epoch **train + val** loss components, brain-masked
  **MAE/PSNR/SSIM**, and **tumour-vs-healthy fidelity** (absolute + intensity-
  normalised "relative" ratio). Saves XAI panels. `--smoke` = 2-epoch plumbing test.
- `cv_project/src/config.py` — VAE fine-tune hyperparameters, dataset/log paths,
  `resolve_vae_source()` (single `USE_FINETUNED_VAE` switch), `snapshot()`.
- `cv_project/requirements.txt` — added `lpips`, `tensorboard` (both optional).
- `.gitignore` — added `logs/`.

## Validation performed (no ML stack in the cloud container)

- `py_compile` on all touched files — pass.
- `symtable` scope-aware undefined-name scan — clean.
- Cross-file symbol resolution: every `C.*` / `utils.*` / re-imported helper
  resolves to a real definition — pass.
- Live runtime test of `config.snapshot()`, `resolve_vae_source()`, and
  `RunLogger` I/O (CSV union, JSONL, run.log, snapshot) — pass.
- VAE encode/decode API mirrors proven existing code (`utils.py:168`,
  `train_healthy_manifold.py:193`).
- **Not executed**: real training (needs torch + data) → run the smoke test on the Mac.

## Run order (your Mac)

```bash
pip install -r cv_project/requirements.txt
python src/make_splits.py                            # if splits/ not built
python src/prepare_vae_dataset.py --max-total 6000
python src/train_vae.py --smoke                       # verify first
python src/train_vae.py                               # full fine-tune
tensorboard --logdir logs/
# then: set USE_FINETUNED_VAE = True in config.py  → proceed to Stage 2
```

## Papers (grouped by purpose)

- **VAE val loss / reconstruction quality:** SSIM — Wang et al. 2004 (*IEEE TIP*);
  LPIPS — Zhang et al. 2018 (*CVPR*); PSNR/MSE standard.
- **Method grounding:** Rombach et al. 2022 (LDM/SD-VAE, λ_kl≈1e-6) · Pinaya et al.
  2022 (*MICCAI*, latent diffusion brain anomaly) · Wyatt et al. 2022 (AnoDDPM) ·
  Baur et al. 2021 (*Med. Image Anal.*, AE brain-anomaly benchmark).
- **Eval metric selection:** Reinke et al. 2024, "Metrics Reloaded" (*Nature
  Methods*) — for imbalanced lesion voxels report **AUPRC** primary, Dice
  (Milletari et al. 2016) at the calibrated operating point.
- **Dataset:** Kazerooni et al. 2024 (BraTS-PEDs) · Bakas et al. 2017 (BraTS).

## Next steps (Stage 2 — not yet done)

1. Load the fine-tuned VAE via `config.resolve_vae_source()` in the diffusion,
   recalibration, and evaluation scripts.
2. Add a **validation denoising loss** (fixed per-slice noise) to
   `train_healthy_manifold.py`.
3. **Before/after XAI**: SAAM + residual panels at epoch 0 vs final.
4. Route diffusion scripts through `logkit.RunLogger`.
