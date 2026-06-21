# PDM — Multi-scale Patch Diffusion Fusion for Unsupervised Brain-MRI Lesion Segmentation

Unsupervised anomaly detection (UAD) for brain MRI: train a diffusion model on
**healthy tissue patches only**, then segment lesions at test time as the places
the model cannot reconstruct as "healthy". No lesion labels are used for training.

This is a from-scratch redesign of an earlier latent-diffusion pipeline that
failed (DICE ≈ 0.0–0.1). The redesign moves to **pixel-space patch diffusion**
with **simplex noise**, **multi-scale scoring**, **Gaussian patch fusion**, and
**dense-CRF refinement**, plus a **counterfactual XAI** suite.

> The *why* behind every design choice — with paper references — lives in
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The XAI rationale is in
> [`docs/XAI.md`](docs/XAI.md). GPU/cloud instructions are in
> [`docs/RUNPOD.md`](docs/RUNPOD.md).

---

## 1. Why this approach (one paragraph)

Whole-image autoencoder / latent-diffusion UAD fails on brain MRI because the
reconstruction error is dominated by **brain-boundary artefacts**, and ×8 latent
compression **erases the fine detail** small lesions live in. We instead diffuse
over **overlapping 96×96 pixel patches**: this (a) turns ~3k healthy slices into
~570k training samples, (b) removes the concept of a global reconstruction edge,
and (c) keeps full pixel resolution. **Simplex noise** (Wyatt et al., 2022) makes
lesions "look like noise" so they are healed toward healthy tissue; **multi-scale
scoring** (Bercea et al., 2023) catches both small and large lesions; **Gaussian
fusion** stitches patch scores without seams; **dense CRF** (Krähenbühl & Koltun,
2011) snaps the result to anatomical edges.

---

## 2. Project layout

```
04_PDM/
├── src/
│   ├── config.py              # all constants (frozen dataclasses, env-overridable)
│   ├── utils/                 # logging, device, io, custom exceptions
│   ├── data/                  # preprocessing, normalization, patches, datasets
│   ├── noise/                 # NoiseStrategy: gaussian | simplex (Strategy pattern)
│   ├── models/                # UNet, diffusion process, EMA
│   ├── scoring/               # residual, multi-scale scorer, CRF
│   ├── calibration/           # threshold calibration on healthy slices
│   ├── evaluation/            # metrics + evaluation loop
│   └── xai/                   # counterfactual, attribution, visualization
├── scripts/                   # 00_preprocess → 01_train → 02_calibrate → 03_evaluate → 04_explain
├── tests/                     # unit tests (patches, noise)
├── docs/                      # ARCHITECTURE.md, XAI.md, RUNPOD.md
└── requirements.txt
```

---

## 3. Setup

```bash
cd 04_PDM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional CRF refinement (skipped gracefully if absent):
pip install pydensecrf            # or: pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

Point the pipeline at your data with three environment variables (defaults are
local folders under the project):

```bash
export PDM_DATA_ROOT=/path/to/BraTS-PED/raw          # patient folders with *.nii.gz
export PDM_PROCESSED_ROOT=/path/to/processed         # slices/patches/manifests
export PDM_OUTPUT_ROOT=/path/to/outputs              # checkpoints/logs/results/xai
```

**Patient splits.** Create `splits/train.txt`, `splits/val.txt`, `splits/test.txt`,
each one patient-ID per line (folder names under `PDM_DATA_ROOT`). Splits are
patient-level to prevent leakage.

---

## 4. Run the pipeline (5 stages)

Each script has a `--smoke` flag that runs in minutes on CPU to verify plumbing
before committing GPU time.

```bash
# Stage 0 — NIfTI → normalized slices + manifests
python scripts/00_preprocess.py --splits splits
#   smoke: python scripts/00_preprocess.py --splits splits --limit-patients 4

# Stage 1 — train the patch diffusion model on healthy patches
python scripts/01_train.py --epochs 300 --bs 128
#   smoke: python scripts/01_train.py --smoke

# Stage 2 — calibrate the anomaly threshold on healthy val slices
python scripts/02_calibrate.py --max-samples 200
#   smoke: python scripts/02_calibrate.py --smoke

# Stage 3 — evaluate on test lesion slices (AUROC / AUPRC / DICE + panels)
python scripts/03_evaluate.py
#   smoke: python scripts/03_evaluate.py --smoke

# Stage 4 — generate counterfactual + attribution XAI panels
python scripts/04_explain.py --n-cases 12
#   smoke: python scripts/04_explain.py --smoke
```

Outputs:
- `outputs/checkpoints/unet_ema/`  — preferred weights for inference
- `outputs/calibration.json`       — operating threshold
- `outputs/results/metrics.json`   — aggregate + per-image AUROC/AUPRC/DICE
- `outputs/results/eval_*.png`     — qualitative panels
- `outputs/xai/xai_*.png`, `outputs/xai/trajectory_*.png` — explanations

---

## 5. Recommended full-run on a GPU

Two walkthroughs:
- [`docs/RUNPOD.md`](docs/RUNPOD.md) — RunPod (network volume, RTX 4090/A100).
- [`docs/COLAB.md`](docs/COLAB.md) — Google Colab with compute credits: preprocess
  locally → two zips → Drive → `/content`, with **resume-safe** training
  (`scripts/01_train.py --resume`) and a ready notebook
  [`notebooks/run_pdm_colab.ipynb`](notebooks/run_pdm_colab.ipynb).

Budget: a single A100 run of the whole pipeline is **~3.5 GPU-hours ≈ €6** (or
~50 Colab credits on A100; ~3× cheaper on L4).

| Stage | A100-40GB time |
|------|----------------|
| Preprocess (CPU-bound) | ~15 min |
| Train (300 ep, bs 128, bf16) | ~2.5 h |
| Calibrate (200 slices) | ~10 min |
| Evaluate (all test slices) | ~30 min |
| Explain (12 cases) | ~5 min |

---

## 6. Expected numbers

| Metric | Old latent pipeline (smoke) | PDM target |
|---|---|---|
| AUROC | 0.53–0.94 (1 patient, unstable) | **0.88–0.93** |
| AUPRC | n/a | **0.40–0.50** |
| DICE @ calibrated | 0.00–0.10 | **0.32–0.40** |
| DICE @ best-global | n/a | **0.38–0.45** |
| DICE oracle (ceiling) | n/a | **0.48–0.55** |

These targets are in line with published diffusion-UAD results on BraTS
(Pinaya et al., 2022; Behrendt et al., 2023; Bercea et al., 2023) — see the
reference list in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## 7. Testing

```bash
pip install pytest
pytest tests/ -q
```

The tests cover patch tiling/fusion correctness and noise-strategy statistics —
the components most likely to break silently — and need no GPU or data.
