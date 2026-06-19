# Pilot run on an 8 GB laptop GPU

Goal: in ~2–3 hours, train a **reduced** VAE + diffusion on a subset, then look at
the numbers and the XAI panels to decide whether the approach is working — before
spending GPU budget on the full RunPod run tomorrow.

This is a **sanity pilot**, not final performance. A 15-epoch / 30-patient run
will not hit the best DICE; it tells you the *mechanism* works (recon is
anatomical, residuals localize the lesion, metrics trend the right way).

---

## 0. Environment (one-time)

You need PyTorch with CUDA for your laptop GPU. Check first:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If it prints `False`, install the CUDA build (CUDA 12.1 wheel shown; pick the one
matching your driver from https://pytorch.org):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then the rest of the deps:

```bash
cd brats-uad
pip install -r requirements.txt
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0), '| VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1))"
```

---

## 1. Build + audit the dev dataset (30 patients)

```bash
bash scripts/make_dataset.sh dev          # → data/processed_dev  + leakage audit
export BUAD_PROCESSED_DIR=$PWD/data/processed_dev
export BUAD_MODEL_DIR=$PWD/pilot/models
export BUAD_RESULTS_DIR=$PWD/pilot/results
export BUAD_LOG_DIR=$PWD/pilot/logs
```
```powershell
# Windows PowerShell equivalent
$env:BUAD_PROCESSED_DIR="$PWD\data\processed_dev"
$env:BUAD_MODEL_DIR="$PWD\pilot\models"; $env:BUAD_RESULTS_DIR="$PWD\pilot\results"; $env:BUAD_LOG_DIR="$PWD\pilot\logs"
```

The audit (`dataset_report.py`, run automatically) prints per-view slice/patient
counts, lesion prevalence, and **patient-level leakage checks** — confirm all
three say `PASS ✅` before training. (Raw data path defaults to
`cv-project/data/BraTS-PEDs-v1/Training`; override with `BUAD_DATA_ROOT`.)

---

## 2. Run the pilot (memory-safe for 8 GB)

Early stopping, weight decay, and overfit-gap warnings are **on by default** —
watch the log for `⚠ overfitting signal`.

```bash
# Stage 1 — VAE: small batch fits 8 GB; grad-accum keeps the effective batch up
python src/train_vae.py --epochs 20 --bs 6 --grad-accum 2 --num-workers 4
python src/compute_scaling_factor.py --n 1500

# Stage 2 — diffusion: latents are 32×32 so batch can be larger
python src/train_diffusion.py --epochs 20 --bs 16 --num-workers 4

# Stage 3 — calibrate thresholds on healthy val slices
python src/calibrate.py

# Stage 4 — numbers + figures + XAI panels
python src/evaluate.py --n-images 40 --max-plots 16
python src/make_report_figures.py
python src/make_xai_panels.py --n 8
```

Each training run writes `pilot/logs/<stage>_<timestamp>/` with `run.log`,
`metrics.csv`, `metrics.jsonl`, `config_snapshot.json`, and (if `pip install
tensorboard`) `tensorboard/`. Watch curves live:
```bash
tensorboard --logdir pilot/logs
```

**If you hit CUDA out-of-memory**, step the batch down (the only knob that matters):
- VAE: `--bs 4 --grad-accum 3`  → or `--bs 2 --grad-accum 6`
- Diffusion: `--bs 8`  → or `--bs 4`

Effective batch = `bs × grad-accum`, so accumulation preserves training quality
at lower memory. Also close other GPU apps (browsers/games share VRAM).

**Want it faster (~1 hr)?** Use `--limit-patients 15` and `--epochs 10`.

---

## 3. What to check — numbers

Open `pilot/results/metrics.json` (or read the final log lines). On a *pilot* you
are looking for **trend and plausibility**, not final scores:

| Signal | Pilot "looks good" | Red flag |
|---|---|---|
| AUROC (mean) | ≳ 0.80 and rising vs a shorter run | ≈ 0.5 (random) or NaN |
| AUPRC | clearly > lesion prevalence (~0.02–0.05) | ≈ prevalence |
| DICE @ best-global | trending up across epochs, > 0.10 | stuck near 0 |
| DICE oracle (ceiling) | meaningfully higher than calibrated | ≈ calibrated and tiny |
| val denoise loss (diffusion log) | steadily decreasing | flat or NaN |

The pilot's calibrated DICE will be modest — that's fine; the **oracle/global DICE
gap** tells you the signal is there and only the threshold needs the full run.

---

## 4. What to check — XAI  (`pilot/results/xai/xai_*.png`)

Each panel is a 3×5 grid. This is the real verification:

1. **Row 2 (healthy reconstruction)** should look like *plausible, lesion-free
   brain anatomy* — sharper than the smoke blob, with visible ventricles/gyri.
   If it's a featureless blur, the VAE needs more epochs.
2. **Row 3 residuals** should **light up on the lesion**, not uniformly on the
   skull/brain edge. Compare against the **GT overlay** (row 1, last panel).
3. **CE residual** (row 2, last) should respond where T1ce enhances — your new
   channel earning its place.
4. **FUSED anomaly map** (row 3, last) should visually overlap the GT.

Also skim `pilot/results/evaluation/eval_*.png` (recon vs prediction vs GT) and
`pilot/results/figures/headline_metrics.png`.

---

## 5. Decision rule for tomorrow

**Green-light the full RunPod run if:** reconstructions are anatomical, residuals
concentrate on lesions (not edges), AUROC ≳ 0.80, and the oracle/global DICE gap
shows real signal. Then tomorrow just drop the `--limit-patients` and raise
`--epochs` (config defaults: VAE 60, diffusion 40) on the 4090.

**Hold and debug if:** recon is a blur after 15 epochs (→ raise VAE epochs / lr),
residuals hug the skull edge (→ brain-mask / normalization issue), or AUROC ≈ 0.5
(→ scaling factor / calibration mismatch — check `scaling_factor.json` is loaded).
