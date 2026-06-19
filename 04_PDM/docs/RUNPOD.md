# Training & Inference on RunPod GPU

A step-by-step guide to running the full PDM pipeline on a [RunPod](https://www.runpod.io)
GPU pod. Total cost for one end-to-end run is **~3.5 GPU-hours ≈ €6** on an
A100-40GB, comfortably inside a €50 budget (leaves room for ablations).

---

## 0. TL;DR

```bash
# on the pod, once:
git clone <your-repo> && cd 04_PDM
pip install -r requirements.txt
export PDM_DATA_ROOT=/workspace/data/raw
export PDM_PROCESSED_ROOT=/workspace/data/processed
export PDM_OUTPUT_ROOT=/workspace/outputs

# pipeline:
python scripts/00_preprocess.py --splits splits
python scripts/01_train.py --epochs 300 --bs 128
python scripts/02_calibrate.py --max-samples 200
python scripts/03_evaluate.py
python scripts/04_explain.py --n-cases 12
```

---

## 1. Pick a pod

- **GPU:** A100-40GB (best value) or A40-48GB. An RTX 4090-24GB also works with
  `--bs 64`.
- **Template:** "RunPod PyTorch 2.x" (CUDA 12.x). bf16 autocast needs Ampere+
  (A100/A40/4090 all qualify).
- **Disk:** attach a **Network Volume** (e.g. 50 GB) mounted at `/workspace` so
  data, checkpoints, and results survive pod restarts. This is the single most
  important setting — without it you lose everything when the pod stops.
- **Type:** "Secure Cloud" for reliability, or "Community Cloud" for lower cost.

When the pod is up, open the **Web Terminal** (or connect over SSH using the key
RunPod shows in the pod's *Connect* panel).

---

## 2. Get the code and data onto the pod

**Code** — clone your git repo (recommended) or upload a zip:
```bash
cd /workspace
git clone <your-repo-url> pdm && cd pdm/04_PDM
```

**Data** — choose one:
- `runpodctl receive <code>` (use `runpodctl send` from your laptop), or
- pull from cloud storage: `aws s3 sync s3://<bucket>/brats /workspace/data/raw`, or
- upload via the RunPod web file browser for small datasets.

Expected raw layout (one folder per patient):
```
/workspace/data/raw/
  BraTS-PED-00001-000/
    BraTS-PED-00001-000-t1n.nii.gz
    BraTS-PED-00001-000-t1c.nii.gz
    BraTS-PED-00001-000-t2w.nii.gz
    BraTS-PED-00001-000-t2f.nii.gz
    BraTS-PED-00001-000-seg.nii.gz
```

---

## 3. Environment

```bash
cd /workspace/pdm/04_PDM
pip install -r requirements.txt
pip install pydensecrf            # optional CRF refinement

# Point the pipeline at the network volume so nothing is lost on restart:
export PDM_DATA_ROOT=/workspace/data/raw
export PDM_PROCESSED_ROOT=/workspace/data/processed
export PDM_OUTPUT_ROOT=/workspace/outputs
# (add these to ~/.bashrc so they persist across terminals)
```

Sanity-check the GPU:
```bash
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"
nvidia-smi
```

---

## 4. Smoke test first (≈2 min, do NOT skip)

Verify the whole pipeline end-to-end on a tiny subset before spending GPU hours:
```bash
python scripts/00_preprocess.py --splits splits --limit-patients 4
python scripts/01_train.py --smoke
python scripts/02_calibrate.py --smoke
python scripts/03_evaluate.py --smoke
python scripts/04_explain.py --smoke
```
If all five complete without error, the plumbing is good.

---

## 5. Full run

Run training inside `tmux`/`nohup` so it survives a dropped SSH connection:
```bash
tmux new -s train
python scripts/00_preprocess.py --splits splits
python scripts/01_train.py --epochs 300 --bs 128 | tee train.out
# detach with Ctrl-b d ; reattach with: tmux attach -t train
```

Monitor:
```bash
watch -n 30 nvidia-smi                       # GPU utilisation
tail -f $PDM_OUTPUT_ROOT/logs/train_*.log    # training log
column -s, -t $PDM_OUTPUT_ROOT/logs/train_metrics.csv | tail   # loss curve
```

Then:
```bash
python scripts/02_calibrate.py --max-samples 200
python scripts/03_evaluate.py
python scripts/04_explain.py --n-cases 12
```

---

## 6. Pull results back to your laptop

```bash
# from your laptop:
runpodctl send <pod>:/workspace/outputs/results ./results
# or zip on the pod and download via the web file browser:
cd /workspace/outputs && zip -r results.zip results xai calibration.json
```

The artefacts you want are:
- `outputs/checkpoints/unet_ema/`  — trained weights (for later inference)
- `outputs/results/metrics.json`   — the numbers
- `outputs/results/eval_*.png`     — qualitative panels
- `outputs/xai/*.png`              — explanations

---

## 7. Inference-only later (no retraining)

Once `unet_ema/` and `calibration.json` exist on the volume, a fresh pod only
needs to re-run scoring:
```bash
export PDM_OUTPUT_ROOT=/workspace/outputs   # where the checkpoint lives
python scripts/03_evaluate.py               # uses the saved EMA UNet + threshold
python scripts/04_explain.py --n-cases 20
```

---

## 8. Cost control

- **Stop the pod** the moment the run finishes — billing is per-second while
  running. A network volume keeps your data for a small storage fee.
- Use `--smoke` to catch bugs on CPU/cheap pods before launching the A100 run.
- One full run ≈ **€6**; you can afford ~7 full runs (e.g. simplex-vs-gaussian
  and patch-size ablations) inside €50.

| Stage | A100-40GB | Notes |
|------|-----------|-------|
| Preprocess | ~15 min | CPU-bound; can run on a cheap pod |
| Train (300 ep, bs 128, bf16) | ~2.5 h | the main cost |
| Calibrate (200 slices) | ~10 min | |
| Evaluate (all test) | ~30 min | scales with #test slices × #noise-scales |
| Explain (12 cases) | ~5 min | |

---

## 9. Common pitfalls

- **Lost data after restart** → you forgot the network volume; always mount at
  `/workspace` and set the `PDM_*` env vars to point there.
- **CUDA OOM** → lower `--bs` (128 → 64 → 32) or reduce `patch_batch` in the
  scorer; bf16 is already on.
- **`pydensecrf` build fails** → it is optional; the pipeline logs a warning and
  continues without CRF. Install later if you want the boundary refinement.
- **bf16 unsupported** → only on pre-Ampere GPUs; set `TrainConfig.amp_dtype` to
  `"float16"` (GradScaler is wired up) or disable AMP.
