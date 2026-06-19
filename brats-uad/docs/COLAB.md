# Training on Google Colab (90 compute units) with Drive persistence

The strategy in one line: **dataset as one tar → extracted to fast local disk;
all outputs (weights, checkpoints, TensorBoard, XAI) written straight to mounted
Drive** so a disconnect never loses progress, and `--resume` continues.

---

## 0. Two hard constraints to plan around

**Compute units.** 90 units is plenty for a **subset / pilot** run but *not* for
full-scale training of both models on all 259 patients. Rough burn:

| GPU | VRAM | ~units/hr | Best for |
|---|---|---|---|
| **T4** | 16 GB | ~1.8 | **most unit-efficient — use this** |
| L4 | 24 GB | ~4.8 | faster wall-clock, burns units quicker |
| A100 | 40 GB | ~12 | only if you must finish fast |

On a **T4**, a 30-patient dev run (VAE 25 ep + diffusion 20 ep + eval) is ~1.5–2 h
≈ **3–4 units**. A ~120-patient run is ~8–12 units. So: **validate on dev, then
scale the patient count up as your unit budget allows.** Full-259 to convergence
is better suited to RunPod — use Colab for the dev/medium runs and the report.

**Drive storage.** Free Drive is 15 GB. The dev tar is ~4.3 GB (fits); a full
processed tar is ~18–20 GB (needs Google One). Outputs add ~5–10 GB. Start with
dev; only pack `full` if you have the Drive space.

---

## 1. Local prep (on your laptop, once)

```bash
cd brats-uad
# (a) build + pack the dataset you want to upload (start with dev)
bash scripts/make_dataset.sh dev                 # → data/processed_dev
python src/data/pack_dataset.py --out data/brats_dev.tar   # → one 4.3 GB file
# later, for a bigger run:
#   python src/data/make_slices.py --limit-patients 120
#   python src/data/pack_dataset.py --out data/brats_120.tar
```
Upload **`data/brats_dev.tar`** to Drive, e.g. `MyDrive/brats-uad/brats_dev.tar`.

Get the **code** onto Colab one of two ways:
- **GitHub (recommended):** push `brats-uad/` to a repo, then clone in Colab.
- **Drive:** zip `brats-uad/` (without `data/`) and upload to `MyDrive/brats-uad/`.

---

## 2. Colab cells (copy-paste, in order)

**Cell 1 — pick GPU + mount Drive**
> Runtime ▸ Change runtime type ▸ **T4 GPU** first.
```python
from google.colab import drive
drive.mount('/content/drive')
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

**Cell 2 — clone the code**
```python
!git clone https://github.com/<you>/brats-uad.git /content/brats-uad
%cd /content/brats-uad
```

**Cell 3 — one-shot setup** (installs deps, extracts dataset to local SSD, makes
Drive output dirs, writes `/content/brats_env.sh`, runs the leakage audit)
```python
!bash scripts/colab_bootstrap.sh brats_dev.tar /content/drive/MyDrive/brats-uad
```
> Args: `<tar-name> [drive-dir] [local-dataset-dir]`. Confirm the audit prints
> `PASS ✅` for all three leakage checks before training.

**Cell 4 — train** (each cell sources the env from step 3; `--resume` is the safety net)
```python
ENV = "/content/brats_env.sh"
!source {ENV} && python src/train_vae.py --epochs 25 --bs 16 --num-workers 2 --resume
!source {ENV} && python src/compute_scaling_factor.py --n 2000
!source {ENV} && python src/train_diffusion.py --epochs 20 --bs 32 --num-workers 2 --resume
```

**Cell 5 — watch curves (run anytime)**
```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/brats-uad/logs
```

**Cell 6 — calibrate, evaluate, XAI (all outputs land on Drive)**
```python
ENV = "/content/brats_env.sh"
!source {ENV} && python src/calibrate.py
!source {ENV} && python src/evaluate.py --n-images 60 --max-plots 16
!source {ENV} && python src/make_report_figures.py
!source {ENV} && python src/make_xai_panels.py --n 8
```

---

## 3. If Colab disconnects (it will)

Nothing is lost — weights, checkpoints, logs, and TensorBoard are already on
Drive. Reconnect and re-run **Cells 1–3** (the bootstrap skips re-extraction if
`/content/processed` already exists), then **Cell 4**: every training script sees
`last.pt` under `BUAD_MODEL_DIR/*_ckpt/` and **continues from the next epoch**
(`--resume`).

> Tip: keep the tab active / use Colab Pro's background execution to reduce idle
> disconnects, but resume makes them harmless either way.

---

## 4. What persists to Drive (your after-training study material)

```
MyDrive/brats-uad/
  models/  vae/  unet/  unet_ema/            # best weights (save_pretrained)
           vae_ckpt/  unet_ckpt/            # last.pt + epoch snapshots (resume)
           scaling_factor.json
  logs/    vae_<ts>/  diffusion_<ts>/        # run.log, metrics.csv/jsonl, config, tensorboard/
  results/ metrics.json  figures/  evaluation/  xai/
```

Every checkpoint is written atomically (temp file → rename), so a kill mid-write
can't corrupt it.

**I/O note:** the UNet resume checkpoint is ~1.4 GB and is written each epoch to
Drive — that's the safety you asked for, at ~30–60 s/epoch overhead. If Drive
writes ever stall, point `BUAD_MODEL_DIR` at `/content/models` (local) and run
`bash scripts/sync.sh "$DRIVE/models"` between stages instead.
