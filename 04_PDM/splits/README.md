# Patient-level splits

Place three files here, one **patient ID per line** (matching the folder names
under `PDM_DATA_ROOT`):

- `train.txt` — patients whose **healthy** slices train the diffusion model
- `val.txt`   — patients whose **healthy** slices calibrate the threshold
- `test.txt`  — patients whose **lesion** slices are used for evaluation

Splits are **patient-level** (never slice-level) so no patient appears in more
than one set — this prevents data leakage, which would inflate metrics.

Example `train.txt`:
```
BraTS-PED-00002-000
BraTS-PED-00005-000
BraTS-PED-00011-000
```

A reasonable split for ~60 patients is 30 / 15 / 15 (train / val / test).
You can generate splits from your dataset folder with a one-liner, e.g.:

```bash
ls "$PDM_DATA_ROOT" | shuf > /tmp/all.txt
head -30 /tmp/all.txt              > splits/train.txt
sed -n '31,45p' /tmp/all.txt       > splits/val.txt
sed -n '46,60p' /tmp/all.txt       > splits/test.txt
```
