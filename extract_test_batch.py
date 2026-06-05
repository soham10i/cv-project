"""
extract_test_batch.py
======================

Memory-efficient extraction of a small, representative 2D *test batch* from the
3D multi-modal BraTS 2021 dataset.

The script is intended as a smoke-test / workflow-validation step prior to a
full-scale latent extraction pipeline. It deliberately processes patients
*sequentially* (one volume set at a time) so that it can run comfortably within
an 8 GB RAM / VRAM budget on a local machine.

Pipeline per patient
---------------------
1.  Load only the ``T1ce``, ``T2``, ``FLAIR`` and ``Seg`` NIfTI volumes.
2.  Apply independent Z-score normalization to each MRI modality, computing the
    mean/std over the *foreground* (non-zero) voxels only.
3.  Stack the three modalities into a tensor of shape ``[3, 240, 240, 155]``.
4.  Iterate over the axial dimension (the 155 slices).
5.  Reject slices whose brain-tissue footprint covers < 15 % of the
    240 x 240 plane.
6.  Use the corresponding ``Seg`` slice to label each kept slice as
    *Healthy* (``sum == 0``) or *Anomalous* (``sum > 0``).
7.  Save each kept slice as a ``[3, 240, 240]`` ``.pt`` tensor.

Execution stops *exactly* when 200 Healthy and 20 Anomalous slices have been
saved.

Dependencies: ``torch``, ``nibabel``, ``numpy``, ``os``, ``glob``.
"""

import os
import glob

import numpy as np
import nibabel as nib
import torch


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Root directory that contains one sub-folder per patient (BraTS 2021 layout):
#   <DATASET_ROOT>/BraTS2021_00000/BraTS2021_00000_t1ce.nii.gz
#   <DATASET_ROOT>/BraTS2021_00000/BraTS2021_00000_t2.nii.gz
#   ...
DATASET_ROOT = os.environ.get("BRATS_ROOT", "data/raw/BraTS2021")

# Output directories for the two classes.
OUT_HEALTHY = os.path.join("data", "processed", "test_batch", "healthy")
OUT_ANOMALOUS = os.path.join("data", "processed", "test_batch", "anomalous")

# Modality file suffixes (BraTS 2021 naming convention).
MODALITY_SUFFIXES = ("t1ce", "t2", "flair")  # order defines channel order
SEG_SUFFIX = "seg"

# Expected per-volume spatial dimensions.
HEIGHT, WIDTH, NUM_SLICES = 240, 240, 155

# Slicing / acceptance thresholds.
FOREGROUND_FRACTION = 0.15           # min brain footprint per slice (15 %)
MIN_FOREGROUND_PIXELS = int(FOREGROUND_FRACTION * HEIGHT * WIDTH)

# Target counts for this minimal test batch.
TARGET_HEALTHY = 200
TARGET_ANOMALOUS = 20


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def find_patient_dirs(dataset_root):
    """Return a sorted list of patient directories under ``dataset_root``.

    A directory is considered a patient if it contains at least one ``.nii.gz``
    file. Sorting guarantees deterministic, reproducible extraction order.
    """
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            f"Dataset root '{dataset_root}' does not exist. "
            "Set the BRATS_ROOT environment variable or edit DATASET_ROOT."
        )
    candidates = sorted(
        d for d in glob.glob(os.path.join(dataset_root, "*")) if os.path.isdir(d)
    )
    return [d for d in candidates if glob.glob(os.path.join(d, "*.nii.gz"))]


def resolve_modality_path(patient_dir, suffix):
    """Locate the NIfTI file for a given modality ``suffix`` inside a patient dir.

    Matches files like ``*_t1ce.nii.gz`` while being tolerant of the exact
    patient-ID prefix. Returns ``None`` if the modality is missing.
    """
    matches = glob.glob(os.path.join(patient_dir, f"*_{suffix}.nii.gz"))
    return matches[0] if matches else None


def load_volume(path, dtype=np.float32):
    """Load a NIfTI volume as a NumPy array.

    ``nibabel`` memory-maps the data on access, so this keeps the resident set
    small until the array is actually materialised.
    """
    return np.asarray(nib.load(path).get_fdata(dtype=np.float64)).astype(dtype)


def zscore_foreground(volume):
    """Independent Z-score normalization over non-zero (foreground) voxels.

    Background voxels (value == 0) are *excluded* from the statistics and left
    untouched, so the brain footprint used for slice rejection is preserved.
    """
    mask = volume > 0
    # Guard against empty / corrupt volumes to avoid divide-by-zero.
    if not np.any(mask):
        return volume

    fg = volume[mask]
    mean = fg.mean()
    std = fg.std()
    if std < 1e-8:  # constant-intensity volume -> nothing to scale
        return volume

    out = np.zeros_like(volume, dtype=np.float32)
    out[mask] = (volume[mask] - mean) / std
    return out


def validate_shape(volume, name, patient_id):
    """Return ``True`` if a volume matches the expected BraTS dimensions."""
    if volume.shape != (HEIGHT, WIDTH, NUM_SLICES):
        print(
            f"  [WARN] {patient_id}: '{name}' has shape {volume.shape}, "
            f"expected {(HEIGHT, WIDTH, NUM_SLICES)} -> skipping patient."
        )
        return False
    return True


# --------------------------------------------------------------------------- #
# Core extraction
# --------------------------------------------------------------------------- #
def process_patient(patient_dir, counts):
    """Extract and save qualifying 2D slices from a single patient.

    ``counts`` is a mutable dict tracking ``{"healthy": int, "anomalous": int}``.
    Returns ``True`` once *both* targets are met (signalling the caller to stop).
    """
    patient_id = os.path.basename(patient_dir.rstrip(os.sep))

    # ----- Resolve & validate all required files up front ----------------- #
    modality_paths = [resolve_modality_path(patient_dir, s) for s in MODALITY_SUFFIXES]
    seg_path = resolve_modality_path(patient_dir, SEG_SUFFIX)

    if any(p is None for p in modality_paths) or seg_path is None:
        print(f"  [SKIP] {patient_id}: missing one or more required modalities.")
        return False

    # ----- Load, normalize and stack the three MRI modalities ------------- #
    # Each volume is loaded, normalized, then released as we move to the next,
    # keeping at most one extra full volume in memory at a time.
    normalized = []
    for suffix, path in zip(MODALITY_SUFFIXES, modality_paths):
        vol = load_volume(path)
        if not validate_shape(vol, suffix, patient_id):
            return False
        normalized.append(zscore_foreground(vol))
        del vol  # explicit release

    # Stack into [3, 240, 240, 155]; np.stack copies, so drop the source list.
    stacked = np.stack(normalized, axis=0)
    del normalized

    seg = load_volume(seg_path)
    if not validate_shape(seg, SEG_SUFFIX, patient_id):
        return False

    # ----- Iterate over axial slices ------------------------------------- #
    saved_this_patient = 0
    for z in range(NUM_SLICES):
        # Early exit if both global targets are already satisfied.
        if counts["healthy"] >= TARGET_HEALTHY and counts["anomalous"] >= TARGET_ANOMALOUS:
            break

        slice_3ch = stacked[:, :, :, z]  # view, shape [3, 240, 240]

        # Brain footprint = any non-zero voxel across the 3 modalities.
        foreground = np.count_nonzero(np.any(slice_3ch != 0, axis=0))
        if foreground < MIN_FOREGROUND_PIXELS:
            continue  # too little brain tissue -> discard

        # Convert to a contiguous tensor only once we know we want the slice.
        seg_slice = torch.from_numpy(np.ascontiguousarray(seg[:, :, z]))

        if torch.sum(seg_slice) > 0:
            label, out_dir, key, target = "anomalous", OUT_ANOMALOUS, "anomalous", TARGET_ANOMALOUS
        else:
            label, out_dir, key, target = "healthy", OUT_HEALTHY, "healthy", TARGET_HEALTHY

        # Respect the per-class cap.
        if counts[key] >= target:
            continue

        tensor = torch.from_numpy(np.ascontiguousarray(slice_3ch)).float()  # [3,240,240]
        out_path = os.path.join(out_dir, f"{patient_id}_{z:03d}.pt")
        torch.save(tensor, out_path)

        counts[key] += 1
        saved_this_patient += 1

        # Progress tracker.
        print(
            f"    [{label:>9}] saved {patient_id}_{z:03d}.pt "
            f"| healthy {counts['healthy']}/{TARGET_HEALTHY} "
            f"| anomalous {counts['anomalous']}/{TARGET_ANOMALOUS}"
        )

    # Release large arrays before moving to the next patient.
    del stacked, seg

    if saved_this_patient:
        print(f"  [DONE] {patient_id}: saved {saved_this_patient} slice(s).")

    return counts["healthy"] >= TARGET_HEALTHY and counts["anomalous"] >= TARGET_ANOMALOUS


def main():
    # Ensure output directories exist.
    os.makedirs(OUT_HEALTHY, exist_ok=True)
    os.makedirs(OUT_ANOMALOUS, exist_ok=True)

    patient_dirs = find_patient_dirs(DATASET_ROOT)
    print(f"Found {len(patient_dirs)} patient(s) under '{DATASET_ROOT}'.")
    print(
        f"Targets -> healthy: {TARGET_HEALTHY}, anomalous: {TARGET_ANOMALOUS}\n"
        f"Foreground threshold -> {MIN_FOREGROUND_PIXELS} px "
        f"({FOREGROUND_FRACTION * 100:.0f}% of {HEIGHT}x{WIDTH})\n"
    )

    counts = {"healthy": 0, "anomalous": 0}

    for idx, patient_dir in enumerate(patient_dirs, start=1):
        print(f"[{idx}/{len(patient_dirs)}] Processing {os.path.basename(patient_dir)} ...")
        done = process_patient(patient_dir, counts)
        if done:
            print("\nBoth targets reached. Stopping extraction.")
            break
    else:
        # Loop finished without hitting both targets.
        print("\nExhausted all patients before reaching targets.")

    # Final summary.
    print(
        "\n=== Extraction summary ===\n"
        f"Healthy   : {counts['healthy']}/{TARGET_HEALTHY}  -> {OUT_HEALTHY}\n"
        f"Anomalous : {counts['anomalous']}/{TARGET_ANOMALOUS}  -> {OUT_ANOMALOUS}"
    )


if __name__ == "__main__":
    main()
