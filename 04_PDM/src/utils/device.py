"""
Device and reproducibility utilities.
=====================================
"""

from __future__ import annotations

import contextlib
import gc
import random
import time
from typing import Iterator

import numpy as np
import torch

from .logging_utils import get_logger

log = get_logger("pdm.device")


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Human-readable device description for logging."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"CUDA — {name} ({mem:.0f} GB)"
    return device.type.upper()


def set_seed(seed: int) -> None:
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_dtype_from_str(name: str) -> torch.dtype:
    """Map a config dtype string to a torch dtype."""
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        name, torch.float32
    )


def clear_cache() -> None:
    """Free cached GPU memory (call between large eval images)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextlib.contextmanager
def timer(label: str) -> Iterator[None]:
    """Context manager that logs the wall-time of a block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        log.info("%s — %.1fs", label, time.perf_counter() - start)
