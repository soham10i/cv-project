"""
Logging utilities.
==================

Single place that configures logging so every module logs consistently to BOTH
stdout (visible in the Colab cell / terminal) and a per-run file.

Design: handlers live on the shared ``pdm`` parent logger. Every module logger is
named ``pdm.<something>`` and *propagates* to that parent, so a single
StreamHandler + (optionally) one FileHandler capture everything. (The previous
version attached a handler per logger with ``propagate=False``, which meant the
file handler on the parent never received child records — log files came out
empty.)
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
_DATEFMT = "%H:%M:%S"
_PARENT = "pdm"
_parent_ready = False


def _ensure_parent() -> logging.Logger:
    """Configure the shared 'pdm' parent logger once (stdout handler)."""
    global _parent_ready
    parent = logging.getLogger(_PARENT)
    if not _parent_ready:
        parent.setLevel(logging.INFO)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        parent.addHandler(sh)
        parent.propagate = False  # don't double-log to the root logger
        _parent_ready = True
    return parent


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module logger that propagates to the shared 'pdm' parent.

    ``name`` should be ``pdm.<module>`` so handlers on the parent capture it.
    """
    _ensure_parent()
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True  # bubble up to the parent's handlers (stdout + file)
    return logger


def setup_file_logging(log_dir: Path, run_name: str) -> Path:
    """Attach a file handler to the 'pdm' parent. Returns the log file path.

    Call once at the start of a script. All ``pdm.*`` loggers then write to both
    stdout and this file.
    """
    parent = _ensure_parent()
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{run_name}_{stamp}.log"

    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    parent.addHandler(fh)
    return log_path
