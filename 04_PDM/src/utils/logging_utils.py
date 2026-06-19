"""
Logging utilities.
==================

Provides a single ``get_logger`` factory and a ``setup_file_logging`` helper so
every script logs consistently to both stdout and a per-run file. We use the
stdlib ``logging`` module (never bare ``print``) so log level, formatting, and
sinks are controlled centrally.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
_DATEFMT = "%H:%M:%S"
_CONFIGURED: set[str] = set()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a console logger, configured once per name."""
    logger = logging.getLogger(name)
    if name not in _CONFIGURED:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED.add(name)
    return logger


def setup_file_logging(log_dir: Path, run_name: str) -> Path:
    """Attach a file handler to the root pipeline logger.

    Returns the path of the created log file. Call once at the start of a script.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{run_name}_{stamp}.log"

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    # Attach to the shared 'pdm' parent so all child loggers propagate to file.
    root = logging.getLogger("pdm")
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    return log_path
