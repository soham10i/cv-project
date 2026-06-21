"""
Shared script bootstrap: make ``src`` importable and set up logging.
====================================================================

Every script imports ``from _bootstrap import init`` and calls ``init(run_name)``
first. Keeps path-juggling and logging setup in one place (DRY).
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import CONFIG  # noqa: E402
from src.utils.logging_utils import get_logger, setup_file_logging  # noqa: E402


def init(run_name: str):
    """Set up file logging for a script run; return (logger, log_path)."""
    log_path = setup_file_logging(CONFIG.paths.logs_dir, run_name)
    log = get_logger(f"pdm.{run_name}")
    log.info("Run '%s' | log file: %s", run_name, log_path)
    return log, log_path
