"""
Unified run logging for every pipeline stage
=============================================
One log folder, one API.  Every training / calibration / evaluation script
creates a ``RunLogger`` which owns a single timestamped run directory under
``logs/`` and fans the same numbers out to three sinks:

    logs/<stage>_<YYYYmmdd_HHMMSS>/
    ├── run.log            # full Python logging stream (console + file)
    ├── metrics.csv        # one row per (epoch, split) — opens in Excel/pandas
    ├── metrics.jsonl      # same rows, append-only, machine-friendly
    ├── config_snapshot.json
    └── tensorboard/       # SummaryWriter event files  (tensorboard --logdir logs)

Design notes
------------
* No hard dependency on TensorBoard — if it (or torch) is missing the writer is
  silently skipped, so the script still runs and CSV/JSONL are always produced.
* ``log_metrics`` accepts an arbitrary dict, so train rows (loss components) and
  val rows (recon metrics) can carry different keys; the CSV header is the union
  of all keys seen so far and is rewritten on each call (cheap at epoch scale).
* Console + file share one formatter so stdout and run.log are identical.

Usage
-----
    from logkit import RunLogger
    rl  = RunLogger("vae", config_snapshot=C.snapshot())
    log = rl.get_logger()
    log.info("starting …")
    rl.log_metrics(epoch=1, split="train", metrics={"loss": 0.21, "l1": 0.18})
    rl.log_metrics(epoch=1, split="val",   metrics={"loss": 0.23, "ssim": 0.81})
    rl.close()
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path


class RunLogger:
    """Owns one timestamped run directory and mirrors metrics to log/CSV/JSONL/TB."""

    def __init__(
        self,
        stage: str,
        log_root: Path | None = None,
        use_tensorboard: bool = True,
        config_snapshot: dict | None = None,
        run_name: str | None = None,
    ):
        # Resolve the log root lazily so config import order never bites us.
        if log_root is None:
            try:
                import config as C  # local import: logkit must stay import-light
                log_root = C.LOG_DIR
            except Exception:
                log_root = Path(__file__).resolve().parents[1] / "logs"

        ts = time.strftime("%Y%m%d_%H%M%S")
        name = run_name or f"{stage}_{ts}"
        self.stage = stage
        self.run_dir = Path(log_root) / name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.run_dir / "metrics.csv"
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self._rows: list[dict] = []          # accumulated metric rows (for CSV union)
        self._keys: list[str] = []           # ordered union of metric keys

        self._logger = self._build_logger()
        self._tb = self._build_tensorboard() if use_tensorboard else None

        if config_snapshot is not None:
            with open(self.run_dir / "config_snapshot.json", "w") as f:
                json.dump(config_snapshot, f, indent=2, default=str)

        self._logger.info("Run directory: %s", self.run_dir)
        if self._tb is None and use_tensorboard:
            self._logger.warning(
                "TensorBoard unavailable — CSV/JSONL logging only "
                "(pip install tensorboard to enable scalar dashboards).")

    # ── setup ────────────────────────────────────────────────────────
    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"cvpipe.{self.stage}.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

        fh = logging.FileHandler(self.run_dir / "run.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        return logger

    def _build_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            tb_dir = str(self.run_dir / "tensorboard")
            writer = SummaryWriter.__new__(SummaryWriter)
            # Init with a short timeout guard — on some Colab runtimes the
            # background writer thread hangs on first file creation.
            import threading
            result = [None]
            exc = [None]
            def _init():
                try:
                    result[0] = SummaryWriter(log_dir=tb_dir)
                except Exception as e:
                    exc[0] = e
            t = threading.Thread(target=_init, daemon=True)
            t.start()
            t.join(timeout=5.0)  # give TB 5 s to init; skip if it hangs
            if t.is_alive() or exc[0] is not None:
                return None
            return result[0]
        except Exception:
            return None

    # ── public API ───────────────────────────────────────────────────
    def get_logger(self) -> logging.Logger:
        return self._logger

    def log_metrics(self, epoch: int, split: str, metrics: dict,
                    step: int | None = None) -> None:
        """Record a metric row for one (epoch, split) to CSV + JSONL + TensorBoard."""
        row = {"epoch": epoch, "split": split}
        row.update({k: _to_py(v) for k, v in metrics.items()})

        # JSONL — append immediately (crash-safe history).
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        # CSV — keep an in-memory union and rewrite (epoch-scale, so cheap).
        self._rows.append(row)
        for k in row:
            if k not in self._keys:
                self._keys.append(k)
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._keys)
            w.writeheader()
            w.writerows(self._rows)

        # TensorBoard — one scalar per metric, tag "<key>/<split>".
        if self._tb is not None:
            tb_step = step if step is not None else epoch
            for k, v in metrics.items():
                pv = _to_py(v)
                if isinstance(pv, (int, float)):
                    self._tb.add_scalar(f"{k}/{split}", pv, tb_step)
            self._tb.flush()

    def log_image(self, tag: str, image_chw, step: int) -> None:
        """Optional: push a CHW image (numpy/tensor) to TensorBoard if available."""
        if self._tb is not None:
            try:
                self._tb.add_image(tag, image_chw, step)
            except Exception:
                pass

    def save_json(self, name: str, obj: dict) -> Path:
        path = self.run_dir / name
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        return path

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
        for h in list(self._logger.handlers):
            h.close()
            self._logger.removeHandler(h)


def _to_py(v):
    """Coerce numpy / torch scalars to plain Python for JSON & CSV."""
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return v
    return v
