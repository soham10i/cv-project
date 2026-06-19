"""
Unified per-run logging for after-training study.
===================================================
Every training run gets its own timestamped folder under LOG_DIR:

    logs/<stage>_<timestamp>/
        run.log              full console log (file sink)
        config_snapshot.json the exact config the run used (reproducibility)
        metrics.jsonl        one JSON object per logged step (machine-readable)
        metrics.csv          flat table (epoch, split, key, value) for quick plots
        tensorboard/         TensorBoard event files (if tensorboard installed)

TensorBoard is optional and degrades gracefully — if the package is missing the
CSV/JSONL still capture everything, and you can `pip install tensorboard` later
to view, or just plot the CSV.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import config as C


class RunLogger:
    def __init__(self, stage: str, config_snapshot: dict | None = None,
                 run_name: str | None = None, use_tensorboard: bool = True):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = C.LOG_DIR / f"{stage}_{run_name or ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stage = stage

        # — file + console logger —
        self.logger = logging.getLogger(f"{stage}.{ts}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self._fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                                      datefmt="%H:%M:%S")
        self._fh = logging.FileHandler(self.run_dir / "run.log", encoding="utf-8")
        self._fh.setFormatter(self._fmt)
        if not self.logger.handlers:
            sh = logging.StreamHandler(); sh.setFormatter(self._fmt)
            self.logger.addHandler(sh)
            self.logger.addHandler(self._fh)

        # — metric sinks —
        self._jsonl = open(self.run_dir / "metrics.jsonl", "a")
        self._csv_path = self.run_dir / "metrics.csv"
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "split", "key", "value"])

        # — tensorboard (optional) —
        self.tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
            except Exception as e:  # missing tensorboard pkg, etc.
                self.logger.warning("TensorBoard disabled (%s) — CSV/JSONL still on. "
                                    "`pip install tensorboard` to enable.", e)

        if config_snapshot is not None:
            self.save_json("config_snapshot.json", config_snapshot)
        self.logger.info("Run dir: %s", self.run_dir)

    def get_logger(self) -> logging.Logger:
        return self.logger

    def attach(self, logger: logging.Logger) -> None:
        """Also route an existing module logger's records into this run's run.log."""
        logger.addHandler(self._fh)

    def log_metrics(self, epoch: int, split: str, metrics: dict) -> None:
        """Record a dict of scalars for one (epoch, split) under all sinks."""
        self._jsonl.write(json.dumps({"epoch": epoch, "split": split, **metrics}) + "\n")
        self._jsonl.flush()
        with open(self._csv_path, "a", newline="") as f:
            w = csv.writer(f)
            for k, v in metrics.items():
                w.writerow([epoch, split, k, v])
        if self.tb is not None:
            for k, v in metrics.items():
                try:
                    self.tb.add_scalar(f"{split}/{k}", float(v), epoch)
                except (TypeError, ValueError):
                    pass

    def save_json(self, name: str, obj: dict) -> None:
        with open(self.run_dir / name, "w") as f:
            json.dump(obj, f, indent=2)

    def close(self) -> None:
        try:
            self._jsonl.close()
        except Exception:
            pass
        if self.tb is not None:
            self.tb.flush(); self.tb.close()
