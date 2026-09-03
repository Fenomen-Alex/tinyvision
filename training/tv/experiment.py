"""Experiment lifecycle management."""
from __future__ import annotations

import json
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from .config import ExperimentConfig, save_config


class ExperimentManager:
    """Manages experiment directory, logging, and status tracking."""

    def __init__(self, config: ExperimentConfig, base_dir: str | Path = "experiments"):
        self.config = config
        self.base_dir = Path(base_dir)
        self.exp_dir = self.base_dir / config.experiment_id
        self.artifacts_dir = self.exp_dir / "artifacts"
        self._start_time: float | None = None

    def setup(self) -> None:
        """Create experiment directories and write initial config."""
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        save_config(self.config, self.exp_dir / "config.yaml")
        self._write_status("started")

    def _write_status(self, status: str, error: str | None = None) -> None:
        record = {
            "experiment_id": self.config.experiment_id,
            "status": status,
        }
        if self._start_time is not None:
            record["elapsed_seconds"] = round(time.time() - self._start_time, 2)
        if error:
            record["error"] = error
        (self.exp_dir / "status.json").write_text(json.dumps(record, indent=2))

    def save_metrics(self, metrics: dict, name: str = "metrics") -> None:
        path = self.artifacts_dir / f"{name}.json"
        path.write_text(json.dumps(metrics, indent=2))

    def save_checkpoint(self, state: dict, filename: str = "checkpoint.pt") -> Path:
        import torch

        path = self.artifacts_dir / filename
        torch.save(state, path)
        return path

    @contextmanager
    def run(self):
        """Context manager that tracks experiment status and timing."""
        self._start_time = time.time()
        self.setup()
        try:
            yield self
            self._write_status("completed")
        except BaseException as exc:
            tb = traceback.format_exc()
            (self.exp_dir / "traceback.txt").write_text(tb)
            if isinstance(exc, KeyboardInterrupt):
                self._write_status("interrupted", error="KeyboardInterrupt")
            else:
                self._write_status("failed", error=str(exc))
            raise
