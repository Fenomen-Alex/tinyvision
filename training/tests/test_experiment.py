"""Tests for experiment lifecycle / status tracking."""
from __future__ import annotations

from pathlib import Path

import pytest

from tv.config import ExperimentConfig
from tv.experiment import ExperimentManager


def test_completed_sets_status(tmp_path: Path) -> None:
    config = ExperimentConfig(model="resnet18", task="classification")
    exp = ExperimentManager(config, base_dir=str(tmp_path))
    with exp.run() as m:
        m.save_metrics({"loss": 0.5})
    import json

    status = json.loads((tmp_path / config.experiment_id / "status.json").read_text())
    assert status["status"] == "completed"
    assert (tmp_path / config.experiment_id / "artifacts" / "metrics.json").exists()


def test_failure_sets_failed_status(tmp_path: Path) -> None:
    config = ExperimentConfig(model="resnet18", task="classification")
    exp = ExperimentManager(config, base_dir=str(tmp_path))
    with pytest.raises(RuntimeError):
        with exp.run():
            raise RuntimeError("boom")

    import json

    status = json.loads((tmp_path / config.experiment_id / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["error"] == "boom"
    assert (tmp_path / config.experiment_id / "traceback.txt").exists()


def test_interrupt_does_not_mark_completed(tmp_path: Path) -> None:
    """SIGINT/Ctrl+C must not leave a misleading 'completed' status."""
    config = ExperimentConfig(model="resnet18", task="classification")
    exp = ExperimentManager(config, base_dir=str(tmp_path))
    with pytest.raises(KeyboardInterrupt):
        with exp.run():
            raise KeyboardInterrupt()

    import json

    status = json.loads((tmp_path / config.experiment_id / "status.json").read_text())
    assert status["status"] != "completed"
    assert status["status"] == "interrupted"