"""Tests for config loading and model registry."""
from __future__ import annotations

from pathlib import Path

import pytest

from tv.config import ExperimentConfig, _dict_to_dataclass, save_config
from tv.models.factory import list_models, get_model_spec, MODEL_REGISTRY


def test_dict_to_dataclass() -> None:
    raw = {"model": "resnet18", "task": "classification", "train": {"epochs": 100}}
    cfg = _dict_to_dataclass(ExperimentConfig, raw)
    assert cfg.model == "resnet18"
    assert cfg.train.epochs == 100


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    cfg = ExperimentConfig(model="mobilenet_v3_small", task="classification")
    save_config(cfg, tmp_path / "config.yaml")
    assert (tmp_path / "config.yaml").exists()

    from tv.config import load_config

    loaded = load_config(tmp_path / "config.yaml")
    assert loaded.model == "mobilenet_v3_small"
    assert loaded.experiment_id != ""
    assert loaded.created_at != ""


def test_experiment_id_is_deterministic(tmp_path: Path) -> None:
    """Same config content should yield the same experiment ID."""
    from tv.config import ExperimentConfig, load_config, save_config

    cfg = ExperimentConfig(model="resnet18", task="classification", created_at="")
    save_config(cfg, tmp_path / "a.yaml")
    save_config(cfg, tmp_path / "b.yaml")

    id_a = load_config(tmp_path / "a.yaml").experiment_id
    id_b = load_config(tmp_path / "b.yaml").experiment_id
    assert id_a == id_b
    assert id_a.startswith("resnet18")


def test_model_registry_has_expected_entries() -> None:
    assert "efficientnet_b0" in MODEL_REGISTRY
    assert "resnet18" in MODEL_REGISTRY
    assert "yolov8n-cls" in MODEL_REGISTRY
    assert "yolov11n" in MODEL_REGISTRY


def test_list_models_filter() -> None:
    cls_models = list_models(task="classification")
    assert all(
        MODEL_REGISTRY[m].task == "classification" for m in cls_models
    )
    det_models = list_models(task="detection")
    assert all(MODEL_REGISTRY[m].task == "detection" for m in det_models)


def test_get_model_spec_raises_on_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_spec("nonexistent_model_xyz")
