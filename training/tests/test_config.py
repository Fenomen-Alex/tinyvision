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


def test_software_versions_are_plain_strings() -> None:
    """Software versions must be plain ``str``.

    ``torch.__version__`` is a ``torch.torch_version.TorchVersion`` (a ``str``
    subclass). PyYAML's representer only treats exact ``str`` as a scalar, so a
    subclass is serialized with a ``!!python/object/new:`` tag that
    ``yaml.safe_load`` (used by :func:`load_config`) refuses to read. Coercing to
    ``str`` keeps written configs loadable.
    """
    from tv.config import record_software

    sw = record_software()
    assert type(sw.torch) is str
    assert type(sw.torchvision) is str
    assert type(sw.ultralytics) is str
    assert type(sw.opencv) is str


def test_save_load_roundtrip_with_populated_software(tmp_path: Path) -> None:
    """A config populated by ``record_software`` must round-trip.

    Regression: ``train`` always populates software metadata, so any config it
    wrote was unreadable by ``evaluate``/``export``.
    """
    from tv.config import load_config, record_software

    cfg = ExperimentConfig(model="resnet18", task="classification")
    cfg.software = record_software()
    save_config(cfg, tmp_path / "config.yaml")

    loaded = load_config(tmp_path / "config.yaml")
    assert loaded.software.torch == cfg.software.torch
    assert loaded.model == "resnet18"


def test_save_config_writes_safe_yaml(tmp_path: Path) -> None:
    """Written configs must be readable by ``yaml.safe_load`` with no python tags."""
    import yaml

    from tv.config import record_software

    cfg = ExperimentConfig(model="resnet18", task="classification")
    cfg.software = record_software()
    save_config(cfg, tmp_path / "config.yaml")

    text = (tmp_path / "config.yaml").read_text()
    assert "python/object" not in text
    assert yaml.safe_load(text)["model"] == "resnet18"


def test_load_config_reads_legacy_torch_version_tag(tmp_path: Path) -> None:
    """Configs written before the fix carry a ``!!python/object/new:`` torch tag.

    They must still load, with the version recovered as a plain string and
    without importing or executing anything.
    """
    from tv.config import load_config

    legacy = (
        "experiment_id: legacy_demo\n"
        "model: resnet18\n"
        "task: classification\n"
        "software:\n"
        "  python: 3.11.15\n"
        "  torch: !!python/object/new:torch.torch_version.TorchVersion\n"
        "  - 2.14.0\n"
        "  torchvision: 0.29.0\n"
        "  ultralytics: 8.4.138\n"
        "  opencv: 5.0.0\n"
        "  os: Darwin 25.6.0\n"
    )
    path = tmp_path / "config.yaml"
    path.write_text(legacy)

    loaded = load_config(path)
    assert loaded.experiment_id == "legacy_demo"
    assert loaded.software.torch == "2.14.0"


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
