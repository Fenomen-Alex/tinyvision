"""Tests for reproducibility: seeds and distinct experiment IDs."""
from __future__ import annotations

from pathlib import Path

import pytest

from tv.config import DatasetConfig, ExperimentConfig, _generate_experiment_id, load_config, save_config
from tv.train import set_seed


def test_set_seed_makes_random_ops_reproducible() -> None:
    """Seeded model forward passes must be identical."""
    import torch

    from tv.models.factory import build_classifier

    def run():
        set_seed(7)
        m = build_classifier("resnet18", 2, pretrained=False)
        x = torch.randn(4, 3, 64, 64)
        with torch.no_grad():
            return m(x)

    a = run()
    b = run()
    assert torch.equal(a, b)


def test_shuffled_loader_is_deterministic_with_generator() -> None:
    """DataLoader shuffle must be repeatable given the same generator."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(torch.arange(100))
    def make():
        return [
            b.tolist() for (b,) in DataLoader(
                ds, batch_size=10, shuffle=True,
                generator=torch.Generator().manual_seed(5),
            )
        ]

    assert make() == make()


def test_experiment_ids_distinct_when_hyperparams_differ() -> None:
    """Two configs differing only in hyperparameters must get distinct IDs."""
    c1 = ExperimentConfig(
        model="efficientnet_b0",
        dataset=DatasetConfig(name="mine", data_dir="/d"),
    )
    # change lr + epochs
    c2 = ExperimentConfig(
        model="efficientnet_b0",
        dataset=DatasetConfig(name="mine", data_dir="/d"),
    )
    c2.train.epochs = 10
    c2.train.learning_rate = 0.1
    id1 = _generate_experiment_id(c1)
    id2 = _generate_experiment_id(c2)
    assert id1 != id2
    assert id1 != "" and id2 != ""


def test_same_config_same_id(tmp_path: Path) -> None:
    p1 = tmp_path / "a.yaml"
    p2 = tmp_path / "b.yaml"
    cfg = ExperimentConfig(model="resnet18", task="classification", created_at="")
    save_config(cfg, p1)
    save_config(cfg, p2)
    assert load_config(p1).experiment_id == load_config(p2).experiment_id
