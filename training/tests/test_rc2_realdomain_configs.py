"""Structural tests for the rc2 real-domain config + reliability suite."""
from __future__ import annotations

from pathlib import Path

from tv.config import load_config
from tv.reliability import load_suite

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_rc2_realdomain_config_loads():
    cfg = load_config(CONFIGS / "planned" / "rc2_realdomain.yaml")
    assert cfg.model == "shufflenet_v2_x1_0"
    assert cfg.dataset.data_dir == "datasets/external-cache/rc2-realdomain"
    assert cfg.dataset.image_size == 224


def test_suite_realdomain_loads():
    suite = load_suite(CONFIGS / "reliability" / "suite-realdomain.yaml")
    assert suite.name == "rc2-realdomain"
    assert suite.positive_class == 0
    assert suite.dataset_root == "datasets/external-cache/rc2-realdomain"
    roles = {s.id: s.role for s in suite.surfaces}
    assert roles["dev_val"] == "dev_validation"
    assert roles["locked_test"] == "locked_test"
    assert roles["real_normal_holdout"] == "normal_holdout"
    for s in suite.surfaces:
        assert s.source_key == "stream"
        assert s.label_map == {"failing": 0, "normal": 1}
