"""Tests for tv/exp2_external.py (Experiment 2: external video-frames augmentation)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training"))

from tv.exp2_external import (  # noqa: E402
    BINARY_LABELS,
    DECISION_VOCAB,
    DECISION_TO_BINARY,
    evaluate_external,
    expanded_rows,
    external_normal_rows,
    prepare_exp2_splits,
    verified_training_rows,
)

BASELINE_CKPT = (
    REPO / "experiments" / "shufflenet_v2_x1_0_printing-errors_cdc9e2cad4"
    / "artifacts" / "best.pt"
)

# External experiment-2 inputs (label JSONs + extracted frames) are not
# committed (see .gitignore). When absent these tests skip explicitly instead
# of erroring, matching test_skz_experiment.py. A skip means "unavailable
# data", never "passing".
EXT_LABELS = REPO / "datasets" / "external" / "video-frames" / "labels"
EXT_VERIFIED = EXT_LABELS / "verified_labels.json"
EXT_MANIFEST = EXT_LABELS / "review_manifest.json"


def _skip_if_external_unavailable(*paths: Path) -> None:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        shown = ", ".join(str(m.relative_to(REPO)) for m in missing)
        pytest.skip(f"external experiment-2 data unavailable: {shown}")


@pytest.fixture(scope="module")
def training_rows():
    _skip_if_external_unavailable(EXT_VERIFIED)
    try:
        return verified_training_rows()
    except FileNotFoundError as exc:
        pytest.skip(f"external experiment-2 frames unavailable: {exc}")


@pytest.fixture(scope="module")
def normal_rows():
    _skip_if_external_unavailable(EXT_MANIFEST)
    try:
        return external_normal_rows()
    except FileNotFoundError as exc:
        pytest.skip(f"external experiment-2 frames unavailable: {exc}")


def test_verified_training_rows_counts(training_rows):
    assert len(training_rows) == 44
    assert sorted({r["source_id"] for r in training_rows}) == ["NORMAL001", "SP001"]
    assert set(r["binary_label"] for r in training_rows) == {"failing"}
    # all merged verified failing maps to failing; no normal/uncertain here
    assert all(r["decision"] == "failing" for r in training_rows)
    # per-source: NORMAL001 21 frames, SP001 23 frames
    from collections import Counter
    counts = Counter(r["source_id"] for r in training_rows)
    assert counts == {"NORMAL001": 21, "SP001": 23}
    for r in training_rows:
        assert Path(r["image_path"]).exists()


def test_external_normal_rows_counts(normal_rows):
    assert len(normal_rows) == 24
    assert {r["source_id"] for r in normal_rows} == {"UE003"}
    assert set(r["binary_label"] for r in normal_rows) == {"normal"}
    units = {r["review_unit_id"] for r in normal_rows}
    assert len(units) == 8
    for r in normal_rows:
        assert Path(r["image_path"]).exists()


def test_binary_mapping_is_faithful():
    assert DECISION_TO_BINARY == {"normal": "normal", "failing": "failing"}
    assert DECISION_VOCAB == {"normal", "failing", "uncertain"}
    assert BINARY_LABELS == {"failing": 0, "normal": 1}


def test_expanded_rows_rejects_unknown_decision():
    with pytest.raises(ValueError):
        expanded_rows([{"review_unit_id": "X", "source_id": "X", "frames": [0],
                        "decision": "bogus", "failure_type": None}])


def test_prepare_exp2_splits(tmp_path, training_rows):
    printing = REPO / "datasets" / "processed" / "printing-errors"
    if not (printing / "splits" / "train.csv").exists():
        pytest.skip("printing-errors splits not present")
    out_dir = tmp_path / "exp2"
    result = prepare_exp2_splits(printing, out_dir)

    splits_dir = out_dir / "splits"
    assert (splits_dir / "splits.json").exists()
    splits_json = json.loads((splits_dir / "splits.json").read_text())

    with open(splits_dir / "train.csv") as f:
        train_rows = list(csv.DictReader(f))
    with open(splits_dir / "val.csv") as f:
        val_rows = list(csv.DictReader(f))
    with open(splits_dir / "test.csv") as f:
        test_rows = list(csv.DictReader(f))

    assert len(train_rows) == 11254 + 44
    assert len(val_rows) == 2174
    assert len(test_rows) == 2859
    assert {r["label"] for r in train_rows} == {"failing", "normal"}
    assert splits_json["data_dir"] == str(printing.resolve())

    # external rows are absolute paths; printing-errors rows stay relative
    external_rows = [r for r in train_rows if r["path"].startswith("/")]
    assert len(external_rows) == 44
    assert all(Path(r["path"]).exists() for r in external_rows)

    # val/test CSVs must be byte-identical to the frozen printing-errors splits
    with open(printing / "splits" / "val.csv") as f:
        orig_val = f.read()
    with open(printing / "splits" / "test.csv") as f:
        orig_test = f.read()
    with open(splits_dir / "val.csv") as f:
        new_val = f.read()
    with open(splits_dir / "test.csv") as f:
        new_test = f.read()
    assert new_val == orig_val
    assert new_test == orig_test

    derived = json.loads((out_dir / "derived_manifest.json").read_text())
    assert derived["training_augmentation"]["frames"] == 44
    assert derived["external_normal_eval"]["frames"] == 24
    assert derived["role_by_source"] == {
        "NORMAL001": "training_augmentation (failing)",
        "SP001": "training_augmentation (failing)",
        "UE003": "external_normal_eval (normal)",
    }


def test_prepare_exp2_does_not_touch_originals(tmp_path):
    printing = REPO / "datasets" / "processed" / "printing-errors"
    if not (printing / "splits" / "train.csv").exists():
        pytest.skip("printing-errors splits not present")
    _skip_if_external_unavailable(EXT_VERIFIED, EXT_MANIFEST)
    before = (printing / "splits" / "train.csv").read_bytes()
    try:
        prepare_exp2_splits(printing, tmp_path / "exp2b")
    except FileNotFoundError as exc:
        pytest.skip(f"external experiment-2 frames unavailable: {exc}")
    assert (printing / "splits" / "train.csv").read_bytes() == before


def test_evaluate_external_normal_subset(normal_rows):
    if not BASELINE_CKPT.exists():
        pytest.skip("baseline checkpoint not present")
    result = evaluate_external(BASELINE_CKPT, subset="normal", kind="generalization")
    assert result["subset"] == "normal"
    assert result["kind"] == "generalization"
    assert result["evaluated_frames"] == 24
    assert result["sources"] == ["UE003"]
    m = result["metrics_failing_positive"]
    assert m["positive_class"] == 0
    assert result["normal_frames"]["frames"] == 24
    assert 0.0 <= result["normal_frames"]["false_positive_rate"] <= 1.0
    assert result["auroc_failing"] is None  # single class -> no AUROC


def test_evaluate_external_failing_subset(training_rows):
    if not BASELINE_CKPT.exists():
        pytest.skip("baseline checkpoint not present")
    result = evaluate_external(BASELINE_CKPT, subset="failing", kind="generalization")
    assert result["evaluated_frames"] == 44
    assert set(result["sources"]) == {"NORMAL001", "SP001"}
    m = result["metrics_failing_positive"]
    assert m["tp"] + m["fn"] == 44
    assert result["auroc_failing"] is None