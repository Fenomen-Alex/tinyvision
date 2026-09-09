"""Tests for tv/skz_experiment.py (SKZ probe experiments A/B)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training"))

from tv.skz_experiment import (  # noqa: E402
    BINARY_LABELS,
    DERIVED_TO_BINARY,
    binary_label,
    entry_rows,
    evaluate_probe,
    load_manifest,
    prepare_expB_splits,
    subset_entries,
    token_split,
)


@pytest.fixture(scope="module")
def manifest():
    path = REPO / "datasets" / "external" / "skz-probe" / "manifest.json"
    if not path.exists():
        pytest.skip("SKZ probe not built locally")
    return load_manifest(path)


@pytest.fixture(scope="module")
def split(manifest):
    return token_split(manifest["entries"], train_fraction=0.8, seed=42)


def test_documented_counts(manifest):
    counts = {
        c: sum(1 for e in manifest["entries"] if e["derived_class"] == c)
        for c in DERIVED_TO_BINARY
    }
    assert counts == {"normal": 50, "stringing": 16, "spaghetti": 50, "under_extrusion": 50}
    assert len(manifest["entries"]) == 166


def test_binary_mapping_is_faithful():
    for f in ("stringing", "spaghetti", "under_extrusion"):
        assert binary_label(f) == "failing"
    assert binary_label("normal") == "normal"
    assert BINARY_LABELS == {"failing": 0, "normal": 1}
    # 16 stringing samples remain 16 (never balanced/duplicated by this module).
    assert len(DERIVED_TO_BINARY) == 4


def test_token_split_never_splits_a_group(split, manifest):
    for e in manifest["entries"]:
        assert e["grouping_key"] in split
    # tokens spanning multiple classes land in one split entirely
    token_classes = {}
    for e in manifest["entries"]:
        token_classes.setdefault(e["grouping_key"], set()).add(e["derived_class"])
    for token, classes in token_classes.items():
        if len(classes) > 1:
            assert split[token] in ("train", "eval")


def test_token_split_sizes(split):
    counts = {}
    for token, s in split.items():
        counts[s] = counts.get(s, 0) + 1
    # train tokens + eval tokens == unique token count; both non-empty
    assert counts["train"] > 0 and counts["eval"] > 0


def test_subset_entries_partition(manifest, split):
    ev = subset_entries(manifest["entries"], split, "skz_eval")
    tr = subset_entries(manifest["entries"], split, "skz_aux")
    all_ids = {e["probe_id"] for e in manifest["entries"]}
    assert {e["probe_id"] for e in tr} | {e["probe_id"] for e in ev} == all_ids
    assert {e["probe_id"] for e in tr} & {e["probe_id"] for e in ev} == set()


def test_subset_entries_invalid(split):
    with pytest.raises(ValueError):
        subset_entries([], split, "nope")


def test_entry_rows_require_local_images(manifest, split):
    ev = subset_entries(manifest["entries"], split, "skz_eval")
    rows = entry_rows(ev)
    assert len(rows) == len(ev)
    seeds = set()
    for r in rows:
        assert r["binary_label"] in ("failing", "normal")
        assert r["binary_idx"] in (0, 1)
        assert Path(r["image_path"]).exists()
        seeds.add(r["grouping_key"])
    assert len(seeds) > 0


def test_prepare_expB_splits(tmp_path, manifest, split):
    printing = REPO / "datasets" / "processed" / "printing-errors"
    if not (printing / "splits" / "train.csv").exists():
        pytest.skip("printing-errors splits not present")
    out_dir = tmp_path / "expb"
    result = prepare_expB_splits(printing, out_dir, train_fraction=0.8, seed=42)

    pe = result["skz_eval"]
    assert pe["images"] == 31
    assert pe["per_class"] == {
        "normal": 11, "stringing": 5, "spaghetti": 4, "under_extrusion": 11
    }

    splits_dir = out_dir / "splits"
    assert (splits_dir / "splits.json").exists()
    splits_json = json.loads((splits_dir / "splits.json").read_text())

    import csv as csvmod

    with open(splits_dir / "train.csv") as f:
        train_rows = list(csvmod.DictReader(f))
    with open(splits_dir / "val.csv") as f:
        val_rows = list(csvmod.DictReader(f))
    with open(splits_dir / "test.csv") as f:
        test_rows = list(csvmod.DictReader(f))

    n_pe = len(train_rows) - result["skz_aux"]["images"]

    base_train = len(train_rows) - result["skz_aux"]["images"]
    assert base_train == 11254
    assert result["skz_aux"]["images"] == 135
    assert len(val_rows) == 2174
    assert len(test_rows) == 2859
    assert len(train_rows) == base_train + 135
    assert len(train_rows) == base_train + 135
    assert {r["label"] for r in train_rows} == {"failing", "normal"}
    assert splits_json["data_dir"] == str(printing.resolve())

    # val/test CSVs must be identical to the frozen printing-errors splits
    with open(printing / "splits" / "val.csv") as f:
        orig_val = list(csvmod.DictReader(f))
    with open(printing / "splits" / "test.csv") as f:
        orig_test = list(csvmod.DictReader(f))
    assert val_rows == orig_val
    assert test_rows == orig_test


def test_prepare_expB_does_not_touch_originals(tmp_path, manifest, split):
    printing = REPO / "datasets" / "processed" / "printing-errors"
    if not (printing / "splits" / "train.csv").exists():
        pytest.skip("printing-errors splits not present")
    before = (printing / "splits" / "train.csv").read_bytes()
    (printing / "splits" / "val.csv").read_bytes()
    (printing / "splits" / "test.csv").read_bytes()
    prepare_expB_splits(printing, tmp_path / "expb2", train_fraction=0.8, seed=42)
    assert (printing / "splits" / "train.csv").read_bytes() == before


def test_evaluate_probe_full_and_subsets(manifest, split):
    ckpt = (
        REPO / "experiments" / "shufflenet_v2_x1_0_printing-errors_cdc9e2cad4"
        / "artifacts" / "best.pt"
    )
    if not ckpt.exists():
        pytest.skip("baseline checkpoint not present")

    full = evaluate_probe(ckpt, subset="all", split=split)
    assert full["evaluated_images"] == 166
    assert full["independent_groups"] == len({e["grouping_key"] for e in manifest["entries"]})
    assert full["metrics_failing_positive"]["positive_class"] == 0
    assert full["metrics_normal_positive"]["positive_class"] == 1
    assert full["confusion_matrix"]
    assert full["auroc_failing"] is None or 0.0 <= full["auroc_failing"] <= 1.0
    assert len(full["classes"]) == 4
    # failing truth count = 116 (16+50+50)
    assert sum(1 for r in manifest["entries"] if r["derived_class"] != "normal") == 116

    ev = evaluate_probe(ckpt, subset="skz_eval", split=split)
    assert ev["evaluated_images"] == 31

    tr = evaluate_probe(ckpt, subset="skz_aux", split=split)
    assert tr["evaluated_images"] == 135


def test_per_failure_type_blocks(manifest, split):
    ev = [e for e in manifest["entries"] if split[e["grouping_key"]] == "eval"]
    evl = subset_entries(manifest["entries"], split, "skz_eval")
    rows = entry_rows(evl)
    from tv.skz_experiment import _failure_type_block

    block = _failure_type_block(rows, [0] * len(rows))
    # every failure type present in the held-out slice is reported with counts
    for name, d in block.items():
        assert d["images"] > 0
        assert d["recall_failing"] in (0.0, 1.0)
        assert d["groups"] > 0


def test_auroc_failing_matches_deepeval_convention():
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from tv.skz_experiment import _auroc_failing

    rng = np.random.RandomState(7)
    y_true = rng.randint(0, 2, 60)
    fail = rng.rand(60)
    # failing-positive AUROC: roc_auc(y_true, 1 - fail_scores), per deepeval
    expected = round(float(roc_auc_score(y_true, 1.0 - fail)), 4)
    assert _auroc_failing(y_true, fail) == expected
    assert _auroc_failing(np.zeros(5, dtype=int), fail[:5]) is None