"""Tests for the Stage 4 dataset-forensics module (deterministic, synthetic data)."""
from __future__ import annotations

import csv
from pathlib import Path

from tv.forensics import (
    MIXED_FAILING_LABEL,
    MIXED_NORMAL_LABEL,
    PURE_FAILING,
    PURE_NORMAL,
    UNKNOWN_AMBIGUOUS,
    build_recording_audit,
    _binary_from_fine,
    _compose,
    _fine_name,
)


def _write_raw_csv(path: Path, rows) -> None:
    """rows: list of dicts with keys image, class, recording, ..."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["image", "class", "recording", "layer"], delimiter=";"
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_split_csv(path: Path, rows) -> None:
    """rows: list of paths like normal/Rec_1/f{ts}.png"""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label"])
        w.writeheader()
        for p in rows:
            w.writerow({"path": p, "label": p.split("/")[0]})


def _fixture(tmp_path: Path):
    raw = tmp_path / "raw.csv"
    _write_raw_csv(
        raw,
        [
            {"image": "ELP_1.png", "class": "0", "recording": "R1", "layer": "1"},
            {"image": "ELP_2.png", "class": "1", "recording": "R1", "layer": "2"},
            {"image": "ELP_3.png", "class": "1", "recording": "R1", "layer": "3"},
            {"image": "ELP_4.png", "class": "0", "recording": "R2", "layer": "1"},
            {"image": "ELP_5.png", "class": "2", "recording": "R3", "layer": "1"},
            {"image": "ELP_6.png", "class": "4", "recording": "R3", "layer": "2"},
        ],
    )
    # Timestamps are the trailing digits of the ELP_<n> filename.
    def ts(n):
        return n.split("_")[1].split(".")[0]

    splits = tmp_path / "splits"
    splits.mkdir()
    _write_split_csv(
        splits / "train.csv",
        [
            f"normal/R1/ELP_1.png",
            f"failing/R1/ELP_2.png",
            f"failing/R1/ELP_3.png",
            f"normal/R2/ELP_4.png",
        ],
    )
    _write_split_csv(
        splits / "val.csv",
        [f"failing/R3/ELP_5.png", f"failing/R3/ELP_6.png"],
    )
    _write_split_csv(splits / "test.csv", [])
    return raw, splits


def test_fine_and_binary_mapping():
    assert _fine_name({"class": "0"}) == "good"
    assert _fine_name({"class": "1"}) == "under_extrusion"
    assert _fine_name({"class": "2"}) == "stringing"
    assert _fine_name({"class": "4"}) == "spaghetti"
    assert _fine_name({"class": "9"}) == "unknown"
    assert _fine_name(None) == "unknown"
    assert _binary_from_fine("good") == "normal"
    assert _binary_from_fine("under_extrusion") == "failing"
    assert _binary_from_fine("unknown") == "unknown"


def test_compose_classes():
    assert _compose({"normal": 5, "failing": 0}, ["good"]) == PURE_NORMAL
    assert _compose({"normal": 0, "failing": 5}, ["stringing"]) == PURE_FAILING
    # mixed with majority failing
    assert _compose({"normal": 2, "failing": 8}, ["good", "under_extrusion"]) == MIXED_FAILING_LABEL
    # mixed with majority normal
    assert _compose({"normal": 9, "failing": 1}, ["good", "under_extrusion"]) == MIXED_NORMAL_LABEL
    assert _compose({"normal": 0, "failing": 0}, []) == UNKNOWN_AMBIGUOUS


def test_build_recording_audit(tmp_path):
    raw, splits = _fixture(tmp_path)
    audit = build_recording_audit(raw, splits)

    recs = audit["recordings"]

    # R1: 1 good + 2 under_extrusion -> mixed, majority failing
    r1 = recs["R1"]
    assert r1["split"] == "train"
    assert r1["frames"] == 3
    assert r1["composition"] == MIXED_FAILING_LABEL
    assert r1["n_transitions"] == 1
    assert r1["starts_good"] is True
    assert r1["fine_runs"] == ["good", "under_extrusion"]

    # R2: all good in train -> PURE_NORMAL
    r2 = recs["R2"]
    assert r2["split"] == "train"
    assert r2["composition"] == PURE_NORMAL
    assert r2["starts_good"] is True

    # R3: stringing + spaghetti in val -> PURE_FAILING
    r3 = recs["R3"]
    assert r3["split"] == "val"
    assert r3["composition"] == PURE_FAILING

    # Each recording assigned to exactly one split.
    splits_seen = {}
    for rec_id, info in recs.items():
        splits_seen.setdefault(info["split"], 0)
        splits_seen[info["split"]] += 1
    assert splits_seen == {"train": 2, "val": 1}


def test_audit_summary(tmp_path):
    raw, splits = _fixture(tmp_path)
    audit = build_recording_audit(raw, splits)
    s = audit["summary"]
    # train: 2 recs (R1 mixed-failing, R2 pure-normal)
    assert s["per_split"]["train"]["n_recordings"] == 2
    assert s["per_split"]["train"]["composition"]["MIXED_FAILING_LABEL"] == 1
    assert s["per_split"]["train"]["composition"]["PURE_NORMAL"] == 1
    # val: 1 rec pure-failing
    assert s["per_split"]["val"]["composition"]["PURE_FAILING"] == 1
    assert s["per_split"]["test"]["n_recordings"] == 0
