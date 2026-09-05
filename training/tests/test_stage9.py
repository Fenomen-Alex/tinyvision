"""Tests for tv/stage9.py (data sufficiency analysis)."""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

from tv.stage9 import (
    RecordingInfo,
    build_inventory,
    frame_concentration,
    failure_type_diversity,
    normal_diversity,
    stringing_investigation,
    normal_domain_investigation,
    split_distribution_comparison,
    data_quality_analysis,
    run_stage9,
    _extract_recording_id,
    _extract_date,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_recording(rid, split, total, normal, failing, fine_counts=None,
                    composition="UNKNOWN", date=None):
    return RecordingInfo(
        recording_id=rid,
        split=split,
        total_frames=total,
        normal_frames=normal,
        failing_frames=failing,
        composition=composition,
        fine_counts=fine_counts or {},
        date=date,
        source_class_dirs={},
    )


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------

def test_extract_recording_id():
    assert _extract_recording_id("failing/Recording_01122022_16_06_44/frame.png") == "Recording_01122022_16_06_44"
    assert _extract_recording_id("a/b/c") == "b"
    assert _extract_recording_id("a/b") is None


def test_extract_date():
    assert _extract_date("Recording_01122022_16_06_44") == "01122022"
    assert _extract_date("Recording_14.02.2023_17_53_55") == "14.02.2023"
    assert _extract_date("other_id") is None


# ---------------------------------------------------------------------------
# Tests: RecordingInfo
# ---------------------------------------------------------------------------

def test_recording_info_properties():
    r = _make_recording("R1", "train", 100, 40, 60)
    assert r.has_normal
    assert r.has_failing
    assert r.is_mixed
    assert not r.is_pure_normal
    assert not r.is_pure_failing


def test_pure_normal():
    r = _make_recording("R1", "train", 100, 100, 0)
    assert r.is_pure_normal
    assert not r.is_mixed
    assert not r.is_pure_failing


def test_pure_failing():
    r = _make_recording("R1", "train", 100, 0, 100, fine_counts={"stringing": 100})
    assert r.is_pure_failing
    assert not r.is_mixed


# ---------------------------------------------------------------------------
# Tests: frame_concentration
# ---------------------------------------------------------------------------

def test_frame_concentration():
    inv = {
        "train": [
            _make_recording("R1", "train", 200, 100, 100),
            _make_recording("R2", "train", 100, 50, 50),
            _make_recording("R3", "train", 50, 25, 25),
        ]
    }
    fc = frame_concentration(inv)
    assert "train" in fc
    assert fc["train"]["n_recordings"] == 3
    assert fc["train"]["total_frames"] == 350
    assert fc["train"]["frames_per_recording"]["max"] == 200
    assert fc["train"]["frames_per_recording"]["min"] == 50
    assert fc["train"]["top5_frame_fraction"] == 1.0  # only 3 recordings, top5 includes all


def test_frame_concentration_real_data():
    inv = build_inventory()
    fc = frame_concentration(inv)
    # Train should have >100 recordings
    assert fc["train"]["n_recordings"] > 100
    assert fc["train"]["total_frames"] > 10000


# ---------------------------------------------------------------------------
# Tests: failure_type_diversity
# ---------------------------------------------------------------------------

def test_failure_type_diversity():
    inv = {
        "train": [
            _make_recording("R1", "train", 100, 20, 80, fine_counts={"stringing": 80}),
            _make_recording("R2", "train", 150, 0, 150, fine_counts={"under_extrusion": 150}),
        ],
        "val": [
            _make_recording("R3", "val", 80, 0, 80, fine_counts={"stringing": 80}),
        ],
    }
    div = failure_type_diversity(inv)
    assert "stringing" in div
    assert "under_extrusion" in div
    assert div["stringing"]["train"]["n_recordings"] == 1
    assert div["stringing"]["train"]["frames"] == 80
    assert div["under_extrusion"]["train"]["n_recordings"] == 1
    assert div["under_extrusion"]["train"]["frames"] == 150
    assert div["stringing"]["val"]["n_recordings"] == 1


# ---------------------------------------------------------------------------
# Tests: normal_diversity
# ---------------------------------------------------------------------------

def test_normal_diversity():
    inv = {
        "train": [
            _make_recording("R1", "train", 100, 100, 0),
            _make_recording("R2", "train", 100, 30, 70),
        ],
    }
    nd = normal_diversity(inv)
    assert nd["train"]["pure_normal_recordings"] == 1
    assert nd["train"]["mixed_recordings_with_normal"] == 1
    assert nd["train"]["total_normal_frames"] == 130


# ---------------------------------------------------------------------------
# Tests: stringing_investigation
# ---------------------------------------------------------------------------

def test_stringing_investigation():
    inv = {
        "train": [
            _make_recording("R1", "train", 200, 0, 200,
                            fine_counts={"stringing": 200}, composition="PURE_FAILING"),
            _make_recording("R2", "train", 5, 0, 5,
                            fine_counts={"stringing": 5}, composition="PURE_FAILING"),
            _make_recording("R3", "train", 100, 0, 100,
                            fine_counts={"under_extrusion": 100}),
        ],
    }
    si = stringing_investigation(inv)
    assert si["n_recordings"] == 2
    assert si["total_stringing_frames"] == 205
    assert si["n_tiny_sessions_lt10_frames"] == 1
    assert "R2" in si["tiny_session_ids"]
    assert "R1" in si["large_session_ids"]


# ---------------------------------------------------------------------------
# Tests: data_quality_analysis
# ---------------------------------------------------------------------------

def test_data_quality_flags_tiny():
    inv = {
        "train": [
            _make_recording("R1", "train", 3, 1, 2),
        ],
    }
    dq = data_quality_analysis(inv)
    assert dq["n_concerns"] >= 1
    types = [c["type"] for c in dq["concerns"]]
    assert "tiny_recording" in types


def test_data_quality_no_concerns_for_good_data():
    inv = {
        "train": [
            _make_recording("R1", "train", 100, 40, 60,
                            fine_counts={"stringing": 60}),
        ],
    }
    dq = data_quality_analysis(inv)
    assert dq["n_concerns"] == 0


# ---------------------------------------------------------------------------
# Tests: split_distribution_comparison
# ---------------------------------------------------------------------------

def test_split_distribution_comparison():
    inv = {
        "train": [
            _make_recording("R1", "train", 100, 40, 60,
                            fine_counts={"stringing": 60}),
        ],
        "val": [
            _make_recording("R2", "val", 80, 0, 80,
                            fine_counts={"under_extrusion": 80}),
        ],
    }
    sd = split_distribution_comparison(inv)
    assert sd["train"]["n_recordings"] == 1
    assert sd["train"]["normal_fraction"] == 0.4
    assert sd["val"]["failure_type_recordings"]["under_extrusion"] == 1


# ---------------------------------------------------------------------------
# Tests: run_stage9 end-to-end (real data)
# ---------------------------------------------------------------------------

def test_run_stage9_end_to_end():
    result = run_stage9()
    assert result["stage"] == 9
    assert "dataset_inventory" in result
    assert "frame_concentration" in result
    assert "failure_type_diversity" in result
    assert "stringing_investigation" in result
    assert "normal_domain_investigation" in result
    assert "split_distribution_comparison" in result
    assert "data_quality" in result
    assert "acquisition_design" in result
    assert "future_experiment" in result
    assert "retraining_gate" in result
    # Verify real data bounds
    assert len(result["dataset_inventory"]["train"]) > 100
    assert len(result["dataset_inventory"]["test"]) > 20
    assert result["retraining_gate"]["A_is_evidence_sufficient_for_immediate_retraining"] is False


def test_stringing_investigation_real_data():
    result = run_stage9()
    si = result["stringing_investigation"]
    assert si["n_recordings"] >= 20  # should have ~23 stringing training recordings
    assert si["total_stringing_frames"] > 2000
    assert len(si["large_session_ids"]) > 10


def test_normal_domain_investigation_real_data():
    result = run_stage9()
    nd = result["normal_domain_investigation"]
    assert nd["training_normal"]["n_pure_normal_recordings"] > 30
    assert nd["training_normal"]["brightness_mean"] is not None
    assert nd["problematic_test_recording"]["brightness_mean"] is not None
    # Brightness difference should be large
    train_bright = nd["training_normal"]["brightness_mean"]
    test_bright = nd["problematic_test_recording"]["brightness_mean"]
    assert abs(train_bright - test_bright) > 50


def test_failure_type_diversity_real_data():
    result = run_stage9()
    ft = result["failure_type_diversity"]
    assert "stringing" in ft
    assert "under_extrusion" in ft
    assert ft["stringing"]["train"]["n_recordings"] >= 20
    assert ft["under_extrusion"]["train"]["n_recordings"] >= 30


def test_retraining_gate_structure():
    result = run_stage9()
    rg = result["retraining_gate"]
    assert "A_is_evidence_sufficient_for_immediate_retraining" in rg
    assert "B_what_evidence_or_data_is_missing" in rg
    assert "C_what_data_acquisition_unblocks_retraining" in rg
    assert "D_what_would_constitute_success" in rg
    assert "priority_order" in rg
    assert len(rg["priority_order"]) >= 4


def test_future_experiment_structure():
    result = run_stage9()
    fe = result["future_experiment"]
    assert "baseline" in fe
    assert "old_test_set" in fe
    assert "new_data_split_policy" in fe
    assert "primary_metrics" in fe
    assert "success_criteria" in fe
    assert fe["old_test_set"]["n_recordings"] == 24
