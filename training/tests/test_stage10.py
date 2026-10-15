"""Tests for tv/stage10.py (cluster and domain generalization audit)."""
from __future__ import annotations

import numpy as np
from pathlib import Path

import pytest

from tv.stage10 import (
    Frame,
    Recording,
    extract_date,
    recording_metrics,
    image_stats,
    stringing_cluster_analysis,
    stringing_variation,
    darkness_audit,
    brightness_buckets,
    under_extrusion_analysis,
    spaghetti_limitation,
    documented_test_observations,
    acquisition_decision,
    future_split_design,
    retraining_gate,
    run_stage10,
    load_recordings,
)

# Real-data tests read the git-ignored model-inference trajectory caches; skip
# when absent.
_REPO = Path(__file__).resolve().parents[2]
_CACHE = _REPO / "docs" / "research" / "evalcaches"
_requires_cache = pytest.mark.skipif(
    not (_CACHE / "frozen_train_trajectories.json").exists(),
    reason="trajectory caches not present (git-ignored)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame(name="f.png", label="normal", fine="normal", score=0.1):
    return Frame(name=name, label=label, fine=fine, score=score)


def _make_recording(rid, split, frames):
    return Recording(recording_id=rid, split=split, frames=frames)


# ---------------------------------------------------------------------------
# extract_date
# ---------------------------------------------------------------------------

def test_extract_date_ddmmyyyy():
    assert extract_date("Recording_01122022_16_06_44") == "01122022"


def test_extract_date_dotformat():
    assert extract_date("Recording_14.02.2023_17_53_55") == "14.02.2023"


def test_extract_date_no_recording_prefix():
    assert extract_date("other_id") is None


def test_extract_date_minimal():
    assert extract_date("Recording_X_time") == "X"


# ---------------------------------------------------------------------------
# recording_metrics
# ---------------------------------------------------------------------------

def test_recording_metrics_all_normal():
    rec = _make_recording("R1", "train", [
        _frame(score=0.1), _frame(score=0.2), _frame(score=0.3),
    ])
    m = recording_metrics(rec, threshold=0.65)
    assert m["n_frames"] == 3
    assert m["tp"] == 0
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["tn"] == 3
    assert m["recall"] is None
    assert m["fpr"] == 0.0


def test_recording_metrics_all_failing():
    rec = _make_recording("R1", "train", [
        _frame(label="failing", fine="stringing", score=0.9),
        _frame(label="failing", fine="stringing", score=0.8),
        _frame(label="failing", fine="stringing", score=0.5),
    ])
    m = recording_metrics(rec, threshold=0.65)
    assert m["tp"] == 2
    assert m["fn"] == 1
    assert m["recall"] is not None
    assert abs(m["recall"] - round(2 / 3, 4)) < 1e-6
    assert abs(m["fnr"] - round(1 / 3, 4)) < 1e-6


def test_recording_metrics_mixed():
    rec = _make_recording("R1", "train", [
        _frame(label="failing", fine="stringing", score=0.9),
        _frame(label="failing", fine="stringing", score=0.5),
        _frame(score=0.1),
        _frame(score=0.8),
    ])
    m = recording_metrics(rec, threshold=0.65)
    assert m["tp"] == 1
    assert m["fn"] == 1
    assert m["fp"] == 1
    assert m["tn"] == 1
    assert abs(m["recall"] - 0.5) < 1e-6
    assert abs(m["fpr"] - 0.5) < 1e-6


def test_recording_metrics_empty():
    rec = _make_recording("R1", "train", [])
    m = recording_metrics(rec, threshold=0.65)
    assert m["n_frames"] == 0
    assert m["present"] is False


def test_recording_metrics_score_quantiles():
    rec = _make_recording("R1", "train", [
        _frame(score=0.1), _frame(score=0.9), _frame(score=0.5),
        _frame(score=0.3), _frame(score=0.7),
    ])
    m = recording_metrics(rec, threshold=0.65)
    assert 0.0 < m["score_q05"] < m["score_q95"] < 1.0
    assert m["score_median"] == 0.5


# ---------------------------------------------------------------------------
# Recording composition
# ---------------------------------------------------------------------------

def test_pure_normal():
    rec = _make_recording("R1", "train", [_frame(), _frame()])
    assert rec.composition == "PURE_NORMAL"


def test_pure_failing():
    rec = _make_recording("R1", "train", [
        _frame(label="failing", fine="stringing"),
        _frame(label="failing", fine="stringing"),
    ])
    assert rec.composition == "PURE_FAILING"


def test_mixed():
    rec = _make_recording("R1", "train", [
        _frame(), _frame(label="failing", fine="stringing"),
    ])
    assert rec.composition == "MIXED"


# ---------------------------------------------------------------------------
# extract_date edge cases
# ---------------------------------------------------------------------------

def test_extract_date_empty_string():
    assert extract_date("") is None


def test_extract_date_single_part():
    assert extract_date("Recording") is None


# ---------------------------------------------------------------------------
# Real-data integration tests
# ---------------------------------------------------------------------------

@_requires_cache
def test_load_recordings_real():
    recs = load_recordings(("train", "val"))
    assert "train" in recs
    assert "val" in recs
    assert len(recs["train"]) > 50
    assert len(recs["val"]) > 10


@_requires_cache
def test_stringing_cluster_analysis_real():
    recs = load_recordings(("train", "val"))
    result = stringing_cluster_analysis(recs)
    assert "clusters" in result
    assert len(result["clusters"]) >= 5
    # 20122022 cluster should exist and have fnr_mean > 0
    assert "20122022" in result["clusters"]
    c = result["clusters"]["20122022"]
    assert c["fnr_mean"] > 0.1


@_requires_cache
def test_stringing_variation_real():
    recs = load_recordings(("train", "val"))
    result = stringing_variation(recs)
    assert result["n_train"] > 15
    assert result["n_val"] > 2
    assert len(result["rows"]) > 20


@_requires_cache
def test_darkness_audit_real():
    recs = load_recordings(("train", "val"))
    result = darkness_audit(recs)
    assert result["n_normal_recordings_train"] > 30
    assert result["brightness_fpr_correlation"] is not None
    # Brightness-FPR correlation should be weak positive (not strongly negative)
    assert result["brightness_fpr_correlation"] > -0.5


@_requires_cache
def test_brightness_buckets_real():
    recs = load_recordings(("train", "val"))
    result = brightness_buckets(recs)
    assert result["available"] is True
    assert "dark" in result["buckets"]
    assert "medium" in result["buckets"]
    assert "bright" in result["buckets"]
    dark = result["buckets"]["dark"]
    medium = result["buckets"]["medium"]
    bright = result["buckets"]["bright"]
    # Dark bucket should have lowest or comparable FPR (NOT highest)
    assert dark["framewise_fpr"] <= medium["framewise_fpr"]


@_requires_cache
def test_under_extrusion_analysis_real():
    recs = load_recordings(("train", "val"))
    result = under_extrusion_analysis(recs)
    assert result["n_recordings"] > 30
    assert result["n_dates"] > 10
    assert result["fnr_mean"] is not None
    assert result["fnr_mean"] < 0.2  # overall mean should be low


@_requires_cache
def test_spaghetti_limitation_real():
    recs = load_recordings(("train", "val"))
    result = spaghetti_limitation(recs)
    assert result["train_spaghetti"]["n_recordings"] == 5
    assert result["val_spaghetti"]["n_recordings"] == 3
    assert result["conclusions"]["frozen_test_spaghetti_performance_measurable"] is False
    assert result["conclusions"]["valid_spaghetti_generalization_conclusion"] is False


@_requires_cache
def test_documented_test_observations_real():
    result = documented_test_observations()
    assert "catastrophic_fn" in result
    assert result["catastrophic_fn"]["fnr_approx"] == 0.898
    assert "high_fp" in result
    assert result["high_fp"]["fp_at_065"] > 0
    assert "firewall_note" in result


def test_acquisition_decision_structure():
    result = acquisition_decision()
    assert "priorities" in result
    assert "stringing" in result["priorities"]
    assert "dark_normal" in result["priorities"]
    assert "spaghetti" in result["priorities"]
    assert "under_extrusion" in result["priorities"]


def test_future_split_design_structure():
    result = future_split_design()
    assert result["current_experiment_1_split_unchanged"] is True
    assert result["principle"] is not None
    assert len(result["dimensions_to_hold_out"]) >= 4


def test_retraining_gate_structure():
    result = retraining_gate()
    assert result["A_retraining_justified"] is False
    assert "B_first_data_to_add" in result
    assert "C_still_missing" in result
    assert "D_minimum_useful_experiment" in result
    assert "E_result_that_justifies_retraining" in result
    assert len(result["notes"]) >= 2


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------

@_requires_cache
def test_run_stage10_end_to_end():
    result = run_stage10()
    assert result["stage"] == 10
    assert result["threshold"] == 0.65
    assert result["test_frozen"] is True
    assert "stringing_cluster_analysis" in result
    assert "darkness_audit" in result
    assert "brightness_buckets" in result
    assert "under_extrusion" in result
    assert "spaghetti_limitation" in result
    assert "documented_test_observations" in result
    assert "retraining_gate" in result
    # Key data bounds
    assert result["under_extrusion"]["n_recordings"] > 30
    assert len(result["stringing_cluster_analysis"]["clusters"]) >= 5
