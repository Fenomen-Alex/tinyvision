"""Tests for tv/stage8.py (threshold & failure-type analysis)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from tv.stage8 import (
    ConfidenceCache,
    calibration_metrics,
    decision_from_threshold,
    evaluate_threshold_metrics,
    extract_hard_examples,
    failure_type_breakdown,
    recording_analysis,
    rank_recordings_by_error,
    reliability_analysis,
    select_best_threshold,
    threshold_sweep,
    load_cached_predictions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_recordings(n_frames: int = 100, n_failing: int = 50, fine: str = "under_extrusion"):
    """Build a synthetic single-recording dict matching the cache schema."""
    frames = []
    for i in range(n_frames):
        label = "failing" if i < n_failing else "normal"
        score = 0.8 if label == "failing" else 0.2
        frames.append({
            "name": f"frame_{i:04d}.png",
            "ts": 1000000 + i,
            "label": label,
            "score": float(score),
            "fine": fine if label == "failing" else "normal",
        })
    return {"Recording_000": {"frames": frames}}


def _make_mixed_recordings():
    """Build a synthetic test dataset with two recordings and varied scores."""
    recs = {}
    frames1 = []
    # Recording A: 30 failing, 20 normal — scores span 0.1–0.9
    for i in range(50):
        label = "failing" if i < 30 else "normal"
        score = 0.1 + 0.02 * i  # 0.1 to 1.0
        score = min(score, 1.0)
        fine = "under_extrusion" if label == "failing" else "normal"
        frames1.append({
            "name": f"frame_{i:04d}.png", "ts": 1000000 + i,
            "label": label, "score": float(score), "fine": fine,
        })
    recs["Recording_A"] = {"frames": frames1}

    frames2 = []
    # Recording B: 20 failing (stringing), 30 normal — scores skewed low
    for i in range(50):
        label = "failing" if i < 20 else "normal"
        score = 0.05 + 0.01 * i  # 0.05 to 0.69
        score = min(score, 1.0)
        fine = "stringing" if label == "failing" else "normal"
        frames2.append({
            "name": f"frame_{i:04d}.png", "ts": 2000000 + i,
            "label": label, "score": float(score), "fine": fine,
        })
    recs["Recording_B"] = {"frames": frames2}
    return recs


def _write_cache(cache_dir: Path, split: str, recordings: dict):
    import json
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"frozen_{split}_trajectories.json"
    path.write_text(json.dumps({"recordings": recordings}))


# ---------------------------------------------------------------------------
# Tests: decision_from_threshold
# ---------------------------------------------------------------------------

def test_decision_below_threshold():
    scores = np.array([0.1, 0.49, 0.499])
    pred = decision_from_threshold(scores, 0.5)
    assert (pred == 0).all()


def test_decision_above_threshold():
    scores = np.array([0.5, 0.6, 1.0])
    pred = decision_from_threshold(scores, 0.5)
    assert (pred == 1).all()


def test_decision_at_boundary():
    scores = np.array([0.3, 0.5, 0.7])
    pred = decision_from_threshold(scores, 0.5)
    np.testing.assert_array_equal(pred, [0, 1, 1])


# ---------------------------------------------------------------------------
# Tests: threshold_sweep + select_best_threshold
# ---------------------------------------------------------------------------

def test_threshold_sweep_cardinality():
    scores = np.array([0.2, 0.4, 0.6, 0.8])
    y_true = np.array([0, 0, 1, 1])
    results = threshold_sweep(y_true, scores, [0.3, 0.5, 0.7])
    assert len(results) == 3
    assert results[0].threshold == 0.3


def test_sweep_selects_best_f1():
    # At t=0.3: perfect recall, good precision on balanced data
    scores = np.array([0.1, 0.29, 0.31, 0.9])
    y_true = np.array([0, 0, 1, 1])
    results = threshold_sweep(y_true, scores, [0.30, 0.50, 0.70])
    best = select_best_threshold(results, metric="f1")
    assert best.threshold in [0.30, 0.50]


def test_select_best_f1_ties_higher_threshold():
    # Both thresholds give identical metrics; tie broken toward higher threshold
    scores = np.array([0.0, 0.0, 0.0, 1.0])
    y_true = np.array([0, 0, 1, 1])
    results = threshold_sweep(y_true, scores, [0.5, 0.8])
    best = select_best_threshold(results, metric="f1")
    assert best.threshold == 0.8


def test_sweep_monotonic_fnr():
    """FNR should be non-increasing as threshold drops."""
    rng = np.random.RandomState(42)
    scores = rng.rand(200)
    y_true = (rng.rand(200) > 0.5).astype(int)
    results = threshold_sweep(y_true, scores, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    fnrs = [r.fnr for r in results]
    # FNR should be non-decreasing as threshold increases
    for i in range(1, len(fnrs)):
        assert fnrs[i] >= fnrs[i - 1] - 0.01  # small tolerance for rounding


# ---------------------------------------------------------------------------
# Tests: ConfidenceCache.from_recordings
# ---------------------------------------------------------------------------

def test_confidence_cache_shapes():
    recs = _make_mixed_recordings()
    cache = ConfidenceCache.from_recordings(recs)
    assert cache.n_frames == 100
    assert len(cache.y_true) == 100
    assert len(cache.scores) == 100
    assert len(cache.recording) == 100
    assert len(cache.fine) == 100


def test_confidence_cache_ordering():
    recs = _make_mixed_recordings()
    cache = ConfidenceCache.from_recordings(recs)
    # First 50 should be Recording_A, next 50 Recording_B
    assert all(r == "Recording_A" for r in cache.recording[:50])
    assert all(r == "Recording_B" for r in cache.recording[50:])


def test_confidence_cache_labels():
    recs = _make_mixed_recordings()
    cache = ConfidenceCache.from_recordings(recs)
    # Recording A: 30 failing
    assert cache.y_true[:30].sum() == 30
    # Recording A: 20 normal
    assert cache.y_true[30:50].sum() == 0
    # Recording B: 20 failing
    assert cache.y_true[50:70].sum() == 20
    # Recording B: 30 normal
    assert cache.y_true[70:100].sum() == 0


# ---------------------------------------------------------------------------
# Tests: evaluate_threshold_metrics
# ---------------------------------------------------------------------------

def test_perfect_metrics():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    m = evaluate_threshold_metrics(y_true, scores, 0.5)
    assert m.accuracy == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.fnr == 0.0
    assert m.fpr == 0.0


def test_high_fnr_scenario():
    y_true = np.array([1, 1, 1, 1, 0, 0])
    scores = np.array([0.55, 0.55, 0.55, 0.55, 0.1, 0.1])
    m = evaluate_threshold_metrics(y_true, scores, 0.6)
    # All failing scores below threshold => all FN, no TP
    assert m.fnr == 1.0
    assert m.recall == 0.0


# ---------------------------------------------------------------------------
# Tests: failure_type_breakdown
# ---------------------------------------------------------------------------

def test_failure_type_breakdown_split():
    recs = _make_mixed_recordings()
    cache = ConfidenceCache.from_recordings(recs)
    breakdown = failure_type_breakdown(cache, threshold=0.5)
    assert "under_extrusion" in breakdown
    assert "stringing" in breakdown
    assert "normal" in breakdown
    # under_extrusion: 30 frames, all failing
    assert breakdown["under_extrusion"]["n_frames"] == 30
    # stringing: 20 frames, all failing
    assert breakdown["stringing"]["n_frames"] == 20
    # normal: 50 frames, all normal
    assert breakdown["normal"]["n_frames"] == 50


# ---------------------------------------------------------------------------
# Tests: recording_analysis
# ---------------------------------------------------------------------------

def test_recording_analysis_counts():
    recs = _make_mixed_recordings()
    cache = ConfidenceCache.from_recordings(recs)
    analysis = recording_analysis(cache, threshold=0.5)
    assert "Recording_A" in analysis
    assert "Recording_B" in analysis
    rec_a = analysis["Recording_A"]
    assert rec_a["n_frames"] == 50
    assert rec_a["n_normal_frames"] == 20
    assert rec_a["n_failing_frames"] == 30


def test_recording_analysis_no_negatives():
    """PURE_FAILING recording: fpr should be None."""
    recs = _make_recordings(n_frames=30, n_failing=30)
    cache = ConfidenceCache.from_recordings(recs)
    analysis = recording_analysis(cache, threshold=0.5)
    rec = analysis["Recording_000"]
    assert rec["n_normal_frames"] == 0
    assert rec["fpr"] is None


# ---------------------------------------------------------------------------
# Tests: rank_recordings_by_error
# ---------------------------------------------------------------------------

def test_rank_recordings_by_error():
    recs = _make_mixed_recordings()
    cache = ConfidenceCache.from_recordings(recs)
    analysis = recording_analysis(cache, threshold=0.5)
    rank = rank_recordings_by_error(analysis)
    assert "highest_fnr" in rank
    assert "highest_fpr" in rank
    assert len(rank["highest_fnr"]) == 2


# ---------------------------------------------------------------------------
# Tests: extract_hard_examples
# ---------------------------------------------------------------------------

def test_extract_hard_examples_cardinality():
    recs = _make_mixed_recordings()
    cache = ConfidenceCache.from_recordings(recs)
    examples = extract_hard_examples(cache, selected_threshold=0.5, top_n=5)
    assert "false_negatives" in examples
    assert "false_positives" in examples
    assert "borderline_errors" in examples
    # top_n caps the count per category
    assert len(examples["false_negatives"]) <= 5
    assert len(examples["false_positives"]) <= 5


def test_extract_hard_examples_captured_fields():
    recs = _make_mixed_recordings()
    cache = ConfidenceCache.from_recordings(recs)
    examples = extract_hard_examples(cache, selected_threshold=0.5, top_n=3)
    for ex in examples["false_positives"]:
        assert "recording" in ex
        assert "frame_index" in ex
        assert "probability" in ex
        assert "source_class" in ex
        assert "failure_type" in ex


# ---------------------------------------------------------------------------
# Tests: calibration_metrics
# ---------------------------------------------------------------------------

def test_calibration_perfect():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.0, 0.0, 1.0, 1.0])
    cal = calibration_metrics(y_true, scores, n_bins=4)
    assert cal["brier_score"] == 0.0
    assert cal["ece"] == 0.0


def test_calibration_worst_case():
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.0, 0.0, 1.0, 1.0])  # all wrong
    cal = calibration_metrics(y_true, scores, n_bins=4)
    assert cal["brier_score"] == 1.0


def test_calibration_brier_range():
    rng = np.random.RandomState(42)
    scores = rng.rand(500)
    y_true = (rng.rand(500) > 0.5).astype(int)
    cal = calibration_metrics(y_true, scores, n_bins=10)
    assert 0.0 <= cal["brier_score"] <= 1.0
    assert 0.0 <= cal["ece"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: reliability_analysis
# ---------------------------------------------------------------------------

def test_reliability_systematic_biases():
    # Synthetic: all bins show mean_conf > observed_freq => overconfident
    cal = {
        "bins": [
            {"n_samples": 100, "mean_confidence": 0.7, "observed_frequency": 0.4, "gap": 0.3},
            {"n_samples": 100, "mean_confidence": 0.8, "observed_frequency": 0.3, "gap": 0.5},
            {"n_samples": 50,  "mean_confidence": 0.5, "observed_frequency": 0.48, "gap": 0.02},
        ]
    }
    reliab = reliability_analysis(cal, reliability_floor=0.8)
    assert reliab["systematic_biases"]["overconfident_failing"] is True
    assert reliab["systematic_biases"]["underconfident_failing"] is False


def test_reliability_empty_bins():
    cal = {"bins": [{"n_samples": 0, "bin": "0.0-0.1"}]}
    reliab = reliability_analysis(cal)
    assert reliab["systematic_biases"]["overconfident_failing"] is False
    assert reliab["systematic_biases"]["underconfident_failing"] is False


# ---------------------------------------------------------------------------
# Tests: load_cached_predictions (integration with real cache if present)
# ---------------------------------------------------------------------------

def test_load_cached_predictions():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        recs = _make_recordings()
        _write_cache(cache_dir, "test", recs)
        loaded = load_cached_predictions(cache_dir, "test")
        assert "Recording_000" in loaded
        assert len(loaded["Recording_000"]["frames"]) == 100


# ---------------------------------------------------------------------------
# Tests: end-to-end sweep consistency
# ---------------------------------------------------------------------------

def test_full_sweep_end_to_end():
    """Run a full sweep and verify selected threshold is valid."""
    recs = _make_mixed_recordings()
    cache = ConfidenceCache.from_recordings(recs)
    results = threshold_sweep(cache.y_true, cache.scores)
    best = select_best_threshold(results)
    # Verify the best metrics are self-consistent
    y_pred = decision_from_threshold(cache.scores, best.threshold)
    n_tp = int(((y_pred == 1) & (cache.y_true == 1)).sum())
    n_fp = int(((y_pred == 1) & (cache.y_true == 0)).sum())
    n_fn = int(((y_pred == 0) & (cache.y_true == 1)).sum())
    n_tn = int(((y_pred == 0) & (cache.y_true == 0)).sum())
    assert n_tp + n_fn == int(cache.y_true.sum())
    assert n_fp + n_tn == int((cache.y_true == 0).sum())
    # Recompute F1 from raw counts
    if n_tp + n_fp > 0:
        prec = n_tp / (n_tp + n_fp)
    else:
        prec = 0.0
    if n_tp + n_fn > 0:
        rec = n_tp / (n_tp + n_fn)
    else:
        rec = 0.0
    if prec + rec > 0:
        f1 = 2 * prec * rec / (prec + rec)
    else:
        f1 = 0.0
    assert abs(f1 - best.f1) < 0.01
