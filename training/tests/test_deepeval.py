"""Tests for deep-eval error-analysis logic (failing-as-positive convention)."""
from __future__ import annotations

import numpy as np
import pytest

from tv.deepeval import _binary_metrics, _threshold_sweep


def test_failing_as_positive_metrics() -> None:
    # failing=0, normal=1. Samples: 2 failing (both correct), 2 normal (1 FP).
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1])  # one normal misclassified as failing
    m = _binary_metrics(y_true, y_pred, pos_integer=0)
    # TP=2 (failing correct), FN=0, FP=1, TN=1
    assert m["tp"] == 2
    assert m["fn"] == 0
    assert m["fp"] == 1
    assert m["tn"] == 1
    assert m["recall"] == 1.0  # all failures caught
    assert m["precision"] == pytest.approx(0.6667)
    assert m["f1"] == pytest.approx(0.8)
    assert m["fnr"] == 0.0
    assert m["fpr"] == 0.5
    assert m["accuracy"] == 0.75


def test_failing_recall_reflects_misses() -> None:
    # Half of failures missed -> failing recall 0.5, FNR 0.5.
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 1])
    m = _binary_metrics(y_true, y_pred, pos_integer=0)
    assert m["recall"] == pytest.approx(0.5)  # 1 of 2 failures caught
    assert m["fnr"] == pytest.approx(0.5)
    assert m["fn"] == 1


def test_normal_positive_is_complement() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 0])
    mf = _binary_metrics(y_true, y_pred, pos_integer=0)  # failing positive
    mn = _binary_metrics(y_true, y_pred, pos_integer=1)  # normal positive
    # failing positive: TP=2 (both failures caught) FN=0 FP=1 TN=1
    # normal  positive: TP=1 FN=1 FP=0 TN=2
    assert mf["tp"] == 2 and mf["tn"] == 1
    assert mf["fp"] == 1 and mf["fn"] == 0
    assert mn["tp"] == 1 and mn["tn"] == 2
    assert mn["fp"] == 0 and mn["fn"] == 1
    # The two views report the same accuracy and sample count.
    assert mf["accuracy"] == mn["accuracy"] == 0.75


def test_threshold_sweep_interprets_score_as_p_failing() -> None:
    # failing=0. faile_scores high => predict failing.
    y_true = np.array([0, 0, 1, 1])
    fail_scores = np.array([0.95, 0.85, 0.20, 0.05])
    res = _threshold_sweep(y_true, fail_scores, {0: "failing", 1: "normal"}, fail_idx=0)
    # At t=0.1: failing predicted for scores >= 0.1 -> samples 0,1,2 = failing,
    # sample 3 = normal. So TP=2 FN=0 FP=1 TN=1.
    t01 = next(r for r in res if r["threshold_p_failing"] == 0.1)
    assert t01["tp"] == 2
    assert t01["fp"] == 1
    assert t01["recall"] == 1.0
    # At t=0.9: only sample 0 predicted failing -> TP=1 FN=1, FP=0 TN=2.
    t09 = next(r for r in res if r["threshold_p_failing"] == 0.9)
    assert t09["tp"] == 1
    assert t09["fn"] == 1
    assert t09["fp"] == 0


def test_threshold_sweep_exhaustive_thresholds() -> None:
    y_true = np.array([0, 1])
    fail_scores = np.array([0.5, 0.5])
    res = _threshold_sweep(y_true, fail_scores, {0: "failing", 1: "normal"}, fail_idx=0)
    # one entry per threshold 0.1..0.9
    assert [r["threshold_p_failing"] for r in res] == [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    ]
