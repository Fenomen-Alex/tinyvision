"""Tests for metrics computation."""
from __future__ import annotations

import numpy as np
import pytest

from tv.metrics import compute_classification_metrics


def test_perfect_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    m = compute_classification_metrics(y_true, y_pred)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.fnr == 0.0
    assert m.fpr == 0.0
    assert m.accuracy == 1.0


def test_all_wrong() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    m = compute_classification_metrics(y_true, y_pred)
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0
    assert m.fnr == 1.0
    assert m.fpr == 1.0


def test_fnr_fpr_calculation() -> None:
    # TN=2, FP=1, FN=1, TP=2
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 0])
    m = compute_classification_metrics(y_true, y_pred)
    assert m.fnr == pytest.approx(1 / 3)
    assert m.fpr == pytest.approx(0.0)  # FP=0


def test_auroc_with_scores() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_classification_metrics(y_true, y_true, y_scores)
    assert m.auroc is not None
    assert m.auroc == pytest.approx(1.0)


def test_auroc_without_scores() -> None:
    y_true = np.array([0, 0, 1, 1])
    m = compute_classification_metrics(y_true, y_true)
    assert m.auroc is None


def test_single_class() -> None:
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    m = compute_classification_metrics(y_true, y_pred)
    assert m.precision == 0.0
    assert m.sample_count == 3
    assert m.auroc is None  # single class -> AUROC undefined -> None


def test_multiclass_does_not_crash() -> None:
    """Three+ classes must not raise (regression: binary-only sklearn misuse)."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])  # perfect
    m = compute_classification_metrics(y_true, y_pred)
    assert m.accuracy == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.sample_count == 6
    assert len(m.classes) == 3
    assert m.confusion_matrix == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_multiclass_wrong_predictions() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([1, 2, 0, 1, 2, 0])  # all wrong
    m = compute_classification_metrics(y_true, y_pred)
    assert m.accuracy == 0.0
    # macro precision: each class precision is 0
    assert m.precision == 0.0
    assert m.recall == 0.0


def test_multiclass_auroc_with_perclass_scores() -> None:
    y_true = np.array([0, 1, 2])
    scores = np.eye(3)  # perfect per-class probabilities
    m = compute_classification_metrics(y_true, y_true, scores)
    assert m.auroc is not None
    assert m.auroc == pytest.approx(1.0)


def test_to_dict_and_summary() -> None:
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    m = compute_classification_metrics(y_true, y_pred)
    d = m.to_dict()
    assert "precision" in d
    assert "confusion_matrix" in d
    s = m.summary()
    assert "Precision" in s
