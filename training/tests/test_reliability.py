"""Tests for the reliability evaluation + release-gate framework.

Deterministic and fast: synthetic fixtures only, no dataset downloads, no
training. Includes a regression test for the reported failure scenario where
every normal sample is predicted failing (FPR must be 1.0 and the gate FAIL).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tv.reliability import (
    DEFAULT_POSITIVE_CLASS,
    STATUS_FAIL,
    STATUS_INCONCLUSIVE,
    STATUS_PASS,
    MissingDataError,
    Suite,
    Surface,
    compare_formats,
    compute_binary_metrics,
    evaluate_gates,
    check_source_leakage,
    load_samples,
    load_suite,
    overall_exit_code,
    predictions_from_scores,
    select_threshold,
    wilson_interval,
    Sample,
)

from tv.reliability import format_summary


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_fpr_definition_all_normal_predicted_failing():
    """Reported scenario: every normal frame predicted failing => FPR == 1.0."""
    y_true = np.array([1, 1, 1, 1])  # all normal
    y_pred = np.array([0, 0, 0, 0])  # all predicted failing
    m = compute_binary_metrics(y_true, y_pred, positive=0)
    assert m["fpr"] == 1.0
    assert m["fp"] == 4 and m["tn"] == 0 and m["tp"] == 0 and m["fn"] == 0
    # No failing samples in truth => FNR/recall undefined, never 0.0.
    assert m["fnr"] is None
    assert m["recall"] is None
    assert "fnr" in m["undefined"] and "recall" in m["undefined"]


def test_fpr_undefined_without_normal_samples():
    y_true = np.array([0, 0, 0])  # all failing
    y_pred = np.array([0, 1, 0])
    m = compute_binary_metrics(y_true, y_pred, positive=0)
    assert m["fpr"] is None
    assert "fpr" in m["undefined"]
    assert m["recall"] == pytest.approx(2 / 3)


def test_confusion_and_rates_known_example():
    # truth: 2 failing, 3 normal; preds: failing, normal, normal, failing, normal
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    m = compute_binary_metrics(y_true, y_pred, positive=0)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 2)
    assert m["fpr"] == pytest.approx(1 / 3)
    assert m["fnr"] == pytest.approx(1 / 2)
    assert m["precision"] == pytest.approx(1 / 2)
    assert m["recall"] == pytest.approx(1 / 2)
    assert m["accuracy"] == pytest.approx(3 / 5)


def test_predictions_threshold_direction():
    scores = np.array([0.49, 0.5, 0.51])
    preds = predictions_from_scores(scores, 0.5, positive=0)
    assert list(preds) == [1, 0, 0]  # >= threshold means failing


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(10, 10)
    assert 0.0 <= lo <= 1.0 and hi == 1.0
    assert wilson_interval(0, 0) is None


def test_auroc_only_when_both_classes_present():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_binary_metrics(y, predictions_from_scores(s, 0.5), s, positive=0)
    assert m["auroc"] == 1.0
    y_single = np.array([1, 1])
    m2 = compute_binary_metrics(y_single, np.array([1, 1]), np.array([0.1, 0.2]), positive=0)
    assert m2["auroc"] is None


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def test_leakage_detects_duplicate_image_and_source_overlap():
    a = [Sample("x.jpg", 1, source_id="s1", surface_id="dev")]
    b = [Sample("x.jpg", 1, source_id="s1", surface_id="hold"),
         Sample("y.jpg", 1, source_id="s2", surface_id="hold")]
    leaks = check_source_leakage({"dev": a, "hold": b},
                                 {"dev": "dev_validation", "hold": "normal_holdout"})
    codes = {leak.code for leak in leaks}
    assert "duplicate_image" in codes
    assert "source_group_leak" in codes


def test_leakage_clean_when_disjoint():
    a = [Sample("x.jpg", 1, source_id="s1", surface_id="dev")]
    b = [Sample("y.jpg", 1, source_id="s2", surface_id="hold")]
    assert check_source_leakage({"dev": a, "hold": b},
                                {"dev": "dev_validation", "hold": "normal_holdout"}) == []


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def test_select_threshold_uses_validation_only_and_respects_fpr():
    # At t=0.5 a normal sample (0.6) is flagged => FPR>0, violating max_fpr=0.
    scores = np.array([0.9, 0.8, 0.6, 0.2])  # first two failing
    y = np.array([0, 0, 1, 1])
    sel = select_threshold(y, scores, objective="max_recall_subject_to_max_fpr",
                           max_fpr=0.0, grid=[0.5, 0.7])
    assert sel["status"] == STATUS_PASS
    assert sel["threshold"] == 0.7
    assert sel["validation_metrics"]["recall"] == 1.0


def test_select_threshold_respects_min_samples():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    y = np.array([0, 0, 1, 1])
    sel = select_threshold(y, scores, objective="max_recall_subject_to_max_fpr",
                           max_fpr=0.0, min_samples=100)
    assert sel["status"] == STATUS_INCONCLUSIVE
    assert sel["threshold"] is None


def test_select_threshold_inconclusive_without_negatives():
    scores = np.array([0.9, 0.8])
    y = np.array([0, 0])
    sel = select_threshold(y, scores, objective="max_recall_subject_to_max_fpr", max_fpr=0.1)
    assert sel["status"] == STATUS_INCONCLUSIVE
    assert sel["threshold"] is None


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def _surface_result(fpr=None, fnr=None, n=200, n_sources=3, n_env=3,
                    recall=None, precision=None):
    return {
        "metrics": {"fpr": fpr, "fnr": fnr, "recall": recall,
                    "precision": precision, "n": n},
        "n_samples": n, "n_sources": n_sources, "n_environments": n_env,
    }


def test_gate_fails_when_fpr_exceeds_limit():
    policy = {"gates": {"normal_holdout": {"required": True, "max_fpr": 0.1,
                                           "min_samples": 100, "min_sources": 2}}}
    res = evaluate_gates({"normal_holdout": _surface_result(fpr=1.0, n=120, n_sources=2)}, policy)
    assert res["overall_status"] == STATUS_FAIL
    assert res["blocks_release"] is True


def test_gate_inconclusive_when_surface_missing():
    policy = {"gates": {"normal_holdout": {"required": True, "max_fpr": 0.1,
                                           "min_samples": 100, "min_sources": 2}}}
    res = evaluate_gates({}, policy)
    assert res["overall_status"] == STATUS_INCONCLUSIVE


def test_gate_inconclusive_when_samples_insufficient():
    policy = {"gates": {"normal_holdout": {"required": True, "max_fpr": 0.5,
                                           "min_samples": 100, "min_sources": 2}}}
    res = evaluate_gates({"normal_holdout": _surface_result(fpr=0.0, n=5, n_sources=2)}, policy)
    assert res["overall_status"] == STATUS_INCONCLUSIVE


def test_gate_inconclusive_when_metric_undefined():
    policy = {"gates": {"normal_holdout": {"required": True, "max_fpr": 0.5,
                                           "min_samples": 10, "min_sources": 1}}}
    res = evaluate_gates({"normal_holdout": _surface_result(fpr=None, n=20, n_sources=1)}, policy)
    assert res["overall_status"] == STATUS_INCONCLUSIVE


def test_gate_pass_and_optional_warn():
    policy = {"gates": {
        "normal_holdout": {"required": True, "max_fpr": 0.1, "min_samples": 100, "min_sources": 2},
        "failure_holdout": {"required": True, "max_fnr": 0.2, "min_recall": 0.8,
                            "min_samples": 100, "min_sources": 2},
        "deployment": {"required": False, "max_model_size_mb": 1},
    }}
    surfaces = {
        "normal_holdout": _surface_result(fpr=0.05, n=200, n_sources=3),
        "failure_holdout": _surface_result(fnr=0.1, recall=0.9, n=200, n_sources=3),
    }
    res = evaluate_gates(surfaces, policy, extra={"deployment": {"model_size_mb": 5.0}})
    assert res["overall_status"] == "WARN"  # only optional deployment gate failed
    assert res["blocks_release"] is False


def test_parity_gate_inconclusive_without_onnx():
    policy = {"gates": {"parity": {"required": True, "max_onnx_score_abs_drift": 0.05}}}
    res = evaluate_gates({}, policy, extra={"parity": None})
    assert res["overall_status"] == STATUS_INCONCLUSIVE


def test_parity_gate_fails_on_large_drift():
    policy = {"gates": {"parity": {"required": True, "max_onnx_score_abs_drift": 0.05}}}
    res = evaluate_gates({}, policy, extra={"parity": {"onnx": {
        "prediction_disagreement": 0.0, "max_score_abs_drift": 0.2}}})
    assert res["overall_status"] == STATUS_FAIL


# ---------------------------------------------------------------------------
# Suite loading
# ---------------------------------------------------------------------------

def test_load_samples_missing_data_raises(tmp_path):
    surface = Surface(id="s", role="normal_holdout", format="csv",
                      path="does-not-exist.csv", label_map={"normal": 1})
    with pytest.raises(MissingDataError):
        load_samples(surface, tmp_path)


def test_load_samples_csv_maps_labels_and_groups(tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text("path,label,source\nimg1.jpg,normal,a\nimg2.jpg,failing,b\n")
    surface = Surface(id="s", role="dev_validation", format="csv",
                      path=str(csv), label_map={"normal": 1, "failing": 0},
                      source_key="source")
    samples = load_samples(surface, None)
    assert [s.label for s in samples] == [1, 0]
    assert samples[0].source_id == "a"


def test_load_suite_rejects_duplicate_ids(tmp_path):
    suite = tmp_path / "s.yaml"
    suite.write_text(
        "name: t\nversion: 1\nsurfaces:\n"
        "  - {id: a, role: normal_holdout, format: csv, path: x.csv}\n"
        "  - {id: a, role: normal_holdout, format: csv, path: y.csv}\n"
    )
    with pytest.raises(ValueError):
        load_suite(suite)


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------

def test_compare_formats_identical_scores():
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    y = np.array([0, 1, 0, 1])
    block = compare_formats(scores, scores.copy(), y, 0.5)
    assert block["prediction_disagreement"] == 0.0
    assert block["max_score_abs_drift"] == 0.0


# ---------------------------------------------------------------------------
# End-to-end (synthetic, framework only)
# ---------------------------------------------------------------------------

def _make_checkpoint(path: Path):
    import torch
    from tv.models.factory import build_classifier

    torch.manual_seed(0)
    model = build_classifier("shufflenet_v2_x1_0", 2, pretrained=False)
    torch.save({"model_state_dict": model.state_dict(), "num_classes": 2,
                "config_model": "shufflenet_v2_x1_0"}, path)


def _make_image(path: Path, color):
    from PIL import Image

    Image.new("RGB", (64, 64), color).save(path)


def test_end_to_end_evaluate_synthetic(tmp_path):
    from tv.reliability import evaluate

    # 2 normal, 2 failing synthetic images
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    rows = []
    for i, (lbl, color) in enumerate([("normal", (10, 200, 10)), ("normal", (12, 190, 20)),
                                      ("failing", (200, 10, 10)), ("failing", (190, 20, 12))]):
        p = imgs / f"{i}.png"
        _make_image(p, color)
        rows.append(f"{p},{lbl},src{i // 2}")
    csv = tmp_path / "samples.csv"
    csv.write_text("path,label,source\n" + "\n".join(rows) + "\n")

    suite_yaml = tmp_path / "suite.yaml"
    suite_yaml.write_text(
        "name: synthetic\nversion: 1\npositive_class: 0\nsurfaces:\n"
        "  - id: dev\n    role: dev_validation\n    format: csv\n"
        f"    path: {csv}\n"
        "    label_map: {normal: 1, failing: 0}\n"
        "    source_key: source\n"
        "  - id: nhold\n    role: normal_holdout\n    format: csv\n"
        f"    path: {csv}\n"
        "    filter: {field: label, equals: normal}\n"
        "    label_map: {normal: 1, failing: 0}\n"
    )
    ckpt = tmp_path / "best.pt"
    _make_checkpoint(ckpt)

    suite = load_suite(suite_yaml)
    report = evaluate(suite, {"m": {"framework": str(ckpt)}},
                      policy={"id": "t", "gates": {"normal_holdout": {
                          "required": True, "max_fpr": 0.5, "min_samples": 2}}},
                      thresholds=[0.5], output_dir=tmp_path / "out")
    model = report["models"]["m"]
    assert "surfaces" in model and "gates" in model
    assert (tmp_path / "out" / "summary.json").exists()
    assert (tmp_path / "out" / "summary.txt").exists()
    assert report["surfaces"][0]["n_loaded"] == 4
    # Suite accepts missing-data surface without crashing the whole run.
    assert isinstance(report["leakage"], list)


def test_exit_code_reflects_gates():
    assert overall_exit_code({"models": {"m": {"gates": {"overall_status": STATUS_FAIL}}}}) == 2
    assert overall_exit_code({"models": {"m": {"gates": {"overall_status": STATUS_INCONCLUSIVE}}}}) == 3
    assert overall_exit_code({"models": {"m": {"gates": {"overall_status": STATUS_PASS}}}}) == 0


def test_exit_code_inconclusive_without_any_gates():
    # No acceptance policy applied => no criteria => cannot claim a pass.
    assert overall_exit_code({"models": {"m": {"gates": None}}}) == 3
    assert overall_exit_code({"models": {}}) == 3


def test_exit_code_leakage_blocks_release():
    # Leakage invalidates holdout claims: block even if every gate passed.
    report = {"models": {"m": {"gates": {"overall_status": STATUS_PASS}}},
              "leakage": [{"code": "source_group_leak", "message": "x"}]}
    assert overall_exit_code(report) == 2


def test_end_to_end_missing_required_surface_inconclusive(tmp_path):
    from tv.reliability import evaluate

    imgs = tmp_path / "imgs"
    imgs.mkdir()
    rows = []
    for i, (lbl, color) in enumerate([("normal", (10, 200, 10)), ("failing", (200, 10, 10))]):
        p = imgs / f"{i}.png"
        _make_image(p, color)
        rows.append(f"{p},{lbl},src{i}")
    csv = tmp_path / "samples.csv"
    csv.write_text("path,label,source\n" + "\n".join(rows) + "\n")

    suite_yaml = tmp_path / "suite.yaml"
    suite_yaml.write_text(
        "name: missing\nversion: 1\npositive_class: 0\nsurfaces:\n"
        "  - id: dev\n    role: dev_validation\n    format: csv\n"
        f"    path: {csv}\n"
        "    label_map: {normal: 1, failing: 0}\n"
        "  - id: nhold\n    role: normal_holdout\n    format: csv\n"
        f"    path: {tmp_path / 'absent.csv'}\n"
        "    label_map: {normal: 1, failing: 0}\n"
    )
    ckpt = tmp_path / "best.pt"
    _make_checkpoint(ckpt)

    suite = load_suite(suite_yaml)
    report = evaluate(suite, {"m": {"framework": str(ckpt)}},
                      policy={"id": "t", "gates": {
                          "normal_holdout": {"required": True, "min_samples": 1}}},
                      thresholds=[0.5])
    assert report["surfaces"][1]["error"] is not None
    gates = report["models"]["m"]["gates"]
    assert gates["overall_status"] == STATUS_INCONCLUSIVE
    assert gates["blocks_release"] is True
    assert overall_exit_code(report) == 3


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------

def test_format_summary_parity_line_shows_scope():
    report = {
        "suite": {"name": "s"},
        "models": {"rc1": {
            "parity": {"int8": {
                "surface": "dev_val", "n_samples": 512,
                "prediction_disagreement": 0.013672,
                "max_score_abs_drift": 0.583879,
            }},
        }},
    }
    text = format_summary(report)
    assert "parity[int8] on dev_val n=512 disagreement=0.013672" in text
    assert "max|dscore|=0.583879" in text


def test_format_summary_parity_line_tolerates_missing_scope():
    report = {
        "suite": {"name": "s"},
        "models": {"rc1": {
            "parity": {"onnx": {
                "prediction_disagreement": 0.0, "max_score_abs_drift": 4e-06,
            }},
        }},
    }
    text = format_summary(report)
    assert "parity[onnx] disagreement=0.0" in text
