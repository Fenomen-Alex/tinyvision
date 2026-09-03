"""Tests for the Stage 3 temporal-evaluation harness (synthetic data, no model)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tv.temporaleval import (
    _first_event_frame,
    _proxy_onset_index,
    _raw_decision,
    evaluate_recording,
    evaluate_split,
)
from tv.temporal import FAILING_EVENT


def _mk_result(recordings):
    return {"split": "test", "checkpoint": "x", "recordings": recordings}


def _rec(label, scores, fine=None):
    frames = []
    for i, s in enumerate(scores):
        frames.append(
            {"name": f"f_{i:03d}.png", "ts": i, "label": label, "score": s, "fine": (fine[i] if fine else "good")}
        )
    return {"label": label, "frames": frames}


def test_proxy_onset_index_clean_transition():
    rec = _rec("failing", [0.1] * 5 + [0.9] * 5, fine=["good"] * 5 + ["under_extrusion"] * 5)
    assert _proxy_onset_index(rec) == 5


def test_proxy_onset_index_none_when_constant():
    rec = _rec("failing", [0.9] * 8, fine=["stringing"] * 8)
    assert _proxy_onset_index(rec) is None


def test_raw_decision_threshold():
    d = _raw_decision([0.1, 0.9, 0.7], threshold=0.5)
    assert [x["state"] for x in d] == ["NORMAL", "FAILING", "FAILING"]
    assert _first_event_frame(d) == 1


def test_evaluate_recording_failing_detected():
    rec = _rec("failing", [0.1] * 10 + [0.95] * 10)
    r = evaluate_recording(rec, {"confirm": 3, "warmup": 5}, "fsm")
    assert r["truth"] == 1
    assert r["seq_detected"] is True
    assert r["fnr"] is not None and 0 <= r["fnr"] < 1.0


def test_evaluate_recording_normal_no_alarm():
    rec = _rec("normal", [0.1] * 20)
    r = evaluate_recording(rec, {"confirm": 3, "warmup": 5}, "fsm")
    assert r["truth"] == 0
    assert r["failing_frames"] == 0
    assert r["fpr"] == 0.0


def test_evaluate_split_detection_and_false_alarm():
    result = _mk_result(
        {
            "R1": _rec("failing", [0.1] * 10 + [0.95] * 10),
            "R2": _rec("failing", [0.1] * 20),
            "N1": _rec("normal", [0.1] * 20),
        }
    )
    s = evaluate_split(result, {"confirm": 3, "warmup": 5}, "fsm")
    assert s["n_failing_seq"] == 2
    assert s["n_detected_seq"] == 1
    assert s["sequence_detection_rate"] == 0.5
    assert s["missed_failure_events"] == 1
    assert s["n_false_trigger_seq"] == 0
    assert s["sequence_false_alarm_rate"] == 0.0


def test_evaluate_split_missed_catastrophic_failure():
    # a failing recording where the model outputs ~0 throughout is a miss that
    # temporal logic cannot recover -- must be reported as such.
    result = _mk_result(
        {"R1": _rec("failing", [0.1] * 30, fine=["good"] * 30)}
    )
    s = evaluate_split(result, {"confirm": 3, "warmup": 5}, "fsm")
    assert s["missed_failure_events"] == 1
    assert s["sequence_detection_rate"] == 0.0
