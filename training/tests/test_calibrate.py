"""Tests for tv/calibrate.py (per-printer threshold calibration)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training"))

from tv.calibrate import (  # noqa: E402
    CalibrationClip,
    CalibrationError,
    calibrate,
    format_summary,
    load_clip_spec,
)


def _clip(cid, label, split, base, n=20, step=0.5, plateau_at=None):
    """Synthetic precomputed scores for one clip.

    ``plateau_at`` raises the score to ``base`` only from that frame index on
    (used to exercise detection latency); before it the score is low.
    """
    scores = []
    for j in range(n):
        if plateau_at is None:
            s = base
        else:
            s = base if j >= plateau_at else 0.02
        scores.append({"timestamp_s": round(j * step, 6), "score": s})
    return CalibrationClip(id=cid, label=label, source="synthetic", split=split,
                           scores=scores)


def _spec(*clips):
    return {"printer": "printer-A", "camera": "cam-0", "checkpoint": None,
            "clips": list(clips)}


# ---------------------------------------------------------------------------
# Spec loading / validation
# ---------------------------------------------------------------------------

def test_load_clip_spec_json_roundtrip(tmp_path):
    spec = {
        "printer": "printer-A", "camera": "cam-0",
        "clips": [
            {"id": "n1", "label": "normal", "source": "n1.mp4"},
            {"id": "f1", "label": "failing", "source": "f1.mp4", "split": "holdout"},
        ],
    }
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    loaded = load_clip_spec(p)
    assert loaded["printer"] == "printer-A"
    assert [c.id for c in loaded["clips"]] == ["n1", "f1"]
    assert loaded["clips"][0].split == "calibration"
    assert loaded["clips"][1].split == "holdout"


def test_load_clip_spec_yaml(tmp_path):
    p = tmp_path / "spec.yaml"
    p.write_text(
        "printer: P\nclips:\n"
        "  - {id: n1, label: normal, source: n1.mp4}\n"
        "  - {id: f1, label: failing, source: f1.mp4}\n"
    )
    loaded = load_clip_spec(p)
    assert len(loaded["clips"]) == 2


def test_load_clip_spec_missing_file():
    with pytest.raises(CalibrationError):
        load_clip_spec("/no/such/spec.yaml")


def test_load_clip_spec_rejects_locked_test_split(tmp_path):
    """A locked test split must never be usable for calibration."""
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"clips": [
        {"id": "x", "label": "normal", "source": "x.mp4", "split": "test"},
    ]}))
    with pytest.raises(CalibrationError, match="split"):
        load_clip_spec(p)


def test_load_clip_spec_rejects_bad_label(tmp_path):
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"clips": [
        {"id": "x", "label": "maybe", "source": "x.mp4"},
    ]}))
    with pytest.raises(CalibrationError, match="label"):
        load_clip_spec(p)


def test_load_clip_spec_rejects_missing_source_and_scores(tmp_path):
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"clips": [{"id": "x", "label": "normal"}]}))
    with pytest.raises(CalibrationError, match="source"):
        load_clip_spec(p)


def test_load_clip_spec_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"clips": [
        {"id": "d", "label": "normal", "source": "a.mp4"},
        {"id": "d", "label": "failing", "source": "b.mp4"},
    ]}))
    with pytest.raises(CalibrationError, match="duplicate"):
        load_clip_spec(p)


# ---------------------------------------------------------------------------
# Selection and metrics
# ---------------------------------------------------------------------------

def test_calibrate_selects_threshold_and_reports_metrics():
    spec = _spec(
        _clip("n1", "normal", "calibration", 0.05),
        _clip("n2", "normal", "calibration", 0.10),
        _clip("n3", "normal", "calibration", 0.08),
        _clip("f1", "failing", "calibration", 0.80),
        _clip("f2", "failing", "calibration", 0.90),
        _clip("f3", "failing", "calibration", 0.60),
        _clip("hn", "normal", "holdout", 0.06),
        _clip("hf", "failing", "holdout", 0.85),
    )
    report = calibrate(spec, thresholds=[i / 100 for i in range(1, 100)])
    assert report["status"] == "ok"
    assert report["selection"]["selected_threshold"] is not None
    cal = report["calibration"]
    assert cal["n_normal"] == 3 and cal["n_failing"] == 3
    assert cal["detection_rate"] == 1.0
    assert cal["false_alarm_rate_per_print"] == 0.0
    # Wilson intervals present where sample size permits
    assert cal["detection_rate_ci"] is not None
    assert cal["false_alarm_rate_ci"] is not None
    # holdout reported separately
    assert report["holdout"]["n_normal"] == 1


def test_holdout_never_influences_selection():
    base = [
        _clip("n1", "normal", "calibration", 0.05),
        _clip("n2", "normal", "calibration", 0.15),
        _clip("f1", "failing", "calibration", 0.70),
        _clip("f2", "failing", "calibration", 0.85),
    ]
    r1 = calibrate(_spec(*base, _clip("hn", "normal", "holdout", 0.02)),
                   thresholds=[i / 100 for i in range(1, 100)])
    # drastically change the holdout; the selected threshold must not move
    r2 = calibrate(_spec(*base, _clip("hn", "normal", "holdout", 0.95)),
                   thresholds=[i / 100 for i in range(1, 100)])
    assert r1["selection"]["selected_threshold"] == r2["selection"]["selected_threshold"]


def test_single_class_calibration_is_insufficient():
    spec = _spec(
        _clip("f1", "failing", "calibration", 0.9),
        _clip("f2", "failing", "calibration", 0.8),
    )
    report = calibrate(spec, thresholds=[i / 100 for i in range(1, 100)])
    assert report["status"] == "insufficient_data"
    assert report["selection"]["selected_threshold"] is None
    assert report["calibration"] is None
    assert any("single class" in w for w in report["warnings"])


def test_no_threshold_meets_fpr_ceiling_is_flagged():
    # Every normal clip already scores as high as the failing clip, so no
    # threshold in the grid can separate them -> FPR cannot reach 0.
    spec = _spec(
        _clip("n1", "normal", "calibration", 0.98),
        _clip("n2", "normal", "calibration", 0.99),
        _clip("f1", "failing", "calibration", 0.99),
    )
    report = calibrate(spec, thresholds=[i / 100 for i in range(1, 100)], fpr_ceiling=0.0)
    assert report["status"] == "no_threshold_meets_fpr_ceiling"
    assert report["selection"]["selected_threshold"] is not None  # conservative point
    assert any("FPR/print" in w for w in report["warnings"])


def test_missing_holdout_warning():
    spec = _spec(
        _clip("n1", "normal", "calibration", 0.05),
        _clip("n2", "normal", "calibration", 0.06),
        _clip("f1", "failing", "calibration", 0.9),
        _clip("f2", "failing", "calibration", 0.85),
    )
    report = calibrate(spec, thresholds=[i / 100 for i in range(1, 100)])
    assert report["holdout"] is None
    assert any("no holdout" in w for w in report["warnings"])


def test_detection_latency_reflects_first_sustained_frame():
    # failing clip rises to high only from frame index 6 (t=3.0s at 0.5s steps)
    spec = _spec(
        _clip("n1", "normal", "calibration", 0.02),
        _clip("n2", "normal", "calibration", 0.03),
        _clip("f1", "failing", "calibration", 0.9, plateau_at=6),
        _clip("f2", "failing", "calibration", 0.9, plateau_at=4),
    )
    report = calibrate(spec, thresholds=[i / 100 for i in range(1, 100)], confirm=1)
    lat = report["calibration"]["detection_latency_s"]
    assert lat is not None
    assert lat["min_s"] == pytest.approx(2.0)  # frame 4 * 0.5s
    assert lat["max_s"] == pytest.approx(3.0)  # frame 6 * 0.5s


def test_confirm_requires_consecutive_frames():
    # a single high frame must not alarm when confirm=2
    clip = CalibrationClip(
        id="npulse", label="failing", source="synthetic", split="calibration",
        scores=[{"timestamp_s": i * 1.0, "score": (0.9 if i == 5 else 0.01)}
                for i in range(10)],
    )
    spec = _spec(_clip("n1", "normal", "calibration", 0.01), clip)
    r1 = calibrate(spec, thresholds=[i / 100 for i in range(1, 100)], confirm=1)
    assert r1["calibration"]["n_failing"] == 1
    assert r1["selection"]["selected_threshold"] is not None
    # With confirm=2 the single-frame pulse never alarms -> no detection at threshold.
    r2 = calibrate(spec, thresholds=[i / 100 for i in range(1, 100)], confirm=2)
    assert r2["calibration"]["detection_rate"] == 0.0


def test_report_is_json_serialisable_and_deterministic():
    spec = _spec(
        _clip("n1", "normal", "calibration", 0.05),
        _clip("n2", "normal", "calibration", 0.2),
        _clip("f1", "failing", "calibration", 0.9),
        _clip("hf", "failing", "holdout", 0.85),
    )
    a = calibrate(spec, thresholds=[i / 100 for i in range(1, 100)])
    b = calibrate(spec, thresholds=[i / 100 for i in range(1, 100)])
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["schema"] == "tinyvision-calibration"
    assert json.loads(json.dumps(a))["schema_version"] == 1
    assert "cross-printer" in a["disclaimer"]


def test_calibrate_requires_checkpoint_when_clips_have_no_scores():
    clip = CalibrationClip(id="n1", label="normal", source="n1.mp4",
                           split="calibration", scores=None)
    with pytest.raises(CalibrationError, match="checkpoint"):
        calibrate(_spec(clip))


def test_invalid_fpr_ceiling_raises():
    spec = _spec(_clip("n1", "normal", "calibration", 0.1))
    with pytest.raises(CalibrationError):
        calibrate(spec, fpr_ceiling=5.0)


def test_format_summary_contains_key_fields():
    spec = _spec(
        _clip("n1", "normal", "calibration", 0.05),
        _clip("f1", "failing", "calibration", 0.9),
    )
    report = calibrate(spec, thresholds=[i / 100 for i in range(1, 100)])
    text = format_summary(report)
    assert "threshold calibration" in text
    assert "selected thresh" in text
    assert "detection=" in text
