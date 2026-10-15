"""Tests for tv/sessions.py (real-world footage session intake)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training"))

from tv.sessions import (  # noqa: E402
    Session,
    SessionSheetError,
    build_calibrate_spec,
    describe_sheet,
    emit_holdout_csv,
    heldout_report,
    load_session_sheet,
    validate_sessions,
    validation_report,
    write_calibrate_spec,
)


def _load_script():
    import importlib.util

    path = REPO / "scripts" / "run_session_intake.py"
    spec = importlib.util.spec_from_file_location("run_session_intake", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scores(value, n=3):
    return [{"timestamp_s": float(i), "score": value} for i in range(n)]


def _session(sid, label, split, **kw):
    base = {"id": sid, "label": label, "split": split, "scores": _scores(0.5)}
    base.update(kw)
    return base


def _valid_payload(**overrides):
    payload = {
        "printer": "P1",
        "camera": "C0",
        "license": "own footage",
        "sessions": [
            _session("c_n1", "normal", "calibration", scores=_scores(0.05)),
            _session("c_f1", "failing", "calibration", failure_type="stringing",
                     onset_s=1.0, duration_s=10.0, scores=_scores(0.9)),
            _session("h_n1", "normal", "holdout", scores=_scores(0.03)),
            _session("h_f1", "failing", "holdout", failure_type="spaghetti",
                     onset_s=2.0, duration_s=10.0, scores=_scores(0.4)),
        ],
    }
    payload.update(overrides)
    return payload


def _load(payload, tmp_path):
    p = tmp_path / "sheet.json"
    p.write_text(json.dumps(payload))
    return load_session_sheet(p)


def _codes(validator):
    return [f.code for f in validator.flags]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_load_json_and_defaults_inherited(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({
        "printer": "P", "camera": "C", "license": "L",
        "sessions": [{"id": "a", "label": "normal", "scores": _scores(0.1)}],
    }))
    sheet = load_session_sheet(p)
    assert sheet.printer == "P"
    s = sheet.sessions[0]
    assert s.printer == "P" and s.camera == "C" and s.license == "L"
    assert s.split == "calibration"


def test_load_yaml(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(
        "printer: P\ncamera: C\nlicense: L\nsessions:\n"
        "  - {id: a, label: normal, split: holdout, scores: [{score: 0.1}]}\n"
    )
    sheet = load_session_sheet(p)
    assert sheet.sessions[0].split == "holdout"


def test_load_bare_list(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"id": "a", "label": "normal", "scores": _scores(0.1)}]))
    sheet = load_session_sheet(p)
    assert sheet.ids() == ["a"]


def test_load_missing_file_raises():
    with pytest.raises(SessionSheetError):
        load_session_sheet("/no/such/sheet.yaml")


def test_load_empty_sessions_raises(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"sessions": []}))
    with pytest.raises(SessionSheetError):
        load_session_sheet(p)


def test_split_alias_locked_holdout(tmp_path):
    sheet = _load(_valid_payload(sessions=[
        _session("c_n1", "normal", "calibration"),
        _session("c_f1", "failing", "calibration", failure_type="stringing",
                 onset_s=1, duration_s=5),
        _session("h_n1", "normal", "locked_holdout"),
        _session("h_f1", "failing", "locked_holdout", failure_type="stringing",
                 onset_s=1, duration_s=5),
    ]), tmp_path)
    assert sheet.by_split("holdout")
    assert validate_sessions(sheet).passes()


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------

def test_valid_sheet_passes(tmp_path):
    v = validate_sessions(_load(_valid_payload(), tmp_path))
    assert v.passes(), _codes(v)


def test_duplicate_id_is_error(tmp_path):
    sheet = _load(_valid_payload(sessions=[
        _session("c_n1", "normal", "calibration"),
        _session("c_n1", "failing", "calibration", failure_type="stringing",
                 onset_s=1, duration_s=5),
        _session("h_n1", "normal", "holdout"),
        _session("h_f1", "failing", "holdout", failure_type="stringing",
                 onset_s=1, duration_s=5),
    ]), tmp_path)
    assert "duplicate_session_id" in _codes(validate_sessions(sheet))


def test_invalid_label_and_split_are_errors(tmp_path):
    sheet = _load(_valid_payload(sessions=[
        _session("x", "maybe", "calibration"),
        _session("c_n1", "normal", "calibration"),
        _session("c_f1", "failing", "calibration", failure_type="stringing",
                 onset_s=1, duration_s=5),
        _session("h_n1", "normal", "lockbox"),
        _session("h_f1", "failing", "holdout", failure_type="stringing",
                 onset_s=1, duration_s=5),
    ]), tmp_path)
    codes = _codes(validate_sessions(sheet))
    assert "invalid_label" in codes and "invalid_split" in codes


def test_missing_source_and_scores_is_error(tmp_path):
    sheet = _load(_valid_payload(sessions=[
        {"id": "c_n1", "label": "normal", "split": "calibration"},
        _session("c_f1", "failing", "calibration", failure_type="stringing",
                 onset_s=1, duration_s=5),
        _session("h_n1", "normal", "holdout"),
        _session("h_f1", "failing", "holdout", failure_type="stringing",
                 onset_s=1, duration_s=5),
    ]), tmp_path)
    assert "missing_source" in _codes(validate_sessions(sheet))


def test_uncertain_label_warns_and_excludes(tmp_path):
    sheet = _load(_valid_payload(sessions=[
        _session("c_n1", "normal", "calibration"),
        _session("c_f1", "failing", "calibration", failure_type="stringing",
                 onset_s=1, duration_s=5),
        _session("u1", "uncertain", "calibration"),
        _session("h_n1", "normal", "holdout"),
        _session("h_f1", "failing", "holdout", failure_type="stringing",
                 onset_s=1, duration_s=5),
    ]), tmp_path)
    v = validate_sessions(sheet)
    assert v.passes(), _codes(v)
    assert "uncertain_label" in _codes(v)
    ids = [c["id"] for c in build_calibrate_spec(sheet)["clips"]]
    assert "u1" not in ids


def test_failing_onset_and_failure_type(tmp_path):
    sheet = _load(_valid_payload(sessions=[
        _session("c_n1", "normal", "calibration"),
        _session("c_f1", "failing", "calibration", failure_type="not_a_type",
                 onset_s=99, duration_s=10),
        _session("h_n1", "normal", "holdout"),
        _session("h_f1", "failing", "holdout",
                 onset_s=1, duration_s=5),
    ]), tmp_path)
    codes = _codes(validate_sessions(sheet))
    assert "unknown_failure_type" in codes  # not_a_type
    assert "onset_out_of_range" in codes  # 99 > 10
    assert "missing_failure_type" in codes  # h_f1 has no failure_type


def test_missing_onset_is_warning(tmp_path):
    sheet = _load(_valid_payload(sessions=[
        _session("c_n1", "normal", "calibration"),
        _session("c_f1", "failing", "calibration", failure_type="stringing",
                 duration_s=5),  # no onset
        _session("h_n1", "normal", "holdout"),
        _session("h_f1", "failing", "holdout", failure_type="stringing",
                 onset_s=1, duration_s=5),
    ]), tmp_path)
    v = validate_sessions(sheet)
    assert v.passes()
    assert "missing_onset" in _codes(v)


def test_normal_with_failure_type_and_onset_warn(tmp_path):
    sheet = _load(_valid_payload(sessions=[
        _session("c_n1", "normal", "calibration", failure_type="stringing",
                 onset_s=1),
        _session("c_f1", "failing", "calibration", failure_type="stringing",
                 onset_s=1, duration_s=5),
        _session("h_n1", "normal", "holdout"),
        _session("h_f1", "failing", "holdout", failure_type="stringing",
                 onset_s=1, duration_s=5),
    ]), tmp_path)
    v = validate_sessions(sheet)
    codes = _codes(v)
    assert "normal_has_failure_type" in codes and "normal_has_onset" in codes


def test_missing_license_is_error(tmp_path):
    sheet = _load(_valid_payload(license=None), tmp_path)
    assert "missing_license" in _codes(validate_sessions(sheet))


def test_multi_printer_is_error(tmp_path):
    sheet = _load(_valid_payload(sessions=[
        _session("c_n1", "normal", "calibration", printer="P1"),
        _session("c_f1", "failing", "calibration", printer="P2",
                 failure_type="stringing", onset_s=1, duration_s=5),
        _session("h_n1", "normal", "holdout", printer="P1"),
        _session("h_f1", "failing", "holdout", printer="P1",
                 failure_type="stringing", onset_s=1, duration_s=5),
    ]), tmp_path)
    assert "multi_printer" in _codes(validate_sessions(sheet))


# ---------------------------------------------------------------------------
# Split independence and the locked-holdout fail-closed gate
# ---------------------------------------------------------------------------

def test_calibration_single_class_is_error(tmp_path):
    sheet = _load(_valid_payload(sessions=[
        _session("c_n1", "normal", "calibration"),
        _session("c_n2", "normal", "calibration"),
        _session("h_n1", "normal", "holdout"),
        _session("h_f1", "failing", "holdout", failure_type="stringing",
                 onset_s=1, duration_s=5),
    ]), tmp_path)
    assert "split_missing_class" in _codes(validate_sessions(sheet))


def test_missing_locked_holdout_fails_closed(tmp_path):
    sheet = _load(_valid_payload(sessions=[
        _session("c_n1", "normal", "calibration"),
        _session("c_f1", "failing", "calibration", failure_type="stringing",
                 onset_s=1, duration_s=5),
    ]), tmp_path)
    v = validate_sessions(sheet)
    assert not v.passes()
    assert "missing_locked_holdout" in _codes(v)


def test_contaminated_holdout_is_error(tmp_path):
    sheet = _load(_valid_payload(sessions=[
        _session("c_n1", "normal", "calibration"),
        _session("c_f1", "failing", "calibration", failure_type="stringing",
                 onset_s=1, duration_s=5),
        _session("h_n1", "normal", "holdout"),
        _session("h_u1", "uncertain", "holdout"),
    ]), tmp_path)
    codes = _codes(validate_sessions(sheet))
    assert "contaminated_holdout" in codes


def test_holdout_missing_class_is_error(tmp_path):
    sheet = _load(_valid_payload(sessions=[
        _session("c_n1", "normal", "calibration"),
        _session("c_f1", "failing", "calibration", failure_type="stringing",
                 onset_s=1, duration_s=5),
        _session("h_n1", "normal", "holdout"),
        _session("h_n2", "normal", "holdout"),
    ]), tmp_path)
    assert "holdout_missing_class" in _codes(validate_sessions(sheet))


def test_declared_md5_duplicate_across_splits_is_error(tmp_path):
    sheet = _load(_valid_payload(sessions=[
        _session("c_n1", "normal", "calibration", md5="deadbeef"),
        _session("c_f1", "failing", "calibration", failure_type="stringing",
                 onset_s=1, duration_s=5),
        _session("h_n1", "normal", "holdout", md5="deadbeef"),
        _session("h_f1", "failing", "holdout", failure_type="stringing",
                 onset_s=1, duration_s=5),
    ]), tmp_path)
    assert "duplicate_media_across_splits" in _codes(validate_sessions(sheet))


def test_declared_md5_duplicate_within_split_is_warning(tmp_path):
    sheet = _load(_valid_payload(sessions=[
        _session("c_n1", "normal", "calibration", md5="deadbeef"),
        _session("c_n2", "normal", "calibration", md5="deadbeef"),
        _session("c_f1", "failing", "calibration", failure_type="stringing",
                 onset_s=1, duration_s=5),
        _session("h_n1", "normal", "holdout"),
        _session("h_f1", "failing", "holdout", failure_type="stringing",
                 onset_s=1, duration_s=5),
    ]), tmp_path)
    v = validate_sessions(sheet)
    assert v.passes()
    assert "duplicate_media" in _codes(v)


# ---------------------------------------------------------------------------
# Physical probing
# ---------------------------------------------------------------------------

def test_probe_missing_file_is_error(tmp_path):
    sheet = _load(_valid_payload(sessions=[
        _session("c_n1", "normal", "calibration", source=str(tmp_path / "nope.mp4")),
        _session("c_f1", "failing", "calibration", failure_type="stringing",
                 onset_s=1, duration_s=5, source=str(tmp_path / "nope2.mp4")),
        _session("h_n1", "normal", "holdout", source=str(tmp_path / "nope3.mp4")),
        _session("h_f1", "failing", "holdout", failure_type="stringing",
                 onset_s=1, duration_s=5, source=str(tmp_path / "nope4.mp4")),
    ]), tmp_path)
    v = validate_sessions(sheet, probe=True)
    assert not v.passes()
    assert _codes(v).count("missing_file") == 4


def test_probe_empty_frame_dir_is_unreadable(tmp_path):
    empty = tmp_path / "frames"
    empty.mkdir()
    frames = tmp_path / "frames_ok"
    frames.mkdir()
    (frames / "frame_0000.jpg").write_bytes(b"x")
    sheet = _load(_valid_payload(sessions=[
        _session("c_n1", "normal", "calibration", source=str(frames)),
        _session("c_f1", "failing", "calibration", failure_type="stringing",
                 onset_s=1, duration_s=5, source=str(empty)),
        _session("h_n1", "normal", "holdout", source=str(frames)),
        _session("h_f1", "failing", "holdout", failure_type="stringing",
                 onset_s=1, duration_s=5, source=str(frames)),
    ]), tmp_path)
    codes = _codes(validate_sessions(sheet, probe=True))
    assert "unreadable_video" in codes


# ---------------------------------------------------------------------------
# Conversion to tv calibrate + held-out evaluation
# ---------------------------------------------------------------------------

def test_build_calibrate_spec_mapping(tmp_path):
    sheet = _load(_valid_payload(sessions=[
        _session("c_n1", "normal", "calibration"),
        _session("c_f1", "failing", "calibration", failure_type="stringing",
                 onset_s=1, duration_s=5),
        _session("v_n1", "normal", "validation"),
        _session("h_n1", "normal", "holdout"),
        _session("h_f1", "failing", "holdout", failure_type="stringing",
                 onset_s=1, duration_s=5),
        _session("u1", "uncertain", "calibration"),
    ]), tmp_path)
    spec = build_calibrate_spec(sheet)
    got = {c["id"]: c["split"] for c in spec["clips"]}
    assert got["c_n1"] == "calibration" and got["c_f1"] == "calibration"
    assert got["v_n1"] == "holdout"  # validation -> calibrate holdout
    assert "h_n1" not in got and "h_f1" not in got  # locked holdout excluded
    assert "u1" not in got  # uncertain excluded
    # The emitted spec must load through tv.calibrate unchanged.
    from tv.calibrate import load_clip_spec
    path = tmp_path / "calib.yaml"
    write_calibrate_spec(spec, path)
    loaded = load_clip_spec(path)
    assert {c.id for c in loaded["clips"]} == {"c_n1", "c_f1", "v_n1"}


def test_build_spec_runs_through_calibrate(tmp_path):
    from tv.calibrate import calibrate, load_clip_spec
    sheet = _load(_valid_payload(sessions=[
        _session("c_n1", "normal", "calibration", scores=_scores(0.05)),
        _session("c_n2", "normal", "calibration", scores=_scores(0.08)),
        _session("c_f1", "failing", "calibration", failure_type="stringing",
                 onset_s=1, duration_s=5, scores=_scores(0.9)),
        _session("v_f1", "failing", "validation", failure_type="stringing",
                 onset_s=1, duration_s=5, scores=_scores(0.85)),
        _session("h_n1", "normal", "holdout", scores=_scores(0.03)),
        _session("h_f1", "failing", "holdout", failure_type="stringing",
                 onset_s=1, duration_s=5, scores=_scores(0.4)),
    ]), tmp_path)
    path = tmp_path / "calib.yaml"
    write_calibrate_spec(build_calibrate_spec(sheet), path)
    report = calibrate(load_clip_spec(path), thresholds=[i / 100 for i in range(1, 100)])
    assert report["status"] == "ok"
    assert report["calibration"]["n_normal"] == 2
    assert report["holdout"] is not None  # validation tier reported


def test_heldout_report_uses_locked_holdout_only(tmp_path):
    sheet = _load(_valid_payload(), tmp_path)
    scored = {s.id: s.scores for s in sheet.sessions}
    # Build scores: holdout normal low, holdout failing high.
    scored["h_n1"] = _scores(0.02)
    scored["h_f1"] = _scores(0.9)
    rep = heldout_report(sheet, scored, threshold=0.5, confirm=1)
    assert rep["n_normal"] == 1 and rep["n_failing"] == 1
    assert rep["detection_rate"] == 1.0
    assert rep["false_alarm_rate_per_print"] == 0.0
    # calibration sessions are NOT part of the held-out report
    assert {s["id"] for s in rep["sessions"]} == {"h_n1", "h_f1"}


def test_emit_holdout_csv(tmp_path):
    sheet = _load(_valid_payload(), tmp_path)
    n = emit_holdout_csv(sheet, tmp_path / "holdout.csv")
    assert n == 2
    text = (tmp_path / "holdout.csv").read_text().splitlines()
    assert text[0].startswith("session,source,label")
    assert len(text) == 3


def test_sheet_to_dict_roundtrips(tmp_path):
    sheet = _load(_valid_payload(), tmp_path)
    data = sheet.to_dict()
    assert data["schema"] == "tinyvision-session-sheet"
    assert len(data["sessions"]) == 4
    s = Session.from_dict(data["sessions"][0])
    assert s.id == "c_n1"


# ---------------------------------------------------------------------------
# CLI smoke test (subprocess, real exit codes)
# ---------------------------------------------------------------------------

def _cli(args, cwd=REPO):
    return subprocess.run(
        [sys.executable, "-m", "tv.cli", *args],
        cwd=str(cwd), capture_output=True, text=True,
    )


def test_cli_sessions_validate_exit_codes(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_valid_payload()))
    p = _cli(["sessions", "validate", "--spec", str(good)], cwd=REPO / "training")
    assert p.returncode == 0, p.stderr

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"printer": "P", "license": "L", "sessions": [
        _session("c_n1", "normal", "calibration"),
    ]}))
    p2 = _cli(["sessions", "validate", "--spec", str(bad)], cwd=REPO / "training")
    assert p2.returncode == 1


def test_cli_sessions_build_fail_closed(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"printer": "P", "license": "L", "sessions": [
        _session("c_n1", "normal", "calibration"),
        _session("c_f1", "failing", "calibration", failure_type="stringing",
                 onset_s=1, duration_s=5),
    ]}))
    out = tmp_path / "calib.yaml"
    p = _cli(["sessions", "build", "--spec", str(bad), "--out", str(out)],
             cwd=REPO / "training")
    assert p.returncode == 1
    assert not out.exists()


def test_cli_sessions_validate_includes_inventory(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_valid_payload()))
    p = _cli(["sessions", "validate", "--spec", str(good)], cwd=REPO / "training")
    assert p.returncode == 0, p.stderr
    report = json.loads(p.stdout[:p.stdout.rindex("}") + 1])
    assert "sheet" in report
    roles = {s["id"]: s["role"] for s in report["sheet"]["sessions"]}
    assert roles["h_f1"] == "locked_holdout"
    assert roles["c_n1"] == "calibration"


# ---------------------------------------------------------------------------
# Report labelling / separation regressions
# ---------------------------------------------------------------------------

def test_describe_sheet_preserves_roles_and_provenance(tmp_path):
    sheet = _load(_valid_payload(provenance="own-capture-2026"), tmp_path)
    inv = describe_sheet(sheet)
    assert inv["provenance"] == "own-capture-2026"
    roles = {s["id"]: s["role"] for s in inv["sessions"]}
    assert roles == {"c_n1": "calibration", "c_f1": "calibration",
                     "h_n1": "locked_holdout", "h_f1": "locked_holdout"}


def test_heldout_report_labelled_as_locked(tmp_path):
    sheet = _load(_valid_payload(), tmp_path)
    scored = {"h_n1": _scores(0.02), "h_f1": _scores(0.9)}
    rep = heldout_report(sheet, scored, threshold=0.5, confirm=1,
                         meta={"checkpoint": "ckpt.pt"})
    assert rep["schema"] == "tinyvision-session-holdout"
    assert rep["role"] == "locked_holdout"
    assert rep["intake_split"] == "holdout"
    assert rep["threshold"] == 0.5 and rep["confirm"] == 1
    assert rep["printer"] == "P1" and rep["camera"] == "C0"
    assert rep["meta"]["checkpoint"] == "ckpt.pt"
    assert {s["id"] for s in rep["sessions"]} == {"h_n1", "h_f1"}
    assert all(s["role"] == "locked_holdout" for s in rep["sessions"])


def test_validation_report_is_labelled_validation_not_holdout(tmp_path):
    from tv.calibrate import calibrate, load_clip_spec
    sheet = _load(_valid_payload(sessions=[
        _session("c_n1", "normal", "calibration", scores=_scores(0.05)),
        _session("c_n2", "normal", "calibration", scores=_scores(0.06)),
        _session("c_f1", "failing", "calibration", failure_type="stringing",
                 onset_s=1, duration_s=5, scores=_scores(0.9)),
        _session("v_n1", "normal", "validation", scores=_scores(0.04)),
        _session("v_f1", "failing", "validation", failure_type="stringing",
                 onset_s=1, duration_s=5, scores=_scores(0.8)),
        _session("h_n1", "normal", "holdout", scores=_scores(0.02)),
        _session("h_f1", "failing", "holdout", failure_type="stringing",
                 onset_s=1, duration_s=5, scores=_scores(0.3)),
    ]), tmp_path)
    path = tmp_path / "calib.yaml"
    write_calibrate_spec(build_calibrate_spec(sheet), path)
    cal = calibrate(load_clip_spec(path), thresholds=[i / 100 for i in range(1, 100)])
    val = validation_report(cal, sheet, meta={"checkpoint": "ckpt.pt"})
    assert val["schema"] == "tinyvision-session-validation"
    assert val["intake_split"] == "validation"
    assert val["role"] == "independent_reporting_only"
    assert val["evaluated"] is True
    assert "NOT the locked holdout" in val["note"]
    assert {s["id"] for s in val["sessions"]} == {"v_n1", "v_f1"}
    # the locked holdout must NEVER appear in the validation report
    assert "h_n1" not in json.dumps(val) and "h_f1" not in json.dumps(val)


def test_validation_report_unevaluated_when_no_validation_split(tmp_path):
    from tv.calibrate import calibrate, load_clip_spec
    sheet = _load(_valid_payload(sessions=[
        _session("c_n1", "normal", "calibration", scores=_scores(0.05)),
        _session("c_f1", "failing", "calibration", failure_type="stringing",
                 onset_s=1, duration_s=5, scores=_scores(0.9)),
        _session("h_n1", "normal", "holdout", scores=_scores(0.02)),
        _session("h_f1", "failing", "holdout", failure_type="stringing",
                 onset_s=1, duration_s=5, scores=_scores(0.3)),
    ]), tmp_path)
    path = tmp_path / "calib.yaml"
    write_calibrate_spec(build_calibrate_spec(sheet), path)
    cal = calibrate(load_clip_spec(path), thresholds=[i / 100 for i in range(1, 100)])
    val = validation_report(cal, sheet)
    assert val["evaluated"] is False
    assert val["sessions"] == []


# ---------------------------------------------------------------------------
# Ordering guarantee: locked holdout is not opened before the threshold is fixed
# ---------------------------------------------------------------------------

def test_run_poc_excludes_locked_holdout(monkeypatch, tmp_path):
    import tv.poc as poc_mod

    seen: list[str] = []

    class FakeConfig:
        def __init__(self, **kw):
            self.kw = kw

    def fake_run_poc(config):
        seen.append(config.kw["source"])
        return {"ok": True}

    monkeypatch.setattr(poc_mod, "PocConfig", FakeConfig)
    monkeypatch.setattr(poc_mod, "run_poc", fake_run_poc)

    sheet = _load(_valid_payload(sessions=[
        _session("c_n1", "normal", "calibration", source="/m/c_n1.mp4"),
        _session("c_f1", "failing", "calibration", failure_type="stringing",
                 onset_s=1, duration_s=5, source="/m/c_f1.mp4"),
        _session("v_n1", "normal", "validation", source="/m/v_n1.mp4"),
        _session("h_n1", "normal", "holdout", source="/m/h_n1.mp4"),
        _session("h_f1", "failing", "holdout", failure_type="stringing",
                 onset_s=1, duration_s=5, source="/m/h_f1.mp4"),
    ]), tmp_path)

    mod = _load_script()
    written, skipped = mod._run_poc(sheet, tmp_path, "ckpt.pt", None, None, None)
    assert written == 3  # calibration (2) + validation (1)
    assert set(seen) == {"/m/c_n1.mp4", "/m/c_f1.mp4", "/m/v_n1.mp4"}
    assert not any("h_" in s for s in seen), "locked holdout must not be opened pre-threshold"


def test_run_poc_splits_constant_excludes_holdout():
    mod = _load_script()
    assert "holdout" not in mod._PRE_THRESHOLD_SPLITS
    assert set(mod._PRE_THRESHOLD_SPLITS) == {"calibration", "validation"}


# ---------------------------------------------------------------------------
# Full automation end-to-end (subprocess; scores-only so it is fast and offline)
# ---------------------------------------------------------------------------

def _full_sheet_payload():
    return _valid_payload(sessions=[
        _session("c_n1", "normal", "calibration", scores=_scores(0.05)),
        _session("c_n2", "normal", "calibration", scores=_scores(0.07)),
        _session("c_f1", "failing", "calibration", failure_type="stringing",
                 onset_s=1, duration_s=5, scores=_scores(0.9)),
        _session("v_n1", "normal", "validation", scores=_scores(0.04)),
        _session("v_f1", "failing", "validation", failure_type="stringing",
                 onset_s=1, duration_s=5, scores=_scores(0.85)),
        _session("h_n1", "normal", "holdout", scores=_scores(0.02)),
        _session("h_f1", "failing", "holdout", failure_type="stringing",
                 onset_s=1, duration_s=5, scores=_scores(0.3)),
    ])


def _run_intake(sheet_payload, tmp_path, extra=()):
    spec = tmp_path / "sheet.json"
    spec.write_text(json.dumps(sheet_payload))
    out = tmp_path / "out"
    p = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_session_intake.py"),
         "--spec", str(spec), "--data-dir", str(out), "--skip-poc", *extra],
        cwd=str(REPO), capture_output=True, text=True,
    )
    return p, out


def test_intake_script_end_to_end_separates_roles(tmp_path):
    p, out = _run_intake(_full_sheet_payload(), tmp_path)
    assert p.returncode == 0, p.stderr

    expected = ["intake_report.json", "calibration-spec.yaml", "calibration.json",
                "validation_report.json", "run_manifest.json",
                "holdout_sessions.csv", "holdout_eval.json"]
    for name in expected:
        assert (out / name).exists(), f"missing {name}"

    # Locked holdout must not enter the calibration spec.
    spec_text = (out / "calibration-spec.yaml").read_text()
    assert "h_n1" not in spec_text and "h_f1" not in spec_text

    he = json.loads((out / "holdout_eval.json").read_text())
    assert {s["id"] for s in he["sessions"]} == {"h_n1", "h_f1"}
    assert he["role"] == "locked_holdout"

    val = json.loads((out / "validation_report.json").read_text())
    assert {s["id"] for s in val["sessions"]} == {"v_n1", "v_f1"}
    assert val["role"] == "independent_reporting_only"

    man = json.loads((out / "run_manifest.json").read_text())
    assert man["calibration"]["select_on"] == "calibration"
    assert man["validation"]["role"] == "independent_reporting_only"


def test_intake_script_fails_closed_on_missing_holdout(tmp_path):
    payload = {"printer": "P", "license": "L", "sessions": [
        _session("c_n1", "normal", "calibration", scores=_scores(0.05)),
        _session("c_f1", "failing", "calibration", failure_type="stringing",
                 onset_s=1, duration_s=5, scores=_scores(0.9)),
    ]}
    p, out = _run_intake(payload, tmp_path)
    assert p.returncode == 1
    assert not (out / "calibration.json").exists()


def test_intake_script_fails_closed_on_leakage(tmp_path):
    payload = _valid_payload(sessions=[
        _session("c_n1", "normal", "calibration", md5="deadbeef",
                 scores=_scores(0.05)),
        _session("c_f1", "failing", "calibration", failure_type="stringing",
                 onset_s=1, duration_s=5, scores=_scores(0.9)),
        _session("h_n1", "normal", "holdout", md5="deadbeef",
                 scores=_scores(0.02)),
        _session("h_f1", "failing", "holdout", failure_type="stringing",
                 onset_s=1, duration_s=5, scores=_scores(0.3)),
    ])
    p, out = _run_intake(payload, tmp_path)
    assert p.returncode == 1
    assert not (out / "calibration.json").exists()
