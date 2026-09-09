"""Tests for tv/vlm_review.py (Stage 12D-extra local VLM review proposals).

The suite must pass without LM Studio running or installed: all network access
is mocked at the HTTP transport.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training"))

import tv.vlm_review as v  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def sample_unit(unit_id="SP001__seq000", source="SP001", utype="sequence",
                frames=None, times=None):
    return v.ReviewUnit(
        review_unit_id=unit_id, source_id=source, unit_type=utype,
        frames=frames or [0], times=times or [0.0],
    )


@pytest.fixture
def manifest():
    units = []
    for sid, n in (("SP001", 6), ("NORMAL001", 3), ("UE003", 111), ("UE001", 5)):
        for i in range(n):
            units.append({
                "review_unit_id": f"{sid}__u{i:03d}",
                "source_id": sid,
                "type": "sequence" if i < 3 else "frame",
                "frames": [i], "times": [float(i)],
                "start_time_s": float(i), "end_time_s": float(i),
                "n_frames": 1,
                "decision": None, "failure_type": None,
                "transition_start": None, "transition_end": None, "notes": "",
            })
    return {"stage": "stage12d", "units": units}


@pytest.fixture
def fs(tmp_path, monkeypatch):
    """Point evidence/manifest/output paths at a temp filesystem."""
    sheets = tmp_path / "sheets"
    extracted = tmp_path / "extracted"
    labels = tmp_path / "labels"
    labels.mkdir()
    monkeypatch.setattr(v, "SHEETS_DIR", sheets)
    monkeypatch.setattr(v, "EXTRACTED_DIR", extracted)
    monkeypatch.setattr(v, "MANIFEST", labels / "review_manifest.json")
    monkeypatch.setattr(v, "PROPOSALS_OUT", labels / "vlm_review_proposals.json")
    monkeypatch.setattr(v, "VERIFIED", labels / "verified_labels.json")
    return {"root": tmp_path, "sheets": sheets, "extracted": extracted,
            "labels": labels}


def make_sheet(fs, unit_id) -> Path:
    (fs["sheets"]).mkdir(exist_ok=True)
    p = fs["sheets"] / f"{unit_id}.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    return p


def make_frame(fs, source, idx) -> Path:
    d = fs["extracted"] / source / "sequence"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"frame_{idx:04d}.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    return p


# ---------------------------------------------------------------------------
# Manifest loading / selection
# ---------------------------------------------------------------------------
def test_load_manifest_ok(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"stage": "stage12d", "units": []}))
    assert v.load_manifest(p)["stage"] == "stage12d"


def test_load_manifest_rejects_invalid(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"stage": "stage12d"}))
    with pytest.raises(ValueError):
        v.load_manifest(p)


def test_select_by_unit(manifest):
    us = v.select_units(manifest, unit_id="NORMAL001__u001")
    assert len(us) == 1
    assert us[0].review_unit_id == "NORMAL001__u001"


def test_select_by_source(manifest):
    us = v.select_units(manifest, source="SP001")
    assert {u.review_unit_id for u in us} == {f"SP001__u{i:03d}" for i in range(6)}


def test_select_requires_selection(manifest):
    with pytest.raises(ValueError):
        v.select_units(manifest)


def test_select_refuses_excluded_source(manifest):
    with pytest.raises(ValueError):
        v.select_units(manifest, source="UE001")
    with pytest.raises(ValueError):
        v.select_units(manifest, unit_id="UE001__u000")


def test_select_refuses_unknown_unit(manifest):
    with pytest.raises(ValueError):
        v.select_units(manifest, unit_id="NOPE__u000")


def test_select_batch_high_value_only(manifest):
    us = v.select_units(manifest, batch=True)
    ids = {u.review_unit_id for u in us}
    assert not any(u.startswith("UE001") for u in ids)
    # 6 + 3 + 111 = 120
    assert len(us) == 120


def test_select_batch_conflicts_with_unit(manifest):
    with pytest.raises(ValueError):
        v.select_units(manifest, batch=True, unit_id="SP001__u000")


# ---------------------------------------------------------------------------
# Image resolution
# ---------------------------------------------------------------------------
def test_resolve_sequence_uses_sheet(fs):
    make_sheet(fs, "SP001__seq000")
    ev = v.resolve_evidence(sample_unit("SP001__seq000"))
    assert ev.kind == "sheet"
    assert ev.primary.name == "SP001__seq000.jpg"
    assert ev.context == []


def test_resolve_sequence_missing(fs):
    with pytest.raises(v.MissingEvidenceError):
        v.resolve_evidence(sample_unit("SP001__seq000"))


def test_resolve_frame_target_and_context(fs):
    for i in (10, 11, 12):
        make_frame(fs, "UE003", i)
    ev = v.resolve_evidence(sample_unit(
        "UE003__frm0011", "UE003", "frame", frames=[11], times=[11.0]))
    assert ev.kind == "frame"
    assert ev.primary.name == "frame_0011.jpg"
    assert sorted(p.name for p in ev.context) == ["frame_0010.jpg", "frame_0012.jpg"]


def test_resolve_frame_context_optional(fs):
    make_frame(fs, "UE003", 11)
    ev = v.resolve_evidence(sample_unit(
        "UE003__frm0011", "UE003", "frame", frames=[11], times=[11.0]),
        include_context=False)
    assert ev.context == []


def test_resolve_frame_missing(fs):
    with pytest.raises(v.MissingEvidenceError):
        v.resolve_evidence(sample_unit(
            "UE003__frm0099", "UE003", "frame", frames=[99], times=[99.0]))


def test_image_to_data_url_png_is_jpeg_mime_placeholder(fs):
    p = fs["root"] / "x.jpg"
    p.write_bytes(b"abc")
    assert v.image_to_data_url(p) == "data:image/jpeg;base64,YWJj"


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------
def _r(obj):
    return v.validate_response(v.extract_json_object(obj))


def test_valid_normal():
    d, ft, c, e = _r(json.dumps({"decision": "normal", "failure_type": None,
                                 "confidence": 0.9,
                                 "visual_evidence": "clean print"}))
    assert d == "normal" and ft is None and c == 0.9


def test_valid_failing():
    d, ft, c, e = _r(json.dumps({"decision": "failing", "failure_type": "under_extrusion",
                                 "confidence": 0.7, "visual_evidence": "gaps"}))
    assert d == "failing" and ft == "under_extrusion"


def test_valid_uncertain():
    d, ft, c, e = _r(json.dumps({"decision": "uncertain", "failure_type": None,
                                 "confidence": 0.2, "visual_evidence": "blur"}))
    assert d == "uncertain"


def test_valid_failure_types_all():
    for ft in v.FAILURE_TYPES:
        d, f, c, e = _r(json.dumps({"decision": "failing", "failure_type": ft,
                                    "confidence": 0.5,
                                    "visual_evidence": "seen"}))
        assert f == ft


def test_failure_type_null_normalized():
    for raw in (None, "null", "None", ""):
        d, f, c, e = _r(json.dumps({"decision": "normal", "failure_type": raw,
                                    "confidence": 0.5, "visual_evidence": "x"}))
        assert f is None


def test_invalid_decision():
    with pytest.raises(v.ResponseValidationError):
        _r(json.dumps({"decision": "maybe", "failure_type": None,
                       "confidence": 0.5, "visual_evidence": "x"}))


def test_invalid_failure_type():
    with pytest.raises(v.ResponseValidationError):
        _r(json.dumps({"decision": "failing", "failure_type": "clogged_nozzle",
                       "confidence": 0.5, "visual_evidence": "x"}))


def test_invalid_confidence():
    with pytest.raises(v.ResponseValidationError):
        _r(json.dumps({"decision": "normal", "failure_type": None,
                       "confidence": 1.5, "visual_evidence": "x"}))
    with pytest.raises(v.ResponseValidationError):
        _r(json.dumps({"decision": "normal", "failure_type": None,
                       "confidence": "high", "visual_evidence": "x"}))
    with pytest.raises(v.ResponseValidationError):
        _r(json.dumps({"decision": "normal", "failure_type": None,
                       "confidence": True, "visual_evidence": "x"}))


def test_malformed_json():
    with pytest.raises(v.ResponseValidationError):
        _r("{not json")


def test_prose_wrapped_valid_json():
    d, ft, c, e = _r('Here is my answer: {"decision":"failing",'
                     '"failure_type":"spaghetti","confidence":0.8,'
                     '"visual_evidence":"tangled"} hope that helps!')
    assert d == "failing"


def test_prose_wrapped_with_preceding_object_rejects_or_uses_last():
    # two objects -> last `}` wins, first `{` extracted; if malformed -> error
    with pytest.raises(v.ResponseValidationError):
        _r('{"a":1} garbage {"decision":')


def test_missing_required_fields():
    with pytest.raises(v.ResponseValidationError):
        _r(json.dumps({"decision": "normal"}))
    with pytest.raises(v.ResponseValidationError):
        _r(json.dumps({"decision": "normal", "failure_type": None,
                       "confidence": 0.5}))


# ---------------------------------------------------------------------------
# API behavior (mocked transport)
# ---------------------------------------------------------------------------
def _envelope(content_text: str) -> tuple[int, str]:
    env = {"choices": [{"message": {"content": content_text}}]}
    return 200, json.dumps(env)


def _proposal_json(decision="normal"):
    return json.dumps({"decision": decision, "failure_type": None,
                       "confidence": 0.9, "visual_evidence": "ok"})


def test_api_success(fs, monkeypatch):
    make_sheet(fs, "SP001__seq000")
    def fake(url, payload, timeout):
        return _envelope(_proposal_json())
    monkeypatch.setattr(v, "_http_post_json", fake)
    prop = v.propose_unit(sample_unit("SP001__seq000"))
    assert prop["status"] == "success"
    assert prop["decision"] == "normal"


def test_api_timeout_then_success_retries(fs, monkeypatch):
    make_sheet(fs, "SP001__seq000")
    calls = {"n": 0}
    def fake(url, payload, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise v.RetryableApiError("timeout", "boom")
        return _envelope(_proposal_json("failing"))
    monkeypatch.setattr(v, "_http_post_json", fake)
    prop = v.propose_unit(sample_unit("SP001__seq000"), retries=2)
    assert prop["status"] == "success"
    assert prop["attempts"] == 2
    assert calls["n"] == 2


def test_api_http_failure_produces_error_proposal(fs, monkeypatch):
    make_sheet(fs, "SP001__seq000")
    def fake(url, payload, timeout):
        raise v.RetryableApiError("http", "HTTP 500")
    monkeypatch.setattr(v, "_http_post_json", fake)
    prop = v.propose_unit(sample_unit("SP001__seq000"), retries=1)
    assert prop["status"] == "error"
    assert prop["error_type"] == "http"


def test_api_validation_retries_then_error(fs, monkeypatch):
    make_sheet(fs, "SP001__seq000")
    def fake(url, payload, timeout):
        return _envelope('{"decision": "garbage"}')
    monkeypatch.setattr(v, "_http_post_json", fake)
    prop = v.propose_unit(sample_unit("SP001__seq000"), retries=2, timeout=5.0)
    assert prop["status"] == "error"
    assert prop["error_type"] == "validation"
    assert prop["attempts"] == 3


def test_api_final_failure_after_retries_bounded(fs, monkeypatch):
    make_sheet(fs, "SP001__seq000")
    calls = {"n": 0}
    def fake(url, payload, timeout):
        calls["n"] += 1
        raise v.RetryableApiError("connection", "down")
    monkeypatch.setattr(v, "_http_post_json", fake)
    prop = v.propose_unit(sample_unit("SP001__seq000"), retries=2)
    assert prop["status"] == "error"
    assert calls["n"] == 3  # bounded: retries + 1


def test_malformed_model_response_represented_as_error(fs, monkeypatch):
    make_sheet(fs, "SP001__seq000")
    def fake(url, payload, timeout):
        return 200, "no json here at all"
    monkeypatch.setattr(v, "_http_post_json", fake)
    prop = v.propose_unit(sample_unit("SP001__seq000"), retries=0)
    assert prop["status"] == "error"


def test_missing_evidence_is_an_error_proposal_not_fabrication(fs, monkeypatch):
    # no sheet written on purpose
    prop = v.propose_unit(sample_unit("SP001__seq000"))
    assert prop["status"] == "error"
    assert prop["error_type"] == "missing_evidence"
    assert "decision" not in prop


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------
def test_manifest_and_verified_untouched(fs, monkeypatch):
    """Run a real review op (mock transport) and verify the manifest is
    byte-for-byte unchanged and verified labels are never created or
    modified."""
    manifest_path = v.MANIFEST
    manifest_bytes = b'{"stage":"stage12d","units":[{"review_unit_id":"SP001__seq000",'
    manifest_bytes += b'"source_id":"SP001","type":"sequence","frames":[0],"times":[0.0]}]}'
    manifest_path.write_bytes(manifest_bytes)

    verified_path = v.VERIFIED
    verified_bytes = b'{"stage":"stage12d","provenance":"human_reviewed","units":[]}'
    verified_path.write_bytes(verified_bytes)

    make_sheet(fs, "SP001__seq000")

    def fake(url, payload, timeout):
        return _envelope(_proposal_json())
    monkeypatch.setattr(v, "_http_post_json", fake)

    store = v.run_review(
        v.select_units(v.load_manifest(manifest_path), unit_id="SP001__seq000"),
        output=fs["labels"] / "vlm_review_proposals.json")

    assert manifest_path.read_bytes() == manifest_bytes
    assert verified_path.read_bytes() == verified_bytes


# ---------------------------------------------------------------------------
# Output / idempotency
# ---------------------------------------------------------------------------
def test_proposal_output_schema_and_repeat_no_duplicates(fs, monkeypatch):
    make_sheet(fs, "SP001__seq000")
    def fake(url, payload, timeout):
        return _envelope(_proposal_json())
    monkeypatch.setattr(v, "_http_post_json", fake)

    output = fs["labels"] / "vlm_review_proposals.json"
    unit = sample_unit("SP001__seq000")
    store1 = v.run_review([unit], output=output)
    assert store1["schema_version"] == 1
    assert any(p["review_unit_id"] == "SP001__seq000" for p in store1["proposals"])

    store2 = v.run_review([unit], output=output)
    matches = [p for p in store2["proposals"] if p["review_unit_id"] == "SP001__seq000"]
    assert len(matches) == 1  # no uncontrolled duplicates


def test_review_preserves_unrelated_proposals(fs, monkeypatch):
    make_sheet(fs, "SP001__seq000")
    make_sheet(fs, "NORMAL001__seq000")
    def fake(url, payload, timeout):
        return _envelope(_proposal_json())
    monkeypatch.setattr(v, "_http_post_json", fake)
    output = fs["labels"] / "vlm_review_proposals.json"

    unit_a = sample_unit("SP001__seq000")
    unit_b = sample_unit("NORMAL001__seq000", "NORMAL001")
    v.run_review([unit_a], output=output)
    v.run_review([unit_b], output=output)

    store = json.loads(output.read_text())
    ids = [p["review_unit_id"] for p in store["proposals"]]
    assert ids == ["SP001__seq000", "NORMAL001__seq000"]


def test_errors_represented_explicitly_in_output(fs, monkeypatch):
    # no sheet -> missing evidence error, but still written to output
    output = fs["labels"] / "vlm_review_proposals.json"
    v.run_review([sample_unit("SP001__seq000")], output=output)
    store = json.loads(output.read_text())
    props = [p for p in store["proposals"] if p["review_unit_id"] == "SP001__seq000"]
    assert props and props[0]["status"] == "error"
    assert props[0]["error_type"] == "missing_evidence"


def test_dry_run_writes_nothing(fs, monkeypatch):
    make_sheet(fs, "SP001__seq000")
    output = fs["labels"] / "vlm_review_proposals.json"
    unit = sample_unit("SP001__seq000")
    result = v.run_review([unit], output=output, dry_run=True)
    assert result["dry_run"][0]["status"] == "dry_run"
    assert not output.exists()


# ---------------------------------------------------------------------------
# Envdata extensions (opt-in; Stage 12D defaults unchanged)
# ---------------------------------------------------------------------------
def test_extended_vocabulary_opt_in_only():
    obj = {"decision": "failing", "failure_type": "layer_shift",
           "confidence": 0.7, "visual_evidence": "shifted layers"}
    # default rejects new types even if the model emits them
    with pytest.raises(v.ResponseValidationError):
        v.validate_response(dict(obj))
    # opt-in accepts them
    d, f, c, e = v.validate_response(dict(obj),
                                     failure_types=v.EXTENDED_FAILURE_TYPES)
    assert d == "failing" and f == "layer_shift"


def test_extended_prompt_covers_extended_types():
    for ft in ("layer_shift", "extrusion_error", "bed_adhesion", "warping"):
        assert ft in v.EXTENDED_REVIEW_PROMPT


def test_propose_unit_accepts_reference_evidence(fs, monkeypatch):
    sheet = make_sheet(fs, "SP001__seq000")
    unit = sample_unit("SP001__seq000")
    evidence = v.Evidence(unit=unit, kind="sheet", primary=sheet, context=[])
    proposal = v.propose_unit(unit, evidence=evidence, dry_run=True)
    assert proposal["kind"] == "sheet"
    assert Path(proposal["primary"]) == sheet


def test_propose_unit_forwards_extra_fields(fs, monkeypatch):
    make_sheet(fs, "SP001__seq000")
    def fake(url, payload, timeout):
        return _envelope(_proposal_json())
    monkeypatch.setattr(v, "_http_post_json", fake)
    unit = sample_unit("SP001__seq000")
    evidence = v.Evidence(unit=unit, kind="sheet",
                          primary=fs["sheets"] / "SP001__seq000.jpg",
                          context=[])
    p = v.propose_unit(unit, evidence=evidence,
                       extra={"environment_id": "env-1",
                              "recording_id": "rec-1",
                              "provenance": "vlm:qwen/qwen3-vl-8b"})
    assert p["environment_id"] == "env-1"
    assert p["recording_id"] == "rec-1"
    assert p["provenance"] == "vlm:qwen/qwen3-vl-8b"


def test_image_to_data_url_scaled(tmp_path):
    from PIL import Image
    big = tmp_path / "big.jpg"
    Image.new("RGB", (400, 200), (10, 10, 10)).save(big, format="JPEG",
                                                     quality=95)
    url = v.image_to_data_url_scaled(big, max_side=100)
    assert url.startswith("data:image/jpeg;base64,")
    assert len(url) > 40