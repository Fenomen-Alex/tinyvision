"""Tests for tv/review.py (Stage 12D human-review workflow)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training"))

from tv.review import (  # noqa: E402
    MIN_SEQUENCE_SECONDS,
    SEQUENCE_CHUNK_SECONDS,
    SampleMeta,
    define_review_units,
    interactive_review,
    is_unstable,
    save_manifest_atomic,
    select_review_units,
    stable_runs,
)


def sm(t, cut=False, frac=0.0, bbox=0.0):
    return SampleMeta(time_s=t, cut_flag=cut, frac_changed=frac,
                      diff_bbox_frac=bbox, seq_index=round(t))


def test_is_unstable_cut():
    assert is_unstable(sm(1.0, cut=True))


def test_is_unstable_broad_change():
    assert is_unstable(sm(1.0, cut=False, frac=0.5, bbox=0.8))
    # not unstable when only localized
    assert not is_unstable(sm(1.0, cut=False, frac=0.5, bbox=0.2))
    assert not is_unstable(sm(1.0, cut=False, frac=0.2, bbox=0.9))


def test_stable_runs_splits_on_cuts():
    samples = [sm(0.0), sm(1.0), sm(2.0, cut=True), sm(3.0), sm(4.0)]
    runs = stable_runs(samples)
    assert [len(r) for r in runs] == [2, 2]
    assert runs[0][0].time_s == 0.0 and runs[1][0].time_s == 3.0


def test_stable_runs_contiguous():
    samples = [sm(i) for i in range(5)]
    runs = stable_runs(samples)
    assert len(runs) == 1 and len(runs[0]) == 5


def test_define_review_units_long_run_becomes_sequences():
    # 25 stable seconds -> sequences of 10
    samples = [sm(i) for i in range(25)]
    units = define_review_units("TEST", samples)
    seqs = [u for u in units if u["type"] == "sequence"]
    assert len(seqs) == 3
    # every sequence has length <= chunk, contiguous times
    for u in seqs:
        assert u["n_frames"] <= SEQUENCE_CHUNK_SECONDS
        assert u["times"] == sorted(u["times"])
    # short run (< min) stays as isolated frames
    samples2 = [sm(i) for i in range(MIN_SEQUENCE_SECONDS - 1)]
    units2 = define_review_units("TEST2", samples2)
    assert all(u["type"] == "frame" for u in units2)


def test_define_review_units_mixed():
    # 4 stable, cut, 8 stable -> first four frames, then a sequence of 8
    samples = [sm(0.0), sm(1.0), sm(2.0), sm(3.0),
               sm(4.0, cut=True),
               sm(5.0), sm(6.0), sm(7.0), sm(8.0), sm(9.0),
               sm(10.0), sm(11.0), sm(12.0)]
    units = define_review_units("MIX", samples, min_seconds=6)
    seqs = [u for u in units if u["type"] == "sequence"]
    frms = [u for u in units if u["type"] == "frame"]
    assert len(seqs) == 1  # the 8-second stable run
    assert seqs[0]["times"] == [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    # first 4 stable seconds (below min) become isolated frames
    assert len(frms) >= 4
    # every second is assigned to exactly one unit
    assigned = []
    for u in units:
        assigned += u["frames"]
    assert all(t in assigned for t in range(13))


def test_review_unit_ids_unique_and_decision_null():
    samples = [sm(i // 2, cut=(i == 6)) for i in range(20)]
    units = define_review_units("UNIQ", samples)
    ids = [u["review_unit_id"] for u in units]
    assert len(ids) == len(set(ids))
    assert all(u["decision"] is None for u in units)


def test_frame_units_cover_every_second_when_no_long_run():
    samples = [sm(i, cut=(i % 2 == 0 and i > 0)) for i in range(10)]
    units = define_review_units("COV", samples)
    # unstable seconds are always represented as isolated frame units
    covered = sorted(t for u in units for t in u["times"])
    assert covered == [float(i) for i in range(10)]


def test_merge_verified_validates_decision_vocab(tmp_path):
    from tv.review import merge_verified
    manifest = {
        "units": [
            {"review_unit_id": "a", "source_id": "S", "type": "frame",
             "frames": [1], "times": [1.0], "decision": "failing",
             "failure_type": "under_extrusion", "transition_start": None,
             "transition_end": None, "notes": ""},
        ]
    }
    mp = tmp_path / "m.json"
    op = tmp_path / "v.json"
    mp.write_text(json.dumps(manifest))
    out = merge_verified(mp, op)
    assert out["n_units_verified"] == 1
    assert out["units"][0]["provenance"] == "human_reviewed"

    # invalid decision -> error
    bad = dict(manifest)
    bad["units"] = [dict(manifest["units"][0], decision="maybe")]
    bp = tmp_path / "bad.json"
    bp.write_text(json.dumps(bad))
    try:
        merge_verified(bp, tmp_path / "x.json")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_merge_verified_requires_all_unless_partial(tmp_path):
    from tv.review import merge_verified
    manifest = {
        "units": [
            {"review_unit_id": "a", "source_id": "S", "type": "frame",
             "frames": [1], "times": [1.0], "decision": "normal"},
            {"review_unit_id": "b", "source_id": "S", "type": "frame",
             "frames": [2], "times": [2.0], "decision": None},
        ]
    }
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(manifest))
    try:
        merge_verified(mp, tmp_path / "v.json", require_all=True)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # partial allowed -> merges only decided unit
    out = merge_verified(mp, tmp_path / "v2.json", require_all=False)
    assert out["n_units_verified"] == 1


# ---------------------------------------------------------------------------
# Interactive human review (Stage 12E preparation)
# ---------------------------------------------------------------------------
def _unit(uid, source="S", utype="sequence", decision=None, failure_type=None):
    return {
        "review_unit_id": uid, "source_id": source, "type": utype,
        "frames": [0], "times": [0.0],
        "start_time_s": 0.0, "end_time_s": 0.0, "n_frames": 1,
        "decision": decision, "failure_type": failure_type,
        "transition_start": None, "transition_end": None, "notes": "",
    }


def _manifest(*units):
    return {
        "stage": "stage12d", "manifest_version": 1,
        "instructions": "", "decision_vocab": ["failing", "normal", "uncertain"],
        "min_sequence_seconds": 6, "sequence_chunk_seconds": 10,
        "per_source": {}, "units": list(units),
    }


def _run(manifest, inputs, unit_ids=None, sources=None, sequence_only=False):
    """Drive an interactive session with scripted inputs (no GUI anywhere)."""
    it = iter(inputs)

    def inp(prompt):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    out, opened, saves = [], [], []
    session = interactive_review(
        manifest,
        unit_ids=unit_ids, sources=sources, sequence_only=sequence_only,
        input_fn=inp,
        print_fn=out.append,
        open_fn=lambda p: (opened.append(str(p)), True)[1],
        save_fn=lambda m: saves.append(json.dumps(m["units"], sort_keys=True)),
    )
    return session, out, opened, saves, manifest


def test_review_stores_normal():
    m = _manifest(_unit("A__seq000", source="A"))
    _, _, opened, saves, m = _run(m, ["n", ""])
    assert m["units"][0]["decision"] == "normal"
    assert m["units"][0]["failure_type"] is None
    assert len(opened) == 1 and opened[0].endswith("sheets/A__seq000.jpg")
    assert len(saves) == 1


def test_review_failing_spaghetti():
    m = _manifest(_unit("A__seq000", "A"))
    _, _, _, saves, m = _run(m, ["f", "s", ""])
    assert m["units"][0]["decision"] == "failing"
    assert m["units"][0]["failure_type"] == "spaghetti"


def test_review_failing_stringing():
    m = _manifest(_unit("A__seq000", "A"))
    _, _, _, saves, m = _run(m, ["f", "t", ""])
    assert m["units"][0]["decision"] == "failing"
    assert m["units"][0]["failure_type"] == "stringing"


def test_review_failing_under_extrusion():
    m = _manifest(_unit("A__seq000", "A"))
    _, _, _, saves, m = _run(m, ["f", "u", ""])
    assert m["units"][0]["decision"] == "failing"
    assert m["units"][0]["failure_type"] == "under_extrusion"


def test_review_failing_requires_failure_type():
    # empty answer at the failure-type prompt is rejected, not accepted as failing
    m = _manifest(_unit("A__seq000", "A"))
    _, out, _, saves, m = _run(m, ["f", "", "u", ""])
    assert m["units"][0]["decision"] == "failing"
    assert m["units"][0]["failure_type"] == "under_extrusion"
    assert any("Invalid choice" in o for o in out)
    assert len(saves) == 1


def test_review_stores_uncertain():
    m = _manifest(_unit("A__seq000", "A"))
    _, _, _, saves, m = _run(m, ["u", ""])
    assert m["units"][0]["decision"] == "uncertain"
    assert m["units"][0]["failure_type"] is None


def test_review_skip_leaves_unchanged():
    m = _manifest(_unit("A__seq000", "A"))
    session, _, _, saves, m = _run(m, ["s"])
    assert m["units"][0]["decision"] is None
    assert saves == []
    assert session["skipped"] == ["A__seq000"]


def test_review_quit_preserves_saved_decisions():
    m = _manifest(_unit("A__seq000", "A"), _unit("B__seq000", "B"))
    session, _, _, saves, m = _run(m, ["n", "", "q"])
    assert m["units"][0]["decision"] == "normal"
    assert m["units"][1]["decision"] is None
    assert len(saves) == 1
    assert session["reviewed"] == ["A__seq000"]


def test_review_existing_decision_not_overwritten_without_confirmation():
    # explicit selection required: the conservative default skips decided units
    m = _manifest(_unit("A__seq000", "A", decision="normal"))
    session, _, _, saves, m = _run(m, [""], unit_ids=["A__seq000"])  # default skip
    assert m["units"][0]["decision"] == "normal"
    assert saves == []
    assert session["skipped"] == ["A__seq000"]

    # explicit [r]eview again permits replacement
    m = _manifest(_unit("A__seq000", "A", decision="normal"))
    session, _, _, saves, m = _run(m, ["r", "f", "t"], unit_ids=["A__seq000"])
    assert m["units"][0]["decision"] == "failing"
    assert m["units"][0]["failure_type"] == "stringing"
    assert len(saves) == 1
    assert session["skipped"] == []


def test_review_rejects_invalid_decision():
    m = _manifest(_unit("A__seq000", "A"))
    _, out, _, _, m = _run(m, ["x", "n", ""])
    assert m["units"][0]["decision"] == "normal"
    assert sum("Invalid choice" in o for o in out) >= 1


def test_review_cancel_failure_type_leaves_unchanged():
    m = _manifest(_unit("A__seq000", "A"))
    session, _, _, saves, m = _run(m, ["f", "x"])
    assert m["units"][0]["decision"] is None
    assert m["units"][0]["failure_type"] is None
    assert saves == []
    assert session["skipped"] == ["A__seq000"]


def test_review_saves_after_each_completed_unit():
    m = _manifest(_unit("A__seq000", "A"), _unit("B__seq000", "B"))
    _, _, _, saves, _ = _run(m, ["n", "", "n", ""])
    assert len(saves) == 2
    s1, s2 = (json.loads(s) for s in saves)
    assert s1[0]["decision"] == "normal" and s1[1]["decision"] is None
    assert s2[0]["decision"] == "normal" and s2[1]["decision"] == "normal"


def test_review_never_reads_vlm_proposals(tmp_path, monkeypatch):
    import builtins
    forbidden = tmp_path / "vlm_review_proposals.json"
    forbidden.write_text("{}")
    real_open = builtins.open
    hits = []

    def guarded(file, *a, **k):
        if isinstance(file, (str, Path)) and str(file).endswith("vlm_review_proposals.json"):
            hits.append(str(file))
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded)
    m = _manifest(_unit("A__seq000", "A"), _unit("B__seq000", "B"))
    _, _, _, saves, _ = _run(m, ["n", "", "n", ""])
    assert saves and hits == []  # the session worked and never touched that file


def test_review_never_writes_verified_labels(tmp_path, monkeypatch):
    import builtins
    real_open = builtins.open

    def guarded(file, *a, **k):
        if isinstance(file, (str, Path)) and str(file).endswith("verified_labels.json"):
            raise AssertionError("review must not write verified_labels.json")
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded)
    m = _manifest(_unit("A__seq000", "A"))
    _, _, _, saves, _ = _run(m, ["n", ""])
    assert len(saves) == 1
    assert not (tmp_path / "verified_labels.json").exists()


def test_review_unit_selection_preserves_cli_order():
    m = _manifest(_unit("A__seq000", "A"), _unit("B__seq000", "B"),
                  _unit("C__seq000", "C"))
    _, _, opened, _, m = _run(m, ["n", "", "n", ""], unit_ids=["C__seq000", "B__seq000"])
    names = [Path(p).stem for p in opened]
    assert names == ["C__seq000", "B__seq000"]
    assert m["units"][0]["decision"] is None  # A untouched


def test_review_source_filters_only_that_source():
    m = _manifest(_unit("NORMAL001__seq000", "NORMAL001"),
                  _unit("SP001__seq000", "SP001"))
    _, _, opened, _, _ = _run(m, ["n", ""], sources=["NORMAL001"])
    assert [Path(p).stem for p in opened] == ["NORMAL001__seq000"]


def test_review_sequence_only_excludes_frame_units():
    m = _manifest(_unit("A__seq000", "A"), _unit("A__frm0000", "A", utype="frame"))
    _, _, opened, _, _ = _run(m, ["n", ""], sources=["A"], sequence_only=True)
    assert [Path(p).stem for p in opened] == ["A__seq000"]


def test_review_default_is_conservative():
    # no selectors -> undecided SEQUENCE units only (not decided, not frames)
    m = _manifest(_unit("A__seq000", "A"),
                  _unit("A__seq001", "A", decision="normal"),
                  _unit("A__frm0000", "A", utype="frame"))
    _, _, opened, _, m = _run(m, ["n", ""])
    assert [Path(p).stem for p in opened] == ["A__seq000"]
    assert m["units"][1]["decision"] == "normal"  # untouched


def test_review_unknown_unit_id_errors():
    m = _manifest(_unit("A__seq000", "A"))
    try:
        select_review_units(m, unit_ids=["NOPE__seq000"])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_review_handles_eof_without_labeling():
    m = _manifest(_unit("A__seq000", "A"), _unit("B__seq000", "B"))
    session, _, _, saves, m = _run(m, ["n", ""])  # EOF at B's prompt
    assert m["units"][0]["decision"] == "normal"
    assert m["units"][1]["decision"] is None
    assert len(saves) == 1  # only completed units were saved


def test_save_manifest_atomic_writes_and_cleans_tmp(tmp_path):
    m = _manifest(_unit("A__seq000", "A", decision="normal"))
    p = tmp_path / "review_manifest.json"
    save_manifest_atomic(m, p)
    data = json.loads(p.read_text())
    assert data["units"][0]["decision"] == "normal"
    assert data["manifest_version"] == 1
    assert not (tmp_path / "review_manifest.json.tmp").exists()


def _merge_manifest():
    done = _unit("A__seq000", "A", decision="failing", failure_type="stringing")
    undecided = _unit("B__seq000", "B")
    return _manifest(done, undecided)


def test_merge_cli_allow_partial_string_arg_boundary(tmp_path):
    """Regression: `merge --allow-partial` must work when the CLI hands
    merge_verified() the default string argparse values (not Paths)."""
    import subprocess
    manifest_path = tmp_path / "review_manifest.json"
    manifest_path.write_text(json.dumps(_merge_manifest()))
    labels = tmp_path / "labels"
    out_path = labels / "verified_labels.json"
    sentinel = labels / "vlm_review_proposals.json"
    labels.mkdir()
    sentinel.write_text("unchanged-by-merge")

    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_human_review.py"),
         "merge", "--manifest", str(manifest_path), "--out", str(out_path),
         "--allow-partial"],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    assert out_path.exists()
    verified = json.loads(out_path.read_text())
    ids = [u["review_unit_id"] for u in verified["units"]]
    # partial merge: only the completed unit is included
    assert ids == ["A__seq000"]
    assert verified["units"][0]["decision"] == "failing"
    assert verified["units"][0]["failure_type"] == "stringing"
    assert verified["units"][0]["provenance"] == "human_reviewed"
    # undecided units stay excluded
    assert "B__seq000" not in ids
    assert verified["n_units_verified"] == 1
    # VLM proposals are never consulted or modified
    assert sentinel.read_text() == "unchanged-by-merge"
