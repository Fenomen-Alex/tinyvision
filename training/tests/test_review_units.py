"""Tests for tv.envdata.review_units."""

import json

import pytest

from tv.envdata.identity import make_environment
from tv.envdata.review_units import (
    assign_priorities,
    build_review_units,
    load_review_units,
    merge_completed,
    render_contact_sheet,
    write_review_units,
)
from tv.envdata.schema import (
    Recording,
    Sample,
    Manifest,
    Dataset,
    write_samples,
)


def _dataset():
    return Dataset(
        dataset_id="test-dataset",
        name="Test Dataset",
        license="CC BY 4.0",
        training_allowed=True,
        evaluation_allowed=True,
        redistribution_allowed=True,
    )


def _env(contributor="contributor-a", **kw):
    return make_environment("trier-test", contributor, "cam-1", "front", **kw)


def _manifest(envs):
    recs = []
    for env in envs:
        recs.append(
            Recording(
                recording_id=f"rec-{env.environment_id}",
                environment_id=env.environment_id,
                source="file:/tmp/x.mp4",
                duration_s=600.0,
                fps=1.0,
                width=640,
                height=480,
                continuous_realtime=True,
            )
        )
    return Manifest(dataset=_dataset(), environments=envs, recordings=recs)


def _samples(rec_id, env_id, indices):
    return [
        Sample(
            sample_id=f"{rec_id}-{i}",
            recording_id=rec_id,
            environment_id=env_id,
            frame_index=i,
            timestamp_s=float(i),
        )
        for i in indices
    ]


def test_sequence_split_on_contiguous_and_gapped_frames():
    env = _env()
    m = _manifest([env])
    samples = _samples(f"rec-{env.environment_id}", env.environment_id, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    units = build_review_units(m, samples, min_sequence_frames=4, chunk_frames=6)
    seqs = [u for u in units if u.type == "sequence"]
    frm = [u for u in units if u.type == "frame"]
    assert len(seqs) == 2  # run 4-12 -> [4-9],[10-12]
    assert any(u.frames == [4, 5, 6, 7, 8, 9] for u in seqs)
    assert any(u.frames == [10, 11, 12] for u in seqs)
    # index 3 is a gap; short run 0-2 becomes frame units
    assert {tuple(u.frames) for u in frm} == {(0,), (1,), (2,)}


def test_failure_env_prioritized_first():
    failing = _env(); failing.metadata = {"failure_type": "spaghetti"}
    normal = _env(contributor="contributor-b")
    normal_holdout = _env(contributor="contributor-c")
    normal_holdout.split_assignment = "external_holdout"
    m = _manifest([normal, failing, normal_holdout])
    units = []
    for env in m.environments:
        units += build_review_units(
            m, _samples(f"rec-{env.environment_id}", env.environment_id, list(range(4)))
        )
    assign_priorities(units, m)
    order = [u.environment_id for u in units]
    by_env = {}
    for env_id in order:
        by_env.setdefault(env_id, len(by_env))
    assert by_env[failing.environment_id] == 0


def test_merge_applies_decisions_and_rejects_invalid():
    env = _env()
    m = _manifest([env])
    samples = _samples(
        f"rec-{env.environment_id}", env.environment_id, list(range(0, 3))
    )
    units = build_review_units(m, samples, min_sequence_frames=4, chunk_frames=6)
    units[0].decision = "failing"
    units[0].failure_type = "spaghetti"
    units[1].decision = "normal"
    units[2].decision = "uncertain"
    units[2].failure_type = "spaghetti"  # invalid combo -> rejected
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        review_path = Path(td) / "review_units.json"
        write_review_units(review_path, units)
        merge_completed(review_path, m, samples)
        failings = [s for s in samples if s.tinyvision_label == "failing"]
        assert len(failings) == 1
        assert any(s.tinyvision_label == "normal" for s in samples)
        assert samples[0].verification == "verified"
        assert samples[0].tinyvision_label == "failing"
        assert samples[0].verification_provenance == "human:release-review"
        assert samples[2].verification == "pending"  # invalid unit rejected


def test_roundtrip_preserves_decisions():
    env = _env()
    m = _manifest([env])
    samples = _samples(
        f"rec-{env.environment_id}", env.environment_id, list(range(0, 3))
    )
    units = build_review_units(m, samples, min_sequence_frames=4, chunk_frames=6)
    units[0].decision = "failing"
    units[0].failure_type = "stringing"
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "review_units.json"
        write_review_units(p, units)
        loaded = load_review_units(p)
    assert loaded[0].decision == "failing"
    assert loaded[0].failure_type == "stringing"


def test_review_confirm_records_single_human_decision(tmp_path):
    import argparse

    from tv.cli import cmd_review_confirm

    env = _env()
    m = _manifest([env])
    samples = _samples(
        f"rec-{env.environment_id}", env.environment_id, list(range(0, 3))
    )
    units = build_review_units(m, samples, min_sequence_frames=4, chunk_frames=6)
    target_id = units[1].review_unit_id
    review_path = tmp_path / "review"
    review_path.mkdir()
    p = review_path / "review_units.json"
    write_review_units(p, units)

    cmd_review_confirm(argparse.Namespace(
        dir=str(tmp_path), unit=target_id, decision="failing",
        failure_type="layer_shift", note="surgical check", review_file="review_units.json",
    ))
    loaded = load_review_units(p)
    target = [u for u in loaded if u.review_unit_id == target_id][0]
    assert target.decision == "failing"
    assert target.failure_type == "layer_shift"
    assert target.notes == "surgical check"

    # failing without failure_type is refused
    with pytest.raises(SystemExit):
        cmd_review_confirm(argparse.Namespace(
            dir=str(tmp_path), unit=target_id, decision="failing",
            failure_type=None, note=None, review_file="review_units.json",
        ))


def test_contact_sheet_renders_or_blanks_missing_frames(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    env = _env()
    m = _manifest([env])
    samples = _samples(
        f"rec-{env.environment_id}", env.environment_id, list(range(0, 3))
    )
    units = build_review_units(m, samples, min_sequence_frames=4, chunk_frames=6)
    units[0].type = "sequence"
    units[0].frames = [0, 1, 2, 3]
    frame_root = tmp_path / "frames"
    rec_dir = frame_root / f"rec-{env.environment_id}"
    rec_dir.mkdir(parents=True)
    for i in [0, 2]:
        cv2.imwrite(str(rec_dir / f"frame_{i:04d}.jpg"), np.zeros((480, 640, 3), np.uint8))
    out = tmp_path / "sheet.jpg"
    render_contact_sheet(frame_root, units[0], out)
    assert out.exists()
    img = cv2.imread(str(out))
    assert img is not None