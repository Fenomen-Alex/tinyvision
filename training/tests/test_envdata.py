"""Tests for the environment-centric staging pipeline (tv.envdata)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tv.envdata.bambu import build_inventory
from tv.envdata.identity import EnvironmentRegistry, make_environment
from tv.envdata.readiness import (
    HoldingsSnapshot,
    HoldingAsset,
    StagedManifestView,
    compute_readiness,
)
from tv.envdata.schema import (
    Consent,
    Dataset,
    Environment,
    Manifest,
    Recording,
    Sample,
    dump_manifest,
    environment_key,
)
from tv.envdata.split import (
    allocate_environments,
    apply_plan,
    check_environment_leakage,
)
from tv.envdata.state import TransitionError, set_verification
from tv.envdata.trier import characterize_from_zenodo, index_trier_tree
from tv.envdata.validate import (
    Validator,
    flag_obvious_duplicates,
    validate_manifest,
    validate_samples,
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


def _env(source="test-dataset", contributor="alice", camera="cam-1", view="front", **kw):
    return make_environment(source, contributor, camera, view, **kw)


def _manifest_with_envs(n=3):
    envs = [_env(contributor=f"u{i}", camera="cam-1", view="front") for i in range(n)]
    recs = [
        Recording(
            recording_id=f"rec-{i}",
            environment_id=env.environment_id,
            source=f"file:/tmp/{i}.mp4",
            duration_s=10.0,
            fps=30.0,
            width=640,
            height=480,
            continuous_realtime=True,
        )
        for i, env in enumerate(envs)
    ]
    return Manifest(dataset=_dataset(), environments=envs, recordings=recs)


# ---------------------------------------------------------------- identity

def test_environment_key_stable_and_normalized():
    assert environment_key("Bambu", "Borillion", "cam", "inside chamber") == \
        "env_bambu_borillion_cam_inside_chamber"
    assert environment_key("bambu", "Borillion", "cam", "inside_chamber") == \
        "env_bambu_borillion_cam_inside_chamber"


def test_registry_prevents_key_rebinding():
    reg = EnvironmentRegistry()
    a = make_environment("src", "alice", "cam", "front", registry=reg)
    reg.register(a.environment_id, a.environment_key)  # same id: idempotent, OK
    with pytest.raises(ValueError):
        reg.register("env-DIFFERENT-1", a.environment_key)  # rebind -> error


def test_registry_stable_across_calls():
    reg = EnvironmentRegistry()
    a = make_environment("src", "alice", "cam", "front", registry=reg)
    b = make_environment("src", "alice", "cam", "front", registry=reg)
    assert a.environment_id == b.environment_id


# ---------------------------------------------------------------- splitting

def test_one_environment_per_split_no_leakage():
    m = _manifest_with_envs(4)
    ids = [e.environment_id for e in m.environments]
    plan = allocate_environments(ids, ratios=(0.5, 0.25, 0.25), seed=7)
    apply_plan(m, plan)
    report = check_environment_leakage(m)
    assert report.ok, report.codes()
    by_split = {}
    for e in m.environments:
        by_split.setdefault(e.split_assignment, []).append(e.environment_id)
    for split, envs in by_split.items():
        assert len(set(envs)) == len(envs)


def test_no_training_when_pool_too_small():
    m = _manifest_with_envs(2)
    ids = [e.environment_id for e in m.environments]
    plan = allocate_environments(ids, ratios=(0.7, 0.15, 0.15), seed=1)
    assert plan.env_count["train"] >= 1
    assert plan.env_count["validation"] == 0


def test_env_assignment_leakage_free():
    m = _manifest_with_envs(2)
    plan = allocate_environments([e.environment_id for e in m.environments],
                                 ratios=(0.5, 0.5, 0.0), seed=42)
    apply_plan(m, plan)
    report = check_environment_leakage(m)
    assert report.ok


def test_dangling_recording_detected():
    m = _manifest_with_envs(1)
    m.recordings[0].environment_id = "env-missing"
    report = check_environment_leakage(m)
    assert not report.ok
    assert "dangling_recording" in report.codes()


def test_recording_cannot_multi_home():
    m = _manifest_with_envs(2)
    m.recordings.append(
        Recording(recording_id="rec-x", environment_id=m.environments[0].environment_id,
                  source="file:/tmp/x.mp4")
    )
    report = check_environment_leakage(m)
    assert not report.ok


def test_duplicate_env_key_detected():
    m = _manifest_with_envs(2)
    m.environments[1].environment_key = m.environments[0].environment_key
    report = check_environment_leakage(m)
    assert not report.ok
    assert "key_collision" in report.codes()


# ---------------------------------------------------------------- validation

def test_validate_missing_usage_rights():
    ds = _dataset()
    ds.redistribution_allowed = False
    ds.training_allowed = False
    ds.evaluation_allowed = False
    m = Manifest(dataset=ds, environments=[], recordings=[])
    v = validate_manifest(m)
    assert any(f.code == "no_usage_rights" for f in v.flags)


def test_validate_accepts_healthy_manifest(tmp_path):
    m = _manifest_with_envs(3)
    dump_manifest(m, tmp_path)
    v = validate_manifest(m, staging_root=tmp_path)
    assert v.passes(), [f.code for f in v.flags]


def test_validate_contributor_requires_consent():
    ds = _dataset()
    ds.acquisition_method = "contributor"
    env = _env()
    m = Manifest(dataset=ds, environments=[env], recordings=[])
    v = validate_manifest(m)
    assert any(f.code == "missing_consent" for f in v.flags)


def test_validate_samples_verified_without_label():
    m = _manifest_with_envs(1)
    samples = [
        Sample(
            sample_id="s-1",
            recording_id=m.recordings[0].recording_id,
            environment_id=m.environments[0].environment_id,
            verification="verified",
            tinyvision_label=None,
        )
    ]
    v = validate_samples(m, samples=samples)
    assert any(f.code == "verified_without_label" for f in v.flags)


# ---------------------------------------------------------------- state machine

def test_verification_transitions():
    s = Sample(sample_id="s-1", recording_id="r-1", environment_id="e-1")
    set_verification(s, "verified", "human:stage12d", tinyvision_label="failing",
                     tinyvision_failure_type="spaghetti")
    assert s.verification == "verified"
    assert s.tinyvision_label == "failing"


def test_verified_requires_human_pipeline_provenance():
    s = Sample(sample_id="s-1", recording_id="r-1", environment_id="e-1")
    with pytest.raises(TransitionError):
        set_verification(s, "verified", "source_auto", tinyvision_label="failing")


def test_rejected_then_reverifiable_by_force():
    s = Sample(sample_id="s-1", recording_id="r-1", environment_id="e-1")
    set_verification(s, "rejected", "human:stage12d")
    with pytest.raises(TransitionError):
        set_verification(s, "pending", "human:stage12d")
    from tv.envdata.state import force_verification
    force_verification(s, "pending", "human:stage12d", reason="reviewer error")
    assert s.verification == "pending"


# ---------------------------------------------------------------- readiness

def test_readiness_gate_deltas():
    holdings = HoldingsSnapshot(
        updated="2026-09-07",
        assets=[HoldingAsset(name="t", kind="train", normal_envs=5, failure_envs=2)],
    )
    staged = []
    report = compute_readiness(holdings, staged)
    assert report["training_ready"] is False
    assert report["gates"]["training_environments_min"]["current"] == 7
    assert report["gates"]["training_environments_min"]["delta"] == 13


def test_readiness_counts_verified_failure_envs():
    holdings = HoldingsSnapshot(updated="2026-09-07", assets=[])
    env = _env()
    env.split_assignment = "train"
    env.metadata["failure_type"] = "under_extrusion"
    m = Manifest(dataset=_dataset(), environments=[env], recordings=[])
    samples = [
        Sample(sample_id="s-1", recording_id="r-1",
               environment_id=env.environment_id,
               verification="verified", tinyvision_label="failing",
               tinyvision_failure_type="under_extrusion",
               verification_provenance="human:stage12d"),
    ]
    report = compute_readiness(holdings, [StagedManifestView(m, samples)])
    assert report["breakdown"]["staged"]["verified_failure_envs"] == 1
    assert report["gates"]["new_failure_envs_human_verified_min"]["current"] == 1


def test_failure_type_coverage_includes_holdings():
    holdings = HoldingsSnapshot(
        updated="2026-09-07",
        assets=[HoldingAsset(name="t", kind="train",
                             failure_types={"spaghetti": 3})],
    )
    report = compute_readiness(holdings, [])
    assert report["breakdown"]["failure_type_coverage"]["spaghetti"]["met"] is True


# ---------------------------------------------------------------- trier/bambu

def test_trier_index_missing_tree(tmp_path):
    idx = index_trier_tree(tmp_path / "nonexistent")
    assert idx["present"] is False


def test_trier_index_file_tree(tmp_path):
    (tmp_path / "defect_free" / "camera1").mkdir(parents=True)
    (tmp_path / "defect_free" / "camera2").mkdir(parents=True)
    (tmp_path / "defect_free" / "camera1" / "a.jpg").write_bytes(b"x")
    (tmp_path / "defect_free" / "camera2" / "b.jpg").write_bytes(b"x")
    idx = index_trier_tree(tmp_path)
    assert idx["total_images"] == 2
    assert idx["by_category"]["normal"] == 2


def test_zenodo_characterization_parses_three_angles():
    record = {
        "id": 18628777,
        "doi": "10.5281/zenodo.18628777",
        "created": "2026-02-13T10:00:00+00:00",
        "modified": "2026-03-19T10:00:00+00:00",
        "metadata": {
            "title": "QC in AM",
            "version": "2.0.0",
            "publication_date": "2026-02-13",
            "license": {"id": "cc-by-4.0"},
            "description": "captured from three different camera angles during production.",
        },
        "files": [{"key": "base_data.zip", "size": 1000, "checksum": "md5:x"}],
    }
    c = characterize_from_zenodo(record)
    assert c["license"] == "cc-by-4.0"
    assert c["observed_claims"]["n_camera_angles"] == 3
    assert c["environment_verified"] is False


def test_bambu_inventory_counts(tmp_path):
    (tmp_path / "timelapses.csv").write_text(
        "user,duration,extension,num_frames,file_name\n"
        "borillion,5.0,.mp4,134,train/timelapses/borillion/a.mp4\n"
        "v2thegreat,5.0,.mp4,134,train/timelapses/v2thegreat/b.mp4\n"
    )
    (tmp_path / "videos.csv").write_text(
        "user,duration,extension,num_frames,file_name\n"
        "borillion,250.0,.mp4,7500,train/videos/borillion/c.mp4\n"
    )
    inv = build_inventory(tmp_path / "timelapses.csv", tmp_path / "videos.csv")
    assert inv["n_contributors"] == 2
    assert inv["n_environment_candidates"] == 1  # only borillion has continuous video
    videos = {s["name"]: s for s in inv["splits"]}
    assert videos["videos"]["rows"] == 1


# ---------------------------------------------------------------- duplicates

def test_duplicate_md5_detected(tmp_path):
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_bytes(b"abc")
    b.write_bytes(b"abc")
    v = Validator()
    probe = {"width": 640, "height": 480, "duration_s": 10.0, "fps": 30.0}
    flag_obvious_duplicates(
        v, [("r-a", "e-1", str(a), probe), ("r-b", "e-1", str(b), probe)]
    )
    assert any(f.code == "exact_duplicate" for f in v.flags)