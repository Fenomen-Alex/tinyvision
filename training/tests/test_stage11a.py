"""Tests for tv/stage11a.py (leave-cluster-out experiment design)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from tv.stage11a import (
    Cluster,
    ExperimentalRun,
    LeakageAuditResult,
    extract_date,
    extract_recording_id,
    load_all_recordings,
    define_stringing_clusters,
    define_under_extrusion_clusters,
    design_runs,
    run_leakage_audit,
    generate_run_csvs,
    build_design,
)


# ---------------------------------------------------------------------------
# extract_date
# ---------------------------------------------------------------------------

def test_extract_date_standard():
    assert extract_date("Recording_22122022_11_00_38") == "22122022"


def test_extract_date_dotformat():
    assert extract_date("Recording_14.02.2023_17_53_55") == "14.02.2023"


def test_extract_date_no_prefix():
    assert extract_date("other_id") is None


# ---------------------------------------------------------------------------
# extract_recording_id
# ---------------------------------------------------------------------------

def test_extract_recording_id():
    assert extract_recording_id("failing/Recording_01122022_16_06_44/frame.png") == "Recording_01122022_16_06_44"


def test_extract_recording_id_short():
    assert extract_recording_id("a/b") is None


# ---------------------------------------------------------------------------
# load_all_recordings (real data)
# ---------------------------------------------------------------------------

def test_load_all_recordings_real():
    data = load_all_recordings()
    assert "train" in data
    assert "val" in data
    assert "test" in data
    assert len(data["train"]) > 80
    assert len(data["val"]) > 10
    assert len(data["test"]) > 10


def test_train_stringing_count_real():
    data = load_all_recordings()
    str_recs = [r for r in data["train"] if r.has_stringing]
    assert len(str_recs) == 23


def test_recording_info_composition():
    data = load_all_recordings()
    for rec in data["train"]:
        assert rec.composition in ("PURE_NORMAL", "PURE_FAILING", "MIXED")
        assert rec.n_frames > 0


# ---------------------------------------------------------------------------
# define_stringing_clusters (real data)
# ---------------------------------------------------------------------------

def test_stringing_clusters_real():
    data = load_all_recordings()
    clusters = define_stringing_clusters(data["train"])
    # Should have 6 date clusters
    assert len(clusters) == 6
    # 22122022 should be the largest
    assert "22122022" in clusters
    assert len(clusters["22122022"].recordings) == 9
    assert clusters["22122022"].stringing_frames == 1260
    # 20122022 should exist
    assert "20122022" in clusters
    assert len(clusters["20122022"].recordings) == 5
    # 15.02.2023 should be exploratory (1 recording)
    assert clusters["15.02.2023"].exploratory is True


def test_stringing_cluster_dates():
    data = load_all_recordings()
    clusters = define_stringing_clusters(data["train"])
    dates = sorted(clusters.keys())
    assert dates == ["14.02.2023", "15.02.2023", "20122022", "21122022", "22122022", "23.02.2023"]


# ---------------------------------------------------------------------------
# define_under_extrusion_clusters (real data)
# ---------------------------------------------------------------------------

def test_under_extrusion_clusters_real():
    data = load_all_recordings()
    clusters = define_under_extrusion_clusters(data["train"])
    assert len(clusters) == 13  # 13 dates
    # 10012023 should have 8 recordings
    assert "10012023" in clusters
    assert len(clusters["10012023"].recordings) == 8
    # 05012023 should have 7 recordings
    assert "05012023" in clusters
    assert len(clusters["05012023"].recordings) == 7


# ---------------------------------------------------------------------------
# design_runs (real data)
# ---------------------------------------------------------------------------

def test_design_runs_real():
    data = load_all_recordings()
    str_cl = define_stringing_clusters(data["train"])
    ue_cl = define_under_extrusion_clusters(data["train"])
    runs = design_runs(data["train"], data["val"], str_cl, ue_cl)
    # Should have 4 runs: 2 stringing + 2 under-extrusion
    assert len(runs) == 4
    run_ids = [r.run_id for r in runs]
    assert "str_holdout_22122022" in run_ids
    assert "str_holdout_20122022" in run_ids
    assert "ue_control_10012023" in run_ids
    assert "ue_control_05012023" in run_ids


def test_run_held_out_recordings():
    data = load_all_recordings()
    str_cl = define_stringing_clusters(data["train"])
    ue_cl = define_under_extrusion_clusters(data["train"])
    runs = design_runs(data["train"], data["val"], str_cl, ue_cl)
    run_22 = next(r for r in runs if r.run_id == "str_holdout_22122022")
    assert len(run_22.held_out_recordings) == 9
    assert run_22.held_out_frame_count == 1260
    assert run_22.train_recording_count == 96  # 105 - 9


def test_run_train_excludes_held_out():
    data = load_all_recordings()
    str_cl = define_stringing_clusters(data["train"])
    ue_cl = define_under_extrusion_clusters(data["train"])
    runs = design_runs(data["train"], data["val"], str_cl, ue_cl)
    all_train_ids = {r.recording_id for r in data["train"]}
    for run in runs:
        held_out = set(run.held_out_recordings)
        # Held-out recordings should be a subset of all train recordings
        assert held_out.issubset(all_train_ids)
        # Train count should exclude held-out
        assert run.train_recording_count == len(all_train_ids - held_out)


# ---------------------------------------------------------------------------
# leakage audit (real data)
# ---------------------------------------------------------------------------

def test_leakage_audit_passes():
    data = load_all_recordings()
    str_cl = define_stringing_clusters(data["train"])
    ue_cl = define_under_extrusion_clusters(data["train"])
    runs = design_runs(data["train"], data["val"], str_cl, ue_cl)
    for run in runs:
        audit = run_leakage_audit(run, data["train"], data["val"], data["test"])
        assert audit.passed is True, f"Leakage audit failed for {run.run_id}"
        assert len(audit.checks) == 8


def test_leakage_audit_catches_held_out_in_train():
    """Verify audit would catch a held-out recording if it leaked into held-in train."""
    data = load_all_recordings()
    str_cl = define_stringing_clusters(data["train"])
    ue_cl = define_under_extrusion_clusters(data["train"])
    runs = design_runs(data["train"], data["val"], str_cl, ue_cl)
    run_22 = next(r for r in runs if r.run_id == "str_holdout_22122022")
    # Create a bad run: held_out is a proper subset, but train_recording_count
    # is wrong (claims held-in count is 105, i.e. includes held-out)
    bad_run = ExperimentalRun(
        run_id="bad",
        held_out_cluster_id="22122022",
        held_out_recordings=run_22.held_out_recordings[:1],  # 1 recording
        held_out_frame_count=100,
        held_out_stringing_frames=100,
        train_recording_count=105,  # wrong: should be 104
        train_frame_count=11254,
        train_stringing_frame_count=2341,
        val_recording_count=22,
        val_frame_count=2174,
    )
    audit = run_leakage_audit(bad_run, data["train"], data["val"], data["test"])
    assert audit.passed is False
    l1 = next(c for c in audit.checks if c["id"] == "L1")
    assert l1["passed"] is False


# ---------------------------------------------------------------------------
# generate_run_csvs (real data)
# ---------------------------------------------------------------------------

def test_generate_run_csvs(tmp_path):
    data = load_all_recordings()
    str_cl = define_stringing_clusters(data["train"])
    ue_cl = define_under_extrusion_clusters(data["train"])
    runs = design_runs(data["train"], data["val"], str_cl, ue_cl)
    run_22 = next(r for r in runs if r.run_id == "str_holdout_22122022")

    out = tmp_path / "run_22122022"
    paths = generate_run_csvs(run_22, data["train"], data["val"], out)

    assert paths["train"].exists()
    assert paths["val"].exists()
    assert paths["splits"].exists()

    import csv as csvmod
    train_rows = list(csvmod.DictReader(open(paths["train"])))
    val_rows = list(csvmod.DictReader(open(paths["val"])))

    # Train should have fewer frames than original
    assert len(train_rows) < 11254
    # No held-out recording should appear in train
    held_out = set(run_22.held_out_recordings)
    for row in train_rows:
        rec_id = row["path"].split("/")[1]
        assert rec_id not in held_out

    # Val should exclude the held-out date
    for row in val_rows:
        rec_id = row["path"].split("/")[1]
        date = extract_date(rec_id)
        assert date != "22122022"


# ---------------------------------------------------------------------------
# build_design (real data, end-to-end)
# ---------------------------------------------------------------------------

def test_build_design_structure():
    d = build_design()
    assert d["stage"] == "11A"
    assert d["no_training"] is True
    assert "recording_inventory" in d
    assert "stringing_recording_table" in d
    assert "stringing_clusters" in d
    assert "experimental_runs" in d
    assert "leakage_audits" in d
    assert "control_condition" in d
    assert "training_config" in d
    assert "metrics" in d
    assert "success_criteria" in d
    assert "test_firewall" in d
    assert "execution_plan" in d


def test_build_design_data_bounds():
    d = build_design()
    inv = d["recording_inventory"]["train"]
    assert inv["total_recordings"] == 105
    assert inv["stringing_recordings"] == 23
    assert inv["under_extrusion_recordings"] == 37


def test_build_design_clusters_correct():
    d = build_design()
    str_cl = d["stringing_clusters"]
    assert len(str_cl) == 6
    assert str_cl["22122022"]["recording_count"] == 9
    assert str_cl["22122022"]["exploratory"] is False
    assert str_cl["15.02.2023"]["exploratory"] is True


def test_build_design_runs_correct():
    d = build_design()
    runs = d["experimental_runs"]
    assert len(runs) == 4
    r22 = runs["str_holdout_22122022"]
    assert r22["held_out_frame_count"] == 1260
    assert r22["train_recording_count"] == 96


def test_build_design_leakage_all_pass():
    d = build_design()
    for run_id, audit in d["leakage_audits"].items():
        assert audit["passed"] is True, f"Leakage failed for {run_id}"


def test_build_design_training_config_matches_experiment1():
    d = build_design()
    tc = d["training_config"]
    assert tc["model"] == "shufflenet_v2_x1_0"
    assert tc["image_size"] == 224
    assert tc["optimizer"] == "adam"
    assert tc["learning_rate"] == 0.001
    assert tc["epochs"] == 30
    assert tc["seed"] == 42
    assert tc["no_hp_search"] is True
