"""Tests for tv/skz_probe.py (SKZ Nozzle-Cam probe builder)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training"))

from tv.skz_probe import (  # noqa: E402
    PROBE,
    PROBE_CLASSES,
    SCHEMA,
    SCHEMA_VERSION,
    all_failure_types,
    defect_ids,
    derive_class,
    entry_metadata,
    frame_number_of,
    group_key_of,
    grouping_verified_of,
    manifest_document,
    multi_defect,
    parse_annotation_line,
    read_label_classes,
    select_eligible,
    select_samples,
    split_by_group,
)


def _rec(image: str, classes: set[int], group: str | None, frame: int) -> dict:
    return {
        "image": image,
        "stem": image[:-4],
        "label_path": f"datasets/external/skz-audit/test/labels/{image[:-4]}.txt",
        "image_path": f"datasets/external/skz-audit/test/images/{image}",
        "classes": frozenset(classes),
        "derived": derive_class(classes),
        "group": group,
        "frame": frame,
        "annotation_count": len(classes),
    }


# ---------------------------------------------------------------- annotation parsing

def test_parse_annotation_line_valid():
    assert parse_annotation_line("5 0.5 0.5 0.1 0.1") == (5.0, 0.5, 0.5, 0.1, 0.1)
    assert parse_annotation_line("3 0.1 0.2 0.3 0.4")[0] == 3.0


def test_parse_annotation_line_invalid_returns_none():
    assert parse_annotation_line("") is None
    assert parse_annotation_line("   ") is None
    assert parse_annotation_line("a b c d e") is None
    assert parse_annotation_line("1 0.5 0.5 0.1") is None          # 4 fields
    assert parse_annotation_line("9 0.5 0.5 0.1 0.1") is None      # class > 8
    assert parse_annotation_line("3 2.0 0.5 0.1 0.1") is None      # coords out of range


def test_read_label_classes_collects_classes_and_count(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.1 0.1\n5 0.2 0.2 0.1 0.1\n5 0.8 0.8 0.1 0.1\n")
    classes, count = read_label_classes(p)
    assert classes == frozenset({0, 5})
    assert count == 3


def test_read_label_classes_rejects_invalid_line(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.1 0.1\njunk\n")
    with pytest.raises(ValueError):
        read_label_classes(p)


# ---------------------------------------------------------------- class mapping

def test_defect_ids():
    assert defect_ids({0, 1, 2}) == frozenset()
    assert defect_ids({3}) == frozenset({3})
    assert defect_ids({4, 7}) == frozenset({4, 7})


def test_derive_class_single_defect():
    assert derive_class({3}) == "spaghetti"
    assert derive_class({4}) == "stringing"
    assert derive_class({5}) == "under_extrusion"
    assert derive_class({0, 4, 1}) == "stringing"


def test_derive_class_no_defect_is_normal_candidate():
    assert derive_class(set()) == "normal"
    assert derive_class({0, 1, 2}) == "normal"


def test_derive_class_non_probe_defects_excluded():
    assert derive_class({6}) is None      # Warping only
    assert derive_class({7}) is None      # bad first layer only
    assert derive_class({8}) is None      # Double Print only
    assert derive_class({6, 7}) is None


def test_derive_class_priority_multi_defect():
    assert derive_class({3, 5}) == "spaghetti"        # spaghetti > under_extrusion
    assert derive_class({4, 5}) == "under_extrusion"  # under_extrusion > stringing
    assert derive_class({3, 4, 7}) == "spaghetti"     # 7 is non-probe, ignored for class


def test_all_failure_types_ordered_by_class_id():
    assert all_failure_types({5, 3}) == ["Spaghetti", "Unterextrusion"]
    assert all_failure_types({0, 1, 2}) == []
    assert all_failure_types({4, 7}) == ["Stringing", "schlechte erste Schicht"]


def test_multi_defect():
    assert multi_defect({3, 5}) is True
    assert multi_defect({5}) is False
    assert multi_defect({0, 4}) is False  # element boxes do not count


# ---------------------------------------------------------------- normal-candidate handling

def test_normal_never_inferred_from_element_boxes():
    # Presence of Nozzle/Object/Purge-Line boxes must NOT produce normal.
    assert derive_class({0}) == "normal"  # only nozzle box -> candidate (no defect)
    assert derive_class({0, 1, 2}) == "normal"
    # A single defect overrides every element box present.
    assert derive_class({0, 1, 2, 3}) == "spaghetti"
    assert derive_class({0, 4}) == "stringing"


def test_normal_candidate_definition():
    # "candidate" is encoded as derived==normal only when no defect-class box exists.
    assert derive_class({0, 1, 2}) == "normal"
    assert defect_ids({0, 1, 2}) == frozenset()


# ---------------------------------------------------------------- grouping / selection

def test_group_key_of():
    assert (
        group_key_of("Profile_210.0_1.2_40.0_0_40.0_0.2_0.0_(1)__b31b986c-frame_01853.png")
        == "Profile_210.0_1.2_40.0_0_40.0_0.2_0.0_(1)"
    )
    assert group_key_of("Latest-Labelled_Images__01aa8e4c-frame_01094.png") is None
    assert group_key_of("project-15-at-2025-10-21-09-17-e5aecb08__6de19ce7-frame_00354.png") is None
    assert group_key_of("not-a-png") is None


def test_frame_number_of():
    assert frame_number_of("Profile_1.0__(1)__aabbccdd-frame_00042.png") == 42


def test_grouping_verified_always_false():
    assert grouping_verified_of("Profile_210.0_1.2_40.0_0_40.0_0.2_0.0_(1)__b31b986c-frame_01853.png") is False


def _synthetic_normal_group(group: str, n: int) -> list[dict]:
    return [
        _rec(f"{group}__{('0' + '%02d' % i)[-2:] * 4}-frame_{100 + i:05d}.png", {0, 1}, group, 100 + i)
        for i in range(n)
    ]


def test_select_eligible_drops_ungrouped_and_other_defect():
    records = _synthetic_normal_group("Profile_210.0_1.2_40.0_0_40.0_0.2_0.0_(1)", 3)
    records.append(_rec("Latest-Labelled_Images__01aa8e4c-frame_01094.png", {0, 1}, None, 10))
    records.append(_rec("Profile_X__aabbccdd-frame_00001.png", {7}, "Profile_X", 1))
    per_class = select_eligible(records)
    assert sum(len(v) for v in per_class.values()) == 3
    assert all(r["group"] is not None for v in per_class.values() for r in v)
    for cls in PROBE_CLASSES:
        for r in per_class[cls]:
            assert r["derived"] == cls


def test_select_samples_group_preserving_and_deterministic():
    targets = {"normal": 4, "stringing": 50, "spaghetti": 50, "under_extrusion": 50}
    records = _synthetic_normal_group("Profile_A", 3) + _synthetic_normal_group("Profile_B", 2) \
        + _synthetic_normal_group("Profile_C", 1)
    per_class = select_eligible(records)
    first = select_samples(per_class, targets, seed=42)["normal"]
    second = select_samples(per_class, targets, seed=42)["normal"]
    assert len(first) == 4
    assert [r["image"] for r in first] == [r["image"] for r in second]
    groups = {r["group"] for r in first}
    assert groups == {"Profile_A", "Profile_B", "Profile_C"}  # one per group in pass 1
    assert first[3]["selection_rationale"].startswith("pass2")
    assert len({r["image"] for r in first}) == 4  # no duplicates


def test_select_samples_never_exceeds_target_or_duplicates():
    targets = {"normal": 2, "stringing": 50, "spaghetti": 50, "under_extrusion": 50}
    per_class = select_eligible(
        _synthetic_normal_group("Profile_A", 5) + _synthetic_normal_group("Profile_B", 5)
    )
    sel = select_samples(per_class, targets, seed=1)["normal"]
    assert len(sel) == 2
    assert len({r["image"] for r in sel}) == 2


def test_median_frame_representative():
    from tv.skz_probe import _median_record

    recs = [
        _rec("Profile_G__aabbccdd-frame_00001.png", {0, 1}, "Profile_G", 5),
        _rec("Profile_G__aabbccdd-frame_00002.png", {0, 1}, "Profile_G", 3),
        _rec("Profile_G__aabbccdd-frame_00003.png", {0, 1}, "Profile_G", 4),
    ]
    assert _median_record(recs)["frame"] == 4


def test_split_by_group_never_splits_a_group():
    group_counts = {"A": 5, "B": 5, "C": 1, "D": 1}
    split = split_by_group(group_counts, train_fraction=0.8, seed=42)
    assert set(split) == set(group_counts)
    assert split["A"] == split["B"]  # same-size groups cannot straddle splits by design
    # every group maps to exactly one split value
    assert all(v in {"train", "eval"} for v in split.values())


# ---------------------------------------------------------------- manifest provenance

def test_entry_metadata_provenance_fields():
    rec = _rec("Profile_210.0_1.2_40.0_0_40.0_0.2_0.0_(1)__b31b986c-frame_01853.png", {3, 5}, "Profile_210.0_1.2_40.0_0_40.0_0.2_0.0_(1)", 1853)
    rec["probe_seq"] = 0
    rec["selection_rationale"] = "pass1 group representative of Profile_210.0_1.2_40.0_0_40.0_0.2_0.0_(1)"
    e = entry_metadata(rec, "spaghetti", build_time="2026-09-05T00:00:00+00:00")
    assert e["probe_id"] == "SKZNC-spaghetti-000"
    assert e["derived_class"] == "spaghetti"
    assert e["source_image_id"] == rec["image"]
    assert e["source_image_path"].endswith("skz-audit/test/images/Profile_210.0_1.2_40.0_0_40.0_0.2_0.0_(1)__b31b986c-frame_01853.png")
    assert e["source_label_path"].endswith(".txt")
    assert e["defect_class_ids"] == [3, 5]
    assert e["all_failure_types"] == ["Spaghetti", "Unterextrusion"]
    assert e["grouping_key"] == "Profile_210.0_1.2_40.0_0_40.0_0.2_0.0_(1)"
    assert e["grouping_verified"] is False
    assert e["multi_defect"] is True
    assert e["selection_rationale"]
    assert e["generation_timestamp"] == "2026-09-05T00:00:00+00:00"


def test_manifest_document_schema_and_required_keys():
    rec = _rec("Profile_A__aabbccdd-frame_00001.png", {3}, "Profile_A", 1)
    rec["probe_seq"] = 0
    rec["selection_rationale"] = "pass1 group representative of Profile_A"
    entries = [entry_metadata(rec, "spaghetti", build_time="2026-09-05T00:00:00+00:00")]
    doc = manifest_document(
        entries,
        counts={"spaghetti": {"selected": 1, "groups": 1, "multi_defect": 0}},
        excluded={"ungrouped": 0, "other_defect": 0},
        seed=42,
        targets={"normal": 50, "stringing": 50, "spaghetti": 50, "under_extrusion": 50},
        build_time="2026-09-05T00:00:00+00:00",
    )
    assert doc["schema"] == SCHEMA
    assert doc["schema_version"] == SCHEMA_VERSION
    for key in (
        "schema", "schema_version", "generated_at", "source_dataset", "source_loc",
        "mapping_rules", "selection", "excluded", "summary", "entries",
    ):
        assert key in doc
    assert doc["source_dataset"]["license_hf_card_observed"] == "CC BY-SA 4.0"
    assert doc["source_dataset"]["license_zenodo_record_observed"] == "CC BY 4.0"
    assert len(doc["entries"]) == 1


# ---------------------------------------------------------------- integration (skips on fresh clone)

def _probe_manifest() -> Path:
    return PROBE / "manifest.json"


@pytest.mark.skipif(not _probe_manifest().exists(), reason="SKZ probe not built locally")
def test_built_probe_manifest_invariants():
    import json

    doc = json.loads(_probe_manifest().read_text())
    entries = doc["entries"]
    assert doc["schema"] == SCHEMA
    assert doc["schema_version"] == SCHEMA_VERSION
    assert len(entries) == doc["summary"]["total_selected"]
    for e in entries:
        assert e["derived_class"] in PROBE_CLASSES
        assert e["grouping_key"] is not None
        assert e["grouping_verified"] is False
        assert e["image_dimensions"]["width"] > 0 and e["image_dimensions"]["height"] > 0
        assert e["annotation_count"] >= 1
        assert _probe_copies_exist(e)


def _probe_copies_exist(e: dict) -> bool:
    img = PROBE / "images" / e["derived_class"] / Path(e["source_image_id"]).name
    lbl = PROBE / "annotations" / e["derived_class"] / (Path(e["source_image_id"]).stem + ".txt")
    return img.exists() and lbl.exists()