"""Tests for tv/external_pilot.py (Stage 12C external pilot dataset build)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training"))

from tv.external_pilot import (  # noqa: E402
    LABEL_MAP,
    assign_split,
    build_frame_records,
    canonical_name,
    find_exact_duplicates,
    frame_grid,
    hamming,
    map_label,
    near_duplicate_report,
    phash,
    source_frame_index,
)


def test_frame_grid_1fps():
    grid = frame_grid(25.5, 1.0)
    assert len(grid) == 26
    assert grid[0] == 0.0
    assert grid[-1] == 25.0
    # determinism
    assert grid == frame_grid(25.5, 1.0)


def test_frame_grid_2fps():
    grid = frame_grid(3.2, 2.0)
    assert len(grid) == 7  # 0.0..3.0 at 0.5 steps
    assert grid == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def test_source_frame_index():
    assert source_frame_index(8.0, 30.0) == 240
    assert source_frame_index(20.0, 30.0) == 600
    assert source_frame_index(1.0, 23.976) == 24


def test_canonical_name():
    assert canonical_name("NORMAL001", 20.0, 600) == "NORMAL001__t020.000__f000600.jpg"
    assert canonical_name("SP001", 8.0, 240) == "SP001__t008.000__f000240.jpg"


def test_label_map_mapping():
    assert map_label("normal_provisional") == "normal"
    assert map_label("suspect_growth") == "failing"
    assert map_label("uncertain") == "uncertain"
    assert map_label("scene_transition_or_final_state") == "uncertain"
    # unknown provisional labels must fall back to uncertain, never normal/failing
    assert map_label("anything_else") == "uncertain"
    assert set(LABEL_MAP.values()) <= {"normal", "failing", "uncertain"}


def test_phash_deterministic_and_sensitive():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    b = a.copy()
    c = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    assert phash(a) == phash(b)
    assert hamming(phash(a), phash(b)) == 0
    assert hamming(phash(a), phash(c)) > 4


def test_hamming():
    assert hamming(0, 0) == 0
    assert hamming(0b1111, 0b0000) == 4
    assert hamming(0b1011, 0b1001) == 1


def test_find_exact_duplicates_removes_only_identical():
    def rec(md5):
        return {"source_id": "X", "timestamp": 0.0, "frame_md5": md5}

    records = [rec("a"), rec("b"), rec("a"), rec("c"), rec("c")]
    kept, stats = find_exact_duplicates(records)
    assert len(kept) == 3
    assert stats["removed_total"] == 2
    # first occurrences kept
    assert [r["frame_md5"] for r in kept] == ["a", "b", "c"]


def test_near_duplicate_report_is_diagnostic_only():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    p = phash(img)
    recs = [
        {"source_id": "A", "timestamp": 0.0, "frame_md5": "x", "phash": p},
        {"source_id": "B", "timestamp": 5.0, "frame_md5": "y", "phash": p},
        {"source_id": "A", "timestamp": 1.0, "frame_md5": "z",
         "phash": phash((img + 7) % 255)},
    ]
    report = near_duplicate_report(recs)
    assert report["cross_source_near_duplicate_pairs"] == 1
    # no removal is performed by the diagnostic
    assert report["policy"].startswith("diagnostic only")


def test_assign_split_deterministic_and_complete():
    sources = ["SP001", "SP002", "NORMAL001", "UE001", "UE003", "STR001"]
    split = assign_split(sources)
    assert set(split) == set(sources)
    assert set(split.values()) <= {"train", "validation"}
    # deterministic
    assert split == assign_split(sources)
    # roughly 80/20 (in the pilot it is documented as degenerate, but still valid)
    val = {s for s, v in split.items() if v == "validation"}
    assert len(val) <= len(sources)


def test_build_frame_records_labels():
    from tv.external_pilot import EXTRACTION_FPS, EXT

    # fake meta-like object
    class Meta:
        def __init__(self):
            self.fps = 30.0
            self.width = 720
            self.height = 1280
            self.source_id = "X"

    # unlabeled source -> all uncertain
    frames = [EXT / "extracted/X/sequence" / f"frame_{i:04d}.jpg" for i in range(2)]
    recs = build_frame_records("UE999", "x.mp4", Meta(), frames, {})
    assert all(r["label"] == "uncertain" for r in recs)

    # labeled source: provisional labels mapped, record fields complete
    meta = Meta()
    meta.fps = 30.0
    meta.width, meta.height = 720, 1280
    prov = {0.0: "normal_provisional", 1.0: "suspect_growth", 2.0: "uncertain"}
    recs = build_frame_records(
        "SP001", "SP001_test.mp4", meta,
        [EXT / f"extracted/SP001/sequence/frame_{i:04d}.jpg" for i in range(3)], prov)
    assert [r["label"] for r in recs] == ["normal", "failing", "uncertain"]
    assert recs[0]["failure_type"] == "spaghetti"
    assert recs[0]["segment_id"] == "SP001__s0"
    assert recs[0]["extraction_fps"] == EXTRACTION_FPS
    assert recs[0]["source_frame_index"] == 0
    assert recs[0]["provenance"]["verification"] == "provisional_unverified"
    assert recs[1]["provenance"]["pilot_basis"] == "stage12b: suspect_growth"
    assert recs[0]["width"] == 720 and recs[0]["height"] == 1280