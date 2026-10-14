import csv

from tv.data.manifest import Manifest, Source
from tv.data.streams import (
    build_streams,
    split_streams,
    stream_summary,
    write_streams,
)


def _source(tmp, sid, weak_label, nframes):
    d = tmp / sid / "frames"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(nframes):
        (d / f"frame_{i:04d}.jpg").write_bytes(b"x")
    return Source(id=sid, url=f"https://y/{sid}", title="t", uploader="u",
                  duration=float(nframes), width=32, height=32,
                  retrieved_at="2026-10-14", license_note="research_only",
                  weak_label=weak_label, frames_dir=str(d), frame_count=nframes)


def test_build_streams_uses_weak_label_then_proposal(tmp_path):
    m = Manifest(version=1, created_at="2026-10-14",
                 sources=[_source(tmp_path, "a", "uncertain", 3),
                          _source(tmp_path, "b", "normal", 2),
                          _source(tmp_path, "c", "uncertain", 2)])
    proposals = {"sources": [
        {"id": "a", "video_label": "failing", "counts": {}},
        {"id": "c", "video_label": "uncertain", "counts": {}},
    ]}
    streams = build_streams(m, proposals=proposals)
    by_id = {s.id: s for s in streams}
    assert by_id["a"].label == "failing"
    assert by_id["a"].label_source == "vlm"
    assert by_id["b"].label == "normal"
    assert by_id["b"].label_source == "title"
    assert "c" not in by_id  # both uncertain -> excluded


def test_build_streams_keeps_uncertain_when_asked(tmp_path):
    m = Manifest(version=1, created_at="2026-10-14",
                 sources=[_source(tmp_path, "c", "uncertain", 2)])
    streams = build_streams(m, proposals=None, include_uncertain=True)
    assert streams[0].label == "uncertain"


def test_write_streams_csv_and_json(tmp_path):
    m = Manifest(version=1, created_at="2026-10-14",
                 sources=[_source(tmp_path, "a", "failing", 3)])
    streams = build_streams(m)
    csv_path = tmp_path / "streams.csv"
    json_path = tmp_path / "streams.json"
    write_streams(streams, csv_path, json_path)
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 3
    assert rows[0]["label"] == "failing"
    assert rows[0]["stream"] == "a"
    assert "path" in rows[0]
    summary = stream_summary(streams)
    assert summary["n_streams"] == 1
    assert summary["n_frames"] == 3
    assert summary["by_label"]["failing"] == 1


def test_split_streams_no_leakage(tmp_path):
    m = Manifest(version=1, created_at="2026-10-14",
                 sources=[_source(tmp_path, f"s{i}", "normal" if i % 2 else "failing", 2)
                          for i in range(10)])
    streams = build_streams(m)
    out = tmp_path / "splits"
    stats = split_streams(streams, out, ratios=(0.6, 0.2, 0.2), seed=1,
                          data_dir=str(tmp_path))
    assert stats["train"] + stats["val"] + stats["test"] == 20
    seen = set()
    for name in ("train", "val", "test"):
        rows = list(csv.DictReader((out / f"{name}.csv").open()))
        streams_here = {r["stream"] for r in rows}
        assert not (streams_here & seen), f"stream leaked across splits: {name}"
        seen |= streams_here
    assert (out / "splits.json").is_file()
