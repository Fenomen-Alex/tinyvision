from tv.data.manifest import (
    Manifest,
    Source,
    load_manifest,
    save_manifest,
    validate_manifest,
)


def _src(tmp, sid="abc", frames=True):
    d = tmp / sid / "frames"
    if frames:
        d.mkdir(parents=True, exist_ok=True)
        (d / "frame_0000.jpg").write_bytes(b"x")
    return Source(id=sid, url=f"https://y/{sid}", title="t", uploader="u",
                  duration=10.0, width=256, height=144, retrieved_at="2026-10-14",
                  license_note="research_only", weak_label="normal",
                  frames_dir=str(d), frame_count=1 if frames else 0)


def test_roundtrip(tmp_path):
    m = Manifest(version=1, created_at="2026-10-14", sources=[_src(tmp_path)])
    p = tmp_path / "manifest.json"
    save_manifest(m, p)
    assert load_manifest(p) == m


def test_validate_flags_missing_dir_and_zero_frames(tmp_path):
    m = Manifest(version=1, created_at="2026-10-14",
                 sources=[_src(tmp_path, "miss", frames=False),
                          _src(tmp_path, "zero", frames=True)])
    m.sources[1].frame_count = 0
    warnings = validate_manifest(m)
    assert any("miss" in w for w in warnings)
    assert any("zero" in w for w in warnings)


def test_validate_flags_duplicate_ids(tmp_path):
    m = Manifest(version=1, created_at="2026-10-14",
                 sources=[_src(tmp_path, "dup"), _src(tmp_path, "dup")])
    assert any("duplicate" in w.lower() for w in validate_manifest(m))
