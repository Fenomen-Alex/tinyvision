"""Tests for the data splitter."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from tv.data.splitter import split_dataset

@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    """Create a small synthetic dataset for testing."""
    data_dir = tmp_path / "data"
    # Layout A: source directories
    for label, n_sources, files_per in [("normal", 5, 3), ("failing", 3, 2)]:
        for s in range(n_sources):
            src = data_dir / label / f"source_{s}"
            src.mkdir(parents=True)
            for f in range(files_per):
                (src / f"img_{f:03d}.jpg").write_bytes(b"\xff")

    # Layout B: flat files in a label dir
    flat_label = data_dir / "flat_label"
    flat_label.mkdir(parents=True)
    for i in range(6):
        (flat_label / f"img_{i:03d}.jpg").write_bytes(b"\xff")

    return data_dir


def test_split_produces_valid_output(sample_dataset: Path, tmp_path: Path) -> None:
    output = tmp_path / "splits"
    result = split_dataset(sample_dataset, output, seed=42)

    assert output.exists()
    assert (output / "splits.json").exists()
    for split_name in ("train", "val", "test"):
        assert (output / f"{split_name}.csv").exists()

    stats = result.stats
    total = sum(stats.values())
    assert total > 0
    assert stats["train"] >= stats["val"]
    assert stats["train"] >= stats["test"]


def test_no_leakage(sample_dataset: Path, tmp_path: Path) -> None:
    """Same source should not appear in multiple splits."""
    result = split_dataset(sample_dataset, tmp_path / "splits", seed=42)

    for label in result.train:
        train_files = set(result.train.get(label, []))
        val_files = set(result.val.get(label, []))
        test_files = set(result.test.get(label, []))
        assert train_files.isdisjoint(val_files), f"Leakage in {label}: train ∩ val"
        assert train_files.isdisjoint(test_files), f"Leakage in {label}: train ∩ test"
        assert val_files.isdisjoint(test_files), f"Leakage in {label}: val ∩ test"


def test_deterministic_with_seed(sample_dataset: Path, tmp_path: Path) -> None:
    r1 = split_dataset(sample_dataset, tmp_path / "a", seed=123)
    r2 = split_dataset(sample_dataset, tmp_path / "b", seed=123)
    assert r1.stats == r2.stats


def test_sums_match(sample_dataset: Path, tmp_path: Path) -> None:
    result = split_dataset(sample_dataset, tmp_path / "splits", seed=42)
    stats = result.stats
    total = sum(stats.values())
    # Should match actual file count
    actual = sum(1 for _ in sample_dataset.rglob("*.jpg"))
    assert total == actual


def test_flat_layout(tmp_path: Path) -> None:
    """Flat files (no source dirs) should work too."""
    data = tmp_path / "data"
    for label in ("ok", "bad"):
        d = data / label
        d.mkdir(parents=True)
        for i in range(4):
            (d / f"img_{i}.jpg").write_bytes(b"\xff")

    result = split_dataset(data, tmp_path / "splits", seed=0)
    assert sum(result.stats.values()) == 8


def test_source_atomicity_layout_a(tmp_path: Path) -> None:
    data = tmp_path / "data"
    # 6 sources per class so each split can be populated
    for label in ("normal", "failing"):
        for s in range(6):
            src = data / label / f"src{s}"
            src.mkdir(parents=True)
            for i in range(5):
                (src / f"{i}.jpg").write_bytes(b"\xff")

    result = split_dataset(data, tmp_path / "splits", seed=3)

    def source_of(path: str) -> str:
        # path .../label/srcNN/i.jpg
        return str(Path(path).parent)

    # Build source -> split membership for each label
    for split_name, split_data in [
        ("train", result.train), ("val", result.val), ("test", result.test),
    ]:
        for label, files in split_data.items():
            seen_sources = set()
            for f in files:
                src = source_of(f)
                # a source can appear multiple times in the same split (fine),
                # but every occurrence must be in THIS split for its label
                assert src not in seen_sources or True
                seen_sources.add(src)
            # assert no source is split across splits: each file's source is
            # present in exactly one split
    # Verify no source directory appears in two different splits
    from collections import defaultdict
    source_splits = defaultdict(set)
    for split_name, split_data in [
        ("train", result.train), ("val", result.val), ("test", result.test),
    ]:
        for label, files in split_data.items():
            for f in files:
                source_splits[source_of(f)].add(split_name)
    for src, splits in source_splits.items():
        assert len(splits) == 1, f"Source {src} leaked across splits {splits}"


def test_cross_class_source_atomicity(tmp_path: Path) -> None:
    """The same source must stay in the same split even across classes.

    A recording may contain frames of both classes (e.g. a print that starts
    normal then fails). The source ID (subdirectory name) is identical across
    label dirs, so the splitter must assign it to exactly one split.
    """
    data = tmp_path / "data"
    recordings = ["recA", "recB", "recC", "recD"]
    for rec in recordings:
        for label in ("normal", "failing"):
            src = data / label / rec
            src.mkdir(parents=True)
            for i in range(4):
                (src / f"{i}.jpg").write_bytes(b"\xff")

    result = split_dataset(data, tmp_path / "splits", seed=7)

    recording_split: dict[str, str] = {}
    for split_name, split_data in [
        ("train", result.train), ("val", result.val), ("test", result.test),
    ]:
        for label, files in split_data.items():
            for f in files:
                rec = Path(f).parent.name
                if rec in recording_split:
                    assert recording_split[rec] == split_name, \
                        f"Recording {rec} leaked across splits"
                else:
                    recording_split[rec] = split_name

    # Every class's files must follow the same recording->split assignment
    for label in ("normal", "failing"):
        for split_name, split_data in [
            ("train", result.train), ("val", result.val), ("test", result.test),
        ]:
            for f in split_data.get(label, []):
                rec = Path(f).parent.name
                assert recording_split[rec] == split_name


def test_split_discovers_nested_frames_subdir(tmp_path: Path) -> None:
    """Layout A with a nested frames/ subdir (target-camera capture) must split
    at the session level and discover frames inside each session's frames/ dir."""
    from tv.data.splitter import build_holdout, split_dataset

    data = tmp_path / "data"
    # normal: frames directly under session dir (legacy) AND under frames/
    src = data / "normal" / "session_legacy"
    src.mkdir(parents=True)
    for i in range(3):
        (src / f"{i}.jpg").write_bytes(b"\xff")
    # failing: session with a frames/ subdir + metadata.json that must be ignored
    for s in range(1, 6):
        sess = data / "failing" / f"session_{s:03d}"
        (sess / "frames").mkdir(parents=True)
        for i in range(4):
            (sess / "frames" / f"{i}.jpg").write_bytes(b"\xff")
        (sess / "metadata.json").write_text('{"session_id": "x"}')

    result = split_dataset(data, tmp_path / "splits", seed=0)
    # 3 normal + 5*4 failing = 23 image files total
    assert sum(result.stats.values()) == 23
    # metadata.json must never be treated as a frame
    assert not any("metadata.json" in f for f in result.train["failing"])
    assert not any("metadata.json" in f for f in result.test["failing"])

    # session-level isolation: every failing session stays in one split
    from collections import defaultdict
    sess_splits = defaultdict(set)
    for split_name, sd in [("train", result.train), ("val", result.val), ("test", result.test)]:
        for f in sd.get("failing", []):
            sess_splits[Path(f).parts[-3]].add(split_name)  # label/session/frames/file
    assert all(len(s) == 1 for s in sess_splits.values())

    # holdout puts everything into a single test set
    hold = build_holdout(data, tmp_path / "holdout")
    assert sum(hold.stats.values()) == 23
    assert hold.train == {} and hold.val == {}
    assert sum(len(v) for v in hold.test.values()) == 23
    assert (tmp_path / "holdout" / "test.csv").exists()
    assert (tmp_path / "holdout" / "splits.json").exists()


def test_holdout_relative_paths(tmp_path: Path) -> None:
    from tv.data.splitter import build_holdout

    data = tmp_path / "data"
    for label in ("normal", "failing"):
        sess = data / label / "session_001"
        (sess / "frames").mkdir(parents=True)
        for i in range(2):
            (sess / "frames" / f"{i}.jpg").write_bytes(b"\xff")

    result = build_holdout(data, tmp_path / "out")
    # paths must be relative to data_dir so loaders resolve against splits.json
    for f in result.test["normal"] + result.test["failing"]:
        assert not Path(f).is_absolute()
        assert str(Path(data / f).resolve()).startswith(str(data.resolve()))

