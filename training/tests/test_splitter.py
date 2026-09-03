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
    """All frames of one source directory must stay in the same split.

    This is the core leakage guarantee: a print/capture session's frames must
    never straddle train/val/test.
    """
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
