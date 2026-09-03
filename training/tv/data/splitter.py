"""Leakage-free dataset splitting by source group."""
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SplitAssignment:
    split: str  # "train", "val", "test"
    source: str
    label: str
    files: list[str]


@dataclass
class SplitResult:
    train: dict[str, list[str]]  # label -> [file paths]
    val: dict[str, list[str]]
    test: dict[str, list[str]]

    @property
    def stats(self) -> dict[str, int]:
        return {
            "train": sum(len(v) for v in self.train.values()),
            "val": sum(len(v) for v in self.val.values()),
            "test": sum(len(v) for v in self.test.values()),
        }


def _discover_sources(data_dir: Path) -> dict[str, dict[str, list[str]]]:
    """Discover sources grouped by label.

    Supports two layouts:
      Layout A (source directories):
          data_dir/label/source_id/file.jpg

      Layout B (flat files):
          data_dir/label/file.jpg
          Each file is treated as its own source.
    """
    sources: dict[str, dict[str, list[str]]] = {}  # label -> source_id -> [files]

    for label_dir in sorted(data_dir.iterdir()):
        if not label_dir.is_dir() or label_dir.name.startswith("."):
            continue
        label = label_dir.name
        sources[label] = {}

        # Check if there are subdirectories (source dirs) or just files
        subdirs = [d for d in label_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        files = sorted(label_dir.glob("*.jpg")) + sorted(label_dir.glob("*.png"))

        if subdirs:
            # Layout A: source directories
            for src_dir in sorted(subdirs):
                src_files = sorted(
                    f for f in src_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
                )
                if src_files:
                    sources[label][f"{label}/{src_dir.name}"] = [str(f) for f in src_files]
        elif files:
            # Layout B: flat files, each is its own source
            for f in files:
                sources[label][f"{label}/{f.stem}"] = [str(f)]

    return sources


def _stratified_source_split(
    sources: dict[str, dict[str, list[str]]],
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    """Split sources into train/val/test while maintaining class balance.

    For each class, sources are shuffled and split according to ratios.
    """
    rng = random.Random(seed)
    train, val, test = {}, {}, {}

    for label, label_sources in sources.items():
        source_ids = list(label_sources.keys())
        rng.shuffle(source_ids)

        n = len(source_ids)
        n_train = max(1, int(n * ratios[0])) if n > 2 else n
        n_val = max(1, int(n * ratios[1])) if n > 2 else 0
        # Ensure test gets the remainder, but at least 0
        n_test = max(0, n - n_train - n_val)
        # Adjust if test is 0 but there are enough sources
        if n_test == 0 and n >= 3 and ratios[2] > 0:
            if n_train > 1:
                n_train -= 1
                n_test = 1

        train_ids = source_ids[:n_train]
        val_ids = source_ids[n_train : n_train + n_val]
        test_ids = source_ids[n_train + n_val :]

        train[label] = [f for sid in train_ids for f in label_sources[sid]]
        val[label] = [f for sid in val_ids for f in label_sources[sid]]
        test[label] = [f for sid in test_ids for f in label_sources[sid]]

    return train, val, test


def split_dataset(
    data_dir: str | Path,
    output_dir: str | Path,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> SplitResult:
    """Split a classification dataset into train/val/test.

    Sources (video prints / capture sessions) are split at the group level
    to prevent data leakage.

    Args:
        data_dir: Root directory with layout ``label/file.jpg`` or
            ``label/source/file.jpg``.
        output_dir: Where to write ``splits.json`` and per-split CSVs.
        ratios: Train / val / test fractions. Must sum to ~1.0.
        seed: Random seed for reproducibility.

    Returns:
        SplitResult with file lists per split and label.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = _discover_sources(data_dir)
    train, val, test = _stratified_source_split(sources, tuple(ratios), seed)

    result = SplitResult(train=train, val=val, test=test)

    # Write splits.json
    manifest = {
        "data_dir": str(data_dir),
        "ratios": list(ratios),
        "seed": seed,
        "stats": result.stats,
        "splits": {
            split_name: {label: files for label, files in split_data.items()}
            for split_name, split_data in [("train", train), ("val", val), ("test", test)]
        },
    }
    with open(output_dir / "splits.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Write per-split CSVs for easy inspection
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        csv_path = output_dir / f"{split_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label"])
            for label, files in split_data.items():
                for fp in files:
                    writer.writerow([fp, label])

    return result
