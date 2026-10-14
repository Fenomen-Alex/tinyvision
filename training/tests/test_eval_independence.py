"""Guard the environment/recording independence of the reliability suite.

The split CSVs are git-ignored and absent in some checkouts, so this test skips
rather than fails when they are missing. When present it asserts the two
`printing-errors` surfaces are recording-disjoint (frames from one recording
must never straddle the dev/locked boundary).
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPLITS = REPO / "datasets" / "processed" / "printing-errors" / "splits"


def _recording_units(csv_path: Path) -> set[str]:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    units = set()
    for r in rows:
        parts = r["path"].split("/")
        units.add(parts[1] if len(parts) >= 3 else r["path"])
    return units


def test_dev_and_locked_surfaces_are_recording_disjoint():
    val = SPLITS / "val.csv"
    test = SPLITS / "test.csv"
    if not val.exists() or not test.exists():
        pytest.skip("release split CSVs are git-ignored / absent in this checkout")
    val_units = _recording_units(val)
    test_units = _recording_units(test)
    assert val_units, "val split has no recording units"
    assert test_units, "test split has no recording units"
    overlap = val_units & test_units
    assert not overlap, f"recording leakage between dev_val and frozen_test: {sorted(overlap)}"
