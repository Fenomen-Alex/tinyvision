#!/usr/bin/env python3
"""Build the first SKZ Nozzle-Cam probe (test split, source-grouped).

Usage:
    python scripts/build_skz_probe.py

Downloads only the selected test-split images into datasets/external/skz-audit/
(if missing), copies them + their original annotations into datasets/external/
skz-probe/, and writes manifest.json, build_summary.json and report.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))

from tv.skz_probe import PROBE, PROBE_CLASSES, build_skz_probe  # noqa: E402


def main() -> int:
    summary = build_skz_probe()

    print("\n--- SKZ probe build summary ---")
    for key in ("total_test_images", "downloaded_test_images"):
        print(f"  {key}: {summary[key]}")
    for cls in PROBE_CLASSES:
        c = summary["counts"][cls]
        print(f"  {cls:16s} selected={c['selected']:3d} groups={c['groups']:3d} "
              f"multi_defect={c['multi_defect']}")
    out = PROBE / "build_summary.json"
    print(f"\nWROTE {PROBE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())