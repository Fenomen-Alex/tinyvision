#!/usr/bin/env python3
"""Stage 12C: build the external pilot dataset (training-ready folders, metadata, splits).

Usage (from inside the training venv):
    python scripts/build_external_pilot.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))

from tv.external_pilot import EXT, build_external_pilot  # noqa: E402


def main() -> int:
    summary = build_external_pilot()

    print("\n--- Stage 12C build summary ---")
    for key in ("total_extracted", "labeled_normal", "labeled_failing", "uncertain",
                "duplicate_count"):
        print(f"  {key}: {summary[key]}")
    print("  split assignment:")
    for sid, split in sorted(summary["splits"].items()):
        print(f"    {sid}: {split}")
    out = EXT / "metadata" / "external_pilot_build_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWROTE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())