#!/usr/bin/env python3
"""Stage 12C: diversity analysis of the external pilot vs existing Kaggle data.

Usage (from inside the training venv):
    python scripts/analyze_external_pilot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))

from tv.external_pilot import run_diversity_analysis  # noqa: E402


def main() -> int:
    result = run_diversity_analysis()
    print("\n--- Stage 12C diversity analysis vs Kaggle ---")
    print("  kaggle:", result["kaggle"])
    print("  external:", result["external"])
    print("  phash:", result["phash_hamming_distance"])
    print("  external dims:", result["image_dimensions_external"])
    print("\nWROTE datasets/external/video-frames/metadata/external_pilot_diversity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())