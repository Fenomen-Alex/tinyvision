#!/usr/bin/env python3
"""Audit evaluation-surface independence for a reliability suite.

Framework leakage checks can only compare the unit IDs a surface exposes. The
`printing-errors` CSV surfaces expose only `path,label`, so this script derives
a recording/venue unit from the path and reports:

  * the recording IDs per surface,
  * pairwise recording overlap between surfaces (must be empty), and
  * the number of distinct recording units (a proxy for environment diversity).

A suite whose surfaces are recording-disjoint but drawn from a single venue is
*not* an environment-independent generalization test; this tool states that
explicitly rather than implying independence.

Usage:
    python scripts/audit_eval_independence.py \
        --suite training/configs/reliability/suite.yaml \
        --dataset-root datasets/processed/printing-errors \
        --out docs/research/evaluation-independence.json
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from datetime import date
from pathlib import Path

import yaml


def _unit_from_path(rel: str) -> str:
    """Heuristic recording unit: the first path component after the class dir.

    e.g. ``failing/Recording_03012023_15_07_03/...png`` -> ``Recording_03012023_15_07_03``.
    """
    parts = rel.split("/")
    return parts[1] if len(parts) >= 3 else rel


def _csv_units(csv_path: Path) -> list[str]:
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return sorted({_unit_from_path(r["path"]) for r in rows if r.get("path")})


def audit(suite_path: Path, dataset_root: Path) -> dict:
    suite = yaml.safe_load(suite_path.read_text())
    surfaces: dict[str, dict] = {}
    for s in suite.get("surfaces", []):
        if s.get("format") != "csv":
            surfaces[s["id"]] = {
                "role": s.get("role"),
                "format": s.get("format"),
                "unit_analysis": "not-analysed (non-CSV surface)",
            }
            continue
        csv_path = dataset_root / s["path"]
        if not csv_path.exists():
            surfaces[s["id"]] = {
                "role": s.get("role"),
                "format": "csv",
                "error": f"CSV not found: {csv_path}",
            }
            continue
        units = _csv_units(csv_path)
        surfaces[s["id"]] = {
            "role": s.get("role"),
            "csv": str(csv_path),
            "n_recording_units": len(units),
            "recording_units": units,
        }

    pairwise = []
    analysed = {k: v for k, v in surfaces.items() if "recording_units" in v}
    for a, b in itertools.combinations(sorted(analysed), 2):
        overlap = set(analysed[a]["recording_units"]) & set(analysed[b]["recording_units"])
        pairwise.append(
            {
                "a": a,
                "b": b,
                "recording_overlap": len(overlap),
                "overlap_units": sorted(overlap),
                "recording_disjoint": not overlap,
            }
        )

    all_units: set[str] = set()
    for v in analysed.values():
        all_units.update(v["recording_units"])

    return {
        "schema": "tinyvision-eval-independence",
        "generated": date.today().isoformat(),
        "suite": str(suite_path),
        "dataset_root": str(dataset_root),
        "surfaces": surfaces,
        "pairwise": pairwise,
        "total_distinct_recording_units": len(all_units),
        "environment_independent": False,
        "verdict": (
            "Recording-disjoint surfaces drawn from a single camera venue are "
            "NOT an environment-independent test. Treat cross-venue/cross-normal "
            "generalization (Trier, stage13 holdout) as a separate, currently "
            "unavailable surface."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="training/configs/reliability/suite.yaml")
    ap.add_argument("--dataset-root", default="datasets/processed/printing-errors")
    ap.add_argument("--out", default="docs/research/evaluation-independence.json")
    args = ap.parse_args()

    report = audit(Path(args.suite), Path(args.dataset_root))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")
    for k, v in report["surfaces"].items():
        n = v.get("n_recording_units", v.get("unit_analysis", v.get("error", "?")))
        print(f"  {k:22} role={v.get('role'):16} units={n}")
    for p in report["pairwise"]:
        print(f"  overlap {p['a']} ∩ {p['b']}: {p['recording_overlap']}")
    print(f"  distinct recording units: {report['total_distinct_recording_units']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
