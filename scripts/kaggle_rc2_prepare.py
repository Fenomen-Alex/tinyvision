#!/usr/bin/env python
"""Prepare a small, printer-disjoint RC2 experiment on the Kaggle dataset.

Reads the full leakage-safe Kaggle splits (stream = print_id, environment =
printer_id) and writes a run directory whose training split is capped to
``--train-per-stream`` evenly-spaced frames per print (CPU-feasible). Evaluation
surfaces are left complete so the reliability suite's ``min_samples`` floors are
met. All output lives outside Git (under the data root by default).

No images are copied; CSVs reference the extracted dataset by absolute path.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _cap(path: Path, cap: int) -> list[dict]:
    by_stream: dict[str, list[dict]] = defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            by_stream[r["stream"]].append(r)
    out: list[dict] = []
    for sid, rows in by_stream.items():
        rows.sort(key=lambda r: r["path"])
        if cap and len(rows) > cap:
            step = len(rows) / cap
            idxs = sorted({min(len(rows) - 1, int(i * step)) for i in range(cap)})
            rows = [rows[i] for i in idxs]
        out.extend(rows)
    out.sort(key=lambda r: r["path"])
    return out


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "stream"])
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Full Kaggle split dir (train/val/test.csv)")
    ap.add_argument("--out", required=True, help="Run directory to create")
    ap.add_argument("--data-dir", required=True, help="Dataset root for splits.json data_dir")
    ap.add_argument("--train-per-stream", type=int, default=100)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = _cap(src / "train.csv", args.train_per_stream)   # capped
    val = _cap(src / "val.csv", 0)                           # full
    test = _cap(src / "test.csv", 0)                         # full

    _write(out / "splits" / "train.csv", train)
    _write(out / "splits" / "val.csv", val)
    _write(out / "splits" / "test.csv", test)
    # Locked test = held-out printer's failing streams; normal holdout = its
    # normal streams. Disjoint groups -> no leakage between surfaces.
    _write(out / "locked_test_fails.csv", [r for r in test if r["label"] == "failing"])
    _write(out / "holdout_normal_test.csv", [r for r in test if r["label"] == "normal"])

    def printers(rows: list[dict]) -> list[str]:
        return sorted({r["stream"].split("__")[0] for r in rows})

    def streams(rows: list[dict]) -> int:
        return len({r["stream"] for r in rows})

    stats = {
        "data_dir": args.data_dir,
        "grouped_by": "print_id (stream); printer_id (environment, split-disjoint)",
        "train_per_stream_cap": args.train_per_stream,
        "train": {"frames": len(train), "streams": streams(train), "printers": printers(train)},
        "val": {"frames": len(val), "streams": streams(val), "printers": printers(val)},
        "locked_test_fails": {"frames": len([r for r in test if r["label"] == "failing"])},
        "normal_holdout": {"frames": len([r for r in test if r["label"] == "normal"]),
                           "streams": len({r["stream"] for r in test if r["label"] == "normal"})},
    }
    # Leakage assertion: train/val/held-out printers must be mutually disjoint.
    seen: dict[str, set[str]] = {}
    for name, rows in [("train", train), ("val", val), ("held", test)]:
        for p in printers(rows):
            seen.setdefault(p, set()).add(name)
    leaks = {p: sorted(v) for p, v in seen.items() if len(v) > 1}
    if leaks:
        raise SystemExit(f"printer leakage across splits: {leaks}")
    stats["leak_check"] = "ok"
    (out / "splits" / "splits.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
