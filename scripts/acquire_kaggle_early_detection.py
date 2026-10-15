#!/usr/bin/env python
"""Register the Kaggle "Early detection of 3D printing issues" dataset.

Dataset: nozzle-camera snapshots of FDM prints.  Print-level binary label
``has_under_extrusion`` (1 = under-extrusion failure, 0 = good/normal print);
all snapshots in one print share the label.  A print is therefore one
*independent* per-print stream; a printer is one capture *environment*.

Licence / access
----------------
The data is governed by the Kaggle **Competition Rules** ("Subject to
Competition Rules"), not an open licence.  It is therefore treated as
RESEARCH-ONLY and NON-REDISTRIBUTABLE: ``noncommercial=True``,
``redistributable=False``; the source images are never committed.

Leakage safety
--------------
Labels are constant within a ``print_id`` and highly correlated with
``printer_id``.  A stream is one ``print_id``; environments (``printer_id``) are
never split across train/val/test, so no environment leaks.

This script only reads CSVs and inspects frame filenames/dimensions; it does not
copy or modify the images.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from tv.data.manifest import Manifest, Source, save_manifest, validate_manifest  # noqa: E402
from tv.data.streams import build_streams, stream_summary  # noqa: E402

COMPETITION_URL = "https://www.kaggle.com/competitions/early-detection-of-3d-printing-issues"
LICENSE = "Subject to Competition Rules"
LICENSE_URL = f"{COMPETITION_URL}/rules"
ATTRIBUTION = ("Kenneth Jiang / Obico — 'Early detection of 3D printing issues' "
               "Kaggle competition")
SOURCE_DATASET = "kaggle-early-detection-of-3d-printing-issues"

# Environment-held-out groups (whole printers).  Chosen so val/test each carry
# both labels and no printer appears in more than one split.
GROUPS = {"train": ["101", "102", "104"], "val": ["022"], "test": ["103"]}


def _dims(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:  # noqa: BLE001
        return 0, 0


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="Extracted dataset root (contains images/, train.csv)")
    ap.add_argument("--out", required=True, help="Output manifest.json path")
    ap.add_argument("--retrieved", default=date.today().isoformat())
    args = ap.parse_args()

    root = Path(args.data_root)
    images = root / "images"
    train = read_csv(root / "train.csv")
    test = read_csv(root / "test.csv")

    # Group train rows by print (stream) and verify the label is constant.
    by_print: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in train:
        by_print[(r["printer_id"], r["print_id"])].append(r)

    sources: list[Source] = []
    for (printer, print_id), rows in sorted(by_print.items()):
        labels = {r["has_under_extrusion"] for r in rows}
        if len(labels) != 1:
            raise SystemExit(f"non-constant label in print {printer}/{print_id}: {labels}")
        label = next(iter(labels))
        # Verify frames exist and are chronological (timestamped names).
        frames = sorted((images / printer / print_id).glob("*.jpg"))
        if len(frames) != len(rows):
            raise SystemExit(
                f"frame mismatch {printer}/{print_id}: {len(frames)} on disk vs "
                f"{len(rows)} in csv")
        w, h = _dims(frames[0])
        sid = f"{printer}__{print_id}"
        group = next((g for g, pids in GROUPS.items() if printer in pids), "")
        sources.append(Source(
            id=sid, url=COMPETITION_URL, title=f"{printer}/{print_id}",
            uploader="kaggle", duration=float(len(frames)), width=w, height=h,
            retrieved_at=args.retrieved,
            license_note="research_only (Subject to Competition Rules)",
            weak_label="normal" if label == "0" else "failing",
            frames_dir=str((images / printer / print_id).resolve()),
            frame_count=len(frames),
            failure_type=None if label == "0" else "under_extrusion",
            frame_glob="*.jpg",
            source_dataset=SOURCE_DATASET, license=LICENSE, license_url=LICENSE_URL,
            attribution=ATTRIBUTION, capture_environment=printer,
            session_id=print_id, video_sha256="",
            noncommercial=True, redistributable=False, split=group,
            notes=("print-level label; nozzle-camera snapshots ~0.5s; "
                   "competition trial data (train.csv)")))

    manifest = Manifest(version=1, created_at=args.retrieved, sources=sources)
    out = Path(args.out)
    save_manifest(manifest, out)

    warnings = validate_manifest(manifest)
    streams = build_streams(manifest)
    summary = stream_summary(streams)

    # Leakage-safe grouped split: environments (printers) never span splits.
    # (``split_streams`` partitions individual streams at random, which would
    # leak a printer across splits; grouping must be done by environment.)
    split_dir = out.parent / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_by_id = {s.id: s.split for s in sources}
    env_by_id = {s.id: s.capture_environment for s in sources}
    stream_of = {f: s.id for s in streams for f in s.frames}
    grouped: dict[str, dict[str, list[str]]] = {"train": {}, "val": {}, "test": {}}
    for s in streams:
        grouped[split_by_id[s.id]].setdefault(s.label, []).extend(s.frames)
    # Assert no environment appears in more than one split.
    env_splits: dict[str, set[str]] = defaultdict(set)
    for s in streams:
        env_splits[env_by_id[s.id]].add(split_by_id[s.id])
    leaks = {e: sorted(v) for e, v in env_splits.items() if len(v) > 1}
    if leaks:
        raise SystemExit(f"environment leakage across splits: {leaks}")
    split_stats = {name: sum(len(v) for v in data.values())
                   for name, data in grouped.items()}
    streams_by_split = {name: len([s for s in streams if split_by_id[s.id] == name])
                         for name in grouped}
    normals_by_split = {name: len([s for s in streams
                                   if split_by_id[s.id] == name and s.label == "normal"])
                         for name in grouped}
    fails_by_split = {name: len([s for s in streams
                                 if split_by_id[s.id] == name and s.label == "failing"])
                       for name in grouped}
    (split_dir / "splits.json").write_text(json.dumps({
        "data_dir": str(root), "grouped_by": "printer_id (environment)", "groups": GROUPS,
        "stats": split_stats,
        "streams_by_split": streams_by_split,
        "normals_by_split": normals_by_split,
        "fails_by_split": fails_by_split,
        "splits": {name: {label: files for label, files in data.items()}
                   for name, data in grouped.items()}}, indent=2) + "\n")
    for name, data in grouped.items():
        with (split_dir / f"{name}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "stream"])
            for label, files in data.items():
                for fp in files:
                    w.writerow([fp, label, stream_of.get(fp, "")])

    # Print-level index (train labelled + test unlabelled).
    index_rows: list[list] = []
    idx = {s.id: s for s in sources}
    for (printer, print_id), rows in sorted(by_print.items()):
        sid = f"{printer}__{print_id}"
        s = idx[sid]
        index_rows.append([sid, s.weak_label, s.failure_type or "", printer, print_id,
                           s.frame_count, s.split, "labeled"])
    test_prints = defaultdict(int)
    for r in test:
        test_prints[(r["printer_id"], r["print_id"])] += 1
    for (printer, print_id), n in sorted(test_prints.items()):
        index_rows.append([f"{printer}__{print_id}", "unknown", "", printer, print_id,
                           n, "", "unlabeled_test"])
    idx_path = out.parent / "index.csv"
    with idx_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stream", "label", "failure_type", "printer", "print",
                    "n_frames", "split", "status"])
        w.writerows(index_rows)

    # Report.
    normals = [s for s in sources if s.weak_label == "normal"]
    fails = [s for s in sources if s.weak_label == "failing"]
    envs = sorted({s.capture_environment for s in sources})
    print(f"manifest: {out} ({len(sources)} streams)")
    print(f"validate warnings: {warnings or 'none'}")
    print(f"streams: {summary}")
    print(f"normal streams: {len(normals)}  failing streams: {len(fails)}")
    print(f"environments (printers): {envs}")
    print(f"grouped split stats: {split_stats}")
    print(f"index: {idx_path} ({len(index_rows)} rows)")
    print(f"normal streams per split: {normals_by_split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
