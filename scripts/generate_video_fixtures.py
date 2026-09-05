#!/usr/bin/env python3
"""Deterministic video-fixture generation from ordered PNG recording sequences.

Creates a small set of representative MP4 video files from the Kaggle
dataset's ordered recording directories.  Each recording is encoded at a
configurable FPS.  The original PNGs are never modified.

Usage:
    python scripts/generate_video_fixtures.py [--output-dir DIR] [--fps 2]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Allow running from repo root or scripts/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from tv.data.video import assemble_video, assemble_video_from_frames, load_labels_from_dir


def combine_recording_dirs(dirs: list[Path]) -> list[tuple[Path, str]]:
    """Combine frames from multiple recording dirs into a sorted sequence.

    Returns list of (png_path, class_label) sorted by filename.
    For MIXED recordings that span normal/ and failing/, this produces the
    correct chronological sequence with per-frame labels.
    """
    frames = []
    for d in dirs:
        class_label = d.parent.name.lower()
        for png in sorted(d.iterdir()):
            if png.suffix.lower() == ".png":
                frames.append((png, class_label))
    # Sort by filename (lexicographic = chronological for ELP naming)
    frames.sort(key=lambda x: x[0].name)
    return frames


# Representative recording set chosen to cover:
# - PURE_NORMAL, PURE_FAILING, MIXED
# - Multiple failure types (under_extrusion, stringing, spaghetti)
# - Both val and test splits
RECORDINGS = [
    # Test split
    {
        "id": "test_pure_normal_68",
        "split": "test",
        "composition": "PURE_NORMAL",
        "recording": "Recording_05012023_13_10_42",
        "class": "normal",
        "failure_types": [],
    },
    {
        "id": "test_pure_failing_underex_147",
        "split": "test",
        "composition": "PURE_FAILING",
        "recording": "Recording_08012023_13_31_14",
        "class": "failing",
        "failure_types": ["under_extrusion"],
    },
    {
        "id": "test_pure_failing_stringing_72",
        "split": "test",
        "composition": "PURE_FAILING",
        "recording": "Recording_14.02.2023_17_38_44",
        "class": "failing",
        "failure_types": ["stringing"],
    },
    {
        "id": "test_mixed_underex_86",
        "split": "test",
        "composition": "MIXED",
        "recording": "Recording_01122022_16_06_44",
        "class": "failing",
        "failure_types": ["under_extrusion"],
    },
    # Val split
    {
        "id": "val_pure_normal_152",
        "split": "val",
        "composition": "PURE_NORMAL",
        "recording": "Recording_01122022_17_44_58",
        "class": "normal",
        "failure_types": [],
    },
    {
        "id": "val_pure_failing_underex_119",
        "split": "val",
        "composition": "PURE_FAILING",
        "recording": "Recording_10012023_18_46_21",
        "class": "failing",
        "failure_types": ["under_extrusion"],
    },
    {
        "id": "val_pure_failing_stringing_40",
        "split": "val",
        "composition": "PURE_FAILING",
        "recording": "Recording_14.02.2023_20_02_21",
        "class": "failing",
        "failure_types": ["stringing"],
    },
    {
        "id": "val_mixed_multi_131",
        "split": "val",
        "composition": "MIXED",
        "recording": "Recording_03012023_15_07_03",
        "class": "failing",
        "failure_types": ["under_extrusion", "spaghetti"],
    },
]

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "processed" / "printing-errors"


def find_recording_dirs(recording_id: str, data_dir: Path) -> list[Path]:
    """Locate all PNG directories for a recording (may span normal/ and failing/)."""
    dirs = []
    for class_dir in ("normal", "failing"):
        candidate = data_dir / class_dir / recording_id
        if candidate.is_dir():
            dirs.append(candidate)
    if not dirs:
        raise FileNotFoundError(
            f"Recording {recording_id} not found in {data_dir}"
        )
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate video fixtures from ordered PNG recordings"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for MP4 fixtures (default: docs/research/stage7-fixtures)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Video FPS for encoding (default: 2.0)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Root data directory (default: datasets/processed/printing-errors)",
    )
    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
        or (Path(__file__).resolve().parent.parent / "docs" / "research" / "stage7-fixtures")
    )
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    fps = args.fps

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "fps": fps,
        "data_dir": str(data_dir),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "recordings": [],
    }

    print(f"Generating video fixtures at {fps} FPS into {output_dir}")
    print(f"Data directory: {data_dir}")
    print()

    for rec in RECORDINGS:
        rec_id = rec["id"]
        recording = rec["recording"]
        print(f"  {rec_id} ({recording})...", end=" ", flush=True)

        try:
            src_dirs = find_recording_dirs(recording, data_dir)
        except FileNotFoundError as e:
            print(f"SKIP ({e})")
            continue

        # Combine frames from all directories (handles MIXED recordings)
        combined = combine_recording_dirs(src_dirs)
        if not combined:
            print(f"SKIP (no PNG frames found)")
            continue

        frame_paths = [f[0] for f in combined]
        frame_classes = [f[1] for f in combined]

        out_path = output_dir / f"{rec_id}.mp4"
        t0 = time.perf_counter()
        info = assemble_video_from_frames(frame_paths, out_path, fps=fps)
        elapsed = time.perf_counter() - t0

        # Save per-frame labels (authoritative ground truth from PNG parent dirs)
        labels = []
        for i, (fp, cls) in enumerate(combined):
            labels.append({
                "frame_index": i,
                "filename": fp.name,
                "label": cls,
            })
        labels_path = output_dir / f"{rec_id}_labels.json"
        with open(labels_path, "w") as f:
            json.dump(labels, f)

        rec_entry = {
            **rec,
            "video_path": info["path"],
            "frame_count": info["frame_count"],
            "resolution": info["resolution"],
            "source_dirs": [str(d) for d in src_dirs],
            "labels_path": str(labels_path),
            "generation_time_s": round(elapsed, 3),
        }
        manifest["recordings"].append(rec_entry)
        print(f"{info['frame_count']} frames, {info['resolution']}, {elapsed:.2f}s")

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_frames = sum(r["frame_count"] for r in manifest["recordings"])
    print(f"\nDone: {len(manifest['recordings'])} fixtures, {total_frames} total frames")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
