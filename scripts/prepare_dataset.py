#!/usr/bin/env python3
"""Reorganize the 3d-print-errors dataset into Layout A for the splitter.

Input:
    datasets/Printing_Errors/
        general_data/all_images_no_filter.csv   (semicolon-delimited)
        images/all_images256/*.png

Output:
    datasets/processed/printing-errors/
        metadata.csv                            (class mapping reference)
        normal/{recording}/*.png
        failing/{recording}/*.png

The 'recording' column in the CSV groups consecutive frames from the same
print session, which is used by the splitter for leakage-free splitting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# Original class indices → binary labels
CLASS_MAP = {
    0: "normal",
    1: "failing",
    2: "failing",
    4: "failing",
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 3d-print-errors dataset")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("datasets/Printing_Errors"),
        help="Path to the raw dataset root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/processed/printing-errors"),
        help="Output path for Layout A directory structure",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        default=True,
        help="Copy files instead of symlinking (default: copy)",
    )
    parser.add_argument(
        "--image-set",
        choices=["all_images256", "test_images_silver265", "test_images_oblique256"],
        default="all_images256",
        help="Which image set to use (default: all_images256)",
    )
    args = parser.parse_args()

    source = args.source
    output = args.output
    image_dir = source / "images" / args.image_set
    csv_path = source / "general_data" / "all_images_no_filter.csv"

    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        sys.exit(1)
    if not image_dir.exists():
        print(f"Error: Image directory not found at {image_dir}")
        sys.exit(1)

    # Read CSV
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rows.append(row)

    print(f"Read {len(rows)} rows from CSV")

    # Group by recording → label → [images]
    recordings: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    skipped = 0
    missing = 0

    for row in rows:
        image_name = row["image"]
        class_idx = int(row["class"])
        recording = row["recording"]

        if class_idx not in CLASS_MAP:
            print(f"  Warning: unknown class {class_idx} for {image_name}, skipping")
            skipped += 1
            continue

        label = CLASS_MAP[class_idx]
        src = image_dir / image_name
        if not src.exists():
            missing += 1
            continue

        recordings[recording][label].append(image_name)

    print(f"Skipped: {skipped} (unknown class), Missing: {missing} (file not found)")

    # Create output directories and copy files
    total_copied = 0
    for recording, labels in sorted(recordings.items()):
        for label, images in labels.items():
            dest_dir = output / label / recording
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img_name in images:
                src = image_dir / img_name
                dst = dest_dir / img_name
                if args.copy:
                    shutil.copy2(src, dst)
                else:
                    dst.symlink_to(src.resolve())
                total_copied += 1

    print(f"Copied {total_copied} images to {output}")

    # Write metadata CSV
    with open(output / "metadata.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "original_class", "recording", "image"])
        for recording, labels in sorted(recordings.items()):
            for label, images in labels.items():
                for img_name in images:
                    writer.writerow(
                        [f"{label}/{recording}/{img_name}", label, 
                         next(r["class"] for r in rows if r["image"] == img_name),
                         recording, img_name]
                    )

    print(f"Metadata written to {output / 'metadata.csv'}")

    # Print summary
    label_counts = defaultdict(int)
    for labels in recordings.values():
        for label, images in labels.items():
            label_counts[label] += len(images)

    print("\nClass distribution:")
    for label in sorted(label_counts):
        print(f"  {label}: {label_counts[label]}")
    print(f"  Total: {sum(label_counts.values())}")
    print(f"  Recordings: {len(recordings)}")


if __name__ == "__main__":
    main()
