"""Dataset audit: integrity, resolution, duplicates, class balance."""
from __future__ import annotations

import hashlib
import json
import struct
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ImageInfo:
    path: Path
    width: int
    height: int
    sha256: str
    size_bytes: int


@dataclass
class AuditReport:
    total_images: int
    corrupted: list[str] = field(default_factory=list)
    duplicates: dict[str, list[str]] = field(default_factory=dict)
    resolutions: dict[str, int] = field(default_factory=dict)
    class_distribution: dict[str, int] = field(default_factory=dict)
    min_size_violations: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_images": self.total_images,
            "corrupted": self.corrupted,
            "duplicates": {h: paths for h, paths in self.duplicates.items()},
            "resolutions": self.resolutions,
            "class_distribution": self.class_distribution,
            "min_size_violations": self.min_size_violations,
            "errors": self.errors,
        }


def _read_png_dimensions(path: Path) -> tuple[int, int] | None:
    """Read PNG dimensions from header without full decode."""
    try:
        with open(path, "rb") as f:
            sig = f.read(8)
            if sig != b"\x89PNG\r\n\x1a\n":
                return None
            # IHDR chunk
            length_bytes = f.read(4)
            chunk_type = f.read(4)
            if chunk_type != b"IHDR":
                return None
            data = f.read(13)
            width = struct.unpack(">I", data[0:4])[0]
            height = struct.unpack(">I", data[4:8])[0]
            return width, height
    except Exception:
        return None


def _read_jpeg_dimensions(path: Path) -> tuple[int, int] | None:
    """Read JPEG dimensions from header."""
    try:
        with open(path, "rb") as f:
            data = f.read()
            i = 0
            while i < len(data) - 1:
                if data[i] == 0xFF:
                    marker = data[i + 1]
                    if marker in (0xC0, 0xC1, 0xC2):
                        height = struct.unpack(">H", data[i + 5 : i + 7])[0]
                        width = struct.unpack(">H", data[i + 7 : i + 9])[0]
                        return width, height
                    elif marker == 0xD8 or marker == 0xD9:
                        i += 2
                    else:
                        length = struct.unpack(">H", data[i + 2 : i + 4])[0]
                        i += 2 + length
                else:
                    i += 1
    except Exception:
        return None


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_dataset(
    data_dir: Path,
    min_width: int = 32,
    min_height: int = 32,
) -> AuditReport:
    """Audit a dataset in Layout A or B format.

    Args:
        data_dir: Dataset root with label/ subdirectories.
        min_width: Minimum acceptable image width.
        min_height: Minimum acceptable image height.

    Returns:
        AuditReport with all findings.
    """
    data_dir = Path(data_dir)
    report = AuditReport(total_images=0)

    # Discover images by label
    class_counts: Counter[str] = Counter()
    sha_to_paths: dict[str, list[str]] = defaultdict(list)
    resolution_counts: Counter[str] = Counter()

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    for label_dir in sorted(data_dir.iterdir()):
        if not label_dir.is_dir() or label_dir.name.startswith(".") or label_dir.name == "splits":
            continue

        label = label_dir.name

        # Walk through all subdirectories (source groups) and files
        for img_path in sorted(label_dir.rglob("*")):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in extensions:
                continue

            report.total_images += 1
            class_counts[label] += 1

            # Check for corruption by verifying file is readable
            try:
                size = img_path.stat().st_size
                if size == 0:
                    report.corrupted.append(str(img_path))
                    continue
            except OSError as e:
                report.errors.append(f"{img_path}: {e}")
                continue

            # Read dimensions
            dims = None
            ext = img_path.suffix.lower()
            if ext == ".png":
                dims = _read_png_dimensions(img_path)
            elif ext in (".jpg", ".jpeg"):
                dims = _read_jpeg_dimensions(img_path)
            else:
                # For other formats, just check file size
                dims = None

            if dims is None:
                report.errors.append(f"{img_path}: could not read dimensions")
            else:
                w, h = dims
                resolution_counts[f"{w}x{h}"] += 1
                if w < min_width or h < min_height:
                    report.min_size_violations.append(
                        f"{img_path}: {w}x{h} < {min_width}x{min_height}"
                    )

            # Compute SHA-256 for duplicate detection
            sha = _compute_sha256(img_path)
            sha_to_paths[sha].append(str(img_path))

    # Build report
    report.class_distribution = dict(class_counts)
    report.resolutions = dict(resolution_counts)
    report.duplicates = {
        h: paths for h, paths in sha_to_paths.items() if len(paths) > 1
    }

    return report


def print_report(report: AuditReport) -> None:
    """Pretty-print the audit report."""
    print(f"\n{'='*60}")
    print(f"DATASET AUDIT REPORT")
    print(f"{'='*60}")
    print(f"\nTotal images: {report.total_images}")

    print(f"\n--- Class Distribution ---")
    for label, count in sorted(report.class_distribution.items()):
        pct = 100.0 * count / report.total_images if report.total_images else 0
        print(f"  {label:15s}  {count:6d}  ({pct:5.1f}%)")

    print(f"\n--- Resolutions ---")
    for res, count in sorted(report.resolutions.items(), key=lambda x: -x[1]):
        print(f"  {res:15s}  {count:6d}")

    print(f"\n--- Integrity ---")
    print(f"  Corrupted files:     {len(report.corrupted)}")
    print(f"  Duplicate sets:      {len(report.duplicates)}")
    print(f"  Total dup images:    {sum(len(v) for v in report.duplicates.values())}")
    print(f"  Min-size violations: {len(report.min_size_violations)}")
    print(f"  Errors:              {len(report.errors)}")

    if report.corrupted:
        print(f"\n  Corrupted files:")
        for p in report.corrupted[:10]:
            print(f"    {p}")
        if len(report.corrupted) > 10:
            print(f"    ... and {len(report.corrupted) - 10} more")

    if report.duplicates:
        print(f"\n  Duplicate sets (showing first 5):")
        for i, (h, paths) in enumerate(sorted(report.duplicates.items())):
            if i >= 5:
                print(f"    ... and {len(report.duplicates) - 5} more sets")
                break
            print(f"    SHA256 {h[:16]}... ({len(paths)} copies):")
            for p in paths[:3]:
                print(f"      {p}")
            if len(paths) > 3:
                print(f"      ... and {len(paths) - 3} more")

    if report.min_size_violations:
        print(f"\n  Min-size violations (first 10):")
        for v in report.min_size_violations[:10]:
            print(f"    {v}")

    if report.errors:
        print(f"\n  Errors (first 10):")
        for e in report.errors[:10]:
            print(f"    {e}")

    print(f"\n{'='*60}")
