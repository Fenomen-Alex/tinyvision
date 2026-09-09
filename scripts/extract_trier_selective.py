#!/usr/bin/env python3
"""Deterministic selective extraction / analysis for Trier FFF-QC zip files.

The full Trier dataset (~48.4 GB) expands to more than the available disk,
so we never extract everything. This tool works directly from a locally
downloaded zip (no full extraction required):

  --analyse path/to/base_data.zip [--json out.json]
      Prints the directory tree with per-directory image counts and byte sizes,
      enough to design environment units (camera / sessions / normal vs defect).

  --extract path/to/base_data.zip --out DIR [--include PREFIX [--include PREFIX ...]]
      Streams only members whose path starts with any PREFIX (append / for dir).
      --stride N subsamples to every Nth member (stable sort by member path),
      so per-environment frame caps and disk use stay bounded.
      --dry-run prints the count and total bytes without writing.

  --verify path/to/base_data.zip
      Runs zipfile.testzip() (integrity check) on the downloaded file.

Filenames are matched as path prefixes so a selection like
  "a3_base_data/0070_front"  extracts only that session subtree.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _member_stats(extracted: dict[str, int]) -> dict:
    ### aggregate member-by-directory counts
    per_dir: dict[str, Counter] = {}
    for name, size in extracted.items():
        parts = name.split("/")
        dirs = ["/".join(parts[: i + 1]) for i in range(len(parts) - 1)]
        for d in dirs:
            per_dir.setdefault(d, Counter()).update(images=0, bytes=0)
        parent = "/".join(parts[:-1]) or "."
        per_dir.setdefault(parent, Counter())
        per_dir[parent].update(images=1 if Path(name).suffix.lower() in IMAGE_EXTS else 0,
                               bytes=size)
    out = {}
    for d, c in per_dir.items():
        out[d] = {"image_files": int(c["images"]), "bytes": int(c["bytes"])}
    return out


def _listing(zip_path: Path) -> list[tuple[str, int]]:
    if not zip_path.exists():
        print(f"error: {zip_path} not found", file=sys.stderr)
        sys.exit(2)
    with zipfile.ZipFile(zip_path) as zf:
        return [(i.filename, i.file_size) for i in zf.infolist()]


def analyse(zip_path: Path, json_out: str | None) -> None:
    members = _listing(zip_path)
    tree = _member_stats(dict(members))
    top_dirs = sorted(
        (d for d in tree if d.count("/") == 1),
        key=lambda d: -tree[d]["bytes"],
    )
    if json_out:
        Path(json_out).write_text(json.dumps({
            "zip": str(zip_path),
            "members": len(members),
            "dirs": tree,
        }, indent=2))
    print(f"zip: {zip_path}  members: {len(members)}")
    for d in top_dirs:
        print(f"{tree[d]['image_files']:>8d} imgs {tree[d]['bytes']:>14d} B  /{d}")


def extract(zip_path: Path, out: Path, prefixes: list[str],
            dry_run: bool, stride: int = 1) -> None:
    members = _listing(zip_path)
    selected = [n for n, s in members
                if any(n.startswith(p) for p in prefixes)]
    if stride > 1:
        selected = selected[::stride]
    size_map = dict(members)
    total = sum(size_map[n] for n in selected)
    print(f"selected {len(selected)}/{len(members)} members (stride {stride}), {total} bytes")
    if dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for i, name in enumerate(selected):
            zf.extract(name, out)
            if i and i % 2500 == 0:
                print(f"  extracted {i}/{len(selected)}")


def verify(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
    print("OK" if bad is None else f"corrupt member: {bad}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analyse", metavar="ZIP")
    ap.add_argument("--extract", metavar="ZIP")
    ap.add_argument("--out", default="datasets/raw/trier-fffqc-v2/extracted")
    ap.add_argument("--include", action="append", default=[])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", metavar="ZIP")
    ap.add_argument("--json", metavar="PATH", default=None)
    args = ap.parse_args()

    if args.analyse:
        analyse(Path(args.analyse), args.json)
    elif args.extract:
        extract(Path(args.extract), Path(args.out), args.include,
                args.dry_run, stride=args.stride)
    elif args.verify:
        verify(Path(args.verify))
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()