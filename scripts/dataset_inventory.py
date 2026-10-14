#!/usr/bin/env python3
"""Build a factual inventory of everything under datasets/.

This is a *reporting* tool: it records what is present on disk and what the
recorded manifests claim, without re-deriving any dataset statistics. It is
deliberately dependency-light (standard library only) so it runs anywhere.

Outputs JSON to docs/research/dataset-inventory.json. Human interpretation
(licensing verdicts, usability classification) lives in
docs/research/dataset-inventory.md and must not be inferred by this script.

Usage:
    python scripts/dataset_inventory.py [--datasets-dir datasets]
                                        [--out docs/research/dataset-inventory.json]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import date
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

MANIFEST_HINTS = (
    "manifest.json",
    "env_manifest.json",
    "derived_manifest.json",
    "canonical_holdings.json",
)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        ).stdout.strip()
    except Exception:  # pragma: no cover - git always present in CI/dev
        return ""


def _dir_stats(root: Path) -> dict:
    n_files = n_images = n_videos = 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            n_files += 1
            try:
                total += p.stat().st_size
            except OSError:
                pass
            ext = p.suffix.lower()
            if ext in IMAGE_EXTS:
                n_images += 1
            elif ext in VIDEO_EXTS:
                n_videos += 1
    return {
        "size_bytes": total,
        "size_mb": round(total / (1024 * 1024), 2),
        "n_files": n_files,
        "n_images": n_images,
        "n_videos": n_videos,
    }


def _find_manifests(root: Path, limit: int = 40) -> list[str]:
    found: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn in MANIFEST_HINTS:
                found.append(str((Path(dirpath) / fn).relative_to(root.parent)))
                if len(found) >= limit:
                    return found
    return found


def _git_tracked_count(rel: str) -> int:
    out = _git("ls-files", rel)
    return len([ln for ln in out.splitlines() if ln.strip()])


def build(datasets_dir: Path) -> dict:
    entries = []
    if datasets_dir.exists():
        for child in sorted(datasets_dir.iterdir()):
            if not child.is_dir():
                continue
            rel = str(child)
            stats = _dir_stats(child)
            entries.append(
                {
                    "path": rel,
                    "name": child.name,
                    **stats,
                    "git_tracked_files": _git_tracked_count(rel),
                    "manifests": _find_manifests(child),
                }
            )

    # Recorded holdings are the project's own summary of frozen assets.
    holdings_path = datasets_dir / "staging" / "readiness" / "canonical_holdings.json"
    holdings = None
    if holdings_path.exists():
        holdings = json.loads(holdings_path.read_text())

    return {
        "schema": "tinyvision-dataset-inventory",
        "generated": date.today().isoformat(),
        "datasets_dir": str(datasets_dir),
        "note": (
            "Facts only. Licensing verdicts and usability classification are "
            "editorial and documented in docs/research/dataset-inventory.md."
        ),
        "top_level": entries,
        "recorded_holdings": holdings,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-dir", default="datasets")
    ap.add_argument("--out", default="docs/research/dataset-inventory.json")
    args = ap.parse_args()

    root = Path(args.datasets_dir).resolve()
    report = build(root)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"wrote {out}")
    print(f"{'path':40} {'MB':>10} {'files':>8} {'imgs':>8} {'tracked':>8}")
    for e in report["top_level"]:
        print(
            f"{e['path']:40} {e['size_mb']:>10.2f} {e['n_files']:>8} "
            f"{e['n_images']:>8} {e['git_tracked_files']:>8}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
