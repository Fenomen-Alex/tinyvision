"""Trier FFF-QC dataset characterization.

Two complementary artifacts:

* ``index_trier_tree(root)`` — a drop-in indexing tool run against an
  *extracted local copy* of the dataset (>48 GB, so it walks and counts; it
  never copies). Safe to run before downloading nothing: if the tree is absent
  it returns an empty inventory with a note.

* ``characterize_from_zenodo(record_dict)`` — builds a metadata-based
  characterization directly from the Zenodo API record. This is the artifact we
  produce now, because downloading the dataset is out of scope.

Environment inventory from this metadata is intentionally marked UNVERIFIED:
frame/printer/camera-env counts cannot be established without inspecting the
extracted tree.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

CATEGORY_KEYWORDS = {
    "normal": ("normal", "defect-free", "defect_free", "ok", "good", "correct"),
    "extrusion_error": ("extrusion",),
    "layer_shift": ("layer", "shift", "layershift"),
    "stringing": ("string",),
    "warping": ("warp",),
    "spaghetti": ("spaghetti",),
    "bed_adhesion": ("adhesion", "bed"),
}

VIEW_KEYWORDS = {
    "front": ("front", "azimuth_0"),
    "side": ("side", "azimuth_90", "lateral"),
    "overhead": ("top", "overhead", "above"),
    "bed_mount": ("bed", "webcam"),
}


def _guess_category(path: str) -> str:
    parts = re.split(r"[/\\_\- .]", path.lower())
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in parts for kw in keywords) or any(
            kw in path.lower() for kw in keywords
        ):
            return cat
    return "unknown"


def _guess_view(path: str) -> str:
    lower = path.lower()
    for view, keywords in VIEW_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return view
    return "unknown"


def _image_files(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [
        p
        for p in root.rglob("*")
        if p.suffix.lower() in exts and not p.name.startswith(".")
    ]


def index_trier_tree(root: str | Path) -> dict[str, Any]:
    """Index a locally-extracted Trier tree. Empty tree -> empty inventory."""
    root = Path(root)
    if not root.exists():
        return {
            "schema": "tinyvision-trier-index",
            "root": str(root),
            "present": False,
            "note": "local copy not present; run again once extracted",
        }
    files = _image_files(root)
    categories = Counter(_guess_category(str(p)) for p in files)
    views = Counter(_guess_view(str(p)) for p in files)
    dirs = sorted(
        (str(d.relative_to(root)), d) for d in root.rglob("*") if d.is_dir()
    )
    per_dir = {
        str(d.relative_to(root)): len(_image_files(d)) for _, d in dirs if len(_image_files(d))
    }
    return {
        "schema": "tinyvision-trier-index",
        "root": str(root),
        "present": True,
        "total_images": len(files),
        "by_category": dict(categories),
        "by_view": dict(views),
        "dirs_with_images": dict(sorted(per_dir.items())),
        "structure_verified": False,
        "environment_count": None,
        "environment_count_verified": False,
        "generated_at_utc": _now_utc(),
    }


def _now_utc() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def characterize_from_zenodo(record: dict[str, Any]) -> dict[str, Any]:
    """Build the metadata-based characterization from a Zenodo API record."""
    md = record.get("metadata", {})
    files = []
    for f in record.get("files", []):
        files.append(
            {
                "key": f.get("key"),
                "size_bytes": f.get("size"),
                "size_human": _human(f.get("size")),
                "checksum": f.get("checksum"),
            }
        )
    description = md.get("description") or ""
    plain = re.sub(r"<[^>]+>", " ", description).replace("\xa0", " ").strip()
    return {
        "schema": "tinyvision-trier-characterization",
        "source": "zenodo-api",
        "record_id": record.get("id"),
        "conceptrecid": record.get("conceptrecid"),
        "doi": record.get("doi"),
        "title": md.get("title"),
        "version": md.get("version"),
        "publication_date": md.get("publication_date"),
        "first_published": record.get("created", "").split("T")[0],
        "modified": record.get("modified", "").split("T")[0],
        "license": (md.get("license") or {}).get("id"),
        "creators": [c.get("name") for c in md.get("creators", [])],
        "description_excerpt": plain[:2000],
        "files": files,
        "observed_claims": {
            "captured_during_production": "three different camera angles" in plain.lower(),
            "n_camera_angles": 3,
            "defects": ["extrusion_error", "layer_shift", "stringing"],
            "has_defect_free": True,
        },
        "environment_units": None,
        "environment_verified": False,
        "note": (
            "Environment units cannot be counted from metadata alone. The "
            "characterization must be completed with index_trier_tree() against "
            "an extracted local copy once one is available."
        ),
        "retrieved_at_utc": _now_utc(),
    }


def _human(size: int | None) -> str | None:
    if size is None:
        return None
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return str(size)


def index_images_by_md5(root: str | Path, limit: int | None = None) -> dict[str, Any]:
    """Walk an extracted tree, computing per-file hashes (expensive; use for
    dedup on smaller datasets only)."""
    root = Path(root)
    files = _image_files(root)
    if limit:
        files = files[:limit]
    hashes: dict[str, Path] = {}
    dup = []
    for p in files:
        digest = _md5(p)
        if digest in hashes:
            dup.append({"path": str(p), "masks": str(hashes[digest])})
        else:
            hashes[digest] = p
    return {
        "schema": "tinyvision-trier-index-hashes",
        "total_hashed": len(files),
        "unique": len(hashes),
        "duplicates": dup,
    }


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()