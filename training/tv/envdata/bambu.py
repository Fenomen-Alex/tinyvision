"""Bambu timelapse dataset environment inventory.

The repository (CC BY 4.0) splits its data into ``train/timelapses`` (3
contributors) and ``train/videos`` (1 contributor). For TinyVision the camera
environment is the contributor's print chamber, so the dataset contributes up to
three candidate *normal* environments. Only the ``videos`` split is continuous
real-time footage; the ``timelapses`` split is skimmed (a few encoded frames per
print) and unsuited to our ~1 fps per-environment training cadence (kept for
robustness probing only).

Inventory is derived from the lightweight metadata CSVs; no media is fetched.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_URL = "https://huggingface.co/datasets/v2thegreat/bambu-timelapse-dataset"
LICENSE = "CC BY 4.0"
RAW_FILE_BASE = (
    "https://huggingface.co/datasets/v2thegreat/bambu-timelapse-dataset/resolve/main"
)


@dataclass
class BambuSplit:
    name: str
    contributors: dict[str, int]  # contributor -> row count
    continuous_realtime: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "continuous_realtime": self.continuous_realtime,
            "rows": sum(self.contributors.values()),
            "contributors": dict(self.contributors),
        }


def _read_csv(path: Path, user_field: str = "user") -> dict[str, int]:
    counts: dict[str, int] = Counter()
    with open(path) as f:
        for row in csv.DictReader(f):
            counts[row.get(user_field, "unknown")] += 1
    return dict(counts)


def build_inventory(
    timelapses_metadata_csv: str | Path,
    videos_metadata_csv: str | Path,
) -> dict[str, Any]:
    """Build the environment inventory from the two metadata CSVs."""
    timelapses = _read_csv(Path(timelapses_metadata_csv))
    videos = _read_csv(Path(videos_metadata_csv))
    splits = [
        BambuSplit("timelapses", timelapses, continuous_realtime=False).to_dict(),
        BambuSplit("videos", videos, continuous_realtime=True).to_dict(),
    ]
    contributors = sorted(set(timelapses) | set(videos))

    environments = []
    for contributor in contributors:
        in_videos = contributor in videos
        environments.append(
            {
                "environment_key": f"env_bambu_{_slug(contributor)}_bonnp-ipcam_inside_chamber",
                "source_dataset": "bambu-timelapse-v1",
                "contributor": contributor,
                "view": "inside_chamber",
                "camera": {"identity": "bambu-pnp-ipcam",
                           "description": "built-in Bambu Lab printer chamber camera"},
                "normal_available": True,
                "failure_type": None,
                "has_continuous_realtime_video": in_videos,
                "timelapse_sequences": timelapses.get(contributor, 0),
                "full_length_videos": videos.get(contributor, 0),
                "notes": (
                    "full-length video useful for ~1fps training if the original "
                    "recording is continuous" if in_videos else
                    "timelapse only; skimmed cadence, unsuited to 1fps training"
                ),
            }
        )

    return {
        "schema": "tinyvision-bambu-inventory",
        "source_dataset": "bambu-timelapse-v1",
        "name": "Bambu Timelapse Dataset",
        "url": REPO_URL,
        "license": LICENSE,
        "license_url": "https://huggingface.co/datasets/v2thegreat/bambu-timelapse-dataset",
        "splits": splits,
        "n_contributors": len(contributors),
        "n_environment_candidates": len([e for e in environments if e["has_continuous_realtime_video"]]),
        "environments": environments,
        "structure_verified_by": "metadata-csv",
        "media_downloaded": False,
        "note": "Environment candidates keyed by contributor's chamber camera.",
    }


def _slug(value: str) -> str:
    import re
    v = value.strip().lower()
    v = re.sub(r"[^a-z0-9._-]+", "_", v)
    return v or "unknown"


def fetch_metadata_csvs(cache_dir: str | Path) -> dict[str, Path]:
    """Download the two small metadata CSVs into ``cache_dir`` (network)."""
    import urllib.request

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    targets = {
        "timelapses": "/train/timelapses/timelapses_metadata.csv",
        "videos": "/train/videos/video_metadata.csv",
    }
    paths: dict[str, Path] = {}
    for name, rel in targets.items():
        out = cache / f"{name}_metadata.csv"
        url = RAW_FILE_BASE + rel
        urllib.request.urlretrieve(url, out)
        paths[name] = out
    return paths