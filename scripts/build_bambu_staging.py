"""Build the Bambu timelapse dataset staging manifest (offline, deterministic).

Reads the two metadata CSVs (already cached under
``datasets/staging/bambu/metadata/``) and writes:

* ``manifests/env_manifest.json`` — 3 environments (one per contributor's
  chamber camera), 135 full-length video recordings (Borillion), timelapse
  counts retained on the environment metadata,
* derived per-environment/per-recording JSON views.

Environment IDs are deterministic (UUID5 from the environment key) so repeated
runs are stable. No media is touched. Run:

    uv run --project training python scripts/build_bambu_staging.py
"""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

from tv.envdata.schema import (
    Consent,
    Dataset,
    Environment,
    Manifest,
    Recording,
)
from tv.envdata.validate import validate_manifest

STAGING = Path("datasets/staging/bambu")
META = STAGING / "metadata"

LICENSE = "CC BY 4.0"
REPO = "https://huggingface.co/datasets/v2thegreat/bambu-timelapse-dataset"
NS = uuid.UUID("6fb8aa1c-8f5e-4e75-972a-1a1b0c1a1b0c")


def env_id(key: str) -> str:
    return "env-" + uuid.uuid5(NS, key).hex


def build() -> Manifest:
    with open(META / "timelapses_metadata.csv") as f:
        tl = list(csv.DictReader(f))
    with open(META / "videos_metadata.csv") as f:
        vid = list(csv.DictReader(f))

    tl_counts: dict[str, list[dict]] = {}
    for r in tl:
        tl_counts.setdefault(r["user"], []).append(r)
    vid_by_user: dict[str, list[dict]] = {}
    for r in vid:
        vid_by_user.setdefault(r["user"], []).append(r)

    envs: list[Environment] = []
    recordings: list[Recording] = []
    for contributor in sorted(set(tl_counts) | set(vid_by_user)):
        key = f"env_bambu_{_slug(contributor)}_bonnp-ipcam_inside_chamber"
        env = Environment(
            environment_id=env_id(key),
            environment_key=key,
            source_dataset="bambu-timelapse-v1",
            contributor=contributor,
            camera={
                "identity": "bambu-pnp-ipcam",
                "description": "built-in Bambu Lab printer chamber camera",
            },
            view="inside_chamber",
            metadata={
                "normal_available": True,
                "failure_type": None,
                "timelapse_sequences": len(tl_counts.get(contributor, [])),
                "timelapse_total_duration_s": sum(
                    float(r["duration"]) for r in tl_counts.get(contributor, [])
                ),
                "full_length_videos": len(vid_by_user.get(contributor, [])),
            },
        )
        envs.append(env)
        # Staging decision: only contributors with continuous real-time video
        # are usable for ~1fps training; Borillion's chamber camera is assigned
        # to train. Timelapse-only environments stay unassigned candidates.
        if vid_by_user.get(contributor):
            env.split_assignment = "train"
        else:
            env.notes = (
                "timelapse-only contributor: candidate_only (unassigned). "
                "Do not promote timelapse footage to training without "
                "reassessment (temporal gaps, frame-rate illusions)."
            )
        for row in vid_by_user.get(contributor, []):
            dur = float(row["duration"]) if row.get("duration") else 0.0
            frames = int(float(row["num_frames"])) if row.get("num_frames") else 0
            rec_id = f"bambu-video-{_slug(row['file_name'].split('/')[-1])}"
            recordings.append(
                Recording(
                    recording_id=rec_id,
                    environment_id=env.environment_id,
                    source=f"{REPO}/resolve/main/{row['file_name']}",
                    duration_s=round(dur, 3),
                    fps=round(frames / dur, 3) if dur > 0 else None,
                    continuous_realtime=True,
                    temporal_note=None,
                    metadata={"n_frames": frames},
                )
            )

    consent = Consent(
        contributor="contributors (v2thegreat, Similar-Ad-1223, Borillion)",
        consent_date="2025-04-17",
        training_allowed=True,
        evaluation_allowed=True,
        redistribution_allowed=True,
        verification="dataset_license",
        attribution="CC BY 4.0 via the Bambu Timelapse Dataset",
        notes="Rights come from the dataset CC BY 4.0 license; contributors are anonymized login names.",
    )

    manifest = Manifest(
        dataset=Dataset(
            dataset_id="bambu-timelapse-v1",
            name="Bambu Timelapse Dataset",
            license=LICENSE,
            license_url=REPO,
            provenance=(
                "Community-contributed timelapse/video footage from Bambu Lab "
                "printers; split into train/timelapses and train/videos."
            ),
            acquisition_method="community_dataset",
            retrieval_date="2026-09-07",
            url=REPO,
            redistribution_allowed=True,
            training_allowed=True,
            evaluation_allowed=True,
            notes=(
                "Timelapses are skimmed (a few encoded frames per print) and are "
                "unsuitable for ~1fps per-environment training; full-length videos "
                "(Borillion, 135 recordings) are real-time continuous footage."
            ),
        ),
        environments=envs,
        recordings=recordings,
        consents=[consent],
    )
    return manifest


def main() -> None:
    manifest = build()
    from tv.envdata.schema import dump_manifest

    dump_manifest(manifest, STAGING)
    v = validate_manifest(manifest, staging_root=STAGING)
    print(json.dumps(v.report(), indent=2))
    print(f"environments={len(manifest.environments)} "
          f"recordings={len(manifest.recordings)}")


def _slug(value: str) -> str:
    import re
    v = value.strip().lower()
    v = re.sub(r"[^a-z0-9._-]+", "_", v)
    return v or "unknown"


if __name__ == "__main__":
    main()