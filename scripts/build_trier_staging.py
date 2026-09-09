#!/usr/bin/env python3
"""Deterministic Trier FFF-QC v2 environment staging.

Builds the environment manifest + per-recording samples from the selectively
extracted tree produced by scripts/extract_trier_selective.py and recorded in
datasets/staging/trier/extraction_plan.json.

Environment units = (print type, camera). The dataset's authors defined three
fixed camera views; we do NOT invent which physical angle a camera represents
(view is recorded as "cameraN-unstated"). Splits follow source intent:
base_data + additional_defect_data -> train, test_data -> external_holdout.
Source labels (normal/failing + failure type) come from the directory name.
Verification is left pending: labels become tinyvision labels only through
human review (tv review-merge; provenance human:release-review).

Run with: uv run --project training python scripts/build_trier_staging.py
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from tv.envdata.schema import (
    Consent,
    Dataset,
    Environment,
    Manifest,
    Recording,
    Sample,
    dump_manifest,
    environment_key,
    write_samples,
)

ROOT = Path("datasets/raw/trier-fffqc-v2/extracted")
STAGING = Path("datasets/staging/trier")
PLAN_PATH = STAGING / "extraction_plan.json"
PRINT_TYPE_TO_FAILURE = {
    "extrusion_error_print": "extrusion_error",
    "layershift_error_print": "layer_shift",
    "stringing_error": "stringing",
    "no_error_print": None,
    "no_stringing_error": None,
}
TRAIN_SOURCE_ROOTS = ("base", "additional_defect_data")

FILE_RE = re.compile(r"(?:camera|cam)(\d+)_(\d+)\.jpg")


def _envs_from_tree() -> list[dict]:
    """Return one entry per (source_root, print_type, camera) populated dir."""
    envs = []
    for image_file in sorted(p for p in ROOT.rglob("*") if p.suffix.lower() == ".jpg"):
        rel = image_file.relative_to(ROOT)
        parts = rel.parts
        camera = parts[3]
        if parts[0] == "base":
            source_root, print_type = parts[0], parts[2]
        elif parts[0] == "additional_defect_data":
            source_root, print_type = parts[0], parts[1]
        elif parts[0] == "test":
            source_root, print_type = parts[0], parts[2]
        else:
            continue
        key = (source_root, print_type, camera)
        if key not in envs:
            envs.append(key)
    return [{"source_root": s, "print_type": p, "camera": c}
            for s, p, c in sorted(envs)]


def _env_metadata(entry: dict) -> dict:
    failure = PRINT_TYPE_TO_FAILURE.get(entry["print_type"])
    return {
        "failure_type": failure or "none",
        "print_type": entry["print_type"],
        "camera": entry["camera"],
        "zip_group": entry["source_root"],
    }


def main() -> None:
    plan = json.loads(PLAN_PATH.read_text())
    entries = _envs_from_tree()
    print(f"environments found: {len(entries)}")

    dataset = Dataset(
        dataset_id="trier-fff-qc-v2",
        name="Quality Control in Additive Manufacturing: Image Dataset for "
             "Defect Detection in Fused Filament Fabrication",
        license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        provenance="downloaded from Zenodo record 18628777 (v2.0.0); "
                   "checksums verified against API",
        acquisition_method="public repository download",
        retrieval_date="2026-09-08",
        url="https://doi.org/10.5281/zenodo.18628777",
        redistribution_allowed=True,
        training_allowed=True,
        evaluation_allowed=True,
        notes="Selective subsampled full-resolution extraction; see "
              "datasets/staging/trier/extraction_plan.json",
    )
    env_registry: dict[str, Environment] = {}
    consent = Consent(
        contributor="trier-team",
        consent_date="2026-09-08",
        training_allowed=True,
        evaluation_allowed=True,
        redistribution_allowed=True,
        attribution="Bast, Scherer, Wahl, Naumann, Dartmann, Didas (CC BY 4.0)",
        preferred_credit="doi:10.5281/zenodo.18628777",
        verification="public CC BY 4.0 license; no individual consent form required",
        notes=None,
    )
    recordings: list[Recording] = []
    all_samples: list[Sample] = []

    for entry in entries:
        failure = PRINT_TYPE_TO_FAILURE.get(entry["print_type"])
        camera = entry["camera"]
        source_root = entry["source_root"]
        print_type = entry["print_type"]
        key = environment_key("trier", "trier-team", camera,
                              f"{camera}-{print_type}-{source_root}")
        env = Environment(
            environment_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{STAGING}/{key}")),
            environment_key=key,
            source_dataset=dataset.dataset_id,
            contributor="trier-team",
            camera={"id": camera, "angle": "unstated in source"},
            view=f"{camera} of {print_type} ({source_root}); physical angle unstated",
            printer={},
            environment_group=print_type,
            split_assignment="train" if source_root in TRAIN_SOURCE_ROOTS else "external_holdout",
            lighting_notes=None,
            background_notes=None,
            notes=f"Trier {print_type} captured with three fixed camera views; "
                  "physical angle not stated by authors",
            metadata=_env_metadata(entry),
        )
        env_registry[key] = env

        # gather frames for this exact env dir
        prefix = _frame_rel_prefix(entry)
        files = sorted(
            (ROOT / prefix).glob("*.jpg"),
            key=lambda p: _frame_number(p.name),
        )
        rec_id = f"trier-{source_root}-{print_type}-{camera}"
        rec = Recording(
            recording_id=rec_id,
            environment_id=env.environment_id,
            source=str(prefix),
            duration_s=float(len(files)),
            fps=1.0,
            width=None,
            height=None,
            continuous_realtime=True,
            temporal_note="1 frame per sampled instance; source fps unknown, "
                          "frames assumed sequential seconds for review",
            md5=None,
            metadata={
                "extraction_stride": _plan_index(plan, source_root),
                "frame_count": len(files),
            },
        )
        recordings.append(rec)
        for i, fp in enumerate(files):
            n = _frame_number(fp.name)
            all_samples.append(Sample(
                sample_id=f"{rec_id}-{n}",
                recording_id=rec_id,
                environment_id=env.environment_id,
                frame_index=n,
                timestamp_s=float(n),
                source_label="failing" if failure else "normal",
                source_failure_type=failure,
                tinyvision_label=None,
                tinyvision_failure_type=None,
                verification="pending",
                verification_provenance=None,
                metadata={"file": str(fp.relative_to(ROOT))},
            ))
        print(f"{rec_id:55s} {len(files):5d} frames  {failure or 'normal':25s} "
              f"split={env.split_assignment}")

    manifest = Manifest(
        dataset=dataset,
        environments=list(env_registry.values()),
        recordings=recordings,
        consents=[consent],
    )
    dump_manifest(manifest, STAGING)
    write_samples(STAGING, all_samples)
    print(f"manifest written: {len(env_registry)} environments, "
          f"{len(recordings)} recordings, {len(all_samples)} samples")


def _frame_number(name: str) -> int:
    m = FILE_RE.search(name)
    return int(m.group(2)) if m else 0


def _frame_rel_prefix(entry: dict) -> str:
    if entry["source_root"] == "additional_defect_data":
        return (f"{entry['source_root']}/{entry['print_type']}/full_res/"
                f"{entry['camera']}/")
    return (f"{entry['source_root']}/full_res/{entry['print_type']}/"
            f"{entry['camera']}/")


def _plan_index(plan: dict, source_root: str) -> int:
    for i, e in enumerate(plan["extractions"]):
        if e["zip"].startswith(source_root.split("_")[0]):
            return int(e["stride"])
    return 0


if __name__ == "__main__":
    main()