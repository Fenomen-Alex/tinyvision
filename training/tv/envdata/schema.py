"""Canonical schema for the TinyVision environment-centric dataset manifest.

The manifest describes a collection of camera *environments* (sensor + optics +
viewpoint + lighting + visible printer setup). Recordings and frames are never
considered independent units; the environment is the atomic unit for splitting.

Layout under ``datasets/staging/<dataset>/``::

    environments/            JSON dumps, one per environment (derived views)
    recordings/              JSON dumps, one per recording (derived views)
    manifests/env_manifest.json   canonical manifest (dataset + envs + recordings + consents)
    metadata/                provenance / inventory / characterization records
    samples/<recording_id>.jsonl   one JSON object per sample frame
    review/                  review-state snapshots
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_NAME = "tinyvision-env-manifest"
SCHEMA_VERSION = 1

SPLITS = ("train", "validation", "external_holdout")
# Canonical vocabulary for labels that TinyVision itself trusts.
TINYVISION_LABELS = ("normal", "failing")
VERIFICATION_STATES = ("pending", "verified", "rejected", "uncertain")
# Provenance sources that are NOT allowed to set a verified label on their own.
BANNED_VERIFICATION_PROVENANCE = ("source_auto", "source", "vlm_only")
# Provenance prefixes that are acceptable for verified labels.
HUMAN_PROVENANCE = ("human",)
PIPELINE_PROVENANCE = ("pipeline",)
ALLOWED_VERIFIED_PROVENANCE = HUMAN_PROVENANCE + PIPELINE_PROVENANCE

FAILURE_TYPE_VOCAB = (
    "spaghetti",
    "under_extrusion",
    "stringing",
    "bed_adhesion",
    "warping",
    "layer_shift",
    "extrusion_error",
    "other",
)

VIEW_VOCAB = ("front", "side", "overhead", "bed_mount", "inside_chamber", "unknown")

_SIMPLE_VALUE = (str, int, float, bool, type(None))


def _norm_value(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _norm_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_norm_value(x) for x in v]
    if isinstance(v, _SIMPLE_VALUE):
        return v
    raise TypeError(f"manifest values must be JSON-simple, got {type(v).__name__}")


@dataclass
class Environment:
    environment_id: str
    environment_key: str
    source_dataset: str
    contributor: str | None = None
    camera: dict[str, str | None] = field(default_factory=dict)
    view: str = "unknown"
    printer: dict[str, str | None] = field(default_factory=dict)
    environment_group: str | None = None
    split_assignment: str | None = None
    lighting_notes: str | None = None
    background_notes: str | None = None
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "environment_key": self.environment_key,
            "source_dataset": self.source_dataset,
            "contributor": self.contributor,
            "camera": self.camera,
            "view": self.view,
            "printer": self.printer,
            "environment_group": self.environment_group,
            "split_assignment": self.split_assignment,
            "lighting_notes": self.lighting_notes,
            "background_notes": self.background_notes,
            "notes": self.notes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Environment":
        return cls(
            environment_id=data["environment_id"],
            environment_key=data["environment_key"],
            source_dataset=data["source_dataset"],
            contributor=data.get("contributor"),
            camera=dict(data.get("camera") or {}),
            view=data.get("view", "unknown"),
            printer=dict(data.get("printer") or {}),
            environment_group=data.get("environment_group"),
            split_assignment=data.get("split_assignment"),
            lighting_notes=data.get("lighting_notes"),
            background_notes=data.get("background_notes"),
            notes=data.get("notes"),
            metadata=dict(data.get("metadata") or {}),
        )

    def has_failure_type(self) -> bool:
        ft = self.metadata.get("failure_type")
        return isinstance(ft, str) and bool(ft.strip()) and ft != "none"


@dataclass
class Recording:
    recording_id: str
    environment_id: str
    source: str
    duration_s: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    continuous_realtime: bool | None = None
    temporal_note: str | None = None
    md5: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "environment_id": self.environment_id,
            "source": self.source,
            "duration_s": self.duration_s,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "continuous_realtime": self.continuous_realtime,
            "temporal_note": self.temporal_note,
            "md5": self.md5,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Recording":
        return cls(
            recording_id=data["recording_id"],
            environment_id=data["environment_id"],
            source=data["source"],
            duration_s=data.get("duration_s"),
            fps=data.get("fps"),
            width=data.get("width"),
            height=data.get("height"),
            continuous_realtime=data.get("continuous_realtime"),
            temporal_note=data.get("temporal_note"),
            md5=data.get("md5"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class Sample:
    sample_id: str
    recording_id: str
    environment_id: str
    frame_index: int | None = None
    timestamp_s: float | None = None
    source_label: str | None = None
    source_failure_type: str | None = None
    tinyvision_label: str | None = None
    tinyvision_failure_type: str | None = None
    verification: str = "pending"
    verification_provenance: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "recording_id": self.recording_id,
            "environment_id": self.environment_id,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "source_label": self.source_label,
            "source_failure_type": self.source_failure_type,
            "tinyvision_label": self.tinyvision_label,
            "tinyvision_failure_type": self.tinyvision_failure_type,
            "verification": self.verification,
            "verification_provenance": self.verification_provenance,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sample":
        return cls(
            sample_id=data["sample_id"],
            recording_id=data["recording_id"],
            environment_id=data["environment_id"],
            frame_index=data.get("frame_index"),
            timestamp_s=data.get("timestamp_s"),
            source_label=data.get("source_label"),
            source_failure_type=data.get("source_failure_type"),
            tinyvision_label=data.get("tinyvision_label"),
            tinyvision_failure_type=data.get("tinyvision_failure_type"),
            verification=data.get("verification", "pending"),
            verification_provenance=data.get("verification_provenance"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class Consent:
    contributor: str
    consent_date: str | None = None
    training_allowed: bool = False
    evaluation_allowed: bool = False
    redistribution_allowed: bool = False
    attribution: str | None = None
    preferred_credit: str | None = None
    verification: str = "template_signed"
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "contributor": self.contributor,
            "consent_date": self.consent_date,
            "training_allowed": self.training_allowed,
            "evaluation_allowed": self.evaluation_allowed,
            "redistribution_allowed": self.redistribution_allowed,
            "attribution": self.attribution,
            "preferred_credit": self.preferred_credit,
            "verification": self.verification,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Consent":
        return cls(
            contributor=data["contributor"],
            consent_date=data.get("consent_date"),
            training_allowed=bool(data.get("training_allowed", False)),
            evaluation_allowed=bool(data.get("evaluation_allowed", False)),
            redistribution_allowed=bool(data.get("redistribution_allowed", False)),
            attribution=data.get("attribution"),
            preferred_credit=data.get("preferred_credit"),
            verification=data.get("verification", "template_signed"),
            notes=data.get("notes"),
        )


@dataclass
class Dataset:
    dataset_id: str
    name: str
    license: str
    license_url: str | None = None
    provenance: str | None = None
    acquisition_method: str | None = None
    retrieval_date: str | None = None
    url: str | None = None
    redistribution_allowed: bool = False
    training_allowed: bool = False
    evaluation_allowed: bool = False
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "license": self.license,
            "license_url": self.license_url,
            "provenance": self.provenance,
            "acquisition_method": self.acquisition_method,
            "retrieval_date": self.retrieval_date,
            "url": self.url,
            "redistribution_allowed": self.redistribution_allowed,
            "training_allowed": self.training_allowed,
            "evaluation_allowed": self.evaluation_allowed,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Dataset":
        return cls(
            dataset_id=data["dataset_id"],
            name=data["name"],
            license=data["license"],
            license_url=data.get("license_url"),
            provenance=data.get("provenance"),
            acquisition_method=data.get("acquisition_method"),
            retrieval_date=data.get("retrieval_date"),
            url=data.get("url"),
            redistribution_allowed=bool(data.get("redistribution_allowed", False)),
            training_allowed=bool(data.get("training_allowed", False)),
            evaluation_allowed=bool(data.get("evaluation_allowed", False)),
            notes=data.get("notes"),
        )


@dataclass
class Manifest:
    dataset: Dataset
    environments: list[Environment] = field(default_factory=list)
    recordings: list[Recording] = field(default_factory=list)
    consents: list[Consent] = field(default_factory=list)

    def env_index(self) -> dict[str, Environment]:
        return {e.environment_id: e for e in self.environments}

    def recording_index(self) -> dict[str, Recording]:
        return {r.recording_id: r for r in self.recordings}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "dataset": self.dataset.to_dict(),
            "environments": [e.to_dict() for e in self.environments],
            "recordings": [r.to_dict() for r in self.recordings],
            "license_consents": [c.to_dict() for c in self.consents],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        if data.get("schema") != SCHEMA_NAME:
            raise ValueError(f"not a {SCHEMA_NAME} manifest")
        return cls(
            dataset=Dataset.from_dict(data["dataset"]),
            environments=[Environment.from_dict(e) for e in data.get("environments", [])],
            recordings=[Recording.from_dict(r) for r in data.get("recordings", [])],
            consents=[Consent.from_dict(c) for c in data.get("license_consents", [])],
        )


def manifest_path(staging_root: Path) -> Path:
    return Path(staging_root) / "manifests" / "env_manifest.json"


def load_manifest(staging_root: str | Path) -> Manifest:
    path = manifest_path(staging_root)
    with open(path) as f:
        return Manifest.from_dict(json.load(f))


def dump_manifest(manifest: Manifest, staging_root: str | Path, write_views: bool = True) -> None:
    """Write the canonical manifest plus per-record and per-env derived views."""
    root = Path(staging_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "environments").mkdir(parents=True, exist_ok=True)
    (root / "recordings").mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "review").mkdir(parents=True, exist_ok=True)
    (root / "samples").mkdir(parents=True, exist_ok=True)
    with open(manifest_path(root), "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    if write_views:
        for env in manifest.environments:
            _write_atomic(root / "environments" / f"{env.environment_id}.json", env.to_dict())
        rec_jsonl = root / "recordings" / "recordings.jsonl"
        with open(rec_jsonl, "w") as f:
            for rec in manifest.recordings:
                f.write(json.dumps(rec.to_dict()) + "\n")


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(_norm_value(data), f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def load_samples(staging_root: str | Path) -> list[Sample]:
    """Load the per-recording JSONL sample files under ``samples/``."""
    root = Path(staging_root)
    samples: list[Sample] = []
    samples_dir = root / "samples"
    if not samples_dir.exists():
        return samples
    for path in sorted(samples_dir.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(Sample.from_dict(json.loads(line)))
    return samples


def write_samples(staging_root: str | Path, samples: list[Sample]) -> None:
    """Write samples as per-recording JSONL files (idempotent by sample_id)."""
    root = Path(staging_root)
    (root / "samples").mkdir(parents=True, exist_ok=True)
    by_rec: dict[str, list[Sample]] = {}
    for s in samples:
        by_rec.setdefault(s.recording_id, []).append(s)
    for rec_id, rec_samples in by_rec.items():
        rec_samples = sorted(rec_samples, key=lambda s: (s.frame_index is None, s.frame_index))
        out = root / "samples" / f"{rec_id}.jsonl"
        tmp = out.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for s in rec_samples:
                f.write(json.dumps(s.to_dict()) + "\n")
        tmp.replace(out)


def _slug(value: str) -> str:
    value = _norm_key(value)
    return value or "unknown"


def _norm_key(value: str) -> str:
    v = value.strip().lower()
    v = re.sub(r"[^a-z0-9_.-]+", "_", v)
    v = re.sub(r"_+", "_", v).strip("_")
    return v


def environment_key(source: str, contributor: str | None, camera: str | None, view: str | None) -> str:
    """Build the human-readable, unique environment key.

    Convention: ``env_<source>_<contributor>_<camera_identity>_<view>``.
    Unknown fields are rendered as ``unknown`` so the key stays stable once set.
    """
    parts = [_slug(source), _slug(contributor or "unknown"), _slug(camera or "unknown"), _slug(view or "unknown")]
    return "env_" + "_".join(parts)