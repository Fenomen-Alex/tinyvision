"""Dataset manifest for acquired real-domain footage (research-only)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Source:
    id: str
    url: str
    title: str
    uploader: str
    duration: float
    width: int
    height: int
    retrieved_at: str
    license_note: str
    weak_label: str
    frames_dir: str
    frame_count: int
    failure_type: str | None = None
    # Optional glob for frame files inside ``frames_dir``; empty means the
    # conventional ``frame_*.jpg`` (acquire.extract_frames). Lets datasets whose
    # frames are named differently (e.g. timestamp snapshots) reuse the tooling.
    frame_glob: str = ""
    # Provenance / licensing (added for real-world acquisition; optional so
    # older manifests keep loading). ``license`` is the dataset's actual license
    # as stated by the source, never inferred from the code host or platform.
    source_dataset: str = ""
    license: str = ""
    license_url: str = ""
    attribution: str = ""
    capture_environment: str = ""
    session_id: str = ""
    video_sha256: str = ""
    noncommercial: bool = False
    redistributable: bool = False
    split: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Source":
        # Tolerate manifests written before newer optional fields existed.
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Manifest:
    version: int
    created_at: str
    sources: list[Source] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"version": self.version, "created_at": self.created_at,
                "sources": [s.to_dict() for s in self.sources]}

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        return cls(version=d["version"], created_at=d["created_at"],
                   sources=[Source.from_dict(s) for s in d.get("sources", [])])


def load_manifest(path: Path) -> Manifest:
    return Manifest.from_dict(json.loads(Path(path).read_text()))


def save_manifest(manifest: Manifest, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n")


def validate_manifest(manifest: Manifest, root: Path | None = None) -> list[str]:
    warnings: list[str] = []
    seen: set[str] = set()
    for s in manifest.sources:
        if s.id in seen:
            warnings.append(f"duplicate source id: {s.id}")
        seen.add(s.id)
        if not s.frames_dir or not Path(s.frames_dir).is_dir():
            warnings.append(f"missing frames dir for source {s.id}: {s.frames_dir}")
        if s.frame_count <= 0:
            warnings.append(f"zero frames for source {s.id}")
    return warnings
