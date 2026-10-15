"""Temporal streams: ordered frames of one acquisition source.

A stream is the unit of weak supervision for the real-domain pipeline. Its label
comes from a VLM proposal if available and non-uncertain, otherwise from the
video title weak label. Streams are the ``source_key``/``environment_key`` in the
reliability suite, so a stream's frames never span train/val/test.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from tv.data.manifest import Manifest


@dataclass
class Stream:
    id: str
    label: str
    label_source: str
    frames: list[str] = field(default_factory=list)
    failure_type: str | None = None


def _proposal_labels(proposals: dict | None) -> dict[str, str]:
    if not proposals:
        return {}
    return {s["id"]: s.get("video_label", "uncertain")
            for s in proposals.get("sources", [])}


def build_streams(manifest: Manifest, proposals: dict | None = None,
                  include_uncertain: bool = False) -> list[Stream]:
    prop = _proposal_labels(proposals)
    streams: list[Stream] = []
    for src in manifest.sources:
        frames = sorted(str(p.resolve())
                        for p in Path(src.frames_dir).glob(src.frame_glob or "frame_*.jpg"))
        # Prefer the video-level weak (title) label; single-frame VLM video
        # labels proved noisy on 20-frame/256px sampling, so they are a fallback
        # only. (See docs/research/rc2-realdomain-results.md.)
        pl = prop.get(src.id)
        if src.weak_label and src.weak_label != "uncertain":
            label, label_source = src.weak_label, "title"
        elif pl and pl != "uncertain":
            label, label_source = pl, "vlm"
        elif include_uncertain:
            label, label_source = "uncertain", "none"
        else:
            continue
        streams.append(Stream(id=src.id, label=label, label_source=label_source,
                              frames=frames, failure_type=src.failure_type))
    return streams


def write_streams(streams: list[Stream], csv_path: Path, json_path: Path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "stream", "source", "failure_type"])
        for s in streams:
            for frame in s.frames:
                w.writerow([frame, s.label, s.id, s.label_source,
                            s.failure_type or ""])
    payload = {"streams": [
        {"id": s.id, "label": s.label, "label_source": s.label_source,
         "failure_type": s.failure_type, "frame_count": len(s.frames),
         "frames": s.frames}
        for s in streams
    ]}
    Path(json_path).write_text(json.dumps(payload, indent=2) + "\n")


def _partition_ids(ids: list[str], ratios: tuple[float, float, float],
                   seed: int) -> tuple[list[str], list[str], list[str]]:
    """Class-aware partition of stream ids into (train, val, test)."""
    import random

    rng = random.Random(seed)
    ids = sorted(ids)
    rng.shuffle(ids)
    c = len(ids)
    if c <= 1:
        return ids, [], []
    if c == 2:
        return ids[:1], ids[1:], []
    n_val = max(1, round(c * ratios[1]))
    n_test = max(1, round(c * ratios[2]))
    n_train = c - n_val - n_test
    if n_train < 1:
        n_train = 1
        while n_val + n_test > c - n_train and n_test > 0:
            n_test -= 1
    return (ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:])


def split_streams(streams: list[Stream], output_dir: Path,
                  ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
                  seed: int = 42, data_dir: str | None = None) -> dict:
    """Split streams (never frames) into train/val/test, class-balanced.

    No stream appears in more than one split. Classes are partitioned within
    each label so val/test see both classes where the counts allow. Writes
    ``splits.json`` and ``{train,val,test}.csv`` with a ``stream`` column
    (usable as the suite ``source_key``).
    """
    by_label: dict[str, list[Stream]] = {}
    stream_of: dict[str, str] = {}
    for s in streams:
        by_label.setdefault(s.label, []).append(s)
        for f in s.frames:
            stream_of[f] = s.id

    assign: dict[str, str] = {}
    for label, group in by_label.items():
        tr, va, te = _partition_ids([s.id for s in group], tuple(ratios), seed)
        for sid in tr:
            assign[sid] = "train"
        for sid in va:
            assign[sid] = "val"
        for sid in te:
            assign[sid] = "test"

    train: dict[str, list[str]] = {}
    val: dict[str, list[str]] = {}
    test: dict[str, list[str]] = {}
    for s in streams:
        target = {"train": train, "val": val, "test": test}[assign[s.id]]
        target.setdefault(s.label, []).extend(s.frames)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"train": sum(len(v) for v in train.values()),
             "val": sum(len(v) for v in val.values()),
             "test": sum(len(v) for v in test.values())}
    manifest = {"data_dir": data_dir or "", "ratios": list(ratios), "seed": seed,
                "stats": stats,
                "splits": {name: {label: files for label, files in data.items()}
                           for name, data in [("train", train), ("val", val), ("test", test)]}}
    (output_dir / "splits.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for name, data in [("train", train), ("val", val), ("test", test)]:
        with (output_dir / f"{name}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "stream"])
            for label, files in data.items():
                for fp in files:
                    w.writerow([fp, label, stream_of.get(fp, "")])
    return stats


def stream_summary(streams: list[Stream]) -> dict:
    by_label: dict[str, int] = {}
    by_source: dict[str, int] = {}
    n_frames = 0
    for s in streams:
        by_label[s.label] = by_label.get(s.label, 0) + 1
        by_source[s.label_source] = by_source.get(s.label_source, 0) + 1
        n_frames += len(s.frames)
    return {"n_streams": len(streams), "n_frames": n_frames,
            "by_label": by_label, "by_source": by_source}
