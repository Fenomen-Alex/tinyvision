"""Review-unit builder for staged environment manifests.

Follows the Stage-12D conventions (see ``tv/review.py``): a HUMAN reviewer
labels sequence/frame units; no inference or automatic labeling happens here.

Units are grounded in 1-fps extracted frame files at ``frame_root/<recording_id>/frame_<NNNN>.jpg``.
A sample gap (missing frame_index step of >1) breaks a sequence: continuity is
required for a reviewer to judge normal vs failing reliably.

Priorities (from the release-dataset build ordering):
  1. new failure environments
  2. new normal train environments
  3. new normal external holdouts
  4. new failure external holdouts

Merge maps completed decisions onto the staged samples through the envdata
verification state machine with provenance ``human:release-review``:
  normal        -> verification=verified, tinyvision_label=normal
  failing       -> verification=verified, tinyvision_label=failing (+failure_type)
  uncertain     -> verification=uncertain (never auto-verified)
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tv.envdata.schema import Manifest, Sample, VERIFICATION_STATES
from tv.envdata.state import set_verification

DECISION_VOCAB = ("normal", "failing", "uncertain")
MIN_SEQUENCE_FRAMES = 6
SEQUENCE_CHUNK_FRAMES = 10
TILE_MAX = 256
PROVENANCE = "human:release-review"


@dataclass
class ReviewUnit:
    review_unit_id: str
    recording_id: str
    environment_id: str
    environment_key: str
    type: str  # sequence | frame
    frames: list[int] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    priority: int = 3
    decision: str | None = None
    failure_type: str | None = None
    notes: str = ""
    instructions: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_unit_id": self.review_unit_id,
            "recording_id": self.recording_id,
            "environment_id": self.environment_id,
            "environment_key": self.environment_key,
            "type": self.type,
            "frames": self.frames,
            "times": self.times,
            "start_time_s": self.times[0] if self.times else None,
            "end_time_s": self.times[-1] if self.times else None,
            "n_frames": len(self.frames),
            "priority": self.priority,
            "decision": self.decision,
            "failure_type": self.failure_type,
            "notes": self.notes,
            "instructions": self.instructions,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReviewUnit":
        return cls(
            review_unit_id=d["review_unit_id"],
            recording_id=d["recording_id"],
            environment_id=d["environment_id"],
            environment_key=d.get("environment_key", ""),
            type=d["type"],
            frames=list(d.get("frames") or []),
            times=list(d.get("times") or []),
            priority=int(d.get("priority", 3)),
            decision=d.get("decision"),
            failure_type=d.get("failure_type"),
            notes=d.get("notes", ""),
            instructions=d.get("instructions", ""),
        )


def _run_sequences(frame_indices: list[int], max_gap: int = 1) -> list[list[int]]:
    """Split frame indices into runs; ``max_gap`` tolerates strided extraction."""
    runs: list[list[int]] = []
    current: list[int] = []
    for i, idx in enumerate(frame_indices):
        if current and idx - current[-1] > max_gap:
            runs.append(current)
            current = []
        current.append(idx)
    if current:
        runs.append(current)
    return runs


def build_review_units(
    manifest: Manifest,
    samples: list[Sample],
    min_sequence_frames: int = MIN_SEQUENCE_FRAMES,
    chunk_frames: int = SEQUENCE_CHUNK_FRAMES,
    max_gap: int = 1,
) -> list[ReviewUnit]:
    """Build review units deterministically from the staged samples.

    ``reviewed_state`` restricts the output to units whose *decision* field is
    currently unset (default None => all samples considered; decisions are
    preserved from existing manifests by the CLI instead).
    """
    env_by_id = manifest.env_index()
    by_rec: dict[str, list[Sample]] = {}
    seen_env: dict[str, str] = {}
    for s in samples:
        by_rec.setdefault(s.recording_id, []).append(s)
        prev = seen_env.get(s.recording_id)
        if prev is not None and prev != s.environment_id:
            raise ValueError(
                f"recording {s.recording_id} has samples in multiple environments"
            )
        seen_env[s.recording_id] = s.environment_id

    units: list[ReviewUnit] = []
    for rec_id in sorted(by_rec):
        rec_samples = sorted(
            by_rec[rec_id], key=lambda s: (s.frame_index is None, s.frame_index)
        )
        if any(s.frame_index is None for s in rec_samples):
            # non-sequence-able samples -> one frame unit each
            for s in rec_samples:
                units.append(ReviewUnit(
                    review_unit_id=f"{rec_id}__frm{s.sample_id}",
                    recording_id=rec_id,
                    environment_id=seen_env[rec_id],
                    environment_key=_env_key(env_by_id, seen_env[rec_id]),
                    type="frame",
                    frames=[],
                    times=[s.timestamp_s] if s.timestamp_s is not None else [],
                    priority=0,
                ))
            continue
        indices = [s.frame_index for s in rec_samples]
        times = {s.frame_index: s.timestamp_s for s in rec_samples}
        seq_counter = 0
        frame_counter = 0
        for run in _run_sequences(indices, max_gap):
            if len(run) >= min_sequence_frames:
                for start in range(0, len(run), chunk_frames):
                    block = run[start:start + chunk_frames]
                    units.append(ReviewUnit(
                        review_unit_id=f"{rec_id}__seq{seq_counter:03d}",
                        recording_id=rec_id,
                        environment_id=seen_env[rec_id],
                        environment_key=_env_key(env_by_id, seen_env[rec_id]),
                        type="sequence",
                        frames=block,
                        times=[times[i] for i in block],
                        instructions=(
                            "Classify the whole captured sequence: a print that "
                            "is advancing normally => normal; a defect is seen => "
                            "failing + failure_type; must see the print in frame "
                            "and continuous footage."
                        ),
                    ))
                    seq_counter += 1
            else:
                for idx in run:
                    units.append(ReviewUnit(
                        review_unit_id=f"{rec_id}__frm{frame_counter:04d}",
                        recording_id=rec_id,
                        environment_id=seen_env[rec_id],
                        environment_key=_env_key(env_by_id, seen_env[rec_id]),
                        type="frame",
                        frames=[idx],
                        times=[times[idx]],
                    ))
                    frame_counter += 1
    units.sort(key=lambda u: (u.priority, u.environment_key, u.recording_id, u.review_unit_id))
    return units


def _env_key(env_by_id: dict[str, Any], env_id: str) -> str:
    env = env_by_id.get(env_id)
    return env.environment_key if env is not None else env_id


def assign_priorities(units: list[ReviewUnit], manifest: Manifest) -> list[ReviewUnit]:
    """Set priority from environment role (spec ordering):
    1 new failure envs, 2 new normal envs, 3 normal holdouts, 4 failure holdouts.
    """
    env_by_id = manifest.env_index()
    for u in units:
        env = env_by_id.get(u.environment_id)
        is_failure = bool(env and env.has_failure_type())
        is_holdout = bool(env and env.split_assignment == "external_holdout")
        if is_failure and is_holdout:
            u.priority = 4
        elif is_failure:
            u.priority = 1
        elif is_holdout:
            u.priority = 3
        else:
            u.priority = 2
    units.sort(key=lambda u: (u.priority, u.environment_key, u.recording_id, u.review_unit_id))
    return units


def frame_image_path(frame_root: Path, recording_id: str, frame_index: int | None) -> Path:
    if frame_index is None:
        raise ValueError("frame unit needs a frame_index")
    return frame_root / recording_id / f"frame_{frame_index:04d}.jpg"


def _resolve(frame_root: Path, recording_id: str, frame_index: int,
             resolver: "Callable[[str, int], Path | None] | None") -> Path:
    if resolver is not None:
        p = resolver(recording_id, frame_index)
        if p is not None:
            return p
    return frame_image_path(frame_root, recording_id, frame_index)


def render_contact_sheet(
    frame_root: Path,
    unit: ReviewUnit,
    out: Path,
    resolver: "Callable[[str, int], Path | None] | None" = None,
) -> Path:
    """Render a JPG contact sheet for a sequence unit (missing frames -> blank)."""
    import cv2
    import numpy as np

    tiles = []
    for idx in unit.frames:
        p = _resolve(frame_root, unit.recording_id, idx, resolver)
        img = cv2.imread(str(p))
        if img is None:
            img = np.zeros((TILE_MAX, TILE_MAX, 3), dtype=np.uint8)
        h, w = img.shape[:2]
        scale = TILE_MAX / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        th, tw = img.shape[:2]
        canvas = np.zeros((TILE_MAX, TILE_MAX, 3), dtype=np.uint8)
        y0, x0 = (TILE_MAX - th) // 2, (TILE_MAX - tw) // 2
        canvas[y0:y0 + th, x0:x0 + tw] = img
        cv2.putText(canvas, f"t={idx}.0s", (4, TILE_MAX - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(canvas)
    cols = min(len(tiles), 5)
    rows = (len(tiles) + cols - 1) // cols
    gap = 4
    sheet = np.full((rows * TILE_MAX + (rows + 1) * gap,
                     cols * TILE_MAX + (cols + 1) * gap, 3), 40, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y = gap + r * (TILE_MAX + gap)
        x = gap + c * (TILE_MAX + gap)
        sheet[y:y + TILE_MAX, x:x + TILE_MAX] = tile
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    return out


def load_review_units(path: Path) -> list[ReviewUnit]:
    data = json.loads(Path(path).read_text())
    return [ReviewUnit.from_dict(u) for u in data["units"]]


def write_review_units(path: Path, units: list[ReviewUnit]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "tinyvision-env-review-units",
        "version": 1,
        "provenance": PROVENANCE,
        "decision_vocab": list(DECISION_VOCAB),
        "n_units": len(units),
        "n_sequence_units": sum(1 for u in units if u.type == "sequence"),
        "n_frame_units": sum(1 for u in units if u.type == "frame"),
        "units": [u.to_dict() for u in units],
    }
    json.dump(payload, open(path, "w"), indent=2)
    return payload


def merge_completed(
    review_path: Path,
    manifest: Manifest,
    samples: list[Sample],
    require_all: bool = False,
) -> dict[str, Any]:
    """Apply completed human decisions back onto the staged samples.

    Only explicit decisions are applied; each is validated by the state machine.
    ``uncertain`` moves samples to state uncertain (never verified).
    Returns counts including verified environments (per gate definition: an
    environment with at least one human verified failing sample).
    """
    units = load_review_units(review_path)
    by_id = {u.review_unit_id: u for u in units}
    rec_for_sample = {s.sample_id: s for s in samples}
    applied = 0
    rejected: list[str] = []
    verified_failing_envs: set[str] = set()
    verified_normal_envs: set[str] = set()
    for u in units:
        if u.decision not in DECISION_VOCAB:
            if require_all:
                rejected.append(f"{u.review_unit_id}: missing decision")
            continue
        if u.failure_type is not None and (u.decision != "failing"):
            rejected.append(f"{u.review_unit_id}: failure_type set but decision != failing")
            continue
        targets = [s for s in samples if s.recording_id == u.recording_id
                   and (not u.frames or s.frame_index in u.frames)]
        if not targets:
            rejected.append(f"{u.review_unit_id}: no samples matched")
            continue
        for s in targets:
            try:
                if u.decision == "normal":
                    set_verification(s, "verified", PROVENANCE, tinyvision_label="normal")
                    verified_normal_envs.add(s.environment_id)
                elif u.decision == "failing":
                    set_verification(s, "verified", PROVENANCE,
                                     tinyvision_label="failing",
                                     tinyvision_failure_type=u.failure_type)
                    verified_failing_envs.add(s.environment_id)
                else:
                    set_verification(s, "uncertain", PROVENANCE)
            except Exception as exc:  # noqa: BLE001
                rejected.append(f"{u.review_unit_id}: {exc}")
        applied += 1
    if rejected and require_all:
        raise ValueError("review merge incomplete:\n" + "\n".join(rejected[:20]))
    return {
        "applied_units": applied,
        "samples_updated": sum(
            1 for s in samples if s.verification_provenance == PROVENANCE
        ),
        "rejected_units": len(rejected),
        "rejections": rejected[:40],
        "verified_failing_environments": sorted(verified_failing_envs),
        "verified_normal_environments": sorted(verified_normal_envs),
    }