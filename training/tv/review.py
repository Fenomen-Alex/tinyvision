"""Stage 12D: human-review workflow for the external pilot footage.

This module builds a REVIEW MANIFEST + contact sheets that a HUMAN reviewer
fills in. It does NOT perform any inference or labeling itself.

Key facts (must be understood before using):
  - This agent has NO image/vision capability. The manifest is explicitly
    produced with empty (null) decisions for a human to complete.
  - Review units are chosen deterministically from the Stage 12B structure
    analysis (`pilot_footage_analysis.json`, per-sample `cut_flag`):
      * SEQUENCE units for continuous stable footage (the analysis strongly
        favors reviewing sequences of ~10 consecutive frames rather than
        isolated frames for failure classification).
      * FRAME units for unstable / cut-heavy regions (a "sequence" is not
        meaningful across scene cuts).
  - No image is inferred from; every unit is grounded in an already-extracted
    1-FPS frame on disk or a freshly-rendered sequence contact sheet.
  - Decisions are recorded as: normal | failing | uncertain, plus optional
    failure_type and optional transition_start/transition_end.
  - The output `verified_labels.json` is only written by the merge step when a
    completed manifest is provided; nothing is labeled automatically.
  - An interactive `review` pass (`scripts/run_human_review.py review`) lets a
    human label units one at a time, opening each unit's visual evidence (contact
    sheet / extracted frame) in the default viewer and saving the manifest after
    every completed unit. It is deliberately BLIND to the Stage 12D-extra VLM
    proposals: it shows only the evidence + neutral metadata, never reads
    `vlm_review_proposals.json`, and never writes `verified_labels.json`.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "datasets" / "external" / "video-frames"
ANALYSIS = EXT / "metadata" / "pilot_footage_analysis.json"
MANIFEST = EXT / "labels" / "review_manifest.json"
SHEETS_DIR = EXT / "labels" / "sheets"
VERIFIED = EXT / "labels" / "verified_labels.json"

# A "stable run" is a maximal contiguous span of seconds during which the
# footage is not unstable. A sample is unstable when it is an explicit camera
# cut (Stage 12B `cut_flag`) or when a broad fraction of the frame changed
# (camera pan / scene change) — matching the Stage 12B definition.
# Minimum run length before we present it as a SEQUENCE unit (seconds).
MIN_SEQUENCE_SECONDS = 6
# Target sequence unit span (seconds). Sequences are grouped in chunks of this
# many consecutive stable seconds so a reviewer sees 10 frames in context.
SEQUENCE_CHUNK_SECONDS = 10
# Maximum width of a contact-sheet tile (frames are downscaled for the sheet).
TILE_MAX = 256

DECISION_VOCAB = {"normal", "failing", "uncertain"}


@dataclass
class SampleMeta:
    time_s: float
    cut_flag: bool
    frac_changed: float
    diff_bbox_frac: float
    seq_index: int  # index of the corresponding extracted sequence frame


def load_analysis() -> dict[str, dict[str, Any]]:
    """Load Stage 12B structure analysis, returning {source_id: {samples...}}.

    The t=0 sample of each source carries no cut/diff fields (no predecessor),
    which we treat as stable (not a cut) by default.
    """
    data = json.loads(ANALYSIS.read_text())
    out: dict[str, dict[str, Any]] = {}
    for sid, v in data["videos"].items():
        samples: list[SampleMeta] = []
        for i, s in enumerate(v["samples"]):
            seq_index = round(float(s["t"]))
            samples.append(SampleMeta(
                time_s=float(s["t"]),
                cut_flag=bool(s.get("cut_flag", False)),
                frac_changed=float(s.get("frac_changed", 0.0)),
                diff_bbox_frac=float(s.get("diff_bbox_frac", 0.0)),
                seq_index=seq_index,
            ))
        out[sid] = {
            "file": v["file"],
            "fps": v["fps"],
            "width": v["width"],
            "height": v["height"],
            "duration_s": v["duration_s"],
            "samples": samples,
        }
    return out


def is_unstable(s: SampleMeta) -> bool:
    """A sample is unstable when it is an explicit cut or a broad scene change.

    Broad change = >30% of pixels changed across >50% of the frame bbox,
    mirroring the Stage 12B unstable definition.
    """
    if s.cut_flag:
        return True
    return s.frac_changed > 0.30 and s.diff_bbox_frac > 0.50


def stable_runs(samples: list[SampleMeta]) -> list[list[SampleMeta]]:
    """Group stable (non-unstable) samples into contiguous runs. Deterministic.

    Each returned list holds consecutive stable SampleMeta objects. The t=0
    sample (no cut info) is treated as stable and begins the first run.
    """
    runs: list[list[SampleMeta]] = []
    current: list[SampleMeta] = []
    for s in samples:
        if is_unstable(s):
            if current:
                runs.append(current)
                current = []
        else:
            current.append(s)
    if current:
        runs.append(current)
    return runs


def define_review_units(source_id: str, samples: list[SampleMeta],
                        min_seconds: int = MIN_SEQUENCE_SECONDS,
                        chunk_seconds: int = SEQUENCE_CHUNK_SECONDS) -> list[dict]:
    """Turn stable runs into review units. Deterministic.

    - Runs of >= min_seconds stable seconds become SEQUENCE units, chunked into
      blocks of `chunk_seconds` consecutive seconds.
    - Every stable second that is not part of a qualifying long run, and every
      unstable/cut second, becomes a FRAME unit (single isolated frame).
    The returned units contain no decision (label is null).
    """
    seq_claimed: set[tuple[str, int]] = set()
    units: list[dict] = []
    for run in stable_runs(samples):
        if len(run) >= min_seconds:
            for start in range(0, len(run), chunk_seconds):
                block = run[start:start + chunk_seconds]
                uid = f"{source_id}__seq{len([u for u in units if u['source_id'] == source_id and u['type'] == 'sequence']):03d}"
                units.append({
                    "review_unit_id": uid,
                    "source_id": source_id,
                    "type": "sequence",
                    "frames": [s.seq_index for s in block],
                    "times": [round(s.time_s, 3) for s in block],
                    "start_time_s": round(block[0].time_s, 3),
                    "end_time_s": round(block[-1].time_s, 3),
                    "n_frames": len(block),
                    "decision": None,
                    "failure_type": None,
                    "transition_start": None,
                    "transition_end": None,
                    "notes": "",
                })
                for s in block:
                    seq_claimed.add((source_id, s.seq_index))
    # Remaining seconds -> isolated frame units (keeps every frame reviewable).
    for s in samples:
        if (source_id, s.seq_index) in seq_claimed:
            continue
        uid = f"{source_id}__frm{s.seq_index:04d}"
        units.append({
            "review_unit_id": uid,
            "source_id": source_id,
            "type": "frame",
            "frames": [s.seq_index],
            "times": [round(s.time_s, 3)],
            "start_time_s": round(s.time_s, 3),
            "end_time_s": round(s.time_s, 3),
            "n_frames": 1,
            "decision": None,
            "failure_type": None,
            "transition_start": None,
            "transition_end": None,
            "notes": "",
        })
    units.sort(key=lambda u: (u["source_id"], u["type"], u["start_time_s"]))
    return units


def frame_path(source_id: str, seq_index: int) -> Path:
    """Path to an extracted 1-FPS sequence frame (index `seq_index`)."""
    return EXT / "extracted" / source_id / "sequence" / f"frame_{seq_index:04d}.jpg"


def sequence_sheet(source_id: str, frames: list[int], out: Path) -> Path:
    """Render a single JPG contact sheet of consecutive frames (labelled tiles)."""
    tiles = []
    for idx in frames:
        p = frame_path(source_id, idx)
        img = cv2.imread(str(p))
        if img is None:
            img = np.zeros((TILE_MAX, TILE_MAX, 3), dtype=np.uint8)
        h, w = img.shape[:2]
        scale = TILE_MAX / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        # pad to square tile
        th, tw = img.shape[:2]
        canvas = np.zeros((TILE_MAX, TILE_MAX, 3), dtype=np.uint8)
        y0, x0 = (TILE_MAX - th) // 2, (TILE_MAX - tw) // 2
        canvas[y0:y0 + th, x0:x0 + tw] = img
        # timestamp label
        label = f"t={idx}.0s"
        cv2.putText(canvas, label, (4, TILE_MAX - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(canvas)
    cols = min(len(tiles), 5)
    rows = (len(tiles) + cols - 1) // cols
    gap = 4
    sheet_h = rows * TILE_MAX + (rows + 1) * gap
    sheet_w = cols * TILE_MAX + (cols + 1) * gap
    sheet = np.full((sheet_h, sheet_w, 3), 40, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y = gap + r * (TILE_MAX + gap)
        x = gap + c * (TILE_MAX + gap)
        sheet[y:y + TILE_MAX, x:x + TILE_MAX] = tile
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    return out


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_manifest(force: bool = False) -> dict:
    """(Re)generate the review manifest + contact sheets. Deterministic.

    All decisions are written as null. Existing decisions are preserved when
    `force=False` (only structural regeneration), or wiped when `force=True`.
    """
    analysis = load_analysis()
    units: list[dict] = []
    per_source: dict[str, dict] = {}
    for sid in sorted(analysis.keys()):
        samples = analysis[sid]["samples"]
        sid_units = define_review_units(sid, samples)
        seq_cnt = sum(1 for u in sid_units if u["type"] == "sequence")
        frm_cnt = sum(1 for u in sid_units if u["type"] == "frame")
        seq_frames = sum(len(u["frames"]) for u in sid_units if u["type"] == "sequence")
        unst = sum(1 for s in samples if is_unstable(s))
        per_source[sid] = {
            "file": analysis[sid]["file"],
            "fps": analysis[sid]["fps"],
            "width": analysis[sid]["width"],
            "height": analysis[sid]["height"],
            "duration_s": analysis[sid]["duration_s"],
            "n_samples": len(samples),
            "unstable_seconds": unst,
            "stable_seconds": len(samples) - unst,
            "n_stable_runs": len(stable_runs(samples)),
            "n_sequence_units": seq_cnt,
            "n_frame_units": frm_cnt,
            "sequence_reviewable_seconds": seq_frames,
            "sequence_reviewable_frames": seq_frames,
        }
        # render sheets for sequence units
        for u in sid_units:
            if u["type"] == "sequence":
                sheet = SHEETS_DIR / f"{u['review_unit_id']}.jpg"
                sequence_sheet(sid, u["frames"], sheet)
        units.extend(sid_units)

    # persist (preserving decisions unless forced)
    old = {}
    if MANIFEST.exists():
        old_data = json.loads(MANIFEST.read_text())
        old = {u["review_unit_id"]: u for u in old_data.get("units", [])}
    new_units = []
    for u in units:
        prev = old.get(u["review_unit_id"])
        if prev is not None and not force:
            u["decision"] = prev.get("decision")
            u["failure_type"] = prev.get("failure_type")
            u["transition_start"] = prev.get("transition_start")
            u["transition_end"] = prev.get("transition_end")
            u["notes"] = prev.get("notes", "")
        new_units.append(u)

    manifest = {
        "stage": "stage12d",
        "manifest_version": 1,
        "instructions": (
            "Human review. For each unit classify the footage as normal, failing, "
            "or uncertain. For SEQUENCE units use the 10-frame contact sheet; a "
            "printing failure is far easier to judge in context than from one frame. "
            "Optionally set failure_type and transition_start/transition_end. "
            "This is ground-truth work; do not rely on any provisional Stage 12B label."
        ),
        "decision_vocab": sorted(DECISION_VOCAB),
        "min_sequence_seconds": MIN_SEQUENCE_SECONDS,
        "sequence_chunk_seconds": SEQUENCE_CHUNK_SECONDS,
        "per_source": per_source,
        "units": new_units,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    return manifest


def completed_count(manifest: dict) -> int:
    return sum(1 for u in manifest["units"] if u.get("decision") in DECISION_VOCAB)


def merge_verified(manifest_path: Path = MANIFEST, out_path: Path = VERIFIED,
                   require_all: bool = True) -> dict:
    """Merge a completed manifest into verified_labels.json (human ground truth).

    Validates every decision against the vocabulary. Writes only human decisions
    (with provenance), never auto-assigned ones. No training occurs here.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    verified: list[dict] = []
    errors: list[str] = []
    for u in manifest["units"]:
        dec = u.get("decision")
        if dec is None:
            if require_all:
                errors.append(f"{u['review_unit_id']}: decision missing")
            continue
        if dec not in DECISION_VOCAB:
            errors.append(f"{u['review_unit_id']}: invalid decision {dec!r}")
            continue
        ft = u.get("failure_type")
        if ft is not None and not isinstance(ft, str):
            errors.append(f"{u['review_unit_id']}: failure_type must be string or null")
            continue
        verified.append({
            "review_unit_id": u["review_unit_id"],
            "source_id": u["source_id"],
            "type": u["type"],
            "frames": u["frames"],
            "times": u["times"],
            "decision": dec,
            "failure_type": ft,
            "transition_start": u.get("transition_start"),
            "transition_end": u.get("transition_end"),
            "notes": u.get("notes", ""),
            "provenance": "human_reviewed",
            "stage": "stage12d",
        })
    if errors:
        raise ValueError("manifest has incomplete/invalid decisions:\n" + "\n".join(errors[:20]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "stage": "stage12d",
        "provenance": "human_reviewed",
        "n_units_verified": len(verified),
        "n_sequence_units": sum(1 for v in verified if v["type"] == "sequence"),
        "n_frame_units": sum(1 for v in verified if v["type"] == "frame"),
        "units": verified,
    }
    out_path.write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Interactive human review
# ---------------------------------------------------------------------------
# One-letter interactive keys -> manifest field values.
_DECISION_KEYS = {"n": "normal", "u": "uncertain"}
_FAILURE_KEYS = {"s": "spaghetti", "t": "stringing", "u": "under_extrusion"}


def select_review_units(
    manifest: dict,
    unit_ids: Sequence[str] | None = None,
    sources: Sequence[str] | None = None,
    sequence_only: bool = False,
) -> list[dict]:
    """Select units for the interactive review in a deterministic order.

    Ordering:
      * ``--unit`` only  -> exactly the supplied order (command-line order).
      * any ``--source`` -> the manifest's existing unit order (stable).
      * neither          -> conservative default: all *undecided* SEQUENCE units
        only. Never the whole 1,006-unit corpus, never FRAME units, and never an
        already-labeled unit (the reviewer must opt in via filters to reach
        those).
    ``sequence_only`` drops FRAME units from any selection.
    Unknown unit ids / sources raise ValueError before anything is shown.
    """
    units = list(manifest["units"])
    by_id = {u["review_unit_id"]: u for u in units}
    known_sources = {u["source_id"] for u in units}
    if unit_ids:
        missing = [uid for uid in unit_ids if uid not in by_id]
        if missing:
            raise ValueError(f"unknown review unit(s): {', '.join(missing)}")
    if sources:
        missing = [s for s in sources if s not in known_sources]
        if missing:
            raise ValueError(f"unknown source(s): {', '.join(missing)}")
    if not unit_ids and not sources:
        return [
            u for u in units
            if u["type"] == "sequence" and u.get("decision") not in DECISION_VOCAB
        ]
    if not sources:
        seen: set[str] = set()
        selected: list[dict] = []
        for uid in unit_ids:
            if uid in seen:
                continue
            seen.add(uid)
            u = by_id[uid]
            if sequence_only and u["type"] != "sequence":
                continue
            selected.append(u)
        return selected
    wanted = set(unit_ids or [])
    selected = [
        u for u in units
        if u["review_unit_id"] in wanted or u["source_id"] in sources
    ]
    if sequence_only:
        selected = [u for u in selected if u["type"] == "sequence"]
    return selected


def visual_evidence_path(unit: dict) -> Path:
    """Return the file a human looks at for this unit.

    SEQUENCE units -> the rendered contact sheet; FRAME units -> the single
    extracted 1-FPS frame. Both are part of the Stage 12D visual evidence.
    """
    if unit.get("type") == "sequence":
        return SHEETS_DIR / f"{unit['review_unit_id']}.jpg"
    return frame_path(unit["source_id"], int(unit["frames"][0]))


def open_visual_evidence(path: Path, print_fn: Callable[[str], None] = print) -> bool:
    """Open an image in the host OS default viewer (non-blocking).

    macOS: `open`; Linux/other Unix: `xdg-open`; Windows: `os.startfile`.
    Returns True when a viewer was launched; False when the file is missing or no
    viewer exists, in which case the absolute path is printed so the reviewer can
    open it manually. Never requires a GUI toolkit.
    """
    path = Path(path)
    if not path.exists():
        print_fn(f"Image not found (open manually): {path.resolve()}")
        return False
    if os.name == "nt":
        try:
            os.startfile(str(path))
            return True
        except OSError:
            print_fn(f"Could not open image (open manually): {path.resolve()}")
            return False
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    try:
        subprocess.Popen([opener, str(path)])
        return True
    except OSError:
        print_fn(f"Could not open image (open manually): {path.resolve()}")
        return False


def save_manifest_atomic(manifest: dict, path: Path = MANIFEST) -> Path:
    """Write the manifest atomically (tmp + fsync + os.replace).

    The full manifest (all units, per-source metadata, instructions) is written
    verbatim apart from the in-memory edits, so unrelated fields are preserved.
    A truncated file can never appear at `path` on interruption.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def _prompt_failure_type(
    input_fn: Callable[[str], str],
    print_fn: Callable[[str], None],
) -> str | None:
    """Ask the reviewer for a failure type after choosing `failing`.

    Returns the canonical name, or None when the reviewer cancels (x) or the
    input stream ends (EOF -> treated as cancel, never a label).
    """
    while True:
        print_fn("Failure type:")
        print_fn("  s = spaghetti")
        print_fn("  t = stringing")
        print_fn("  u = under_extrusion")
        print_fn("  x = cancel")
        try:
            raw = input_fn("Choice: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw in _FAILURE_KEYS:
            return _FAILURE_KEYS[raw]
        if raw == "x":
            return None
        print_fn(f"Invalid choice: {raw!r}")


def interactive_review(
    manifest: dict,
    unit_ids: Sequence[str] | None = None,
    sources: Sequence[str] | None = None,
    sequence_only: bool = False,
    *,
    manifest_path: Path = MANIFEST,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    open_fn: Callable[[Path], bool] | None = None,
    save_fn: Callable[[dict], None] | None = None,
) -> dict:
    """Run one interactive review session over the selected units.

    For each unit it prints neutral identity metadata, opens the visual evidence
    (contact sheet / frame) in the default viewer, and accepts a decision. The
    manifest is saved to disk after every completed unit, so an interrupt or `q`
    never loses already-entered decisions. A `failing` decision always requires a
    failure type (spaghetti/stringing/under_extrusion).

    Blindness: the session never reads `vlm_review_proposals.json`, never shows a
    predicted label, and never writes `verified_labels.json` (that is `merge`'s
    job). EOF/quit is never recorded as a label.
    """
    if save_fn is None:
        def save_fn(m: dict) -> None:
            save_manifest_atomic(m, manifest_path)
    if open_fn is None:
        def open_fn(p: Path) -> bool:
            return open_visual_evidence(p, print_fn=print_fn)

    session = {"reviewed": [], "skipped": []}
    for unit in select_review_units(manifest, unit_ids=unit_ids,
                                    sources=sources, sequence_only=sequence_only):
        if unit.get("decision") in DECISION_VOCAB:
            print_fn(f"Already labeled: {unit['decision']}")
            print_fn("[r] review again")
            print_fn("[s] skip")
            try:
                redo = input_fn("Choice (default s): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print_fn("Stopping; saved decisions preserved.")
                return session
            if redo != "r":
                session["skipped"].append(unit["review_unit_id"])
                continue

        print_fn(f"Review unit: {unit['review_unit_id']}")
        print_fn(f"Type: {unit['type']}")
        print_fn(f"Source: {unit['source_id']}")
        print_fn(f"Start time: {unit.get('start_time_s')} s | "
                 f"End time: {unit.get('end_time_s')} s")
        open_fn(visual_evidence_path(unit))

        outcome = None
        while outcome is None:
            print_fn("Decision:")
            print_fn("  n = normal")
            print_fn("  f = failing")
            print_fn("  u = uncertain")
            print_fn("  s = skip")
            print_fn("  q = quit")
            try:
                raw = input_fn("Choice: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print_fn("Stopping; nothing written for this unit, "
                         "prior units remain saved.")
                return session
            if raw == "q":
                print_fn("Quit; prior decisions remain saved.")
                return session
            if raw == "s":
                session["skipped"].append(unit["review_unit_id"])
                break
            if raw in _DECISION_KEYS:
                unit["decision"] = _DECISION_KEYS[raw]
                unit["failure_type"] = None
                outcome = unit["decision"]
            elif raw == "f":
                failure = _prompt_failure_type(input_fn, print_fn)
                if failure is None:
                    session["skipped"].append(unit["review_unit_id"])
                    break
                unit["decision"] = "failing"
                unit["failure_type"] = failure
                outcome = "failing"
            else:
                print_fn(f"Invalid choice: {raw!r}")
        if outcome is None:
            continue
        try:
            notes = input_fn("Notes (Enter to leave blank): ").strip()
        except (EOFError, KeyboardInterrupt):
            notes = ""
        if notes:
            unit["notes"] = notes
        session["reviewed"].append(unit["review_unit_id"])
        save_fn(manifest)
    return session
