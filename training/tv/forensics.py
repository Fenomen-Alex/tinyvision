"""Dataset forensics: authoritative recording-level audit for the 3D-print-errors data.

The frozen TinyVision splits (train/val/test) are per-frame labeled. The binary
``normal``/``failing`` folder label is a *per-frame* mapping of the raw CSV
``class`` column (0 -> normal; 1/2/4 -> failing), so a single recording can span
both folders when its frames transition good -> defect during the print. There is
**no recording-level label in the source dataset**: the "recording label" used in
earlier stages was a TinyVision construction, not source ground truth.

This module derives the authoritative per-recording facts directly from the raw
metadata CSV joined to the frozen split CSVs, and classifies each recording by
*composition* (PURE_NORMAL / MIXED_* / PURE_FAILING / UNKNOWN_AMBIGUOUS). It
never invents a recording-level ground-truth label.
"""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

TS_RE = re.compile(r"_(\d+)\.(?:png|jpg|jpeg)$")

# Raw-CSV fine-class meanings (the dataset's own per-image label).
FINE_CLASS_NAMES = {
    "0": "good",
    "1": "under_extrusion",
    "2": "stringing",
    "4": "spaghetti",
}
# Binary mapping used to build the prepared folders (see scripts/prepare_dataset.py).
BINARY_BY_CLASS = {"0": "normal", "1": "failing", "2": "failing", "4": "failing"}

# Recording composition taxonomy.
PURE_NORMAL = "PURE_NORMAL"
MIXED_NORMAL_LABEL = "MIXED_NORMAL_LABEL"
MIXED_FAILING_LABEL = "MIXED_FAILING_LABEL"
PURE_FAILING = "PURE_FAILING"
UNKNOWN_AMBIGUOUS = "UNKNOWN_AMBIGUOUS"

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _recording_from_path(path: str) -> Optional[str]:
    parts = path.split("/")
    return parts[1] if len(parts) >= 3 else None


def _timestamp_from_name(name: str) -> Optional[int]:
    m = TS_RE.search(name)
    return int(m.group(1)) if m else None


def load_raw_rows(raw_csv: Path) -> Dict[str, dict]:
    """Map image filename -> raw metadata row (source per-frame class)."""
    out = {}
    with open(raw_csv, newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            out[r["image"]] = r
    return out


def load_split_rows(split_csv: Path) -> List[dict]:
    with open(split_csv) as f:
        return list(csv.DictReader(f))


def _fine_name(raw_row: Optional[dict]) -> str:
    if not raw_row or raw_row.get("class") not in FINE_CLASS_NAMES:
        return "unknown"
    return FINE_CLASS_NAMES[raw_row["class"]]


def _binary_from_fine(fine: str) -> str:
    if fine == "good":
        return "normal"
    if fine == "unknown":
        return "unknown"
    return "failing"


def build_recording_audit(
    raw_csv: Path,
    split_dir: Path,
    splits: tuple = SPLITS,
    seed_name: str = "origin",
) -> Dict[str, dict]:
    """Build a per-recording audit.

    Each recording's split membership and per-frame fine-class labels are the
    *only* source of truth used. The returned structure never assigns a single
    bool to a recording; instead it records composition facts.

    Returns a dict::

        {
          "generated_by": "tv.forensics.build_recording_audit",
          "source": {"raw_csv": ..., "split_dir": ...},
          "recording_label_note": "...",
          "recordings": {
             recording_id: {
                "split": "train"|"val"|"test",
                "frames": <n>,
                "fine_counts": {"good": n, "under_extrusion": n, ...},
                "binary_counts": {"normal": n, "failing": n, "unknown": n},
                "composition": PURE_NORMAL|MIXED_*|PURE_FAILING|UNKNOWN_AMBIGUOUS,
                "first_ts": int|None,
                "last_ts": int|None,
                "starts_good": bool,
                "transitions": n,   # number of good->defect (or defect->good) run boundaries
                "fine_runs": [ "good", ... ],  # collapsed class runs in time order
             }
          },
          "summary": {...per-split + global composition counts...},
        }
    """
    raw = load_raw_rows(raw_csv)

    # recording -> split assignment (a recording belongs to exactly one split).
    rec_split: Dict[str, str] = {}
    # recording -> {fine: count}
    rec_fine: Dict[str, Counter] = defaultdict(Counter)
    rec_ts: Dict[str, List[int]] = defaultdict(list)

    for split in splits:
        scsv = split_dir / f"{split}.csv"
        if not scsv.exists():
            continue
        for row in load_split_rows(scsv):
            rec = _recording_from_path(row["path"])
            if rec is None:
                continue
            rec_split[rec] = split
            fine = _fine_name(raw.get(Path(row["path"]).name))
            rec_fine[rec][fine] += 1
            ts = _timestamp_from_name(Path(row["path"]).name)
            if ts is not None:
                rec_ts[rec].append(ts)

    recordings: Dict[str, dict] = {}
    for rec in sorted(rec_split):
        fine = rec_fine[rec]
        binary = Counter()
        starts_good = None
        fine_runs: List[str] = []
        for f, c in fine.items():
            binary[_binary_from_fine(f)] += c

        # time-ordered fine runs (needs per-frame order; approximate via ts sort here).
        frames_for_runs = _ordered_fine_runs(raw, rec, split_dir, rec_split[rec], rec_fine[rec])
        fine_runs = frames_for_runs["runs"]
        starts_good = frames_for_runs["starts_good"]
        n_transitions = frames_for_runs["transitions"]

        comp = _compose(binary, fine_runs)
        ts_list = sorted(rec_ts.get(rec, []))
        recordings[rec] = {
            "split": rec_split[rec],
            "frames": int(sum(fine.values())),
            "fine_counts": dict(fine),
            "binary_counts": dict(binary),
            "composition": comp,
            "first_ts": ts_list[0] if ts_list else None,
            "last_ts": ts_list[-1] if ts_list else None,
            "n_transitions": n_transitions,
            "fine_runs": fine_runs,
            "starts_good": starts_good,
        }

    summary = _summarize(recordings)
    return {
        "generated_by": "tv.forensics.build_recording_audit",
        "seed_name": seed_name,
        "source": {
            "raw_csv": str(raw_csv),
            "split_dir": str(split_dir),
        },
        "recording_label_note": (
            "The source dataset has no recording-level label; binary labels are "
            "per-frame. Composition describes each recording's frame mix."
        ),
        "recordings": recordings,
        "summary": summary,
    }


def _ordered_fine_runs(raw, rec, split_dir, split, fine_total):
    """Collapse each recording's frames (time-ordered) into fine-class runs.

    Uses the split CSV row order (already a recording grouping) joined to raw
    class per frame, ordered by timestamp. Falls back gracefully when a frame
    has no timestamp (kept in place).
    """
    rows = []
    scsv = split_dir / f"{split}.csv"
    if scsv.exists():
        for row in load_split_rows(scsv):
            if _recording_from_path(row["path"]) != rec:
                continue
            name = Path(row["path"]).name
            fine = _fine_name(raw.get(name))
            rows.append((_timestamp_from_name(name), fine))
    rows.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0))

    runs: List[str] = []
    transitions = 0
    for _ts, fine in rows:
        if not runs or runs[-1] != fine:
            if runs:
                transitions += 1
            runs.append(fine)
    starts_good = (runs[0] == "good") if runs else None
    return {"runs": runs, "starts_good": starts_good, "transitions": transitions}


def _compose(binary: Counter, fine_runs: List[str]) -> str:
    """Classify a recording by composition (no invented recording label)."""
    n_normal = binary.get("normal", 0)
    n_failing = binary.get("failing", 0)
    n_unknown = binary.get("unknown", 0)
    total = n_normal + n_failing + n_unknown
    if total == 0 or not fine_runs:
        return UNKNOWN_AMBIGUOUS
    if n_failing == 0:
        return PURE_NORMAL if n_unknown == 0 else UNKNOWN_AMBIGUOUS
    if n_normal == 0:
        return PURE_FAILING if n_unknown == 0 else UNKNOWN_AMBIGUOUS
    # mixed: has both normal and failing frames
    dominant = "normal" if n_normal >= n_failing else "failing"
    return MIXED_NORMAL_LABEL if dominant == "normal" else MIXED_FAILING_LABEL


def _summarize(recordings: Dict[str, dict]) -> dict:
    comp = Counter()
    per_split = {s: {"recordings": Counter(), "frames": Counter(), "mixed": 0} for s in SPLITS}
    rec_per_split = defaultdict(set)
    for rec, info in recordings.items():
        split = info["split"]
        c = info["composition"]
        comp[c] += 1
        pd = per_split[split]
        pd["recordings"][c] += 1
        pd["frames"]["total"] += info["frames"]
        rec_per_split[split].add(rec)
        if c in (MIXED_NORMAL_LABEL, MIXED_FAILING_LABEL):
            pd["mixed"] += 1
            pd["frames"]["mixed"] = pd["frames"].get("mixed", 0) + info["frames"]
    out = {"composition_total": dict(comp)}
    out["per_split"] = {}
    for s in SPLITS:
        out["per_split"][s] = {
            "n_recordings": len(rec_per_split[s]),
            "composition": dict(per_split[s]["recordings"]),
            "frames_total": per_split[s]["frames"]["total"],
            "frames_in_mixed": per_split[s]["frames"].get("mixed", 0),
            "mixed_recordings": per_split[s]["mixed"],
        }
    return out
