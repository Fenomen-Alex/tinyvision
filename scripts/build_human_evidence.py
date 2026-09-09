"""Render focused, high-resolution evidence sheets for human verification candidates.

The default review sheets downscale source frames to 256 px tiles, which hides
extrusion defects. For VLM-supported human-verification candidates this script
re-renders a small number of sequence units directly from the full-resolution
source frames at large tile size, writing sheets to

    <staging>/review/evidence/<environment_key>/<review_unit_id>.jpg

and (re)generating the human review set at

    <staging>/review/human_candidates.json

with the shape:

    environment_id, environment_key, failure_type (hypothesis),
    candidate_units [{ review_unit_id, role, vlm_confidence, frames,
                       evidence_jpg, source_frames }], vlm_confidence, reason

Usage:

    python scripts/build_human_evidence.py --dir datasets/staging/trier \
        --candidates <path-to-candidates.json>   # optional: reuse definitions

The candidate definitions live in ``datasets/staging/trier/review/human_candidates.json``
(payload `.definitions`); if absent this script only renders the definitions it
is given via ``--candidates``. Re-running is idempotent.

This script records proposals only: it never writes review units, samples, or
verified labels, and it never confirms or merges anything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

TILE = 768
COLS = 5

DEFAULT_CANDIDATES = None  # provided via --candidates or <dir>/review/candidate_definitions.json


def load_samples_map(samples_dir: Path) -> dict[str, dict[int, Path]]:
    """recording_id -> {frame_index: absolute source path} from staged samples."""
    out: dict[str, dict[int, Path]] = {}
    for f in sorted(samples_dir.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            p = Path(r["metadata"]["file"])
            if not p.is_absolute():
                p = ROOT / p
            out.setdefault(r["recording_id"], {})[r["frame_index"]] = p
    return out


def load_units(path: Path) -> dict[str, dict]:
    units = json.loads(path.read_text())["units"]
    return {u["review_unit_id"]: u for u in units}


def load_proposals(path: Path) -> dict[str, dict]:
    store = json.loads(path.read_text())["proposals"]
    return {p["review_unit_id"]: p for p in store}


def render_focused_sheet(unit: dict, frames: list[int], times: list[float],
                         samples: dict[int, Path], out: Path) -> list[str]:
    """Render selected source frames as a large-tile sheet. Returns source paths."""
    src: list[str] = []
    tiles = []
    for idx, t in zip(frames, times):
        p = samples.get(idx)
        src.append(str(p) if p else None)
        img = cv2.imread(str(p)) if p else None
        if img is None:
            tile = np.zeros((TILE, TILE, 3), dtype=np.uint8)
            cv2.putText(tile, "missing", (60, TILE // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
        else:
            h, w = img.shape[:2]
            scale = TILE / max(h, w)
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
            th, tw = img.shape[:2]
            tile = np.zeros((TILE, TILE, 3), dtype=np.uint8)
            y0, x0 = (TILE - th) // 2, (TILE - tw) // 2
            tile[y0:y0 + th, x0:x0 + tw] = img
        cv2.putText(tile, f"t={t:.0f}s", (8, TILE - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 2, cv2.LINE_AA)
        tiles.append(tile)
    cols = min(len(tiles), COLS)
    rows = (len(tiles) + cols - 1) // cols
    gap = 8
    sheet = np.full((rows * TILE + (rows + 1) * gap, cols * TILE + (cols + 1) * gap, 3),
                    30, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y = gap + r * (TILE + gap)
        x = gap + c * (TILE + gap)
        sheet[y:y + TILE, x:x + TILE] = tile
    title = f"{unit['recording_id']}  {unit['review_unit_id']}"
    cv2.putText(sheet, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    return [s for s in src if s]


def main() -> None:
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--candidates", type=Path, default=None,
                    help="JSON with {'definitions': [{environment_key, failure_type, "
                         "environment_id, units:[{id, role}]}]}")
    args = ap.parse_args()

    staging = Path(args.dir)
    ROOT = Path(json.loads((staging / "metadata" / "index.json").read_text())["root"]).resolve()

    units = load_units(staging / "review" / "review_units.json")
    props = load_proposals(staging / "review" / "vlm_proposals.json")
    manifest = json.loads((staging / "manifests" / "env_manifest.json").read_text())["environments"]
    env_key_by_id = {e["environment_id"]: e["environment_key"] for e in manifest}
    samples = load_samples_map(staging / "samples")

    defs_path = args.candidates if args.candidates else staging / "review" / "candidate_definitions.json"
    if defs_path.exists():
        defs = json.loads(defs_path.read_text()).get("definitions", [])
    else:
        raise SystemExit(f"no candidate definitions found at {defs_path} (use --candidates)")

    entries = []
    for d in defs:
        cand_units = []
        for cu in d["units"]:
            u = units.get(cu["id"])
            if not u:
                print(f"skip: unknown unit {cu['id']}")
                continue
            eff = list(cu.get("frames") or u.get("frames"))
            times = list(cu.get("times") or u.get("times"))
            step = max(1, len(eff) // 8)
            picks = eff[::step][:8]
            times_p = [times[eff.index(f)] for f in picks]
            env_key = env_key_by_id.get(u["environment_id"], d["environment_key"])
            name = cu["id"] + ("_" + cu["role"] if cu.get("role") else "")
            out = staging / "review" / "evidence" / env_key / f"{name}.jpg"
            src = render_focused_sheet(u, picks, times_p, samples.get(u["recording_id"], {}), out)
            prop = props.get(cu["id"], {})
            cand_units.append({
                "review_unit_id": cu["id"],
                "role": cu.get("role", "failing"),
                "vlm_decision": prop.get("decision"),
                "vlm_failure_type": prop.get("failure_type"),
                "vlm_confidence": prop.get("confidence"),
                "start_time_s": u["start_time_s"],
                "end_time_s": u["end_time_s"],
                "evidence_jpg": str(out),
                "fullres_source_frames": src,
            })
        if not cand_units:
            continue
        unit0 = units[cand_units[0]["review_unit_id"]]
        entries.append({
            "environment_id": d.get("environment_id") or unit0["environment_id"],
            "environment_key": env_key_by_id.get(unit0["environment_id"], d["environment_key"]),
            "failure_type": d["failure_type"],
            "candidate_units": cand_units,
            "vlm_confidence": max((c["vlm_confidence"] or 0 for c in cand_units), default=None),
            "reason": d.get("reason", ""),
        })

    payload = {
        "schema": "tinyvision-human-candidates",
        "version": 2,
        "note": "VLM-supported candidate set for surgical human verification. "
                "Proposals only; nothing is confirmed or merged. Review the high-res "
                "evidence_jpg files, not the 256px default sheets.",
        "definitions": defs,
        "environments": entries,
    }
    out_path = staging / "review" / "human_candidates.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path}  ({len(entries)} environments)")


if __name__ == "__main__":
    main()