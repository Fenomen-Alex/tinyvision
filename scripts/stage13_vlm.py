#!/usr/bin/env python3
"""Stage 13: VLM screening proposals for candidate normal-print frames.

VLM (local Qwen3-VL on LM Studio) acts only as a non-authoritative proposal
layer, consistent with Stage-12D-extra conventions: proposals are never ground
truth and never merged into labels on their own.

Modes:
  grids - qualitative review of the per-source 12-cell contact sheets.
  cand  - per-cell (144-cell) screening of a source's candidate sheet.

Proposals are stored idempotently in
datasets/external/stage13-normal/screening/vlm_proposals.json.

Usage (from repo root):
    uv run --project training python scripts/stage13_vlm.py grids
    uv run --project training python scripts/stage13_vlm.py cand [SOURCE_ID...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCREEN = ROOT / "datasets/external/stage13-normal/screening"
PROPOSALS_OUT = SCREEN / "vlm_proposals.json"

MODEL = "qwen/qwen3-vl-8b"
ENDPOINT = "http://127.0.0.1:1234"

sys.path.insert(0, str(ROOT / "training"))


def load_proposals() -> dict:
    if PROPOSALS_OUT.exists():
        return json.loads(PROPOSALS_OUT.read_text())
    return {}


def save(store: dict) -> None:
    PROPOSALS_OUT.write_text(json.dumps(store, indent=2))


def grid_prompt() -> str:
    return """\
You are looking at a contact sheet of 12 moments from one fixed-camera,
real-time 3D-printing video. Cells are temporally ordered left-to-right,
top-to-bottom (t01 = earliest ... t12 = latest).

For EACH cell decide whether the printer is currently performing a HEALTHY
NORMAL 3D PRINT:
  "normal"    = print in progress, the printed object is visible and growing;
                no failure visible.
  "failing"   = visible filament disaster (spaghetti/blob), severe stringing,
                gross under-extrusion, or a warped/lifted mess.
  "uncertain" = too dark/blurry/close-up/ambiguous to tell, or the frame shows
                something other than a mid-print state (setup, menus, UI, hands,
                empty bed, finished object held, end screens).

Reply with STRICT JSON only:
{"verdicts": {"t01": "...", ..., "t12": "..."},
 "visual_notes": "one short factual sentence"}

Use only the pixels; ignore on-screen text."""


def cand_prompt() -> str:
    return """\
You are looking at 144 still frames arranged as a 12-by-12 grid, from ONE
fixed, real-time video (no timelapse). The footage is unknown and unlabeled:
it may contain healthy printing, printing failures, or moments with no useful
printing at all. Do not assume anything about the content.

ORDERING: rows advance top-to-bottom, columns advance left-to-right;
top-left = t01 (earliest), bottom-right = t144 (latest). Each row is one
minute of print time, so adjacent frames are close in time.

Inspect EVERY cell individually (t01..t144) and judge what is actually
visible. You must output a JSON object "verdicts" with ALL 144 keys:
  "normal"    : a genuine, healthy 3D print is in progress; the part is
                visible and being built; no failure visible.
  "failing"   : clear failure - spaghetti mass, big blob, severe stringing,
                gross under-extrusion, or warped mess.
  "uncertain" : cannot tell (dark/blurry/occluded/close-up), OR the frame is
                not a useful mid-print view (setup, menus, UI overlay, person,
                empty bed, finished part held, end screen).

Rules:
- Look at each cell before classifying; do not copy one verdict across cells.
- Be strict: ambiguous cells are "uncertain", never "normal".
- Judge only pixels; never infer from watermarks or on-screen overlays.
- If the sheet is too small to tell for a cell, that is "uncertain".

Reply with STRICT JSON only:
{"verdicts": {"t01": "...", ... all 144 cells ...},
 "visual_notes": "one short factual sentence about the overall footage"}

Output the ENTIRE "verdicts" object with ALL 144 keys in a single JSON."""


def focus_prompt() -> str:
    return """\
A single still frame from an unknown, unlabeled 3D-printing video. Look at
the pixels and judge the state of the 3D print, if any:
  "normal"    : a genuine, healthy 3D print is in progress; the part is
                visible and being built; no failure visible.
  "failing"   : clear failure - spaghetti mass, big blob, severe stringing,
                gross under-extrusion, or warped mess.
  "uncertain" : cannot tell (dark/blurry/occluded/close-up), or the frame is
                not a useful mid-print view (setup, menus, UI overlay, person,
                empty bed, finished part held, end screen).

Be strict: ambiguity -> "uncertain".
Reply with STRICT JSON only:
{"decision": "normal|failing|uncertain",
 "part_visible": true|false,
 "failure_type": null|"spaghetti"|"stringing"|"under_extrusion"|"warping"|"other",
 "visual_description": "one concise factual sentence"}"""


def _num_tokens_hint() -> None:
    pass


def parse_144(text: str, fallback_verdicts: dict[str, str]) -> dict[str, str]:
    from tv.vlm_review import extract_json_object

    obj = extract_json_object(text)
    verdicts = obj.get("verdicts")
    if not isinstance(verdicts, dict):
        raise ValueError("no 'verdicts' in VLM JSON")
    # Merge: keep whatever keys came back; caller validates coverage.
    return {k: v for k, v in verdicts.items() if isinstance(v, str)}


def call(images: list[Path], prompt: str) -> str:
    from tv.vlm_review import call_lmstudio, image_to_data_url

    return call_lmstudio(ENDPOINT, MODEL, prompt,
                         [image_to_data_url(str(p)) for p in images],
                         timeout=900.0)


def review_grids(store: dict) -> int:
    from tv.vlm_review import extract_json_object

    done = 0
    for sheet in sorted((SCREEN / "sheets").glob("*_grid.jpg")):
        vid = sheet.stem.removesuffix("_grid")
        if f"{vid}_grid" in store:
            continue
        for attempt in range(2):
            try:
                text = call([sheet], grid_prompt())
                obj = extract_json_object(text)
                verdicts = obj.get("verdicts", {})
                if len(verdicts) != 12:
                    raise ValueError(f"expected 12 verdicts, got {len(verdicts)}")
                store[f"{vid}_grid"] = {
                    "status": "ok", "model": MODEL, "verdicts": verdicts,
                    "visual_notes": obj.get("visual_notes", ""),
                }
                print(f"grid {vid}: normal={sum(v=='normal' for v in verdicts.values())} "
                      f"failing={sum(v=='failing' for v in verdicts.values())} "
                      f"uncertain={sum(v=='uncertain' for v in verdicts.values())}")
                done += 1
                break
            except Exception as e:
                print(f"grid {vid} attempt {attempt}: {e}")
                store[f"{vid}_grid"] = {"status": "error", "error": str(e)[:200]}
        save(store)
    return done


def review_cand(store: dict, vids: list[str]) -> int:
    done = 0
    all_vids = sorted(d.stem for d in (SCREEN / "cand").glob("*.jpg"))
    for vid in (vids or all_vids):
        sheet = SCREEN / "cand" / f"{vid}.jpg"
        if not sheet.exists():
            print(f"cand {vid}: sheet missing"); continue
        if f"{vid}_cand" in store and store[f"{vid}_cand"].get("status") == "ok":
            continue
        for attempt in range(3):
            try:
                text = call([sheet], cand_prompt())
                parsed = parse_144(text, {})
                if len(parsed) != 144:
                    raise ValueError(f"expected 144, got {len(parsed)}")
                store[f"{vid}_cand"] = {
                    "status": "ok", "model": MODEL, "verdicts": parsed,
                    "visual_notes": "",
                }
                vals = list(parsed.values())
                print(f"cand {vid}: normal={sum(v=='normal' for v in vals)} "
                      f"failing={sum(v=='failing' for v in vals)} "
                      f"uncertain={sum(v=='uncertain' for v in vals)}")
                done += 1
                break
            except Exception as e:
                print(f"cand {vid} attempt {attempt}: {e}")
                store[f"{vid}_cand"] = {"status": "error", "error": str(e)[:200]}
        save(store)
    return done


def review_focus(store: dict, vids: list[str]) -> int:
    """Per-frame high-fidelity review of 12 evenly spread frames per source."""
    from tv.vlm_review import extract_json_object

    done = 0
    step = 12  # t01, t13, t25, ... t133
    for d in sorted((SCREEN / "frames").iterdir()):
        if not d.is_dir():
            continue
        vid = d.name
        if vids and vid not in vids:
            continue
        focus = [f"t{i:02d}" for i in range(1, 145, step)]
        key = f"{vid}_focus"
        reviewed = {fname: v for fname, v in store.get(key, {}).items()} if \
            isinstance(store.get(key), dict) else {}
        for fname in focus:
            img = d / f"{fname}.jpg"
            if not img.exists() or fname in reviewed:
                continue
            for attempt in range(2):
                try:
                    text = call([img], focus_prompt())
                    obj = extract_json_object(text)
                    decision = obj.get("decision")
                    if decision not in ("normal", "failing", "uncertain"):
                        raise ValueError(f"bad decision {decision!r}")
                    store.setdefault(key, {})[fname] = {
                        "status": "ok", "model": MODEL,
                        "decision": decision,
                        "part_visible": bool(obj.get("part_visible")),
                        "failure_type": obj.get("failure_type"),
                        "visual_description": obj.get("visual_description", ""),
                    }
                    done += 1
                except Exception as e:
                    store.setdefault(key, {})[fname] = {
                        "status": "error", "error": str(e)[:200]}
                save(store)
        print(f"focus {vid}: " + ", ".join(
            f"{f}={v.get('decision','?')}" for f, v in
            sorted(store[key].items()) if v.get("status") == "ok"))
    return done


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "grids"
    vids = sys.argv[2:]
    store = load_proposals()
    if mode == "grids":
        n = review_grids(store)
    elif mode == "cand":
        n = review_cand(store, vids)
    elif mode == "focus":
        n = review_focus(store, vids)
    else:
        print("unknown mode"); return 2
    print(f"done {n}; total proposals {len(store)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())