#!/usr/bin/env python3
"""VLM visual review proposals for the real-world evaluation set.

This agent has no image-input capability, so a local Qwen3-VL-8B (LM Studio,
127.0.0.1:1234) plays the role of a *second, non-authoritative* visual observer
— exactly the convention established in Stage 12D-extra: VLM output is an
opinion/proposal, never ground truth, never merged into labels.

Images reviewed (per video):
  * 12-frame labelled grid (temporally ordered) -> overall visual category
  * 5-frame strip around the runtime's first emitted failure event
  * single frames at the model's lowest-P and highest-P sampled frames

The prompt is neutrally-worded (no titles, no channel, no prior predictions, no
runtime output). Proposals are upserted idempotently into
datasets/external/real-world-eval/vlm_proposals.json.

Usage (from repo root):
    uv run --project training python scripts/real_world_vlm.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from tv.vlm_review import call_lmstudio, extract_json_object

ROOT = Path(__file__).resolve().parent.parent
REVIEW_DIR = ROOT / "datasets/external/real-world-eval/review"
PROPOSALS_OUT = ROOT / "datasets/external/real-world-eval/vlm_proposals.json"

MODEL = "qwen/qwen3-vl-8b"
ENDPOINT = "http://127.0.0.1:1234"

PROMPT = """\
You are reviewing frames from one 3D-printing video. Lighter-colored
experiments exist; report only what is actually visible in the pixels.

Grid/strip images are temporally ordered left-to-right, top-to-bottom.

Reply with STRICT JSON only:
{
  "decision": "normal" | "failing" | "uncertain",
  "failure_type": null | "spaghetti" | "stringing" | "under_extrusion",
  "playback": "continuous" | "timelapse" | "edited",
  "visible_evidence": "one factual sentence describing what you actually see",
  "confidence": 0.0-1.0
}

Definitions:
- "normal": a clean object is being built layer by layer with visible structure.
- "failing": visible filament disaster (loose spaghetti/blob mass), or gaps/
  holes/sparse layers (under-extrusion), or thin stringing webs.
- "uncertain": frames too dark, blurry, close-up, occluded, or ambiguous to
  judge; use this rather than guessing.
Use only the pixels; ignore any on-screen text."""

_QUALIFIER = {
    "grid": "A contact sheet of 12 moments from the print.",
    "eventstrip": "These 5 frames span the moments around an alarm.",
    "minP": "Single frame.",
    "maxP": "Single frame.",
}


def load_proposals() -> dict:
    if PROPOSALS_OUT.exists():
        return json.loads(PROPOSALS_OUT.read_text())
    return {}


def main() -> int:
    from tv.vlm_review import image_to_data_url

    store = load_proposals()
    images = sorted(REVIEW_DIR.glob("*.jpg"))
    units = []
    for img in images:
        key = img.stem
        if key in store and store[key].get("status") == "ok":
            continue
        units.append((key, img))

    if not units:
        print("no pending review units")
        return 0

    for key, img in units:
        kind = "grid" if key.endswith("_grid") else (
            "eventstrip" if key.endswith("_eventstrip") else (
                "minP" if key.endswith("_minP") else "maxP"))
        prompt = _QUALIFIER[kind] + "\n\n" + PROMPT
        try:
            text = call_lmstudio(ENDPOINT, MODEL, prompt, [image_to_data_url(img)])
            obj = extract_json_object(text)
            decision = obj.get("decision")
            if decision not in ("normal", "failing", "uncertain"):
                raise ValueError(f"bad decision {decision!r}")
            store[key] = {
                "status": "ok",
                "model": MODEL,
                "image": str(img.relative_to(ROOT)),
                **obj,
            }
            print(f"OK  {key}: decision={decision} conf={obj.get('confidence')}")
        except Exception as e:
            store[key] = {"status": "error", "image": str(img.relative_to(ROOT)),
                          "error": str(e)[:300]}
            print(f"ERR {key}: {e}")
        PROPOSALS_OUT.write_text(json.dumps(store, indent=2))
    print("wrote", PROPOSALS_OUT)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "training"))
    sys.exit(main())