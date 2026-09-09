#!/usr/bin/env python3
"""Stage 13: per-frame VLM review of stage13 sample frames (parallel).

Images: datasets/external/stage13-normal/screening/samples/<id>/<sNN>.jpg
Prompt: strict single-frame neutral prompt (see focus_prompt in stage13_vlm).

Reviews run concurrently against local LM Studio (default 4 workers) and are
saved incrementally into screening/vlm_proposals.json under key `<id>_samples`.

Usage (from repo root):
    uv run --project training python scripts/stage13_review.py [--workers 4] [SOURCE...]
"""
from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(ROOT := Path(__file__).resolve().parent.parent / "training"))

from tv.vlm_review import call_lmstudio, extract_json_object, image_to_data_url

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "datasets/external/stage13-normal/screening/samples"
PROPOSALS_OUT = ROOT / "datasets/external/stage13-normal/screening/vlm_proposals.json"

MODEL = "qwen/qwen3-vl-8b"
ENDPOINT = "http://127.0.0.1:1234"

PROMPT = """\
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
 "visual_description": "one concise factual sentence"}"""


def load_proposals() -> dict:
    return json.loads(PROPOSALS_OUT.read_text()) if PROPOSALS_OUT.exists() else {}


def review_one(img: Path) -> dict:
    text = call_lmstudio(ENDPOINT, MODEL, PROMPT, [image_to_data_url(str(img))],
                         timeout=180.0)
    obj = extract_json_object(text)
    decision = obj.get("decision")
    if decision not in ("normal", "failing", "uncertain"):
        raise ValueError(f"bad decision {decision!r}")
    return {
        "status": "ok",
        "model": MODEL,
        "decision": decision,
        "part_visible": bool(obj.get("part_visible")),
        "visual_description": obj.get("visual_description", ""),
    }


def main() -> int:
    args = [a for a in sys.argv[1:]]
    workers = 4
    only = []
    for a in args:
        if a.startswith("--workers"):
            workers = int(a.split("=", 1)[1])
        else:
            only.append(a)

    store = load_proposals()
    lock = threading.Lock()
    tasks: list[tuple[Path, str, str]] = []
    for d in sorted(SAMPLES.iterdir()):
        if not d.is_dir():
            continue
        if only and d.name not in only:
            continue
        key = f"{d.name}_samples"
        reviewed = store.get(key, {})
        for img in sorted(d.rglob("s*.jpg")):
            skey = f"{img.parent.name}_{img.stem}"
            if skey in reviewed and reviewed[skey].get("status") == "ok":
                continue
            tasks.append((img, key, skey))
    print(f"{len(tasks)} samples pending across "
          f"{len({k for _, k, _ in tasks})} sources")

    def work(item: tuple[Path, str, str]) -> None:
        img, key, skey = item
        try:
            res = review_one(img)
        except Exception as e:
            res = {"status": "error", "error": str(e)[:200]}
        with lock:
            store.setdefault(key, {})[skey] = res
            if res["status"] == "ok":
                print(f"  OK {key} {skey}: {res['decision']}"
                      + ("" if res.get("part_visible") else " (part? )"),
                      flush=True)
            PROPOSALS_OUT.write_text(json.dumps(store, indent=2))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in as_completed([pool.submit(work, t) for t in tasks]):
            pass

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())