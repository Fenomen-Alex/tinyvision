#!/usr/bin/env python3
"""Stage 14: candidate footage screening via local VLM (Qwen3-VL-8B).

For each candidate video under datasets/external/stage14-failure/raw/<id>/ the
script extracts a fixed set of frames spread across the duration and requests a
neutral, structured single-frame review. The VLM produces PROPOSALS only;
no ground truth is created here, and the prompt never receives source IDs,
filenames, titles, or any recorded metadata.

Output: datasets/external/stage14-failure/screening/candidate_vlm_proposals.json
Incremental upsert by (video_id, frame), resumable.

Usage:
    python scripts/stage14_candidates_review.py [--ids a,b] [--workers N]
    python scripts/stage14_candidates_review.py --dry-run
"""
import argparse
import datetime as _dt
import json
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from tv.vlm_review import (  # noqa: E402
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    call_lmstudio,
    extract_json_object,
    image_to_data_url,
)

BASE = Path("datasets/external/stage14-failure")
RAW = BASE / "raw"
SCREEN = BASE / "screening"
PROPOSALS = SCREEN / "candidate_vlm_proposals.json"
N_FRAMES = 12
FRACTIONS = [0.05 + i * (0.90 / (N_FRAMES - 1)) for i in range(N_FRAMES)]

SCENE_OPTS = {
    "printer_whole", "printer_partial", "nozzle_closeup", "print_closeup",
    "no_printer", "screen_recording", "other",
}
FAILURE_OPTS = {
    "none", "spaghetti", "blobs", "warping", "detached_part", "stringing",
    "under_extrusion", "layers", "other",
}
MAGNITUDE_OPTS = {"none", "mild", "moderate", "severe"}

SCREEN_PROMPT = """Review one frame of 3D-printing footage captured by a monitoring setup, for a candidate-screening step. The image may or may not show a printer actively printing. Answer strictly from visible evidence and return only valid JSON with these fields:
{"scene": "printer_whole | printer_partial | nozzle_closeup | print_closeup | no_printer | screen_recording | other",
"print_in_progress": true | false | uncertain,
"camera_fixed": true | false | uncertain  (rigid stationary angle, no pan/tilt/zoom/cuts),
"printer_model": "short model guess if identifiable, else null",
"visible_failure": "none | spaghetti | blobs | warping | detached_part | stringing | under_extrusion | layers | other",
"failure_magnitude": "none | mild | moderate | severe",
"print_facing_failure_window": true | false | uncertain  (print has recently begun to visibly degrade or is in an obvious failure state),
"real_time": true | false | uncertain  (normal-speed active printing rather than a timelapse or edits),
"visual_evidence": "one or two sentences describing what is visibly happening, including any out-of-place filament"}"""


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def iter_video_files():
    for did in sorted(os.listdir(RAW)):
        dpath = RAW / did
        if not dpath.is_dir():
            continue
        vids = sorted(p for p in dpath.iterdir()
                      if p.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov"))
        if vids:
            yield did, vids[0]


def sample_frames(video_path: Path, out_dir: Path):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    dur = n / fps if fps else 0
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for i, frac in enumerate(FRACTIONS):
        t = dur * frac
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        fname = out_dir / f"r{i:02d}.jpg"
        cv2.imwrite(str(fname), fr)
        samples.append({"index": i, "time_s": round(t, 2),
                        "fraction": round(frac, 3), "file": fname.name})
    cap.release()
    return samples


def validate(obj: dict) -> dict:
    scene = str(obj.get("scene", "")).strip().lower()
    if scene not in SCENE_OPTS:
        raise ValueError(f"invalid scene: {obj.get('scene')!r}")
    pin = str(obj.get("print_in_progress", "")).strip().lower()
    if pin not in ("true", "false", "uncertain"):
        raise ValueError(f"invalid print_in_progress: {obj.get('print_in_progress')!r}")
    cf = str(obj.get("camera_fixed", "")).strip().lower()
    if cf not in ("true", "false", "uncertain"):
        raise ValueError(f"invalid camera_fixed: {obj.get('camera_fixed')!r}")
    vf = str(obj.get("visible_failure", "")).strip().lower()
    if vf not in FAILURE_OPTS:
        raise ValueError(f"invalid visible_failure: {obj.get('visible_failure')!r}")
    mag = str(obj.get("failure_magnitude", "")).strip().lower()
    if mag not in MAGNITUDE_OPTS:
        raise ValueError(f"invalid failure_magnitude: {obj.get('failure_magnitude')!r}")
    fw = str(obj.get("print_facing_failure_window", "")).strip().lower()
    if fw not in ("true", "false", "uncertain"):
        raise ValueError(f"invalid print_facing_failure_window: {obj.get('print_facing_failure_window')!r}")
    rt = str(obj.get("real_time", "")).strip().lower()
    if rt not in ("true", "false", "uncertain"):
        raise ValueError(f"invalid real_time: {obj.get('real_time')!r}")
    ev = obj.get("visual_evidence")
    if not isinstance(ev, str) or not ev.strip():
        raise ValueError("visual_evidence missing or empty")
    pm = obj.get("printer_model")
    if not isinstance(pm, str) and pm is not None:
        raise ValueError(f"invalid printer_model: {pm!r}")
    return {
        "scene": scene,
        "print_in_progress": pin,
        "camera_fixed": cf,
        "printer_model": pm if isinstance(pm, str) else None,
        "visible_failure": vf,
        "failure_magnitude": mag,
        "print_facing_failure_window": fw,
        "real_time": rt,
        "visual_evidence": ev.strip(),
    }


def review_item(did: str, frame: dict, endpoint: str, model: str) -> dict:
    path = RAW / did / "samples" / frame["file"]
    try:
        obj = extract_json_object(
            call_lmstudio(endpoint, model, SCREEN_PROMPT, [image_to_data_url(path)]))
        parsed = validate(obj)
        return {
            "video_id": did,
            "frame_index": frame["index"],
            "time_s": frame["time_s"],
            "fraction": frame["fraction"],
            "file": frame["file"],
            "status": "success",
            **parsed,
            "model": model,
            "generated_at": _now_iso(),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "video_id": did,
            "frame_index": frame["index"],
            "time_s": frame["time_s"],
            "fraction": frame["fraction"],
            "file": frame["file"],
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "generated_at": _now_iso(),
        }


def load_store(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 1, "proposals": []}
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "proposals" not in data:
        raise ValueError(f"malformed proposal store: {path}")
    return data


def write_store(store: dict, path: Path):
    path.write_text(json.dumps(store, indent=2, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=None, help="comma-separated video ids")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    wanted = set(args.ids.split(",")) if args.ids else None
    jobs = []
    for did, vpath in iter_video_files():
        if wanted and did not in wanted:
            continue
        samples = sample_frames(vpath, RAW / did / "samples")
        jobs.append((did, samples))
    if not jobs:
        print("no candidate videos", file=sys.stderr)
        return 2
    total = sum(len(s) for _, s in jobs)
    print(f"plan: {len(jobs)} videos, {total} frames to review via {args.model}")

    if args.dry_run:
        for did, samples in jobs:
            for s in samples:
                print(f"  {did}  t={s['time_s']:>7.1f}s  f={s['fraction']:.3f}")
        return 0

    store = load_store(PROPOSALS)
    done = {(p["video_id"], p["frame_index"]) for p in store.get("proposals", [])
            if p.get("status") == "success"}

    items = [(did, s) for did, samples in jobs for s in samples
             if (did, s["index"]) not in done]
    print(f"existing: {total - len(items)}  to review: {len(items)}")
    if not items:
        print("nothing to do")
        return 0

    def run(item):
        return review_item(item[0], item[1], args.endpoint, args.model)

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run, it): it for it in items}
        for fut in as_completed(futs):
            prop = fut.result()
            existed = [p for p in store["proposals"]
                       if p["video_id"] == prop["video_id"]
                       and p["frame_index"] == prop["frame_index"]]
            store["proposals"] = [p for p in store["proposals"] if p not in existed]
            store["proposals"].append(prop)
            store["generated_at"] = _now_iso()
            write_store(store, PROPOSALS)
            completed += 1
            tag = "OK " if prop["status"] == "success" else "ERR"
            print(f"[{completed}/{len(items)}] {tag} {prop['video_id']} "
                  f"t={prop['time_s']:>7.1f}s f={prop['fraction']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())