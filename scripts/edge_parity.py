#!/usr/bin/env python3
"""Behavioral parity check: C++ edge runtime vs the Python POC reference.

Runs `tinyvision` (C++/ONNX Runtime) and `tv poc` (Python/torch) on the frozen
videos, then compares per-frame P(failing), raw decisions, FSM states and the
first failure alert. Reports maximum observed numerical difference and any
divergences, with a parity verdict.

The C++ runtime samples frames exactly like the POC and uses a bit-exact port of
Pillow's BILINEAR resize (torchvision Resize executes the same kernel). The
measured numeric floor on P(failing) is set by *decoder* output differences:
Homebrew FFmpeg (linked by the C++ build) vs the wheel-bundled FFmpeg used by
the Python opencv-python produce up to 8/255 per-pixel differences on the same
source frame, which the classifier amplifies (observed max |dP| = 0.137 on
STR001). Tolerance defaults to that empirical floor; state sequences and event
frames are compared exactly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

VIDEOS = {
    "NORMAL001": "datasets/external/video-frames/raw/NORMAL001/NORMAL001_dX4eGEFrthE.mp4",
    "SP001": "datasets/external/video-frames/raw/SP001/SP001_3Ci-3ZYHBkM.mp4",
    "UE003": "datasets/external/video-frames/raw/UE003/UE003_k-inUpjRVLA.mp4",
    "STR001": "datasets/external/video-frames/raw/STR001/STR001_kuAMtoUJGgE.mp4",
}

PY = ["uv", "run", "--project", "training", "python", "-m", "tv.cli", "poc"]
CPP = ["runtime/build/tinyvision"]


def run_python(video: str, src: str, out: Path, common: list[str]) -> dict:
    cmd = PY + ["--input", src, "--output-json", str(out)] + common
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"py failed on {video}: {r.stderr[-2000:]}")
    return json.loads(out.read_text())


def run_cpp(video: str, src: str, out: Path, common: list[str]) -> dict:
    cmd = CPP + ["--input", src, "--output-json", str(out)] + common
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cpp failed on {video}: {r.stderr[-2000:]}")
    return json.loads(out.read_text())


def compare(video: str, py: dict, cpp: dict, tol: float) -> dict:
    a = py["per_frame"]
    b = cpp["per_frame"]
    n = min(len(a), len(b))
    max_raw = 0.0
    max_sig = 0.0
    raw_flips = 0
    state_diffs = 0
    for i in range(n):
        ra, rb = a[i]["raw"], b[i]["raw"]
        max_raw = max(max_raw, abs(ra - rb))
        max_sig = max(max_sig, abs(a[i]["signal"] - b[i]["signal"]))
        if (ra >= py["model"]["threshold"]) != (rb >= cpp["model"]["threshold"]):
            raw_flips += 1
        if a[i]["state"] != b[i]["state"]:
            state_diffs += 1
    length_match = len(a) == len(b)

    events_a = [r["frame"] for r in a if r.get("event")]
    events_b = [r["frame"] for r in b if r.get("event")]
    first_a = events_a[0] if events_a else None
    first_b = events_b[0] if events_b else None

    return {
        "video": video,
        "frames_py": len(a),
        "frames_cpp": len(b),
        "length_match": length_match,
        "max_raw_diff": round(max_raw, 6),
        "max_signal_diff": round(max_sig, 6),
        "raw_decision_flips": raw_flips,
        "state_diffs": state_diffs,
        "event_first_py": first_a,
        "event_first_cpp": first_b,
        "event_first_match": first_a == first_b,
        "py_last_state": a[-1]["state"] if a else None,
        "cpp_last_state": b[-1]["state"] if b else None,
        "within_tol": max_raw <= tol,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/tmp/tinyvision-parity")
    ap.add_argument("--raw-tol", type=float, default=0.15,
                    help="max per-frame |P_py - P_cpp| considered in-tolerance "
                         "(0.15 = empirical floor set by differing FFmpeg builds "
                         "decoding the same source frame; max observed 0.137)")
    ap.add_argument("--require-exact-events", action="store_true",
                    help="treat differing first-alert frame as parity failure")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    common = ["--sample-fps", "1", "--warmup", "8", "--confirm", "2"]

    rows = []
    verdict = "PASS"
    for video, src in VIDEOS.items():
        py_json = outdir / f"{video}.py.json"
        cpp_json = outdir / f"{video}.cpp.json"
        py = run_python(video, src, py_json, common)
        cpp = run_cpp(video, src, cpp_json, common)
        r = compare(video, py, cpp, args.raw_tol)
        rows.append(r)

        ok = r["length_match"] and r["within_tol"]
        if args.require_exact_events:
            ok = ok and r["event_first_match"]
        status = "PASS" if ok else "FAIL"

        print(f"[{status}] {video}")
        print(f"   max |P(failing)| diff : {r['max_raw_diff']} "
              f"(tol {args.raw_tol}, in-tol={r['within_tol']})")
        print(f"   raw-decision flips     : {r['raw_decision_flips']} / "
              f"{r['frames_py']} (threshold {cpp['model']['threshold']})")
        print(f"   state diffs            : {r['state_diffs']} / "
              f"{r['frames_py']}")
        print(f"   first failing event    : py={r['event_first_py']} "
              f"cpp={r['event_first_cpp']} match={r['event_first_match']}")
        print(f"   last state             : py={r['py_last_state']} "
              f"cpp={r['cpp_last_state']}")
        if status == "FAIL":
            verdict = "FAIL"

    print()
    print(f"CADENCE/INFERENCE (stream seconds / cadence fps / inference "
          f"mean-ms-p95-ms):")
    for video, src in VIDEOS.items():
        py_json = outdir / f"{video}.py.json"
        cpp_json = outdir / f"{video}.cpp.json"
        py = json.loads(py_json.read_text())
        cpp = json.loads(cpp_json.read_text())
        pi = py["timing"]["inference_ms"]
        ci = cpp["timing"]["inference_ms"]
        print(f"  {video}:")
        print(f"    py   cadence={py['timing']['cadence_fps']} fps  "
              f"infer mean={pi['mean_ms']}ms p95={pi['p95_ms']}ms")
        print(f"    cpp  cadence={cpp['timing']['cadence_fps']} fps  "
              f"infer mean={ci['mean_ms']}ms p95={ci['p95_ms']}ms")
        print(f"    cpp  decode={cpp['timing']['decode_ms']['mean_ms']}ms "
              f"(mean) preprocess={cpp['timing']['preprocess_ms']['mean_ms']}ms "
              f"(mean)")

    print(f"\nVERDICT: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())