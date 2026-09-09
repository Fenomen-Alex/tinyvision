#!/usr/bin/env python3
"""Edge-runtime benchmark: measures the C++ `tinyvision` binary end to end.

For each frozen validation video it runs the C++ runtime at the POC sampling
configuration (--sample-fps 1) and, for reference, at native (every-frame)
cadence, then aggregates the JSON report's per-stage timings and measures the
process's CPU usage and peak RSS using psutil.

This is the C++ deployment benchmark (baseline, not optimized). The Python
`tv poc` reference numbers live in docs/research/poc-inference-architecture.md.

Usage (run from repo root):
    python3 scripts/benchmark_edge_runtime.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parent.parent

VIDEOS = {
    "NORMAL001": "datasets/external/video-frames/raw/NORMAL001/NORMAL001_dX4eGEFrthE.mp4",
    "SP001": "datasets/external/video-frames/raw/SP001/SP001_3Ci-3ZYHBkM.mp4",
    "UE003": "datasets/external/video-frames/raw/UE003/UE003_k-inUpjRVLA.mp4",
    "STR001": "datasets/external/video-frames/raw/STR001/STR001_kuAMtoUJGgE.mp4",
}
BIN = REPO / "runtime" / "build" / "tinyvision"
DEFAULT_MODEL = (
    "experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/model.onnx"
)


def _agg(stats: dict) -> dict:
    if not stats or stats.get("count", 0) == 0:
        return {"count": 0}
    return {
        "count": stats["count"],
        "mean_ms": stats["mean_ms"],
        "p50_ms": stats.get("p50_ms", stats["mean_ms"]),
        "p95_ms": stats.get("p95_ms", stats["mean_ms"]),
    }


def run_cpp(video: str, src: str, sample_fps: float, out: Path,
            model: str) -> dict:
    cmd = [
        str(BIN), "--input", src, "--output-json", str(out),
        "--model", model, "--sample-fps", str(sample_fps),
    ]
    proc = psutil.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                        text=True, cwd=str(REPO))
    start = time.perf_counter()
    cpu_sum = 0.0
    peak_rss = 0
    while proc.poll() is None:
        try:
            pm = proc.memory_info()
            peak_rss = max(peak_rss, pm.rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        try:
            ct = proc.cpu_times()
            cpu_sum = max(cpu_sum, ct.user + ct.system)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        threading.Event().wait(0.02)
    wall = time.perf_counter() - start
    err = proc.stderr.read()
    if proc.returncode != 0:
        raise RuntimeError(f"tinyvision failed on {video}: {err[-2000:]}")
    wall = time.perf_counter() - start
    report = json.loads(out.read_text())
    cpu_pct = 100.0 * cpu_sum / wall if wall > 0 else 0.0
    return {"report": report, "peak_rss": peak_rss, "cpu_pct": cpu_pct}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--outdir", default=str(REPO / "docs" / "research" / "edge-bench"))
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, src in VIDEOS.items():
        for sample_fps in (1.0, 0.0):
            tag = "sample1" if sample_fps else "native"
            json_path = out / f"{name}.{tag}.json"
            ctx = run_cpp(name, src, sample_fps, json_path, args.model)
            report = ctx["report"]
            tim = report["timing"]
            dec = _agg(tim["decode_ms"])
            pre = _agg(tim["preprocess_ms"])
            inf = _agg(tim["inference_ms"])
            total = dec["mean_ms"] + pre["mean_ms"] + inf["mean_ms"]
            rows.append({
                "video": name,
                "sampling": tag,
                "frames": tim["processed_frames"],
                "decode_mean_ms": dec["mean_ms"],
                "decode_p95_ms": dec["p95_ms"],
                "preprocess_mean_ms": pre["mean_ms"],
                "inference_mean_ms": inf["mean_ms"],
                "inference_p95_ms": inf["p95_ms"],
                "total_mean_ms": round(total, 4),
                "cadence_fps": round(tim["cadence_fps"], 2),
                "cpu_pct": round(ctx["cpu_pct"], 1),
                "peak_rss_mb": round(ctx["peak_rss"] / 1e6, 1),
            })
            print(f"{name} {tag}: frames={tim['processed_frames']} "
                  f"infer={inf['mean_ms']}ms pre={pre['mean_ms']}ms "
                  f"dec={dec['mean_ms']}ms total={total:.2f}ms "
                  f"fps={tim['cadence_fps']:.1f} cpu={ctx['cpu_pct']:.1f}% "
                  f"rss={ctx['peak_rss']/1e6:.1f}MB")

    result = {
        "benchmark": "edge-runtime-cpp",
        "runtime": "cpp-onnxruntime",
        "model": args.model,
        "platform": sys.platform,
        "videos": rows,
    }
    outpath = REPO / "docs" / "research" / "edge-runtime-benchmark.json"
    outpath.write_text(json.dumps(result, indent=2))
    print("wrote", outpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
