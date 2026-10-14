#!/usr/bin/env python3
"""Evaluate temporal/runtime outputs on real external footage.

The 7 real-world-eval source videos are NOT committed (licensing); what is
committed are the derived per-frame runtime outputs (C++ ONNX Runtime) and a
set of VLM (qwen3-vl-8b) video-level proposals. This script turns those into an
honest, clearly-scoped statement about real-domain behaviour:

  * per-video raw score distribution (P(failing), failing = positive class),
  * whether the temporal FSM emitted a FAILING event (video-level alarm),
  * agreement with the VLM video-level labels (weak, unverified reference).

It never fabricates labels: the VLM judgements are the only video-level
reference available and are reported as unverified. Read JSON from the working
tree when present, else from ``git show HEAD:<path>`` so it also runs in a
worktree that has deliberately pruned the (non-redistributable) artifacts.

Usage:
    python scripts/temporal_real_footage_analysis.py \
        --output docs/research/real-footage-analysis.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "datasets" / "external" / "real-world-eval"
DEFAULT_LABELS = REPO / "docs" / "research" / "real-footage-vlm-labels.json"

# run directory -> model identity (baseline == the model rc1 replaced).
MODEL_DIRS = {
    "runs": "baseline_shufflenet_v2_x1_0",
    "rc1_runs": "rc1_release_candidate",
    "exp3_runs": "exp3_target_normal_aug",
    "exp4_runs": "exp4_target_balanced",
    "py_runs": "baseline_python_reference",
}


def _read_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    rel = path.relative_to(REPO).as_posix()
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "show", f"HEAD:{rel}"],
            check=True, capture_output=True, text=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise FileNotFoundError(f"{rel} not on disk and not in git HEAD") from exc
    return json.loads(out)


def _stats(raws: list[float]) -> dict:
    if not raws:
        return {"n": 0}
    s = sorted(raws)
    def q(p: float) -> float:
        return s[min(len(s) - 1, int(p * len(s)))]
    return {
        "n": len(raws),
        "min": round(min(raws), 4),
        "median": round(statistics.median(raws), 4),
        "mean": round(statistics.mean(raws), 4),
        "p90": round(q(0.9), 4),
        "max": round(max(raws), 4),
        "frac_ge_0.5": round(sum(r >= 0.5 for r in raws) / len(raws), 4),
        "frac_ge_0.9": round(sum(r >= 0.9 for r in raws) / len(raws), 4),
        "median_ge_0.5": statistics.median(raws) >= 0.5,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    labels = json.loads(args.labels.read_text())["videos"]

    report: dict = {
        "schema": "real-footage-analysis",
        "schema_version": 1,
        "scope": (
            "Real external 3D-print footage, models executed by the C++ ONNX "
            "Runtime. Video-level labels are VLM (qwen3-vl-8b), UNVERIFIED."
        ),
        "labels_provenance": "docs/research/real-footage-vlm-labels.json",
        "labels_verification": "unverified (machine-generated)",
        "models": {},
    }

    for subdir, model in MODEL_DIRS.items():
        d = args.root / subdir
        if d.exists():
            vids = sorted(p.stem for p in d.glob("*.json"))
        else:
            tree = subprocess.run(
                ["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", "HEAD",
                 f"{d.relative_to(REPO).as_posix()}"],
                capture_output=True, text=True,
            ).stdout.split()
            vids = sorted(Path(t).stem for t in tree if t.endswith(".json"))
        if not vids:
            print(f"note: no run artifacts for {subdir}", file=sys.stderr)
            continue

        entry: dict = {"per_video": {}}
        normal_fire = failing_fire = normal_n = failing_n = 0
        for vid in vids:
            data = _read_json(d / f"{vid}.json")
            pf = data.get("per_frame") or []
            raws = [p["raw"] for p in pf]
            summary = data.get("summary") or {}
            fired = int(summary.get("first_failing_event_frame", -1)) >= 0
            st = _stats(raws)
            label = (labels.get(vid) or {}).get("decision", "unknown")
            entry["per_video"][vid] = {
                "vlm_label": label,
                "fired_failing_event": fired,
                "first_failing_event_frame": summary.get("first_failing_event_frame"),
                "failing_frames": summary.get("failing_frames"),
                "last_state": summary.get("last_state"),
                **st,
            }
            if label == "normal":
                normal_n += 1
                normal_fire += int(fired)
            elif label == "failing":
                failing_n += 1
                failing_fire += int(fired)

        entry["summary"] = {
            "normal_videos": normal_n,
            "normal_videos_fired": normal_fire,
            "normal_video_false_alarm_rate": (normal_fire / normal_n) if normal_n else None,
            "failing_videos": failing_n,
            "failing_videos_fired": failing_fire,
            "failing_video_detection_rate": (failing_fire / failing_n) if failing_n else None,
        }
        report["models"][model] = entry

    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
