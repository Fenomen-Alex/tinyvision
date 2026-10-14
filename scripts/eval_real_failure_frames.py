#!/usr/bin/env python3
"""Run a frozen TinyVision classifier over the Stage-14 real failure frames.

Stage-14 holds real 3D-printer failure frames extracted from two external
videos. Their labels are **VLM-verified but not human-reviewed**
(``pending_human_review: true`` in the manifest), so this is a real-data,
weakly-labelled diagnostic — not a benchmark. It reports the raw P(failing)
distribution over the committed frames so we can see whether a model that looks
good on the frozen test also fires on real failure footage.

Frames are read from ``datasets/external/stage14-failure/frames``. If they are
absent (pruned worktree), they are materialised from ``git HEAD`` into a
temporary directory so nothing touches the working tree.

Usage:
    python scripts/eval_real_failure_frames.py \
        --model experiments/rc1_release_candidate_shufflenet_v2_x1_0/artifacts/best.pt \
        --output docs/research/stage14-real-frames-eval.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

STAGE14_REL = "datasets/external/stage14-failure"
DEFAULT_CHECKPOINT = (
    "experiments/rc1_release_candidate_shufflenet_v2_x1_0/artifacts/best.pt"
)


def _read_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    rel = path.relative_to(REPO).as_posix()
    out = subprocess.run(
        ["git", "-C", str(REPO), "show", f"HEAD:{rel}"],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)


def _materialise_frames(tmp: Path) -> Path:
    """Export HEAD's Stage-14 tree into ``tmp`` and return the frames root."""
    tar_path = tmp / "stage14.tar"
    with open(tar_path, "wb") as fh:
        subprocess.run(
            ["git", "-C", str(REPO), "archive", "HEAD", STAGE14_REL],
            check=True, stdout=fh,
        )
    with tarfile.open(tar_path) as tf:
        tf.extractall(tmp)
    return tmp / STAGE14_REL


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
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path,
                    default=REPO / STAGE14_REL / "manifest.json")
    ap.add_argument("--model", default=str(REPO / DEFAULT_CHECKPOINT))
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    from tv.reliability import Sample, score_framework, sha256_file

    manifest = _read_json(args.manifest)
    entries = manifest["frames"]

    tmp = None
    frames_root = REPO / STAGE14_REL
    if not (frames_root / "frames").exists():
        tmp = tempfile.TemporaryDirectory(prefix="stage14-")
        frames_root = _materialise_frames(Path(tmp.name))
        print(f"note: frames absent on disk; materialised from HEAD to {frames_root}",
              file=sys.stderr)

    samples = []
    for e in entries:
        # window == "fail" -> failing (positive=0); "final" -> not failing (1)
        label = 0 if e["window"] == "fail" else 1
        samples.append(Sample(
            image_path=str(frames_root / e["file"]),
            label=label,
            source_id=e.get("source_id"),
            surface_id="stage14",
            extra={"window": e["window"], "failure_type": e.get("failure_type"),
                   "magnitude": e.get("failure_magnitude")},
        ))

    missing = [s.image_path for s in samples if not Path(s.image_path).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} frame(s) missing, e.g. {missing[0]}")

    scores = score_framework(args.model, samples, args.image_size, args.batch_size)

    by_window: dict[str, list[float]] = {}
    by_source: dict[str, list[float]] = {}
    by_type: dict[str, list[float]] = {}
    for s, sc in zip(samples, scores):
        by_window.setdefault(s.extra["window"], []).append(float(sc))
        by_source.setdefault(s.source_id or "?", []).append(float(sc))
        by_type.setdefault(s.extra["failure_type"] or "?", []).append(float(sc))

    thr = args.threshold
    fail_scores = by_window.get("fail", [])
    det = {
        "threshold": thr,
        "fail_frames": len(fail_scores),
        "fail_frames_above_threshold": int(sum(s >= thr for s in fail_scores)),
        "fail_frame_detection_rate": (sum(s >= thr for s in fail_scores) / len(fail_scores))
        if fail_scores else None,
        "final_idle_frames": len(by_window.get("final", [])),
        "final_idle_above_threshold": int(sum(s >= thr for s in by_window.get("final", []))),
    }

    report = {
        "schema": "stage14-real-frames-eval",
        "schema_version": 1,
        "model": str(Path(args.model).relative_to(REPO)),
        "model_sha256": sha256_file(args.model),
        "label_provenance": {
            "source": "datasets/external/stage14-failure/manifest.json",
            "verification": "VLM-verified, pending_human_review=true (NOT human-verified)",
            "note": "weak labels; treat as a real-data diagnostic, not a benchmark",
        },
        "positive_class": 0,
        "window_stats": {k: _stats(v) for k, v in by_window.items()},
        "source_stats": {k: _stats(v) for k, v in by_source.items()},
        "failure_type_stats": {k: _stats(v) for k, v in by_type.items()},
        "detection": det,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        print(f"wrote {args.output}")
    else:
        print(text)
    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
