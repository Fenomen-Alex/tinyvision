#!/usr/bin/env python
"""Per-frame and per-stream metrics for the Kaggle RC2 experiment.

Scores one or more checkpoints on the run's evaluation surfaces, then reports:
  * per-frame metrics (P/R/F1/FPR/FNR/AUROC) via the audited evaluator;
  * per-stream metrics (stream = print_id): a stream is flagged failing iff the
    median P(failing) of its frames >= threshold;
  * per-stream normal false-alarm rate and failure-event recall with Wilson 95%
    confidence intervals.

Scores are cached as .npy keyed by (checkpoint sha256, surface) so re-runs are
cheap. CPU-only and deterministic.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from tv.reliability import (  # noqa: E402
    Sample, evaluate_surface, score_framework, wilson_interval, FAIL_IDX,
)

SURFACES = {
    "dev_val": "splits/val.csv",
    "locked_test": "locked_test_fails.csv",
    "normal_holdout": "holdout_normal_test.csv",
}

# Class-disjoint surfaces from the same (held-out) printer, combined so that a
# threshold-independent metric (AUROC) is defined on the held-out environment.
COMBINED = {
    "holdout103_combined": ["locked_test", "normal_holdout"],
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_samples(csv_path: Path) -> list[Sample]:
    label_map = {"failing": 0, "normal": 1}
    samples: list[Sample] = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            sid = r["stream"]
            samples.append(Sample(
                image_path=r["path"], label=label_map[r["label"]],
                source_id=sid, environment_id=sid.split("__")[0],
            ))
    return samples


def per_stream(samples: list[Sample], scores: np.ndarray, threshold: float) -> dict:
    by_stream: dict[str, list[float]] = defaultdict(list)
    label_of: dict[str, int] = {}
    for s, sc in zip(samples, scores):
        by_stream[s.source_id].append(float(sc))
        label_of[s.source_id] = s.label
    streams = []
    for sid, scs in by_stream.items():
        med = float(np.median(scs))
        streams.append({
            "stream": sid, "environment": sid.split("__")[0],
            "label": "failing" if label_of[sid] == 0 else "normal",
            "n_frames": len(scs), "median_p_failing": round(med, 4),
            "flagged_failing": bool(med >= threshold),
        })
    streams.sort(key=lambda d: (d["label"], d["stream"]))

    normal = [s for s in streams if s["label"] == "normal"]
    failing = [s for s in streams if s["label"] == "failing"]
    n_alarm = sum(s["flagged_failing"] for s in normal)
    n_detected = sum(s["flagged_failing"] for s in failing)
    far = (n_alarm / len(normal)) if normal else None
    recall = (n_detected / len(failing)) if failing else None
    return {
        "threshold": threshold,
        "n_streams": len(streams),
        "normal_streams": len(normal),
        "failing_streams": len(failing),
        "normal_false_alarms": n_alarm,
        "stream_false_alarm_rate": far,
        "stream_far_ci95": wilson_interval(n_alarm, len(normal)) if normal else None,
        "failing_detected": n_detected,
        "failure_event_recall": recall,
        "stream_recall_ci95": wilson_interval(n_detected, len(failing)) if failing else None,
        "streams": streams,
    }


def select_threshold(samples: list[Sample], scores: np.ndarray,
                     fpr_ceiling: float, positive: int = FAIL_IDX) -> dict:
    """Pick an operating threshold from a single (validation) surface.

    Maximise recall subject to ``FPR <= fpr_ceiling``; ties break toward the
    highest threshold (fewer alarms). If no threshold is feasible, fall back to
    Youden's J. The returned threshold is only ever derived from the surface it
    is given, so callers must pass the validation surface.
    """
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    best = None  # feasible: (recall, threshold)
    best_j = None  # fallback: (j, threshold)
    for t in grid:
        m = evaluate_surface(samples, scores, threshold=float(t), positive=positive)["metrics"]
        fpr, rec = m["fpr"], m["recall"]
        if fpr is None or rec is None:
            continue
        if fpr <= fpr_ceiling:
            key = (rec, float(t))
            if best is None or key > best:
                best = key
        j = rec - (fpr if fpr is not None else 1.0)
        if best_j is None or j > best_j[0] + 1e-12 or (
                abs(j - best_j[0]) <= 1e-12 and t > best_j[1]):
            best_j = (j, float(t))
    if best is not None:
        return {"threshold": best[1], "objective": "max_recall_fpr_ceiling",
                "fpr_ceiling": fpr_ceiling, "recall": best[0]}
    return {"threshold": best_j[1], "objective": "youden_j_fallback",
            "fpr_ceiling": fpr_ceiling, "recall": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run dir (has splits/ etc.)")
    ap.add_argument("--model", action="append", required=True, metavar="NAME=CKPT")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cache", required=True, help="Score cache dir")
    ap.add_argument("--out", required=True, help="Output JSON")
    ap.add_argument("--select-on", default="dev_val",
                    help="Surface id used for threshold selection (validation only)")
    ap.add_argument("--fpr-ceiling", type=float, default=0.1)
    args = ap.parse_args()

    run, cache = Path(args.run), Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    report: dict = {"run": str(run), "threshold": args.threshold, "models": {}}

    surface_cache: dict[str, tuple[list[Sample], dict[str, np.ndarray]]] = {}
    for sid, rel in SURFACES.items():
        surface_cache[sid] = (load_samples(run / rel), {})

    for spec in args.model:
        name, path = spec.split("=", 1)
        ckpt = Path(path)
        sha = _sha256(ckpt)

        scores_by_sid: dict[str, np.ndarray] = {}
        for sid, (samples, _) in surface_cache.items():
            cpath = cache / f"{sha[:16]}_{sid}.npy"
            if cpath.exists():
                scores = np.load(cpath)
            else:
                scores = score_framework(ckpt, samples, image_size=args.image_size,
                                         batch_size=args.batch_size)
                np.save(cpath, scores)
            scores_by_sid[sid] = scores

        # Threshold selected on the validation surface only.
        selection = select_threshold(
            surface_cache[args.select_on][0], scores_by_sid[args.select_on],
            args.fpr_ceiling)
        sel_t = selection["threshold"]

        model_out: dict = {
            "checkpoint": str(ckpt), "checkpoint_sha256": sha,
            "val_selection": {**selection, "surface": args.select_on},
            "surfaces": {},
        }
        for sid, (samples, _) in surface_cache.items():
            scores = scores_by_sid[sid]
            model_out["surfaces"][sid] = {
                "per_frame": evaluate_surface(samples, scores,
                                              threshold=args.threshold, positive=FAIL_IDX),
                "per_stream": per_stream(samples, scores, args.threshold),
                "per_frame_at_val_threshold": evaluate_surface(
                    samples, scores, threshold=sel_t, positive=FAIL_IDX),
                "per_stream_at_val_threshold": per_stream(samples, scores, sel_t),
            }
        for cid, parts in COMBINED.items():
            csamples = [s for p in parts for s in surface_cache[p][0]]
            cscores = np.concatenate([scores_by_sid[p] for p in parts])
            model_out["surfaces"][cid] = {
                "per_frame": evaluate_surface(csamples, cscores,
                                              threshold=args.threshold, positive=FAIL_IDX),
                "per_stream": per_stream(csamples, cscores, args.threshold),
                "per_frame_at_val_threshold": evaluate_surface(
                    csamples, cscores, threshold=sel_t, positive=FAIL_IDX),
                "per_stream_at_val_threshold": per_stream(csamples, cscores, sel_t),
            }
        report["models"][name] = model_out

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    for name, mo in report["models"].items():
        sel = mo["val_selection"]
        print(f"\n=== {name} ({mo['checkpoint_sha256'][:12]}) ===")
        print(f"  val-selected threshold={sel['threshold']} "
              f"(objective={sel['objective']}, fpr_ceiling={sel['fpr_ceiling']})")
        for sid, d in mo["surfaces"].items():
            f, s = d["per_frame"]["metrics"], d["per_stream"]
            print(f"  [{sid}] frame: n={f['n']} FPR={f['fpr']} FNR={f['fnr']} "
                  f"P={f['precision']} R={f['recall']} AUROC={f['auroc']}")
            print(f"           stream: normals={s['normal_streams']} "
                  f"FAR={s['stream_false_alarm_rate']} (ci {s['stream_far_ci95']}) "
                  f"fails={s['failing_streams']} recall={s['failure_event_recall']} "
                  f"(ci {s['stream_recall_ci95']})")
            fs = d["per_frame_at_val_threshold"]["metrics"]
            ss = d["per_stream_at_val_threshold"]
            print(f"           @val_t={sel['threshold']} frame: FPR={fs['fpr']} "
                  f"FNR={fs['fnr']} R={fs['recall']} | stream: "
                  f"FAR={ss['stream_false_alarm_rate']} recall={ss['failure_event_recall']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
