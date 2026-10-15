#!/usr/bin/env python3
"""Video-level threshold sweep and operating-point derivation (rc2 real-domain).

Decision unit = one stream (= one acquisition source). A stream is flagged
``failing`` at threshold ``t`` iff its **median** frame score >= t (the same rule
``scripts/realdomain_report.py`` uses). The median rule is chosen because the
target decision is per-print and single frames are noisy.

Threshold SELECTION uses the validation split ONLY. The test split is scored for
REPORTING (evaluation at a preselected operating point) and must never be used to
choose one. The oracle band printed at the end is explicitly evaluation-only: it
shows where a threshold *would* have to sit to separate the held-out classes, to
quantify whether the validation split could have found it.

This is a DIAGNOSTIC over weak/VLM labels, not the release gate.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

FAIL_IDX = 0  # score = P(failing); classes alphabetical: failing=0, normal=1


def _build_model(model_name: str, ckpt: Path):
    from tv.models.factory import build_classifier

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model = build_classifier(model_name, 2, pretrained=False)
    model.load_state_dict(sd)
    model.eval()
    return model


def _tfm(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def _score(model, tfm, paths: list[str], batch: int = 32) -> list[float]:
    scores: list[float] = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        xs = torch.stack([tfm(Image.open(p).convert("RGB")) for p in chunk])
        probs = torch.softmax(model(xs), dim=1)[:, FAIL_IDX]
        scores.extend(probs.tolist())
    return scores


def per_stream(csv_path: Path, model, tfm) -> dict[str, dict]:
    streams: dict[str, dict] = defaultdict(lambda: {"paths": [], "label": None})
    for row in csv.DictReader(csv_path.open()):
        s = streams[row["stream"]]
        s["paths"].append(row["path"])
        s["label"] = row["label"]
    out: dict[str, dict] = {}
    for sid, s in streams.items():
        paths = sorted(s["paths"])
        sc = _score(model, tfm, paths)
        med = statistics.median(sc)
        out[sid] = {
            "label": s["label"], "n": len(sc),
            "median": round(med, 6), "mean": round(statistics.mean(sc), 6),
            "max": round(max(sc), 6), "min": round(min(sc), 6),
            "scores": [round(x, 6) for x in sc],
        }
    return out


def _rates(streams: dict[str, dict], t: float) -> dict:
    """Video-level FAR / detection at threshold t (median decision rule)."""
    normals = [v for v in streams.values() if v["label"] == "normal"]
    failings = [v for v in streams.values() if v["label"] == "failing"]
    fa = sum(v["median"] >= t for v in normals)
    det = sum(v["median"] >= t for v in failings)
    return {
        "n_normal": len(normals), "n_failing": len(failings),
        "false_alarms": fa, "detections": det,
        "far": (fa / len(normals)) if normals else None,
        "detection": (det / len(failings)) if failings else None,
    }


def _f1(r: dict) -> float | None:
    if not r["n_failing"] or not r["n_normal"]:
        return None
    recall = r["detections"] / r["n_failing"]
    prec_den = r["detections"] + r["false_alarms"]
    precision = (r["detections"] / prec_den) if prec_den else 1.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def derive_operating_points(val: dict[str, dict], grid: list[float],
                            max_fpr: float) -> dict:
    rows = []
    for t in grid:
        r = _rates(val, t)
        r["threshold"] = round(t, 6)
        r["f1"] = _f1(r)
        rows.append(r)

    normals_med = [v["median"] for v in val.values() if v["label"] == "normal"]
    fail_med = [v["median"] for v in val.values() if v["label"] == "failing"]
    vs = sorted(val.values(), key=lambda v: v["median"])
    val_normals = [v for v in vs if v["label"] == "normal"]
    val_fail = [v for v in vs if v["label"] == "failing"]
    highest_val_normal = max(normals_med) if normals_med else None
    lowest_val_failing = min(fail_med) if fail_med else None

    # 1. The repository's formulation (tv.reliability.select_threshold): max
    #    recall subject to FAR <= max_fpr, tie-broken toward the LOWEST
    #    threshold (most sensitive; the FPR ceiling is already enforced).
    feasible = [r for r in rows if r["far"] is not None and r["far"] <= max_fpr
                and r["n_failing"] and r["detection"] and r["detection"] > 0]
    full = [r for r in feasible if r["detection"] == r["n_failing"]]
    repo_op = (sorted(full, key=lambda r: r["threshold"])[0]["threshold"]
               if full else None)

    # 2. Conservative alternative: highest feasible threshold (largest margin
    #    over the validation normals). Included to show a *defensible but
    #    different* choice that can miss a low-scoring held-out failure.
    conservative_op = (sorted(full, key=lambda r: -r["threshold"])[0]["threshold"]
                       if full else None)

    # 3. Max-F1 band on validation.
    f1_vals = [r["f1"] for r in rows if r["f1"] is not None]
    max_f1 = max(f1_vals) if f1_vals else None
    f1_band = [r["threshold"] for r in rows if r["f1"] is not None and r["f1"] == max_f1]

    # 4. Margin heuristic: midpoint between the highest val normal median and the
    #    lowest val failing median.
    midpoint_op = (None if highest_val_normal is None or lowest_val_failing is None
                   else round((highest_val_normal + lowest_val_failing) / 2, 6))

    return {
        "sweep": rows,
        "val_highest_normal_median": highest_val_normal,
        "val_lowest_failing_median": lowest_val_failing,
        "max_fpr": max_fpr,
        "operating_points": {
            "repo_max_recall_lowest_threshold": repo_op,
            "conservative_highest_threshold": conservative_op,
            "max_f1_band": f1_band,
            "midpoint_margin": midpoint_op,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default="shufflenet_v2_x1_0")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--grid", default=None,
                    help="comma list; default 0.02..0.98 step 0.02")
    ap.add_argument("--max-fpr", type=float, default=0.0,
                    help="normal false-alarm ceiling for threshold selection")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    grid = ([round(float(x), 6) for x in args.grid.split(",")] if args.grid
            else [round(0.02 * i, 2) for i in range(1, 50)])

    model = _build_model(args.model, Path(args.checkpoint))
    tfm = _tfm(args.image_size)
    val = per_stream(Path(args.val), model, tfm)
    test = per_stream(Path(args.test), model, tfm)

    derived = derive_operating_points(val, grid, args.max_fpr)

    # Evaluation-only: apply each validation-derived operating point to test.
    eval_points: dict[str, dict] = {}
    for name, t in derived["operating_points"].items():
        if isinstance(t, list):
            t = t[0] if t else None
        if t is None:
            eval_points[name] = {"threshold": None}
            continue
        eval_points[name] = {"threshold": t, "test": _rates(test, t), "val": _rates(val, t)}

    # Oracle band (evaluation-only): what test itself would require.
    test_normals = [v["median"] for v in test.values() if v["label"] == "normal"]
    test_fail = [v["median"] for v in test.values() if v["label"] == "failing"]
    oracle = {
        "test_highest_normal_median": max(test_normals) if test_normals else None,
        "test_lowest_failing_median": min(test_fail) if test_fail else None,
        "separable": (max(test_normals) < min(test_fail)
                      if test_normals and test_fail else None),
    }

    # Detection delay at the sensitivity operating point: first frame index >= t.
    delay = {}
    sens = derived["operating_points"]["repo_max_recall_lowest_threshold"]
    if sens is not None:
        for sid, v in test.items():
            if v["label"] == "failing":
                idx = next((i for i, x in enumerate(v["scores"]) if x >= sens), None)
                delay[sid] = {"threshold": sens, "first_frame_ge": idx,
                              "n_frames": v["n"], "median": v["median"]}

    result = {
        "checkpoint": str(args.checkpoint),
        "decision_rule": "stream flagged failing iff median(frame scores) >= threshold",
        "selection_surface": str(args.val),
        "evaluation_surface": str(args.test),
        "val_per_stream": val,
        "test_per_stream": test,
        "selection": derived,
        "held_out_evaluation": eval_points,
        "held_out_oracle_band": oracle,
        "held_out_detection_delay": delay,
    }
    text = json.dumps(result, indent=2)
    print(json.dumps({"selection": {
        "val_highest_normal_median": derived["val_highest_normal_median"],
        "val_lowest_failing_median": derived["val_lowest_failing_median"],
        "operating_points": derived["operating_points"],
    }, "held_out_evaluation": eval_points,
        "held_out_oracle_band": oracle,
        "held_out_detection_delay": delay}, indent=2))
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
