#!/usr/bin/env python3
"""Video-level report for the rc2 real-domain diagnostic.

Scores every frame of each stream (a stream = one acquisition source) and
aggregates to a video-level decision, then reports per-stream and aggregate
metrics. Video-level is the honest unit here: the target decision is per-print,
and single frames are noisy. This is a DIAGNOSTIC, not the release gate.

Decision (at threshold t): a stream is flagged failing iff the median frame
score >= t. Also reports the max-score ("any frame") rule as a secondary view.
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

from tv.models.factory import build_classifier

FAIL_IDX = 0  # score = P(failing); classes are alphabetical: failing=0, normal=1


def _load_model(model_name: str, ckpt: Path):
    model = build_classifier(model_name, 2, pretrained=False)
    state = torch.load(ckpt, map_location="cpu")
    sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
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


def _load_streams(csv_path: Path) -> dict[str, dict]:
    streams: dict[str, dict] = defaultdict(lambda: {"paths": [], "label": None})
    for row in csv.DictReader(csv_path.open()):
        s = streams[row["stream"]]
        s["paths"].append(row["path"])
        s["label"] = row["label"]
    return streams


def run(csv_path: Path, model, tfm, threshold: float) -> dict:
    streams = _load_streams(csv_path)
    per_stream = {}
    for sid, s in streams.items():
        sc = _score(model, tfm, sorted(s["paths"]))
        med = statistics.median(sc)
        per_stream[sid] = {
            "label": s["label"], "n": len(sc),
            "median": round(med, 4), "mean": round(statistics.mean(sc), 4),
            "max": round(max(sc), 4), "min": round(min(sc), 4),
            "frac_ge": round(sum(x >= threshold for x in sc) / len(sc), 4),
            "decision_median": "failing" if med >= threshold else "normal",
            "decision_any": "failing" if max(sc) >= threshold else "normal",
        }
    normals = [v for v in per_stream.values() if v["label"] == "normal"]
    failings = [v for v in per_stream.values() if v["label"] == "failing"]
    far = (sum(v["decision_median"] == "failing" for v in normals) / len(normals)
           if normals else None)
    det = (sum(v["decision_median"] == "failing" for v in failings) / len(failings)
           if failings else None)
    far_any = (sum(v["decision_any"] == "failing" for v in normals) / len(normals)
               if normals else None)
    det_any = (sum(v["decision_any"] == "failing" for v in failings) / len(failings)
               if failings else None)
    return {"threshold": threshold, "n_streams": len(per_stream),
            "n_normal": len(normals), "n_failing": len(failings),
            "video_far_median": far, "video_detection_median": det,
            "video_far_any": far_any, "video_detection_any": det_any,
            "per_stream": per_stream}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="Split CSV (path,label,stream)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default="shufflenet_v2_x1_0")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = _load_model(args.model, Path(args.checkpoint))
    result = run(Path(args.split), model, _tfm(args.image_size), args.threshold)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
