#!/usr/bin/env python3
"""Stage 7 — Offline video/pipeline benchmark.

Processes video files through the TinyVision preprocessing + frozen ShuffleNetV2
pipeline.  Measures decode, preprocessing, inference, and end-to-end timing.

This is a file-based benchmark — no physical cameras are accessed.

Inputs:
  - Labelled MP4 fixtures (from generate_video_fixtures.py)
  - User-supplied video files (unlabelled domain-shift probes)

Usage:
    python scripts/benchmark_video.py [--fixtures-dir DIR] [--video FILE ...]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from tv.data.video import replay_video, compare_png_vs_video
from tv.models.factory import build_classifier

EXPERIMENT_ID = "shufflenet_v2_x1_0_printing-errors_cdc9e2cad4"
CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / EXPERIMENT_ID
    / "artifacts"
    / "best.pt"
)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_model(device: str = "cpu") -> tuple[torch.nn.Module, dict, int]:
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    num_classes = ckpt.get("num_classes", 2)
    model_name = ckpt.get("config_model", "shufflenet_v2_x1_0")
    class_to_idx = ckpt.get("class_to_idx", {})
    model = build_classifier(model_name, num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, class_to_idx, num_classes


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _mem_mb() -> float | None:
    try:
        return round(psutil.Process(os.getpid()).memory_info().rss / 1e6, 1)
    except Exception:
        return None


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    sv = sorted(values)
    n = len(sv)
    return {
        "count": n,
        "mean_ms": round(statistics.mean(values), 4),
        "std_ms": round(statistics.stdev(values), 4) if n > 1 else 0.0,
        "p50_ms": round(sv[int(n * 0.5)], 4),
        "p95_ms": round(sv[min(int(n * 0.95), n - 1)], 4),
        "p99_ms": round(sv[min(int(n * 0.99), n - 1)], 4),
        "min_ms": round(sv[0], 4),
        "max_ms": round(sv[-1], 4),
    }


def run_labelled_benchmark(
    video_path: Path,
    labels_path: Path,
    model: torch.nn.Module,
    transform: transforms.Compose,
    device: str,
    sample_rates: list[float | None],
) -> dict:
    """Run a labelled fixture through the pipeline and verify correctness."""
    with open(labels_path) as f:
        png_labels = json.load(f)

    # Build PNG reference predictions by running the original PNGs through the pipeline
    source_video_dir = video_path.parent
    # Find the source PNGs from the manifest or labels
    png_predictions = []
    video_dir = None
    for key in ("source_dirs", "source_dir"):
        # Try to find source dirs from manifest
        pass

    # Run PNG images through pipeline to get reference predictions
    # We need to locate the original PNGs
    # The labels have filenames; search in known data dirs
    data_dir = Path(__file__).resolve().parent.parent / "datasets" / "processed" / "printing-errors"

    # Determine which class dirs to search
    label_set = set(l["label"] for l in png_labels)
    search_dirs = []
    for cls in label_set:
        for recording_name in video_path.stem.split("_", 1):
            pass
        # Search all class dirs for this recording's PNGs
    # Actually, use the manifest to find source dirs
    manifest_path = source_video_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        # Find this recording in the manifest
        video_stem = video_path.stem
        for rec in manifest.get("recordings", []):
            if rec.get("video_path", "").endswith(video_stem + ".mp4") or rec.get("id") == video_stem:
                for sd in rec.get("source_dirs", []):
                    search_dirs.append(Path(sd))
                break

    if not search_dirs:
        # Fallback: search both class dirs
        for cls in ("normal", "failing"):
            candidate = data_dir / cls
            if candidate.is_dir():
                search_dirs.append(candidate)

    # Build frame list from labels
    png_frame_paths = []
    for label in png_labels:
        fname = label["filename"]
        for sd in search_dirs:
            candidate = sd / fname
            if candidate.exists():
                png_frame_paths.append((candidate, label["label"]))
                break

    # Run PNG predictions
    import time as _time
    for fp, cls in sorted(png_frame_paths, key=lambda x: x[0].name):
        img_bgr = cv2.imread(str(fp))
        if img_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)
        tensor = transform(pil).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor.to(device))
            probs = torch.softmax(output, dim=1)
        fail_score = float(probs[0, 0])
        png_predictions.append({
            "frame_index": len(png_predictions),
            "fail_score": round(fail_score, 6),
            "prediction": 0 if fail_score >= 0.5 else 1,
        })

    results = {"video_path": str(video_path), "png_frame_count": len(png_predictions)}

    for rate in sample_rates:
        rate_label = f"native" if rate is None else f"{rate:.0f}fps"
        replay = replay_video(video_path, transform, model, device, sample_fps=rate)

        # Compare against PNG predictions (align by frame index in the processed set)
        png_subset = png_predictions[:len(replay.get("frame_results", []))]
        comparison = compare_png_vs_video(png_subset, replay)

        results[f"replay_{rate_label}"] = {
            "sample_fps": rate,
            "frames_processed": replay["frames_processed"],
            "decode_ms": replay["decode_ms"],
            "preprocess_ms": replay["preprocess_ms"],
            "inference_ms": replay["inference_ms"],
            "correctness": comparison,
        }

    return results


def run_unlabelled_benchmark(
    video_path: Path,
    model: torch.nn.Module,
    transform: transforms.Compose,
    device: str,
    sample_rates: list[float | None],
) -> dict:
    """Run an unlabelled video through the pipeline. No accuracy claims."""
    results = {"video_path": str(video_path), "labelled": False}

    for rate in sample_rates:
        rate_label = f"native" if rate is None else f"{rate:.0f}fps"
        replay = replay_video(video_path, transform, model, device, sample_fps=rate)

        fps_values = []
        if replay["inference_ms"].get("count", 0) > 0:
            inf_mean = replay["inference_ms"]["mean_ms"]
            if inf_mean > 0:
                fps_values = [1000.0 / (replay["preprocess_ms"]["mean_ms"] + inf_mean)]

        results[f"replay_{rate_label}"] = {
            "sample_fps": rate,
            "frames_processed": replay["frames_processed"],
            "decode_ms": replay["decode_ms"],
            "preprocess_ms": replay["preprocess_ms"],
            "inference_ms": replay["inference_ms"],
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 7 Offline Video Pipeline Benchmark")
    parser.add_argument(
        "--fixtures-dir",
        type=str,
        default=None,
        help="Directory with labelled MP4 fixtures + manifest.json",
    )
    parser.add_argument(
        "--video",
        type=str,
        nargs="*",
        default=[],
        help="Additional unlabelled video file paths",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device",
    )
    args = parser.parse_args()

    fixtures_dir = Path(
        args.fixtures_dir
        or (Path(__file__).resolve().parent.parent / "docs" / "research" / "stage7-fixtures")
    )
    output_path = Path(
        args.output
        or (Path(__file__).resolve().parent.parent / "docs" / "research" / "stage7-video-benchmark.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 7 — OFFLINE VIDEO/PIPELINE BENCHMARK")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Fixtures: {fixtures_dir}")
    print(f"Output: {output_path}")
    print()

    # Load model
    print("[1/4] Loading frozen ShuffleNetV2...")
    model, class_to_idx, num_classes = load_model(args.device)
    transform = build_transform()
    mem_before = _mem_mb()
    print(f"  Model loaded: {num_classes} classes, RSS={mem_before} MB")

    results: dict[str, Any] = {
        "stage": 7,
        "device": args.device,
        "checkpoint": str(CHECKPOINT_PATH),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "memory_before_mb": mem_before,
    }

    # Sample rates to test
    sample_rates = [None, 2.0, 1.0]  # native, 2 FPS, 1 FPS

    # === Labelled fixtures ===
    print("\n[2/4] Processing labelled fixtures...")
    manifest_path = fixtures_dir / "manifest.json"
    labelled_results = {}

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

        for rec in manifest.get("recordings", []):
            rec_id = rec["id"]
            video_path = Path(rec["video_path"])
            labels_path = Path(rec.get("labels_path", ""))

            if not video_path.exists():
                print(f"  {rec_id}: SKIP (video not found)")
                continue
            if not labels_path.exists():
                print(f"  {rec_id}: SKIP (labels not found)")
                continue

            print(f"  {rec_id} ({rec['composition']}, {rec['frame_count']} frames)...", end=" ", flush=True)
            t0 = time.perf_counter()
            bench = run_labelled_benchmark(video_path, labels_path, model, transform, args.device, sample_rates)
            elapsed = time.perf_counter() - t0

            # Print correctness summary
            native_key = "replay_native"
            if native_key in bench:
                correctness = bench[native_key].get("correctness", {})
                score_match = correctness.get("score_match_rate", 0)
                pred_match = correctness.get("pred_match_rate", 0)
                max_diff = correctness.get("max_score_diff", 0)
                score_mismatches = correctness.get("score_mismatches", 0)
                pred_mismatches = correctness.get("pred_mismatches", 0)
                print(f"score_match={score_match:.2%} pred_match={pred_match:.2%} max_diff={max_diff:.4f} score_miss={score_mismatches} pred_miss={pred_mismatches} ({elapsed:.1f}s)")
            else:
                print(f"({elapsed:.1f}s)")

            labelled_results[rec_id] = bench

    results["labelled_benchmarks"] = labelled_results

    # === Unlabelled videos ===
    print("\n[3/4] Processing unlabelled videos...")
    unlabelled_results = {}
    for video_str in args.video:
        video_path = Path(video_str)
        if not video_path.exists():
            print(f"  {video_path.name}: SKIP (not found)")
            continue
        print(f"  {video_path.name}...", end=" ", flush=True)
        t0 = time.perf_counter()
        bench = run_unlabelled_benchmark(video_path, model, transform, args.device, sample_rates)
        elapsed = time.perf_counter() - t0
        print(f"({elapsed:.1f}s)")
        unlabelled_results[video_path.name] = bench

    results["unlabelled_benchmarks"] = unlabelled_results

    # === Summary ===
    print("\n[4/4] Summary...")
    results["memory_after_mb"] = _mem_mb()

    # Build summary table
    summary_rows = []
    for rec_id, bench in labelled_results.items():
        for rate in sample_rates:
            rate_label = f"native" if rate is None else f"{rate:.0f}fps"
            key = f"replay_{rate_label}"
            if key not in bench:
                continue
            rb = bench[key]
            correctness = rb.get("correctness", {})
            summary_rows.append({
                "fixture": rec_id,
                "sample_rate": rate_label,
                "frames": rb.get("frames_processed", 0),
                "decode_p50_ms": rb.get("decode_ms", {}).get("p50_ms", 0),
                "preprocess_p50_ms": rb.get("preprocess_ms", {}).get("p50_ms", 0),
                "inference_p50_ms": rb.get("inference_ms", {}).get("p50_ms", 0),
                "e2e_p50_ms": round(
                    rb.get("preprocess_ms", {}).get("p50_ms", 0)
                    + rb.get("inference_ms", {}).get("p50_ms", 0),
                    4,
                ),
                "match_rate": correctness.get("score_match_rate"),
                "pred_match_rate": correctness.get("pred_match_rate"),
                "max_diff": correctness.get("max_score_diff"),
            })

    results["summary_table"] = summary_rows

    # Print summary
    print(f"\n{'fixture':<35} {'rate':>8} {'frames':>7} {'decode':>8} {'preproc':>8} {'infer':>8} {'E2E':>8} {'score%':>8} {'pred%':>8}")
    print("-" * 115)
    for row in summary_rows:
        score_str = f"{row['match_rate']:.1%}" if row['match_rate'] is not None else "—"
        pred_str = f"{row['pred_match_rate']:.1%}" if row['pred_match_rate'] is not None else "—"
        print(
            f"{row['fixture']:<35} {row['sample_rate']:>8} {row['frames']:>7} "
            f"{row['decode_p50_ms']:>7.1f}ms {row['preprocess_p50_ms']:>7.1f}ms "
            f"{row['inference_p50_ms']:>7.1f}ms {row['e2e_p50_ms']:>7.1f}ms {score_str:>8} {pred_str:>8}"
        )

    # Write results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
