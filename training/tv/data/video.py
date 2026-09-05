"""Video fixture generation and replay utilities for TinyVision.

Provides deterministic conversion of ordered PNG recording sequences into
MP4 video files, and a frame-by-frame replay path through the TinyVision
preprocessing + inference pipeline.

The original PNG frames and their per-frame labels remain the authoritative
ground truth.  MP4 encoding introduces JPEG compression; the replay/compare
tools verify that this does not materially change predictions.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


def assemble_video(
    image_dir: Path,
    output_path: Path,
    fps: float = 2.0,
    codec: str = "mp4v",
    quality: int = 95,
) -> dict:
    """Deterministically encode an ordered PNG directory into an MP4 video.

    Frames are sorted by filename (lexicographic).  The original PNGs are
    never modified.

    Args:
        image_dir: Directory containing ordered PNG frames.
        output_path: Destination .mp4 path (parent dirs created automatically).
        fps: Frames per second for the output video.
        codec: FourCC codec string (default ``mp4v``; ``avc1``/``H264`` if
            available).
        quality: JPEG quality for ``imencode`` fallback (unused for mp4v
            but kept for future codec variants).

    Returns:
        Dict with generation metadata: frame count, resolution, fps, path.
    """
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pngs = sorted(p for p in image_dir.iterdir() if p.suffix.lower() == ".png")
    if not pngs:
        raise FileNotFoundError(f"No PNG files in {image_dir}")

    # Read first frame to get dimensions
    first = cv2.imread(str(pngs[0]))
    if first is None:
        raise RuntimeError(f"Cannot read {pngs[0]}")
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        # Fallback to XVID if mp4v not available
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_fallback = output_path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(out_fallback), fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter for {output_path}")
        output_path = out_fallback

    frame_count = 0
    for png in pngs:
        img = cv2.imread(str(png))
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)
        frame_count += 1

    writer.release()

    return {
        "path": str(output_path),
        "frame_count": frame_count,
        "resolution": f"{w}x{h}",
        "fps": fps,
        "codec": codec,
        "source_dir": str(image_dir),
    }


def assemble_video_from_frames(
    frame_paths: list[Path],
    output_path: Path,
    fps: float = 2.0,
    codec: str = "mp4v",
) -> dict:
    """Encode an ordered list of frame paths into an MP4 video.

    Used for MIXED recordings whose frames span multiple directories.
    The caller is responsible for correct sort order.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not frame_paths:
        raise ValueError("No frame paths provided")

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Cannot read {frame_paths[0]}")
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_fallback = output_path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(out_fallback), fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter for {output_path}")
        output_path = out_fallback

    frame_count = 0
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)
        frame_count += 1

    writer.release()

    return {
        "path": str(output_path),
        "frame_count": frame_count,
        "resolution": f"{w}x{h}",
        "fps": fps,
        "codec": codec,
    }


def load_labels_from_dir(image_dir: Path) -> list[dict]:
    """Load per-frame labels from a recording directory.

    Infers the label from the parent directory name (``normal`` or ``failing``).
    Returns a list of dicts with ``frame_index``, ``filename``, ``label``,
    ``fine_label``.

    The fine label is inferred from the image filename prefix when possible
    (ELP filenames encode the recording timestamp, not the failure type, so
    fine labels come from the audit or are left as the coarse label).
    """
    image_dir = Path(image_dir)
    pngs = sorted(p for p in image_dir.iterdir() if p.suffix.lower() == ".png")

    # Determine coarse label from parent dir
    parent_name = image_dir.parent.name.lower()
    if parent_name in ("normal", "failing"):
        coarse_label = parent_name
    else:
        coarse_label = "unknown"

    labels = []
    for i, png in enumerate(pngs):
        labels.append({
            "frame_index": i,
            "filename": png.name,
            "label": coarse_label,
        })
    return labels


def replay_video(
    video_path: Path,
    transform: transforms.Compose,
    model=None,
    device: str = "cpu",
    sample_fps: float | None = None,
) -> dict:
    """Replay a video file through the TinyVision preprocessing + inference path.

    If ``model`` is None, only preprocessing timing is measured (no inference).

    Args:
        video_path: Path to the MP4/AVI video file.
        transform: The TinyVision eval-time transform.
        model: Loaded PyTorch model (optional).
        device: Inference device.
        sample_fps: If set, only process every N-th frame to simulate this
            sampling cadence.  ``None`` means process every frame.

    Returns:
        Dict with per-frame results and timing statistics.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine frame skip for sample_fps
    if sample_fps is not None and native_fps > 0 and sample_fps < native_fps:
        skip = max(1, round(native_fps / sample_fps))
    else:
        skip = 1

    frame_results = []
    preprocess_times = []
    inference_times = []
    decode_times = []
    frame_indices = []

    frame_idx = 0
    while True:
        t0 = __import__("time").perf_counter()
        ret, frame = cap.read()
        t1 = __import__("time").perf_counter()

        if not ret or frame is None:
            break

        decode_times.append((t1 - t0) * 1000)

        # Apply skip
        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        frame_indices.append(frame_idx)

        # Preprocess: BGR → RGB → PIL → tensor
        t_pre0 = __import__("time").perf_counter()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)
        tensor = transform(pil).unsqueeze(0)
        t_pre1 = __import__("time").perf_counter()
        preprocess_times.append((t_pre1 - t_pre0) * 1000)

        # Inference
        if model is not None:
            t_inf0 = __import__("time").perf_counter()
            import torch
            with torch.no_grad():
                output = model(tensor.to(device))
                probs = torch.softmax(output, dim=1)
            t_inf1 = __import__("time").perf_counter()
            inference_times.append((t_inf1 - t_inf0) * 1000)

            fail_score = float(probs[0, 0])
            pred = 0 if fail_score >= 0.5 else 1
            frame_results.append({
                "frame_index": frame_idx,
                "fail_score": round(fail_score, 6),
                "prediction": pred,
            })
        else:
            frame_results.append({"frame_index": frame_idx})

        frame_idx += 1

    cap.release()

    def _stats(vals):
        if not vals:
            return {"count": 0}
        import statistics
        sv = sorted(vals)
        n = len(sv)
        return {
            "count": n,
            "mean_ms": round(statistics.mean(vals), 4),
            "p50_ms": round(sv[int(n * 0.5)], 4) if n > 0 else 0,
            "p95_ms": round(sv[min(int(n * 0.95), n - 1)], 4),
            "p99_ms": round(sv[min(int(n * 0.99), n - 1)], 4),
            "min_ms": round(sv[0], 4),
            "max_ms": round(sv[-1], 4),
        }

    return {
        "video_path": str(video_path),
        "native_fps": round(native_fps, 2),
        "total_frames_in_video": total_frames,
        "resolution": f"{width}x{height}",
        "sample_fps": sample_fps,
        "skip": skip,
        "frames_processed": len(frame_results),
        "frame_results": frame_results,
        "decode_ms": _stats(decode_times),
        "preprocess_ms": _stats(preprocess_times),
        "inference_ms": _stats(inference_times),
        "frame_indices": frame_indices,
    }


def compare_png_vs_video(
    png_predictions: list[dict],
    video_result: dict,
    score_tolerance: float = 0.05,
    prediction_match: bool = True,
) -> dict:
    """Compare per-frame predictions from PNG path vs video replay path.

    Two levels of comparison:
    - **Score-level**: absolute difference in fail_score within ``score_tolerance``.
      Default 0.05 accounts for JPEG/H.264 compression artefacts on 256×256 images.
    - **Prediction-level**: whether the binarised prediction (threshold 0.5) agrees.

    Args:
        png_predictions: List of dicts with ``frame_index``, ``fail_score``.
        video_result: Output of ``replay_video``.
        score_tolerance: Maximum allowed absolute difference in fail_score for
            a frame to count as a "score match".
        prediction_match: If True, also check that the predicted class agrees.

    Returns:
        Dict with match/mismatch counts at both levels, worst-case difference.
    """
    video_frames = {r["frame_index"]: r for r in video_result.get("frame_results", [])}

    score_matches = 0
    pred_matches = 0
    total = 0
    max_diff = 0.0
    mismatch_details = []

    for png_pred in png_predictions:
        idx = png_pred["frame_index"]
        if idx not in video_frames:
            continue
        v_pred = video_frames[idx]
        diff = abs(png_pred.get("fail_score", 0) - v_pred.get("fail_score", 0))
        total += 1
        if diff > max_diff:
            max_diff = diff

        score_ok = diff <= score_tolerance
        pred_ok = (
            png_pred.get("prediction") == v_pred.get("prediction")
        )

        if score_ok:
            score_matches += 1
        if pred_ok:
            pred_matches += 1

        if not (score_ok and pred_ok):
            mismatch_details.append({
                "frame_index": idx,
                "png_score": png_pred.get("fail_score"),
                "video_score": v_pred.get("fail_score"),
                "diff": round(diff, 6),
                "png_pred": png_pred.get("prediction"),
                "video_pred": v_pred.get("prediction"),
                "pred_agree": pred_ok,
            })

    return {
        "total_frames_compared": total,
        "score_matches": score_matches,
        "score_mismatches": total - score_matches,
        "score_match_rate": round(score_matches / total, 4) if total > 0 else 0,
        "pred_matches": pred_matches,
        "pred_mismatches": total - pred_matches,
        "pred_match_rate": round(pred_matches / total, 4) if total > 0 else 0,
        "max_score_diff": round(max_diff, 6),
        "score_tolerance": score_tolerance,
        "mismatch_details": mismatch_details[:20],
    }
