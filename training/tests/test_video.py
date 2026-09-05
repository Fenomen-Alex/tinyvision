"""Tests for tv/data/video.py (video fixture generation, replay, comparison)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from tv.data.video import (
    assemble_video,
    assemble_video_from_frames,
    compare_png_vs_video,
    load_labels_from_dir,
    replay_video,
)


def _make_frames(tmp: Path, n: int = 5, w: int = 256, h: int = 256) -> Path:
    """Create n ordered synthetic PNG frames in a directory."""
    d = tmp / "frames"
    d.mkdir()
    for i in range(n):
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        cv2.imwrite(str(d / f"frame_{i:04d}.png"), img)
    return d


def _make_toy_video(tmp: Path, n: int = 10, fps: float = 2.0) -> Path:
    """Create a small MP4 video from synthetic frames."""
    frames_dir = _make_frames(tmp, n)
    out = tmp / "test.mp4"
    assemble_video(frames_dir, out, fps=fps)
    return out


def _toy_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def test_assemble_video_basic():
    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = _make_frames(Path(tmp), n=5)
        out = Path(tmp) / "out.mp4"
        info = assemble_video(frames_dir, out, fps=2.0)
        assert info["frame_count"] == 5
        assert info["resolution"] == "256x256"
        assert info["fps"] == 2.0
        assert out.exists()
        # Verify with OpenCV
        cap = cv2.VideoCapture(str(out))
        assert cap.isOpened()
        count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            count += 1
        cap.release()
        assert count == 5


def test_assemble_video_empty_raises():
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "empty"
        empty.mkdir()
        out = Path(tmp) / "out.mp4"
        try:
            assemble_video(empty, out)
            assert False, "Should have raised"
        except FileNotFoundError:
            pass


def test_assemble_video_from_frames_basic():
    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = _make_frames(Path(tmp), n=8)
        frame_paths = sorted(frames_dir.glob("*.png"))
        out = Path(tmp) / "out.mp4"
        info = assemble_video_from_frames(frame_paths, out, fps=2.0)
        assert info["frame_count"] == 8
        assert out.exists()


def test_assemble_video_from_frames_empty_raises():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.mp4"
        try:
            assemble_video_from_frames([], out)
            assert False, "Should have raised"
        except ValueError:
            pass


def test_load_labels_from_dir():
    with tempfile.TemporaryDirectory() as tmp:
        # Create frames in a "failing" class dir
        class_dir = Path(tmp) / "failing" / "rec_001"
        class_dir.mkdir(parents=True)
        for i in range(3):
            cv2.imwrite(str(class_dir / f"frame_{i:04d}.png"), np.zeros((64, 64, 3), dtype=np.uint8))

        labels = load_labels_from_dir(class_dir)
        assert len(labels) == 3
        assert all(l["label"] == "failing" for l in labels)
        assert labels[0]["frame_index"] == 0
        assert labels[2]["frame_index"] == 2


def test_load_labels_normal_dir():
    with tempfile.TemporaryDirectory() as tmp:
        class_dir = Path(tmp) / "normal" / "rec_002"
        class_dir.mkdir(parents=True)
        for i in range(2):
            cv2.imwrite(str(class_dir / f"img_{i:04d}.png"), np.zeros((64, 64, 3), dtype=np.uint8))

        labels = load_labels_from_dir(class_dir)
        assert len(labels) == 2
        assert all(l["label"] == "normal" for l in labels)


def test_replay_video_basic():
    with tempfile.TemporaryDirectory() as tmp:
        video = _make_toy_video(Path(tmp), n=10, fps=2.0)
        transform = _toy_transform()
        result = replay_video(video, transform, sample_fps=None)
        assert result["frames_processed"] == 10
        assert result["native_fps"] == 2.0
        assert len(result["frame_results"]) == 10
        assert result["decode_ms"]["count"] > 0
        assert result["preprocess_ms"]["count"] == 10


def test_replay_video_with_model():
    with tempfile.TemporaryDirectory() as tmp:
        video = _make_toy_video(Path(tmp), n=5, fps=2.0)
        transform = _toy_transform()

        # Build a tiny model for testing
        from tv.models.factory import build_classifier
        model = build_classifier("shufflenet_v2_x1_0", 2, pretrained=False)
        model.eval()

        result = replay_video(video, transform, model=model, sample_fps=None)
        assert result["frames_processed"] == 5
        assert len(result["inference_ms"]) > 0
        # Check predictions exist
        for fr in result["frame_results"]:
            assert "fail_score" in fr
            assert "prediction" in fr
            assert fr["prediction"] in (0, 1)


def test_replay_video_sample_fps():
    with tempfile.TemporaryDirectory() as tmp:
        # Create a 10 FPS video with 20 frames
        frames_dir = _make_frames(Path(tmp), n=20)
        out = Path(tmp) / "fast.mp4"
        assemble_video(frames_dir, out, fps=10.0)

        transform = _toy_transform()
        # Sample at 2 FPS -> skip = round(10/2) = 5
        result = replay_video(out, transform, sample_fps=2.0)
        assert result["skip"] == 5
        assert result["frames_processed"] == 4  # 20/5 = 4


def test_compare_png_vs_video_perfect_match():
    png_preds = [
        {"frame_index": 0, "fail_score": 0.3, "prediction": 1},
        {"frame_index": 1, "fail_score": 0.8, "prediction": 0},
    ]
    video_result = {
        "frame_results": [
            {"frame_index": 0, "fail_score": 0.3, "prediction": 1},
            {"frame_index": 1, "fail_score": 0.8, "prediction": 0},
        ]
    }
    result = compare_png_vs_video(png_preds, video_result)
    assert result["total_frames_compared"] == 2
    assert result["score_matches"] == 2
    assert result["pred_matches"] == 2
    assert result["max_score_diff"] == 0.0


def test_compare_png_vs_video_tolerance():
    png_preds = [
        {"frame_index": 0, "fail_score": 0.5, "prediction": 0},
        {"frame_index": 1, "fail_score": 0.5, "prediction": 0},
    ]
    video_result = {
        "frame_results": [
            {"frame_index": 0, "fail_score": 0.52, "prediction": 0},  # within tolerance
            {"frame_index": 1, "fail_score": 0.7, "prediction": 0},   # beyond tolerance
        ]
    }
    result = compare_png_vs_video(png_preds, video_result, score_tolerance=0.05)
    assert result["score_matches"] == 1
    assert result["score_mismatches"] == 1
    assert result["pred_matches"] == 2  # both predictions agree despite score diff


def test_compare_png_vs_video_prediction_mismatch():
    png_preds = [
        {"frame_index": 0, "fail_score": 0.6, "prediction": 0},
    ]
    video_result = {
        "frame_results": [
            {"frame_index": 0, "fail_score": 0.3, "prediction": 1},
        ]
    }
    result = compare_png_vs_video(png_preds, video_result)
    assert result["pred_matches"] == 0
    assert result["pred_mismatches"] == 1


def test_compare_png_vs_video_empty():
    result = compare_png_vs_video([], {"frame_results": []})
    assert result["total_frames_compared"] == 0


def test_compare_png_vs_video_partial_index_overlap():
    """PNG preds have frames 0,1,2 but video only has 0,1."""
    png_preds = [
        {"frame_index": 0, "fail_score": 0.3, "prediction": 1},
        {"frame_index": 1, "fail_score": 0.7, "prediction": 0},
        {"frame_index": 2, "fail_score": 0.4, "prediction": 1},
    ]
    video_result = {
        "frame_results": [
            {"frame_index": 0, "fail_score": 0.3, "prediction": 1},
            {"frame_index": 1, "fail_score": 0.7, "prediction": 0},
        ]
    }
    result = compare_png_vs_video(png_preds, video_result)
    assert result["total_frames_compared"] == 2  # frame 2 not in video


def test_assemble_video_preserves_all_frames():
    """Verify all source frames are encoded."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "frames"
        d.mkdir()
        for i in range(5):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(d / f"frame_{i:04d}.png"), img)

        out = Path(tmp) / "out.mp4"
        info = assemble_video(d, out, fps=1.0)
        assert info["frame_count"] == 5
        assert info["source_dir"] == str(d)
