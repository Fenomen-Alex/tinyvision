"""CLI-level tests for `tv poc` and `tv calibrate` (end-to-end, subprocess)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TRAINING = REPO / "training"
BASELINE_CKPT = (
    REPO / "experiments" / "shufflenet_v2_x1_0_printing-errors_cdc9e2cad4"
    / "artifacts" / "best.pt"
)


def _run(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(TRAINING) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-m", "tv.cli", *args],
                          cwd=REPO, env=env, capture_output=True, text=True)


def _make_video(path: Path, n_frames: int = 8, fps: float = 4.0, size: int = 48) -> Path:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    assert writer.isOpened()
    for i in range(n_frames):
        writer.write(np.full((size, size, 3), 30 + i * 5, dtype=np.uint8))
    writer.release()
    return path


cv2_available = True
try:
    import cv2  # noqa: F401
except Exception:  # pragma: no cover
    cv2_available = False


# ---------------------------------------------------------------------------
# tv poc
# ---------------------------------------------------------------------------

needs = pytest.mark.skipif(
    not (cv2_available and BASELINE_CKPT.exists()),
    reason="OpenCV or baseline checkpoint not available",
)


@needs
def test_poc_cli_writes_json_and_summary(tmp_path):
    video = _make_video(tmp_path / "clip.mp4", n_frames=8)
    out = tmp_path / "report.json"
    r = _run("poc", "--input", str(video), "--checkpoint", str(BASELINE_CKPT),
             "--output-json", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "POC inference summary" in r.stdout
    report = json.loads(out.read_text())
    assert report["schema"] == "poc-inference"
    assert report["timing"]["processed_frames"] == 8
    assert len(report["per_frame"]) == 8


@needs
def test_poc_cli_default_json_path_next_to_input(tmp_path):
    video = _make_video(tmp_path / "clip.mp4", n_frames=4)
    r = _run("poc", "--input", str(video), "--checkpoint", str(BASELINE_CKPT))
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "clip.poc.json").exists()


def test_poc_cli_missing_source_exit_2(tmp_path):
    r = _run("poc", "--input", str(tmp_path / "nope.mp4"),
             "--checkpoint", str(BASELINE_CKPT))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "error:" in r.stderr


def test_poc_cli_invalid_threshold_exit_2(tmp_path):
    r = _run("poc", "--input", "x.mp4", "--threshold", "5.0",
             "--checkpoint", str(BASELINE_CKPT))
    assert r.returncode == 2
    assert "threshold" in r.stderr


def test_poc_cli_input_and_camera_mutually_exclusive():
    r = _run("poc", "--input", "x.mp4", "--camera", "0",
             "--checkpoint", str(BASELINE_CKPT))
    assert r.returncode == 2
    assert "not both" in r.stderr


# ---------------------------------------------------------------------------
# tv calibrate
# ---------------------------------------------------------------------------

def _score_clip(cid, label, split, base, n=10):
    return {
        "id": cid, "label": label, "split": split, "source": "synthetic",
        "scores": [{"timestamp_s": i * 0.5, "score": base} for i in range(n)],
    }


def test_calibrate_cli_writes_report(tmp_path):
    spec = {"printer": "P", "camera": "cam0", "clips": [
        _score_clip("n1", "normal", "calibration", 0.05),
        _score_clip("n2", "normal", "calibration", 0.10),
        _score_clip("f1", "failing", "calibration", 0.90),
        _score_clip("f2", "failing", "calibration", 0.85),
    ]}
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec))
    out = tmp_path / "cal.json"
    r = _run("calibrate", "--clips", str(sp), "--output-json", str(out),
             "--thresholds", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "threshold calibration" in r.stdout
    report = json.loads(out.read_text())
    assert report["schema"] == "tinyvision-calibration"
    assert report["status"] == "ok"
    assert report["calibration"]["detection_rate"] == 1.0


def test_calibrate_cli_bad_spec_exit_2(tmp_path):
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps({"nope": True}))
    r = _run("calibrate", "--clips", str(sp))
    assert r.returncode == 2
    assert "error:" in r.stderr


def test_calibrate_cli_locked_test_split_rejected(tmp_path):
    spec = {"clips": [_score_clip("x", "normal", "test", 0.1)]}
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec))
    r = _run("calibrate", "--clips", str(sp))
    assert r.returncode == 2
    assert "split" in r.stderr


def test_calibrate_cli_missing_checkpoint_exit_2(tmp_path):
    spec = {"clips": [{"id": "n1", "label": "normal", "source": "x.mp4"}]}
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec))
    r = _run("calibrate", "--clips", str(sp))
    assert r.returncode == 2
    assert "checkpoint" in r.stderr
