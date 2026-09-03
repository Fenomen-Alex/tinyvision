"""Tests for session-oriented capture (provenance, layout, class safety)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tv.data.capture import CAPTURE_CLASSES, capture_session


class _FakeCap:
    """Minimal stand-in for cv2.VideoCapture, injected via monkeypatch."""

    def __init__(self, index: int):
        self._open = True
        self._width = 640
        self._height = 480
        self._frames = 3
        self._read = 0

    def isOpened(self) -> bool:
        return self._open

    def open(self, index: int) -> bool:  # noqa: A003
        self._open = True
        return True

    def set(self, prop: int, value) -> bool:
        if prop == 3:  # CAP_PROP_FRAME_WIDTH
            self._width = int(value)
        elif prop == 4:  # CAP_PROP_FRAME_HEIGHT
            self._height = int(value)
        return True

    def get(self, prop: int) -> float:
        if prop == 3:
            return self._width
        if prop == 4:
            return self._height
        if prop == 5:  # CAP_PROP_FPS
            return 30.0
        return 0.0

    def getBackendName(self) -> str:
        return "FAKE"

    def read(self):
        if self._read >= self._frames:
            return False, None
        self._read += 1
        import numpy as np
        return True, np.zeros((self._height, self._width, 3), dtype=np.uint8)

    def release(self) -> None:
        self._open = False


@pytest.fixture(autouse=True)
def _fake_cv2(monkeypatch):
    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", _FakeCap)
    # imwrite just writes bytes
    def _imwrite(path, frame):
        Path(path).write_bytes(b"\xff\xff\xff")
        return True
    monkeypatch.setattr(cv2, "imwrite", _imwrite)


def _frames_saved(session_dir: Path, expected: int) -> bool:
    frames = list((session_dir / "frames").glob("*.jpg"))
    return len(frames) == expected


def test_capture_session_layout_and_metadata(tmp_path: Path) -> None:
    result = capture_session(
        tmp_path / "target-camera",
        session_class="normal",
        duration=0.0,  # requires 3 frames then ends because fake cap runs out
        fps=1000.0,
        printer="printer-A",
        object="object-1",
        notes="test note",
    )
    # Layout A structure: <root>/normal/session_001/{metadata.json,frames/}
    session_dir = tmp_path / "target-camera" / "normal" / "session_001"
    assert session_dir.is_dir()
    assert (session_dir / "frames").is_dir()
    assert (session_dir / "metadata.json").is_file()
    assert result["class"] == "normal"
    # frames were written (fake cap yields 3 frames at high fps)
    assert _frames_saved(session_dir, 3)

    meta = json.loads((session_dir / "metadata.json").read_text())
    assert meta["session_id"] == "session_001"
    assert meta["class"] == "normal"
    assert meta["printer"] == "printer-A"
    assert meta["object"] == "object-1"
    assert meta["notes"] == "test note"
    # failure_type must NOT be invented when not provided
    assert "failure_type" not in meta
    assert "captured_at" in meta
    assert "camera" in meta
    assert meta["camera"]["device_index"] == 0


def test_capture_failure_type_only_when_provided(tmp_path: Path) -> None:
    result = capture_session(
        tmp_path / "target-camera",
        session_class="failing",
        duration=0.0,
        fps=1000.0,
        failure_type="stringing",
    )
    session_dir = tmp_path / "target-camera" / "failing" / "session_001"
    meta = json.loads((session_dir / "metadata.json").read_text())
    assert meta["class"] == "failing"
    assert meta["failure_type"] == "stringing"
    # not-provided optional fields must be absent
    assert "printer" not in meta
    assert "object" not in meta
    assert "notes" not in meta


def test_capture_rejects_invalid_class(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        capture_session(tmp_path / "target-camera", session_class="safe", duration=0.0)


def test_capture_auto_sequences_sessions_per_class(tmp_path: Path) -> None:
    # Two normal sessions -> session_001, session_002
    capture_session(tmp_path / "target-camera", session_class="normal", duration=0.0, fps=1000.0)
    r2 = capture_session(tmp_path / "target-camera", session_class="normal", duration=0.0, fps=1000.0)
    assert r2["session_id"] == "session_002"
    # A failing session is numbered independently within its class dir.
    r3 = capture_session(tmp_path / "target-camera", session_class="failing", duration=0.0, fps=1000.0)
    assert r3["session_id"] == "session_001"


def test_capture_classes_constant() -> None:
    assert set(CAPTURE_CLASSES) == {"normal", "failing"}
