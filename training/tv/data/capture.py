"""Webcam frame capture for data collection.

Two capture paths are supported:

* ``capture_frames`` — a low-level flat capture that writes frames into a single
  directory. Kept for backward compatibility / automated use.
* ``capture_session`` — a session-oriented capture for building a curated,
  provenance-tracked target-camera dataset in Layout A:

      output_dir/<class>/<session_id>/
          metadata.json
          frames/
              <session_id>_000000.jpg
              ...

The session path REQUIRES an explicit ``class`` (``normal`` or ``failing``) and
records provenance in ``metadata.json`` at capture time, making it hard to
accidentally file a session under the wrong class.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

# Valid capture classes for a classification dataset. A session must be labelled
# as one of these before recording begins.
CAPTURE_CLASSES = ("normal", "failing")

# Fields that are only written to metadata when the caller actually supplies a
# value. We never invent metadata values.
_OPTIONAL_METADATA_FIELDS = ("printer", "object", "failure_type", "notes")


def _next_session_id(output_dir: Path, session_class: str) -> str:
    """Return the next session_XXX id that does not already exist.

    Sessions are numbered independently per class directory, so the id is unique
    within ``output_dir/<class>/``. The first session in a fresh class dir is
    ``session_001``.
    """
    class_dir = output_dir / session_class
    if not class_dir.is_dir():
        return "session_001"
    existing = [
        int(d.name.removeprefix("session_"))
        for d in class_dir.iterdir()
        if d.is_dir() and d.name.startswith("session_")
    ]
    nxt = max(existing, default=0) + 1
    return f"session_{nxt:03d}"


def _session_metadata(
    session_id: str,
    session_class: str,
    captured_at: str,
    camera_info: dict,
    resolution: tuple[int, int],
    frames: int,
    fps: float,
    duration: float,
    **extra,
) -> dict:
    """Assemble the session metadata dict.

    Only caller-supplied optional fields (``printer``, ``object``,
    ``failure_type``, ``notes``) are included.
    """
    meta = {
        "session_id": session_id,
        "class": session_class,
        "captured_at": captured_at,
        "camera": camera_info,
        "resolution": f"{resolution[0]}x{resolution[1]}",
        "frames": frames,
        "actual_fps": round(fps, 2),
        "duration_seconds": round(duration, 2),
    }
    for key in _OPTIONAL_METADATA_FIELDS:
        value = extra.get(key)
        if value is not None and str(value).strip() != "":
            meta[key] = value
    return meta


def _camera_info(cap, camera_index: int) -> dict:
    """Best-effort camera metadata. Values come from OpenCV, never invented."""
    info = {"device_index": camera_index}
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps_prop = cap.get(cv2.CAP_PROP_FPS)
    if w and h:
        info["width"] = int(w)
        info["height"] = int(h)
    if fps_prop and fps_prop > 0:
        info["fps"] = round(float(fps_prop), 2)
    backend_name = None
    try:
        backend_name = cap.getBackendName()
    except Exception:
        pass
    if backend_name:
        info["backend"] = backend_name
    return info


def capture_frames(
    output_dir: str | Path,
    session_id: str | None = None,
    duration: float = 10.0,
    fps: float = 2.0,
    camera: int = 0,
    resolution: tuple[int, int] | None = None,
) -> dict:
    """Capture frames from a webcam into a single directory (legacy, flat).

    Args:
        output_dir: Directory to save frames.
        session_id: Session prefix for filenames. Auto-generated if omitted.
        duration: Capture duration in seconds. 0 = until interrupted.
        fps: Target frames per second.
        camera: Camera device index.
        resolution: Optional (width, height) override.

    Returns:
        Dict with capture metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if session_id is None:
        session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera}")

    if resolution:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    interval = 1.0 / fps if fps > 0 else 0
    frame_count = 0
    start_time = time.time()
    consecutive_failures = 0
    max_consecutive_failures = 5

    print(f"Capturing to {output_dir} (session: {session_id})")
    print(f"Resolution: {actual_w}x{actual_h}, target FPS: {fps}")
    if duration > 0:
        print(f"Duration: {duration}s (Ctrl+C to stop early)")
    else:
        print("Press Ctrl+C to stop")

    try:
        while True:
            elapsed = time.time() - start_time
            if duration > 0 and elapsed >= duration:
                break

            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print("\nCamera stopped returning frames; ending capture")
                    break
                time.sleep(0.1)
                continue

            consecutive_failures = 0
            fname = f"{session_id}_{frame_count:06d}.jpg"
            cv2.imwrite(str(output_dir / fname), frame)
            frame_count += 1

            if interval > 0:
                time.sleep(max(0, interval - (time.time() - start_time - (frame_count - 1) * interval)))

    except KeyboardInterrupt:
        print("\nCapture interrupted")
    finally:
        cap.release()

    elapsed = time.time() - start_time
    return {
        "session_id": session_id,
        "output_dir": str(output_dir),
        "frames_captured": frame_count,
        "duration_seconds": round(elapsed, 2),
        "actual_fps": round(frame_count / max(elapsed, 0.001), 2),
        "resolution": f"{actual_w}x{actual_h}",
    }


def capture_session(
    output_dir: str | Path,
    session_class: str,
    session_id: str | None = None,
    duration: float = 10.0,
    fps: float = 2.0,
    camera: int = 0,
    resolution: tuple[int, int] | None = None,
    printer: str | None = None,
    object: str | None = None,
    failure_type: str | None = None,
    notes: str | None = None,
) -> dict:
    """Capture a labelled session into Layout A with provenance metadata.

    Creates ``output_dir/<class>/<session_id>/frames/`` and writes
    ``metadata.json`` recording provenance at capture time. The class is
    validated up-front; session IDs are auto-sequenced per class so sessions
    never collide and the class is structurally enforced by the directory.

    Args:
        output_dir: Root dataset directory (e.g. ``datasets/target-camera``).
        session_class: ``normal`` or ``failing`` (required and validated).
        session_id: Optional explicit id (``session_XXX``). Auto-assigned if
            omitted using the next available number within the class.
        duration: Capture duration in seconds. 0 = until interrupted.
        fps: Target frames per second.
        camera: Camera device index.
        resolution: Optional (width, height) override.
        printer: Optional printer identifier recorded in metadata.
        object: Optional print/object identifier recorded in metadata.
        failure_type: Optional failure type (e.g. ``stringing``, ``under_extrusion``).
        notes: Optional free-text notes.

    Returns:
        Dict with capture and session metadata.
    """
    session_class = session_class.lower()
    if session_class not in CAPTURE_CLASSES:
        raise ValueError(
            f"Invalid session class {session_class!r}; must be one of "
            f"{CAPTURE_CLASSES}"
        )

    output_dir = Path(output_dir)
    if session_id is None:
        session_id = _next_session_id(output_dir, session_class)

    frames_dir = output_dir / session_class / session_id / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera}")

    if resolution:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    interval = 1.0 / fps if fps > 0 else 0
    frame_count = 0
    start_time = time.time()
    consecutive_failures = 0
    max_consecutive_failures = 5

    print(f"Class      : {session_class}")
    print(f"Session    : {session_id}")
    print(f"Frames dir : {frames_dir}")
    print(f"Resolution : {actual_w}x{actual_h}, target FPS: {fps}")
    if duration > 0:
        print(f"Duration   : {duration}s (Ctrl+C to stop early)")
    else:
        print("Duration   : until Ctrl+C")

    try:
        while True:
            elapsed = time.time() - start_time
            if duration > 0 and elapsed >= duration:
                break

            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print("\nCamera stopped returning frames; ending capture")
                    break
                time.sleep(0.1)
                continue

            consecutive_failures = 0
            fname = f"{session_id}_{frame_count:06d}.jpg"
            cv2.imwrite(str(frames_dir / fname), frame)
            frame_count += 1

            if interval > 0:
                time.sleep(max(0, interval - (time.time() - start_time - (frame_count - 1) * interval)))

    except KeyboardInterrupt:
        print("\nCapture interrupted")
    finally:
        # Read camera metadata while the capture handle is still open.
        cam = _camera_info(cap, camera)
        cam.setdefault("width", actual_w)
        cam.setdefault("height", actual_h)
        cap.release()

    elapsed = time.time() - start_time
    actual_fps = frame_count / max(elapsed, 0.001)

    captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    metadata = _session_metadata(
        session_id,
        session_class,
        captured_at,
        cam,
        (actual_w, actual_h),
        frame_count,
        actual_fps,
        elapsed,
        printer=printer,
        object=object,
        failure_type=failure_type,
        notes=notes,
    )
    meta_path = output_dir / session_class / session_id / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "session_id": session_id,
        "class": session_class,
        "session_dir": str(output_dir / session_class / session_id),
        "metadata_path": str(meta_path),
        "frames_captured": frame_count,
        "duration_seconds": round(elapsed, 2),
        "actual_fps": round(actual_fps, 2),
        "resolution": f"{actual_w}x{actual_h}",
        "metadata": metadata,
    }
