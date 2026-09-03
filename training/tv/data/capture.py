"""Webcam frame capture for data collection."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2


def capture_frames(
    output_dir: str | Path,
    session_id: str | None = None,
    duration: float = 10.0,
    fps: float = 2.0,
    camera: int = 0,
    resolution: tuple[int, int] | None = None,
) -> dict:
    """Capture frames from a webcam.

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
                print("Frame read failed, retrying...")
                time.sleep(0.1)
                continue

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
