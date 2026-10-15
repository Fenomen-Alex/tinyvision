#!/usr/bin/env python3
"""Generate a deterministic synthetic video fixture for the TinyVision demo.

This produces a short, fully synthetic clip so the local demo and its tests can
be exercised without any real 3D-print footage, printer, or camera.

IMPORTANT: the clip is a **software test fixture**. It is generated from a fixed
random seed and it is NOT evidence of real-world detection accuracy. It is not a
dataset, it contains no real printing, and any detection shown on it proves only
that the inference pipeline runs end to end.

The clip has a quiet first half and then a synthetic, growing visual anomaly
(a dark blob plus thin "stringing" lines) so the demo has a visible change to
show. Whether the frozen classifier reacts to it is not the point.

Usage:
    python scripts/make_synthetic_demo_video.py \\
        --out /Users/alex/tinyvision-data/tinyvision-demo/fixtures/synthetic_print.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SEED = 20261015
FPS = 10.0
FRAMES = 120
SIZE = 320


def _base_layer(rng: np.random.Generator) -> np.ndarray:
    """A deterministic 'printed layer' backdrop."""
    img = np.full((SIZE, SIZE, 3), 40, dtype=np.uint8)
    for y in range(40, SIZE - 40, 12):
        cv2.line(img, (40, y), (SIZE - 40, y), (150, 150, 155), 2, cv2.LINE_AA)
    noise = rng.integers(-6, 7, size=(SIZE, SIZE, 3), dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def generate(out_path: Path, fps: float = FPS, frames: int = FRAMES,
             size: int = SIZE, seed: int = SEED) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (size, size))
    if not writer.isOpened():
        raise SystemExit(f"could not open video writer for {out_path}")

    anomaly_start = frames // 2
    for i in range(frames):
        frame = _base_layer(rng)
        if i >= anomaly_start:
            frac = (i - anomaly_start) / max(1, (frames - anomaly_start - 1))
            radius = int(18 + 70 * frac)
            cx, cy = size // 2 + int(20 * frac), size // 2
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), radius, (10, 10, 15), -1, cv2.LINE_AA)
            frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
            for k in range(4):
                x = 60 + k * 60
                cv2.line(frame, (x, 60), (x + int(30 * frac), size - 60),
                         (200, 200, 205), max(1, int(3 * frac)), cv2.LINE_AA)
        writer.write(frame)
    writer.release()

    meta = {
        "schema": "tinyvision-synthetic-fixture",
        "schema_version": 1,
        "synthetic": True,
        "disclaimer": (
            "SOFTWARE TEST FIXTURE — not real footage, not a dataset, and not "
            "evidence of real-world detection accuracy."
        ),
        "seed": seed,
        "fps": fps,
        "frames": frames,
        "size": size,
        "duration_s": round(frames / fps, 3),
        "anomaly_start_frame": anomaly_start,
        "path": str(out_path),
    }
    (out_path.with_suffix(".fixture.json")).write_text(json.dumps(meta, indent=2))
    return meta


def _default_out() -> str:
    """Default output path, matched to the demo's per-user data directory.

    Uses the package default when ``tv`` is importable from this checkout, so the
    fixture lands where the demo looks (respecting ``TINYVISION_DEMO_DATA``).
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
        from tv.webdemo import default_data_dir

        return str(default_data_dir() / "fixtures" / "synthetic_print.mp4")
    except Exception:  # pragma: no cover - fall back when run standalone
        return "/Users/alex/tinyvision-data/tinyvision-demo/fixtures/synthetic_print.mp4"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=_default_out())
    p.add_argument("--fps", type=float, default=FPS)
    p.add_argument("--frames", type=int, default=FRAMES)
    p.add_argument("--size", type=int, default=SIZE)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    meta = generate(Path(args.out), args.fps, args.frames, args.size, args.seed)
    print("Synthetic fixture written (SOFTWARE TEST FIXTURE — not real data):")
    print(f"  video : {meta['path']}")
    print(f"  meta  : {Path(meta['path']).with_suffix('.fixture.json')}")
    print(f"  {meta['frames']} frames @ {meta['fps']} fps = {meta['duration_s']}s, "
          f"{meta['size']}x{meta['size']}, seed {meta['seed']}")
    print("  Synthetic anomaly starts at "
          f"frame {meta['anomaly_start_frame']} (~{meta['anomaly_start_frame']/meta['fps']}s).")


if __name__ == "__main__":
    main()
