#!/usr/bin/env python3
"""Stage 13: per-minute candidate sampling for stage13-normal sources.

Protocol: one sample every OFFSET seconds (default 35) + every
SAMPLE_EVERY_S (30 s), capped at MAX_SAMPLES (60) per source, written into a
per-source subdirectory (pass_a / pass_b). Two offset passes (35 s and 50 s)
avoid near-duplicate frames while increasing per-source density for short
sources.

Outputs per source under datasets/external/stage13-normal/screening/samples/:
  * <id>/<pass>/<sNN>.jpg - full-resolution sample frame (s01 = earliest)
Additionally writes samples/manifest.json with per-sample timestamp, raw frame
index, and a crude object-coverage signal.

Usage (from repo root):
    uv run --project training python scripts/stage13_candidates.py --pass a --offset 35
    uv run --project training python scripts/stage13_candidates.py --pass b --offset 50
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets/external/stage13-normal/raw"
OUT = ROOT / "datasets/external/stage13-normal" / "screening" / "samples"

MAX_SAMPLES = 60
SAMPLE_EVERY_S = 30
START_OFFSET_S = 35
SPAN_CAP_H = 6.0

INCLUDE = [
    "DHpo4Za2--M",    # Ender 3 Pro benchy (training)
    "LwHaHzVV_iE",    # Creality K1 benchy (training)
    "PscyiwiZ9Mc",    # XIMBRIUM test print (training)
    "R0xnULlObrs",    # AnyCubic Chiron wall (training)
    "kZ06bpskj9I",    # K1 Max benchy (training)
    "uCCXJzvYxKA",    # Prusa Core One benchy (training)
    "zFX80aMaYFg",    # C270 webcam live (training)
    "LY3jlcO34f4",    # Civilized Barbarian live (EVAL holdout)
    "zkJgLanCSUw",    # Martinez benchy (EVAL holdout)
]


def probe(video: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, float(fps), w, h


def sample_seconds(n_frames: int, fps: float, offset: float, max_samples: int,
                   step: float) -> list[float]:
    duration = n_frames / fps
    span = min(duration, SPAN_CAP_H * 3600.0)
    n = min(max_samples, int(max(0, (span - offset)) // step) + 1)
    return[offset + i * step for i in range(n)]  # noqa: C413


def coverage(img: np.ndarray, white_bg: bool) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)[1]
    frac_bright = float(np.count_nonzero(thr)) / thr.size
    return frac_bright if not white_bg else 1.0 - frac_bright


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_name", default="a",
                    help="pass id, used as output subdir (a=offset 35, b=offset 50)")
    ap.add_argument("--offset", type=float, default=START_OFFSET_S)
    ap.add_argument("--step", type=float, default=SAMPLE_EVERY_S)
    ap.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    args = ap.parse_args()

    manifest: dict = {}
    for vid_id in INCLUDE:
        v = next((RAW / vid_id).glob("video.*"), None)
        if v is None:
            print(vid_id, "no video"); continue
        n, fps, w, h = probe(v)
        secs = sample_seconds(n, fps, args.offset, args.max_samples, args.step)
        raw_idx = [min(int(round(t * fps)), n - 1) for t in secs]

        out_dir = OUT / vid_id / args.pass_name
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(v))
        want = set(raw_idx)
        got = {}
        idx = -1
        while want:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx in want:
                want.discard(idx)
                got[idx] = frame
        cap.release()

        rows = []
        for i, (t, r) in enumerate(zip(secs, raw_idx)):
            if r not in got:
                continue
            img = got[r]
            name = f"s{i + 1:02d}"
            cv2.imwrite(str(out_dir / f"{name}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append({
                "sample": name,
                "pass": args.pass_name,
                "time_s": round(t, 1),
                "raw_frame": int(r),
                "coverage_bright": round(coverage(img, white_bg=False), 4),
            })
        manifest[vid_id] = {
            "samples": rows,
            "resolution": f"{w}x{h}",
            "fps": fps,
            "n_raw_frames": int(n),
            "trust": "sample",
        }
        print(f"[pass {args.pass_name}] {vid_id}: {len(rows)} samples "
              f"({secs[0]:.0f}s..{secs[-1]:.0f}s)")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote samples manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())