#!/usr/bin/env python3
"""Extract review images + structural signals for the real-world evaluation set.

For each video in datasets/external/real-world-eval/, this produces:

  * a 12-frame labelled contact sheet (4 cols x 3 rows, temporally ordered),
  * a 5-frame ongoing-state strip centred on every runtime event frame (the
    raw sampled frame at event and the two sampled frames before/after),
  * single-frame images for the max-P and min-P sampled frames,
  * a per-sample structural time series (object-coverage fraction) computed on
    the same 12 grid timestamps used for the contact sheet.

The sampled runtime frames (frame_idx) are taken from the runtime JSON report so
the review is anchored to the exact frames the model saw. Output goes to
datasets/external/real-world-eval/review/. This is evaluation-only data and is
kept separate from training/verified-label data.

Usage (from repo root):
    uv run --project training python scripts/real_world_extract.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
VID_DIR = ROOT / "datasets/external/real-world-eval"
RUN_DIR = VID_DIR / "runs"
REVIEW_DIR = VID_DIR / "review"

GRID_FRAMES = 12
CELL_W, CELL_H = 384, 216


def _font(size: int) -> ImageFont.ImageFont:
    for p in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def probe(video: Path) -> tuple[int, int, float]:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, w, h


def extract_frames(video: Path, indices: set[int]) -> dict[int, np.ndarray]:
    """Frame-index -> BGR image. Reads the whole file once (videos are small)."""
    cap = cv2.VideoCapture(str(video))
    want = set(indices)
    out: dict[int, np.ndarray] = {}
    idx = -1
    while want:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx in want:
            want.discard(idx)
            out[idx] = frame
    cap.release()
    return out


def coverage(img: np.ndarray) -> float:
    """Non-background pixel fraction, a crude object-growth signal."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    return round(float(np.count_nonzero(thr)) / thr.size, 4)


def tile(frames: list[tuple[str, np.ndarray]]) -> Image.Image:
    cols = 4 if len(frames) >= 4 else len(frames)
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * CELL_W, rows * CELL_H), (240, 240, 240))
    draw = ImageDraw.Draw(sheet)
    for i, (label, img) in enumerate(frames):
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil = pil.resize((CELL_W, CELL_H), Image.BILINEAR)
        x, y = (i % cols) * CELL_W, (i // cols) * CELL_H
        sheet.paste(pil, (x, y))
        draw.text((x + 6, y + 4), label, fill=(255, 32, 32), font=_font(22))
    return sheet


def main() -> int:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    signal: dict = {}
    for vid_dir in sorted(VID_DIR.iterdir()):
        if not vid_dir.is_dir() or vid_dir.name in ("runs", "review"):
            continue
        vid_id = vid_dir.name
        video_path = next(vid_dir.glob("video.*"), None)
        report_path = RUN_DIR / f"{vid_id}.json"
        if video_path is None or not report_path.exists():
            print(f"{vid_id}: skip (no video or report)")
            continue
        report = json.loads(report_path.read_text())
        per_frame = report["per_frame"]
        n, w, h = probe(video_path)
        fps = float(report["source"]["native_fps"] or 30.0)
        total = float(report["source"].get("total_frames_in_video") or n)

        # Grid timestamps spread across the video.
        grid_raw = sorted({round(i * (total - 1) / (GRID_FRAMES - 1)) for i in range(GRID_FRAMES)})
        grid_t = [int(x) for x in grid_raw]

        # Runtime event frames (raw video index of the sampled frame that fired).
        events = [r["frame_idx"] for r in per_frame if r.get("event")]
        # Ongoing-state strip: the sampled frames around the first event.
        event_strip: list[int] = []
        if events:
            ev = events[0]
            idxs = [i for i, r in enumerate(per_frame) if r["frame_idx"] == ev]
            if idxs:
                i = idxs[0]
                event_strip = [per_frame[j]["frame_idx"] for j in range(max(0, i - 2), min(len(per_frame), i + 3))]

        extrema = []
        if per_frame:
            low = min(per_frame, key=lambda r: r["raw"])
            high = max(per_frame, key=lambda r: r["raw"])
            extrema = [("minP", low["frame_idx"]), ("maxP", high["frame_idx"])]

        want = set(grid_t) | set(event_strip) | {x[1] for x in extrema}
        frames = extract_frames(video_path, want)

        # Coverage signal on grid timestamps.
        cov = [coverage(frames[t]) if t in frames else None for t in grid_t]
        signal[vid_id] = {
            "n_frames": n,
            "native_fps": fps,
            "grid_raw_frames": grid_t,
            "grid_times_s": [round(t / fps, 2) for t in grid_t],
            "coverage": cov,
        }

        # Contact sheet (labelled t01..t12).
        grid_images = [
            (f"t{i + 1:02d}", frames[t])
            for i, t in enumerate(grid_t)
            if t in frames
        ]
        if grid_images:
            sheet = tile(grid_images)
            sheet.save(REVIEW_DIR / f"{vid_id}_grid.jpg", quality=88)

        # Event strip (first event, 5 contiguous sampled frames).
        if event_strip and all(f in frames for f in event_strip):
            strip = tile(
                [("event-%d" % (j - 2), frames[f]) for j, f in enumerate(event_strip)]
            )
            strip.save(REVIEW_DIR / f"{vid_id}_eventstrip.jpg", quality=88)

        # Extrema single frames.
        for tag, f in extrema:
            if f in frames:
                Image.fromarray(
                    cv2.cvtColor(frames[f], cv2.COLOR_BGR2RGB)
                ).save(REVIEW_DIR / f"{vid_id}_{tag}.jpg", quality=88)

        print(
            f"{vid_id}: grid={len(grid_images)}/12 cov={cov[0]}..{cov[-1]} "
            f"events={len(events)} first_event_raw={events[0] if events else '-'}"
        )

    (REVIEW_DIR / "structural_signal.json").write_text(
        json.dumps(signal, indent=2)
    )
    print("wrote", REVIEW_DIR / "structural_signal.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())