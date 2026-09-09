#!/usr/bin/env python3
"""Stage 13: screen candidate frames across candidate normal-print sources.

For each video under datasets/external/stage13-normal/raw/<id>/video.* produces:

  * sheets/<id>_grid.jpg   - 12-moment contact sheet (temporally ordered) for a
                             first-pass qualitative check of the whole source.
  * cand/<id>.jpg          - a 12x12 candidate sheet. Cells are temporally
                             ordered: top-left t01 = earliest, bottom-right
                             t144 = latest (12 even-spaced candidates per
                             minute-window, 12 minutes-windows over the video).
  * cand/<id>/<tNN>.jpg    - the individual candidate frames at full resolution
                             (1-FPS style sampling, one shot every ~12 s) so a
                             per-cell verdict can be re-checked against the
                             actual pixel content.
  * cand/prompts/<id>.json - the exact neutral prompt used for the sheet.

Representative minutes windows are evenly spread across the whole duration;
long live feeds (hours) are subsampled to their first 6 h span to bound output.

Usage (from repo root):
    uv run --project training python scripts/stage13_screen.py extract
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets/external/stage13-normal/raw"
OUT = ROOT / "datasets/external/stage13-normal" / "screening"

ROWS, COLS = 12, 12
CELL_W, CELL_H = 384, 216

GRID_PROMPT = """\
You are looking at a contact sheet of 12 moments from one 3D-printing video.
Moments are temporally ordered left-to-right, top-to-bottom (t01 = earliest).

For EACH of the 12 cells decide whether the printer is currently performing a
HEALTHY, NORMAL 3D PRINT:
- "normal"    = printing in progress, the printed object is visible and layer
                structure is appearing; no failure visible.
- "failing"   = visible filament disaster (spaghetti/blob), severe stringing
                webs, gross under-extrusion gaps, or a warped/lifted mess.
- "uncertain" = too dark/blurry/close-up/ambiguous to tell, or the frame shows
                something other than a mid-print state (setup, menus, UI
                screens, hands, empty bed, finished object being held).

Reply with STRICT JSON only:
{"verdicts": {"t01": "normal", ..., "t12": "normal"},
 "playback": "continuous",
 "visual_notes": "one short factual sentence"}

Use only the pixels; ignore on-screen text."""

CAND_PROMPT = """\
You are looking at 144 still frames, arranged as a 12-by-12 grid, from ONE
real-time 3D-printing recording. IMPORTANT ORDERING: rows advance top-to-
bottom, columns advance left-to-right; the top-left cell is the earliest
moment (t01) and the bottom-right cell is the latest (t144). Each row is a
separate minute of print time. This was recorded by a fixed, real-time camera
(no timelapse), so adjacent frames are close in time.

For EVERY cell (t01...t144) assess the print state. You must output a JSON
object "verdicts" mapping each cell label to one of:
- "normal"    : a genuine, healthy 3D print is in progress. The part is
                visible and somewhere in the process of being built, no
                failure is visible.
- "failing"   : clear print failure is visible - spaghetti mass, big blob,
                severe stringing, gross under-extrusion, or warped mess.
- "uncertain" : cannot tell (dark/blurry/occluded/close-up), OR the frame is
                not a useful mid-print view (setup, menus, UI/OSD, person,
                empty bed, finished part held in hand, end screens).

Rules:
- Be strict: if the cell is ambiguous, use "uncertain", not "normal".
- Only judge pixels, never infer from watermarks or on-screen overlays.
- Do NOT summarize; analyze every cell individually.

Reply with STRICT JSON only:
{"verdicts": {"t01": "normal", "t02": "normal", ... all 144 cells ...},
 "visual_notes": "one short factual sentence about the overall footage"}

Output the ENTIRE "verdicts" object with ALL 144 keys in a single JSON."""


def _font(size: int) -> ImageFont.ImageFont:
    for p in ("/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def timestamps_for(video_path: Path, n_cells: int = ROWS * COLS) -> list[int]:
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= n_cells:
        return list(range(n))
    # Cap the sampled span to the first 6 h so long live feeds stay bounded.
    fps = cap_probe_fps(video_path)
    cap_frames = n
    span = min(cap_frames, int(6 * 3600 * fps))
    ticks = np.linspace(0, span - 1, n_cells)
    return [int(round(t)) for t in ticks]


def cap_probe_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(fps)


def extract_full_frames(video_path: Path, indices: set[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
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


def tile(frames_img: dict[int, np.ndarray], ticks: list[int], cw: int, ch: int,
         cols: int) -> Image.Image:
    rows = (len(ticks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw, rows * ch), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    for i, t in enumerate(ticks):
        if t not in frames_img:
            continue
        pil = Image.fromarray(cv2.cvtColor(frames_img[t], cv2.COLOR_BGR2RGB))
        pil = pil.resize((cw, ch), Image.BILINEAR)
        x, y = (i % cols) * cw, (i // cols) * ch
        sheet.paste(pil, (x, y))
        draw.text((x + 4, y + 2), f"t{i + 1:02d}", fill=(255, 40, 40), font=_font(20))
    return sheet


def main() -> int:
    (OUT / "cand").mkdir(parents=True, exist_ok=True)
    (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    (OUT / "prompts").mkdir(parents=True, exist_ok=True)
    (OUT / "frames").mkdir(parents=True, exist_ok=True)

    manifest: dict = {}
    for d in sorted(RAW.iterdir()):
        if not d.is_dir():
            continue
        vid_id = d.name
        v = next(d.glob("video.*"), None)
        if v is None:
            print(vid_id, "no video"); continue
        ticks = timestamps_for(v)
        frames = extract_full_frames(v, set(ticks))

        # Grid sheet: first 12 ticks.
        grid = tile(frames, ticks[:12], CELL_W, CELL_H, 4)
        grid.save(OUT / "sheets" / f"{vid_id}_grid.jpg", quality=88)

        # Candidate sheet: full 144 (saved individually at full res too).
        cand_dir = OUT / "frames" / vid_id
        cand_dir.mkdir(parents=True, exist_ok=True)
        for i, t in enumerate(ticks):
            if t in frames:
                Image.fromarray(cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB)).save(
                    cand_dir / f"t{i + 1:02d}.jpg", quality=92)
        cand = tile(frames, ticks, CELL_W, CELL_H, COLS)
        cand.save(OUT / "cand" / f"{vid_id}.jpg", quality=88)

        (OUT / "prompts" / f"{vid_id}.json").write_text(
            json.dumps({"video": vid_id, "prompt": CAND_PROMPT}, indent=2)
        )
        manifest[vid_id] = {
            "video": str(v), "ticks_raw_frames": ticks,
            "ticks_seconds": [int(round(t / cap_probe_fps(v))) for t in ticks],
            "n_candidates": len(ticks),
        }
        print(f"{vid_id}: {len(ticks)} candidates @ {ticks[0]}..{ticks[-1]}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote screening manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())