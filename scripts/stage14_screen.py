#!/usr/bin/env python3
"""Stage-14 candidate screening: extract frames + contact sheets.

For each video under datasets/external/stage14-failure/raw/<id>/, extract a set
of frames spread evenly across the duration, compute an adjacent-frame-diff
proxy, and render a contact sheet. Output goes under
datasets/external/stage14-failure/screening/.

Usage:
    python scripts/stage14_screen.py [--raw DIR] [--max-frames N]
"""

import argparse
import hashlib
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

RAW = os.path.join("datasets", "external", "stage14-failure", "raw")
OUT = os.path.join("datasets", "external", "stage14-failure", "screening")
COLS = 6


def iter_videos(raw):
    if not os.path.isdir(raw):
        return
    for did in sorted(os.listdir(raw)):
        dpath = os.path.join(raw, did)
        if not os.path.isdir(dpath):
            continue
        vids = sorted(
            v for v in os.listdir(dpath)
            if v.lower().endswith((".mp4", ".webm", ".mkv", ".mov"))
        )
        if not vids:
            print(f"skip {did}: no video file", file=sys.stderr)
            continue
        yield did, os.path.join(dpath, vids[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--ids", default=None, help="comma-separated subset")
    ap.add_argument("--max-frames", type=int, default=42)
    args = ap.parse_args()

    wanted = set(args.ids.split(",")) if args.ids else None
    frames_dir = os.path.join(OUT, "frames")
    mont_dir = os.path.join(OUT, "montages")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(mont_dir, exist_ok=True)

    for did, vpath in iter_videos(args.raw):
        if wanted and did not in wanted:
            continue
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"skip {did}: cannot open", file=sys.stderr)
            continue
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        dur = n / fps if fps else 0
        step = max(0.5, dur / args.max_frames)
        t_from = min(30.0, dur / 4)
        times = [t_from + i * step for i in range(args.max_frames)]
        times = [t for t in times if t < dur - 1]
        if not times:
            cap.release()
            continue

        out_dir = os.path.join(frames_dir, did)
        os.makedirs(out_dir, exist_ok=True)
        frames = []
        prev = None
        diffs = []
        for i, t in enumerate(times):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                d = float(np.mean(np.abs(gray.astype(np.int16) - prev.astype(np.int16))))
                diffs.append(d)
            prev = gray
            fname = os.path.join(out_dir, f"s{i:03d}.jpg")
            cv2.imwrite(fname, fr)
            frames.append((t, fname, fr))
        cap.release()

        summary = {
            "video": vpath,
            "duration_s": round(dur, 1),
            "fps": round(fps, 3),
            "frames_extracted": len(frames),
            "mean_adjacent_diff": round(float(np.mean(diffs)), 2) if diffs else None,
            "max_adjacent_diff": round(float(np.max(diffs)), 2) if diffs else None,
            "pct_frames_almost_static": round(
                100.0 * sum(1 for d in diffs if d < 4.0) / len(diffs), 1
            ) if diffs else None,
            "file_md5": hashlib.md5(open(vpath, "rb").read()).hexdigest(),
        }

        fic = os.path.join(OUT, "candidate_meta.json")
        meta = {}
        if os.path.exists(fic):
            meta = json.load(open(fic))
        meta[did] = summary
        json.dump(meta, open(fic, "w"), indent=2)

        thumb_w = 384
        rows = (len(frames) + COLS - 1) // COLS
        cell_h = 216
        margin = 10
        W = COLS * thumb_w + (COLS + 1) * margin
        H = rows * cell_h + (rows + 1) * margin + 24
        canvas = Image.new("RGB", (W, H), (16, 16, 16))
        dr = ImageDraw.Draw(canvas)
        dr.text((margin, 4), f"{did}  dur={dur:.0f}s fps={fps:.1f} mean_diff={summary['mean_adjacent_diff']} static%={summary['pct_frames_almost_static']}", fill=(220, 220, 220))
        for idx, (t, fname, fr) in enumerate(frames):
            im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            im.thumbnail((thumb_w, cell_h))
            r, c = divmod(idx, COLS)
            x = margin + c * (thumb_w + margin)
            y = 28 + r * (cell_h + margin)
            canvas.paste(im, (x, y))
        mp = os.path.join(mont_dir, f"{did}.png")
        canvas.save(mp)
        print(json.dumps(summary))
        print(f"  montage: {mp}")


if __name__ == "__main__":
    main()