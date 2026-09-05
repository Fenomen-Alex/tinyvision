#!/usr/bin/env python3
"""Stage 12B: deterministic footage analysis (no visual model available).

Computes objective content metrics at 1 Hz:
  - gray mean/std, RMS diff vs prev sample, fraction of changed pixels
  - bbox coverage of changed pixels (high+broad => camera move/cut; localized => print activity)
  - cut_flag
For spaghetti candidates: foreground blob growth vs an early reference frame in the central crop.
Records all parameters for reproducibility. Evidence only; nothing here labels a failure.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "datasets" / "external" / "video-frames"
DIFF_THRESH = 20
CUT_FRAC = 0.55
CUT_RMS = 35.0
FG_DIST = 45
FG_MIN_AREA = 0.08


def analyze(cap, n, fps, interval):
    rows, prev = [], None
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    step = max(1, round(fps * interval))
    fidx = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if fidx % step == 0:
            t = fidx / fps
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
            r = {"t": round(t, 3), "gray_mean": round(float(g.mean()), 2), "gray_std": round(float(g.std()), 2)}
            if prev is not None:
                d = np.abs(g - prev)
                rms = float(np.sqrt((d ** 2).mean()))
                m = d > DIFF_THRESH
                frac = float(m.mean())
                coords = np.argwhere(m)
                bf, cx, cy = 0.0, 0.5, 0.5
                if coords.size:
                    y0, x0 = coords.min(axis=0)
                    y1, x1 = coords.max(axis=0)
                    bf = float((y1 - y0) * (x1 - x0) / (h * w))
                    cx = float((x0 + x1) / 2 / w)
                    cy = float((y0 + y1) / 2 / h)
                cut = (frac >= CUT_FRAC and rms >= CUT_RMS) or (bf >= 0.9 and rms >= CUT_RMS)
                r.update({"rms_diff": round(rms, 2), "frac_changed": round(frac, 4),
                          "diff_bbox_frac": round(bf, 4), "diff_cx": round(cx, 3),
                          "diff_cy": round(cy, 3), "cut_flag": bool(cut)})
            prev = g
            rows.append(r)
        fidx += 1
    return rows


def fg_growth(cap, rows, h, w, fps, interval):
    if not rows:
        return rows
    ref_t = rows[0]["t"]
    ref = None
    x0, y0 = int(w * 0.2), int(h * 0.2)
    x1, y1 = int(w * 0.8), int(h * 0.8)
    cap.set(cv2.CAP_PROP_POS_MSEC, ref_t * 1000.0)
    ok, frame = cap.read()
    if ok:
        ref = frame[y0:y1, x0:x1].astype(np.float32)
    if ref is None:
        for r in rows:
            r["fg_area"] = r["fg_largest_frac"] = None
        return rows
    step = max(1, round(fps * interval))
    results = {}
    fidx = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if fidx % step == 0:
            t = fidx / fps
            c = f[y0:y1, x0:x1].astype(np.float32)
            fg = (np.abs(c - ref).mean(axis=2) > FG_DIST).astype(np.uint8)
            area = float(fg.mean())
            largest = 0.0
            if area >= FG_MIN_AREA:
                nlab, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
                sizes = stats[1:, cv2.CC_STAT_AREA] if nlab > 1 else np.array([0])
                largest = float(sizes.max()) / fg.size
            results[round(t, 2)] = (round(area, 4), round(largest, 4))
        fidx += 1
    for r in rows:
        k = round(r["t"], 2)
        if k in results:
            r["fg_area"], r["fg_largest_frac"] = results[k]
        else:
            t = round(r["t"], 3)
            keys = sorted(results.keys())
            near = min(keys, key=lambda z: abs(z - t)) if keys else None
            if near is not None and abs(near - t) <= interval:
                r["fg_area"], r["fg_largest_frac"] = results[near]
            else:
                r["fg_area"] = r["fg_largest_frac"] = None
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--spaghetti", default="SP001,SP002")
    args = ap.parse_args()

    videos = {
        "SP001": "SP001_3Ci-3ZYHBkM.mp4",
        "SP002": "SP002_qcVFnRcJcP4.mp4",
        "NORMAL001": "NORMAL001_dX4eGEFrthE.mp4",
        "UE001": "UE001_6V319bseO3U.mp4",
        "UE003": "UE003_k-inUpjRVLA.mp4",
        "STR001": "STR001_kuAMtoUJGgE.mp4",
    }
    spag = {x.strip() for x in args.spaghetti.split(",")}
    results = {"interval_s": args.interval, "videos": {}}
    for sid, fname in videos.items():
        src = ROOT / "raw" / sid / fname
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rows = analyze(cap, int(n // (fps * args.interval)) + 1, fps, args.interval)
        if sid in spag:
            rows = fg_growth(cap, rows, h, w, fps, args.interval)
        results["videos"][sid] = {"file": fname, "fps": fps, "width": w, "height": h,
                                  "nb_frames": n, "duration_s": round(n / fps, 3),
                                  "samples": rows}
        print(f"OK {sid}: {len(rows)} samples")
        cap.release()

    out = ROOT / "metadata" / "pilot_footage_analysis.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()