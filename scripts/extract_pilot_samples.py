#!/usr/bin/env python3
"""Stage 12B pilot: extract a small inspection sample from each acquired video.

Deterministic given input. Records extraction parameters alongside extracted frames.
"""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "datasets" / "external" / "video-frames"


def parse_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def parse_frame_count(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "default=noprint_wrappers=1:nokey=1",
         str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(out)


def timestamps_for(duration: float, target: int) -> list[float]:
    if duration <= 60.0:
        return [i for i in range(0, int(duration) + 1)]
    step = (duration - 0.5) / (target + 1)
    return [round(0.5 + step * i, 3) for i in range(target)]


def extract_one(src: Path, out_dir: Path, ts: float, index: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"frame_{index:04d}_t{ts:08.3f}.jpg"
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{ts}", "-i", str(src), "-frames:v", "1",
         "-vf", "scale=iw:-2", "-q:v", "2", str(out)],
        capture_output=True, check=True,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pilot inspection samples")
    parser.add_argument("--target-per-video", type=int, default=35)
    args = parser.parse_args()

    videos = {
        "SP001": "SP001_3Ci-3ZYHBkM.mp4",
        "SP002": "SP002_qcVFnRcJcP4.mp4",
        "NORMAL001": "NORMAL001_dX4eGEFrthE.mp4",
        "UE001": "UE001_6V319bseO3U.mp4",
        "UE003": "UE003_k-inUpjRVLA.mp4",
        "STR001": "STR001_kuAMtoUJGgE.mp4",
    }

    extraction_records = {}
    for source_id, fname in videos.items():
        src = ROOT / "raw" / source_id / fname
        if not src.exists():
            print(f"SKIP {source_id}: {src} missing")
            continue
        duration = parse_duration(src)
        target = len(timestamps_for(duration, args.target_per_video))
        out_dir = ROOT / "extracted" / source_id / "sample"
        tss = timestamps_for(duration, args.target_per_video)
        extracted = []
        for i, ts in enumerate(tss):
            out = extract_one(src, out_dir, ts, i)
            extracted.append({"index": i, "timestamp_s": ts, "path": str(out.relative_to(ROOT))})
        extraction_records[source_id] = {
            "source_video": fname,
            "duration_s": duration,
            "target_frames": target,
            "spacing_s": [round(tss[1] - tss[0], 3)] if len(tss) > 1 else None,
            "frames": extracted,
        }
        print(f"OK   {source_id}: {len(extracted)} frames (dur={duration:.1f}s) -> {out_dir}")

    meta_dir = ROOT / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_json = meta_dir / "pilot_extraction_params.json"
    out_json.write_text(json.dumps(extraction_records, indent=2))
    print(f"WROTE {out_json}")


if __name__ == "__main__":
    main()