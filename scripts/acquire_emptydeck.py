#!/usr/bin/env python
"""Acquire a targeted subset of the EmptyDeck bed-adhesion failure-onset dataset.

Dataset: "Bed-Adhesion Failure-Onset Dataset for FDM 3D Printing"
(EmptyDeck/3DPrint-Failure-Video-Dataset).  The annotations, ground-truth masks
and metadata are released under **CC BY-NC 4.0** (noncommercial); the externally
hosted timelapse videos are "provided for research use under the same terms".

Consequence: this data is RESEARCH-ONLY.  It is never written into a
release-bound or training split.  ``Source.noncommercial`` is set True and
``Source.redistributable`` False; source media is not redistributed.

Each sequence is one independent *failing* print (bed-adhesion onset is the
change point ``onset_frame``).  Frames ``[0, onset_frame)`` are the normal
baseline of that same print and are NOT an independent fully-normal stream.

Download is deliberately limited: low-resolution (360p) timelapses, shortest
sequences first, under a byte budget.  Full videos are tens of GB in total and
are not needed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

LICENSE = "CC BY-NC 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"
ATTRIBUTION = ("EmptyDeck, 'Bed-Adhesion Failure-Onset Dataset for FDM 3D "
               "Printing' (https://github.com/EmptyDeck/3DPrint-Failure-Video-Dataset), "
               "CC BY-NC 4.0")


def sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sequences(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def select(rows: list[dict], envs: set[str], min_per_env: int,
           max_bytes: int, order: str = "shortest") -> list[dict]:
    rows = [r for r in rows if not envs or r["env"] in envs]
    rows.sort(key=lambda r: (int(r["n_frames"]) if order == "shortest"
                             else -int(r["n_frames"])))
    # 360p mp4 is ~45-160 kbit/s; use a conservative 0.9 MB per 300 s estimate
    # to stop before the budget is exceeded.
    est = lambda r: int(r["n_frames"]) * 0.9 / 300 * 1e6  # noqa: E731
    chosen, per_env = [], {}
    for r in rows:
        need = per_env.get(r["env"], 0) < min_per_env
        total = sum(est(c) for c in chosen) + est(r)
        if need or total <= max_bytes:
            chosen.append(r)
            per_env[r["env"]] = per_env.get(r["env"], 0) + 1
    return chosen


def download(url: str, out_dir: Path, max_filesize: int = 0) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    found = sorted(out_dir.glob("*.mp4"))
    if found:
        return found[0]
    cmd = ["yt-dlp", "--no-warnings", "-f", "134/160",
           "-o", str(out_dir / "%(id)s.%(ext)s")]
    if max_filesize:
        cmd += ["--max-filesize", str(max_filesize)]
    cmd.append(url)
    subprocess.run(cmd, check=True, capture_output=True)
    found = sorted(out_dir.glob("*.mp4"))
    if not found:
        raise RuntimeError(f"yt-dlp produced no mp4 for {url}")
    return found[0]


def extract_frames(video: Path, out_dir: Path, fps: int = 1) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("frame_*.jpg")):
        return len(list(out_dir.glob("frame_*.jpg")))
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-vf", f"fps={fps}",
         "-q:v", "2", str(out_dir / "frame_%04d.jpg")],
        check=True, capture_output=True)
    return len(list(out_dir.glob("frame_*.jpg")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True,
                    help="Cloned EmptyDeck repo (has metadata/sequences.csv)")
    ap.add_argument("--out-root", required=True, help="Output root (outside repo)")
    ap.add_argument("--envs", default="", help="Comma list, e.g. P1,P2")
    ap.add_argument("--min-per-env", type=int, default=3)
    ap.add_argument("--max-bytes", type=float, default=1.5e9)
    ap.add_argument("--order", default="shortest", choices=["shortest", "longest"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="",
                    help="Explicit comma list of sequence names (bypasses budget)")
    ap.add_argument("--max-filesize", type=float, default=0,
                    help="yt-dlp per-video cap in bytes (0 = none)")
    args = ap.parse_args()

    repo = Path(args.repo)
    rows = load_sequences(repo / "metadata" / "sequences.csv")
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        chosen = [r for r in rows if r["sequence"] in want]
    else:
        envs = {e.strip() for e in args.envs.split(",") if e.strip()}
        chosen = select(rows, envs, args.min_per_env, args.max_bytes, args.order)
        if args.limit:
            chosen = chosen[: args.limit]

    out_root = Path(args.out_root)
    videodir = out_root / "videos"
    manifest_path = out_root / "manifest.json"
    manifest = {"version": 1, "created_at": date.today().isoformat(),
                "sources": []}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    have = {s["id"]: s for s in manifest["sources"]}

    def save() -> None:
        manifest["sources"] = list(have.values())
        out_root.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"selected {len(chosen)} sequences "
          f"(envs={sorted({c['env'] for c in chosen})})")
    for r in chosen:
        seq, env, vid = r["sequence"], r["env"], r["video_url"]
        sid = f"{env}__{seq}"
        try:
            video = download(vid, videodir / seq, max_filesize=int(args.max_filesize))
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {sid}: {str(e)[:120]}")
            continue
        n = extract_frames(video, videodir / seq / "frames", fps=int(r["fps"]))
        info = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "json", str(video)],
            capture_output=True, text=True, check=True)
        st = json.loads(info.stdout)["streams"][0]
        have[sid] = {
            "id": sid, "url": vid, "title": seq, "uploader": "EmptyDeck",
            "duration": float(r["n_frames"]), "width": int(st["width"]),
            "height": int(st["height"]), "retrieved_at": date.today().isoformat(),
            "license_note": "research_only (CC BY-NC 4.0)",
            "weak_label": "failing", "frames_dir": str(video.parent / "frames"),
            "frame_count": n, "failure_type": "bed_adhesion",
            "source_dataset": "emptydeck-bed-adhesion-onset",
            "license": LICENSE, "license_url": LICENSE_URL,
            "attribution": ATTRIBUTION, "capture_environment": env,
            "session_id": seq, "video_sha256": sha256(video),
            "noncommercial": True, "redistributable": False, "split": "",
            "notes": f"onset_frame={r['onset_frame']}; "
                     f"n_normal_baseline={r['n_normal_baseline']} "
                     f"(same print, NOT independent normal)",
        }
        save()
        print(f"  ok {sid}: {n} frames, onset={r['onset_frame']}, "
              f"sha256={have[sid]['video_sha256'][:12]}", flush=True)

    save()
    print(f"manifest: {manifest_path} ({len(have)} sources)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
