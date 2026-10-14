"""Acquire real video footage and extract frames (yt-dlp + ffmpeg).

All external commands go through an injectable ``runner`` so the module can be
tested without network or ffmpeg.
"""
from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

from tv.data.manifest import Source


class AcquireError(RuntimeError):
    """Raised when an external acquisition command fails."""


def _run(runner, cmd, **kw):
    try:
        return runner(cmd, capture_output=True, check=True, **kw)
    except subprocess.CalledProcessError as e:
        err = getattr(e, "stderr", b"")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        raise AcquireError(f"command failed: {' '.join(cmd)}\n{err}") from e


def _as_url(source: str) -> str:
    if source.startswith("http"):
        return source
    return f"https://www.youtube.com/watch?v={source}"


def _out(result) -> str:
    out = getattr(result, "stdout", b"")
    return out.decode() if isinstance(out, bytes) else out


def probe(video_path: Path, runner=subprocess.run) -> dict:
    """Return ``{"duration", "width", "height"}`` via ffprobe."""
    res = _run(runner, ["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,duration",
                        "-show_entries", "format=duration", "-of", "json",
                        str(video_path)])
    data = json.loads(_out(res))
    stream = (data.get("streams") or [{}])[0]
    duration = stream.get("duration") or (data.get("format") or {}).get("duration")
    return {"duration": float(duration or 0.0),
            "width": int(stream.get("width") or 0),
            "height": int(stream.get("height") or 0)}


def fetch_video(source: str, out_dir: Path, max_seconds: int | None = None,
                format_selector: str = "160", runner=subprocess.run) -> Path:
    """Download ``source`` (URL or bare id) as an mp4; idempotent."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.mp4"))
    if existing:
        return existing[0]
    cmd = ["yt-dlp", "--no-warnings", "-f", format_selector,
           "-o", str(out_dir / "%(id)s.%(ext)s")]
    if max_seconds:
        cmd += ["--download-sections", f"*0-{max_seconds}",
                "--force-keyframes-at-cuts"]
    cmd.append(_as_url(source))
    _run(runner, cmd)
    found = sorted(out_dir.glob("*.mp4"))
    if not found:
        raise AcquireError(f"yt-dlp produced no mp4 for {source}")
    return found[0]


def extract_frames(video_path: Path, out_dir: Path, fps: float = 1.0,
                   duration: float | None = None,
                   runner=subprocess.run) -> list[Path]:
    """Extract one frame per grid point; idempotent (reuses existing frames)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("frame_*.jpg"))
    if existing:
        return existing
    if duration is None:
        duration = probe(video_path, runner=runner)["duration"]
    step = 1.0 / fps if fps else 1.0
    # Never request a timestamp at or past the clip end (ffmpeg errors there).
    last = max(0.0, duration - min(step, 0.5)) if duration > 0 else 0.0
    grid: list[float] = []
    t = 0.0
    while t < duration or not grid:
        grid.append(min(t, last))
        t += step
        if t >= duration:
            break
    seen: set[float] = set()
    paths: list[Path] = []
    for ts in grid:
        if ts in seen:
            continue
        seen.add(ts)
        out = out_dir / f"frame_{len(paths):04d}.jpg"
        try:
            _run(runner, ["ffmpeg", "-y", "-ss", f"{ts}", "-i", str(video_path),
                          "-frames:v", "1", "-q:v", "2", str(out)])
        except AcquireError:
            continue
        paths.append(out)
    return paths


def acquire_source(source: str, cache_root: Path, max_seconds: int | None = None,
                   fps: float = 1.0, meta: dict | None = None,
                   runner=subprocess.run) -> Source:
    """Download + extract one source and return its manifest ``Source``."""
    sid = source.rsplit("=", 1)[-1].rstrip("/").split("/")[-1]
    root = Path(cache_root) / sid
    video = fetch_video(source, root, max_seconds=max_seconds, runner=runner)
    info = probe(video, runner=runner)
    frames = extract_frames(video, root / "frames", fps=fps,
                            duration=info["duration"], runner=runner)
    m = meta or {}
    return Source(id=sid, url=_as_url(source), title=m.get("title", ""),
                  uploader=m.get("uploader", ""), duration=info["duration"],
                  width=info["width"], height=info["height"],
                  retrieved_at=date.today().isoformat(),
                  license_note="research_only",
                  weak_label=m.get("weak_label", "uncertain"),
                  frames_dir=str(root / "frames"), frame_count=len(frames))
