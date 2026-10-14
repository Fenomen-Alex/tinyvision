"""Offline end-to-end test of `tv acquire` using stub external tools."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAINING = REPO / "training"


def _stub(bin_dir: Path, name: str, body: str) -> None:
    p = bin_dir / name
    p.write_text(f"#!{sys.executable}\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run(*args: str, env: dict) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e.update(env)
    return subprocess.run(
        [sys.executable, "-m", "tv.cli", *args],
        cwd=REPO, env=e, capture_output=True, text=True,
    )


def test_acquire_offline_pipeline(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir, "yt-dlp", (
        "import sys, pathlib\n"
        "a = sys.argv\n"
        "tmpl = a[a.index('-o') + 1]\n"
        "d = pathlib.Path(tmpl).parent\n"
        "d.mkdir(parents=True, exist_ok=True)\n"
        "(d / 'clip.mp4').write_bytes(b'v')\n"
    ))
    _stub(bin_dir, "ffprobe", (
        "import json\n"
        "print(json.dumps({'streams':[{'width':640,'height':480,'duration':'2.0'}],"
        "'format':{'duration':'2.0'}}))\n"
    ))
    _stub(bin_dir, "ffmpeg", (
        "import sys, pathlib\n"
        "pathlib.Path(sys.argv[-1]).write_bytes(b'jpg')\n"
    ))

    cache = tmp_path / "cache"
    sources = tmp_path / "sources.yaml"
    sources.write_text(
        "sources:\n"
        "  - id: abc123\n    url: abc123\n    weak_label: normal\n    max_seconds: 4\n"
    )
    env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
           "PYTHONPATH": str(TRAINING) + os.pathsep + os.environ.get("PYTHONPATH", "")}

    r = _run("acquire", "--sources", str(sources), "--cache-dir", str(cache),
             "--fps", "1.0", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    manifest = json.loads((cache / "manifest.json").read_text())
    assert manifest["sources"][0]["id"] == "abc123"
    assert manifest["sources"][0]["frame_count"] == 2
    assert manifest["sources"][0]["weak_label"] == "normal"

    r2 = _run("acquire", "--sources", str(sources), "--cache-dir", str(cache), env=env)
    assert "skip (already acquired): abc123" in r2.stdout


def test_acquire_help():
    r = _run("acquire", "--help", env={})
    assert r.returncode == 0
    assert "cache-dir" in r.stdout
