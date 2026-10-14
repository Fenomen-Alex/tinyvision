"""Offline test of `tv streams`."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAINING = REPO / "training"


def _run(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(TRAINING) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-m", "tv.cli", *args],
                          cwd=REPO, env=env, capture_output=True, text=True)


def test_streams_cli(tmp_path):
    frames = tmp_path / "a" / "frames"
    frames.mkdir(parents=True)
    for i in range(2):
        (frames / f"frame_{i:04d}.jpg").write_bytes(b"x")
    manifest = {
        "version": 1, "created_at": "2026-10-14",
        "sources": [{
            "id": "a", "url": "https://y/a", "title": "t", "uploader": "u",
            "duration": 2.0, "width": 32, "height": 32,
            "retrieved_at": "2026-10-14", "license_note": "research_only",
            "weak_label": "failing", "frames_dir": str(frames),
            "frame_count": 2, "failure_type": None,
        }],
    }
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest))

    r = _run("streams", "--manifest", str(mp))
    assert r.returncode == 0, r.stdout + r.stderr
    rows = list(csv.DictReader((tmp_path / "streams.csv").open()))
    assert len(rows) == 2
    assert rows[0]["label"] == "failing"
    assert rows[0]["stream"] == "a"
    assert "1 streams" in r.stdout


def test_streams_cli_no_labelled_exit_1(tmp_path):
    manifest = {"version": 1, "created_at": "2026-10-14", "sources": []}
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest))
    r = _run("streams", "--manifest", str(mp))
    assert r.returncode == 1, r.stdout + r.stderr
