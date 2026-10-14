"""Offline end-to-end test of `tv autolabel` against a stub chat endpoint."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAINING = REPO / "training"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        body = json.dumps({
            "choices": [{"message": {"content": '{"label": "failing", "confidence": 0.8}'}}]
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_autolabel_cli(tmp_path):
    from PIL import Image

    frames = tmp_path / "abc" / "frames"
    frames.mkdir(parents=True)
    for i in range(2):
        Image.new("RGB", (32, 32), (i, 0, 0)).save(frames / f"frame_{i:04d}.jpg")
    manifest = {
        "version": 1, "created_at": "2026-10-14",
        "sources": [{
            "id": "abc", "url": "https://y/abc", "title": "t", "uploader": "u",
            "duration": 2.0, "width": 32, "height": 32,
            "retrieved_at": "2026-10-14", "license_note": "research_only",
            "weak_label": "uncertain", "frames_dir": str(frames),
            "frame_count": 2, "failure_type": None,
        }],
    }
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest))

    srv = _serve()
    try:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(TRAINING) + os.pathsep + env.get("PYTHONPATH", "")
        r = subprocess.run(
            [sys.executable, "-m", "tv.cli", "autolabel", "--manifest", str(mp),
             "--endpoint", f"http://127.0.0.1:{srv.server_address[1]}",
             "--model", "stub", "--out", str(tmp_path / "proposals.json")],
            cwd=REPO, env=env, capture_output=True, text=True)
    finally:
        srv.shutdown()
    assert r.returncode == 0, r.stdout + r.stderr
    proposals = json.loads((tmp_path / "proposals.json").read_text())
    assert proposals["sources"][0]["video_label"] == "failing"
    assert proposals["sources"][0]["counts"]["failing"] == 2
