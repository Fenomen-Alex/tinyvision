"""Offline vertical slice: acquire -> autolabel -> streams -> split -> load.

Uses stub yt-dlp/ffprobe/ffmpeg on PATH and a stub chat endpoint, so the whole
real-domain pipeline is exercised with no network and no large downloads.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAINING = REPO / "training"


def _stub(bin_dir: Path, name: str, body: str) -> None:
    p = bin_dir / name
    p.write_text(f"#!{sys.executable}\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        body = json.dumps({
            "choices": [{"message": {"content": '{"label": "uncertain"}'}}]
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_realdomain_vertical_slice(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir, "yt-dlp", (
        "import sys, pathlib\n"
        "a = sys.argv\n"
        "d = pathlib.Path(a[a.index('-o') + 1]).parent\n"
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

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["PYTHONPATH"] = str(TRAINING) + os.pathsep + env.get("PYTHONPATH", "")

    def _run(*args: str, endpoint: str | None = None):
        a = list(args)
        if endpoint:
            a += ["--endpoint", endpoint]
        return subprocess.run([sys.executable, "-m", "tv.cli", *a],
                              cwd=REPO, env=env, capture_output=True, text=True)

    sources = tmp_path / "sources.yaml"
    lines = ["sources:"]
    ids = []
    for i in range(8):
        sid = f"vid{i:02d}"
        ids.append(sid)
        label = "failing" if i < 4 else "normal"
        lines.append(f"  - id: {sid}\n    url: {sid}\n    weak_label: {label}\n    max_seconds: 3")
    sources.write_text("\n".join(lines) + "\n")

    cache = tmp_path / "cache"

    r = _run("acquire", "--sources", str(sources), "--cache-dir", str(cache), "--fps", "1.0")
    assert r.returncode == 0, r.stdout + r.stderr
    manifest = cache / "manifest.json"
    assert manifest.is_file()

    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ep = f"http://127.0.0.1:{srv.server_address[1]}"
        r = _run("autolabel", "--manifest", str(manifest), "--model", "stub",
                 endpoint=ep)
        assert r.returncode == 0, r.stdout + r.stderr

        split_dir = cache / "rc2-realdomain" / "splits"
        r = _run("streams", "--manifest", str(manifest),
                 "--proposals", str(cache / "autolabel.json"),
                 "--split-dir", str(split_dir),
                 "--data-dir", str(cache / "rc2-realdomain"))
        assert r.returncode == 0, r.stdout + r.stderr
    finally:
        srv.shutdown()

    assert (split_dir / "splits.json").is_file()
    total = 0
    for name in ("train", "val", "test"):
        p = split_dir / f"{name}.csv"
        assert p.is_file()
        total += sum(1 for _ in p.open()) - 1
    assert total == 16

    # The split CSVs must load through the training loader with failing=0.
    sys.path.insert(0, str(TRAINING))
    from tv.data.loaders import CSVImageDataset

    train = CSVImageDataset(split_dir / "train.csv",
                            data_dir=cache / "rc2-realdomain")
    assert len(train) > 0
    if "failing" in train.class_to_idx and "normal" in train.class_to_idx:
        assert train.class_to_idx["failing"] == 0
        assert train.class_to_idx["normal"] == 1
