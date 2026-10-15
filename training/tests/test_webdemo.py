"""Tests for the local demo server (tv/webdemo.py) and synthetic fixture."""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training"))

from tv.poc import PocConfig  # noqa: E402
from tv.webdemo import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEMO_DISCLAIMER,
    MAX_UPLOAD_BYTES,
    DemoError,
    Job,
    build_settings,
    create_server,
    default_data_dir,
    list_models,
    load_ui,
)

BASELINE_CKPT = (
    REPO / "experiments" / "shufflenet_v2_x1_0_printing-errors_cdc9e2cad4"
    / "artifacts" / "best.pt"
)


def _cv2():
    try:
        import cv2
        return cv2
    except Exception:  # pragma: no cover
        return None


def _make_video(path: Path, n_frames: int = 10, fps: float = 5.0, size: int = 48) -> Path:
    cv2 = _cv2()
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    assert writer.isOpened()
    for i in range(n_frames):
        writer.write(np.full((size, size, 3), 20 + i * 3, dtype=np.uint8))
    writer.release()
    return path


def _e2e_available() -> bool:
    return _cv2() is not None and BASELINE_CKPT.exists()


E2E = pytest.mark.skipif(not _e2e_available(),
                         reason="baseline checkpoint or OpenCV not available")


class _Client:
    def __init__(self, base: str):
        self.base = base

    def get(self, path: str):
        try:
            with urllib.request.urlopen(self.base + path) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def post_json(self, path: str, obj) -> tuple[int, dict]:
        req = urllib.request.Request(self.base + path, data=json.dumps(obj).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def post_bytes(self, path: str, data: bytes, headers: dict) -> tuple[int, dict]:
        req = urllib.request.Request(self.base + path, data=data, headers=headers,
                                     method="POST")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")


@pytest.fixture
def server(tmp_path):
    httpd = create_server(host="127.0.0.1", port=0, data_dir=tmp_path)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd, _Client(f"http://127.0.0.1:{port}")
    finally:
        httpd.shutdown()
        httpd.server_close()


def _wait(client: _Client, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        _, st = client.get(f"/api/jobs/{job_id}")
        if st.get("status") in ("done", "error"):
            return st
        time.sleep(0.2)
    raise AssertionError("job did not finish in time")


# ---------------------------------------------------------------------------
# Static / discovery / defaults
# ---------------------------------------------------------------------------

def test_default_data_dir_is_outside_repo():
    assert not str(DEFAULT_DATA_DIR).startswith(str(REPO))


def test_default_data_dir_honours_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TINYVISION_DEMO_DATA", str(tmp_path / "d"))
    assert default_data_dir() == tmp_path / "d"


def test_health_and_models(server):
    httpd, client = server
    status, health = client.get("/api/health")
    assert status == 200 and health["status"] == "ok"
    assert health["experimental"] is True
    assert "not validated" in health["disclaimer"].lower()
    # health reports where data lives and the retention policy (for the footer)
    assert health["data_dir"] == str(headless_data_dir(httpd))
    assert health["retention"]

    status, models = client.get("/api/models")
    assert status == 200
    assert isinstance(models["models"], list)
    if BASELINE_CKPT.exists():
        assert models["default"] and models["default"].endswith("best.pt")


def headless_data_dir(httpd) -> Path:
    return Path(httpd.job_manager.data_dir)


def test_load_ui_reads_packaged_asset():
    html = load_ui()
    assert "TinyVision" in html and "EXPERIMENTAL" in html


def test_index_serves_html(server):
    httpd, client = server
    with urllib.request.urlopen(client.base + "/") as r:
        body = r.read().decode("utf-8")
    assert "TinyVision" in body and "EXPERIMENTAL" in body


def test_list_models_shape():
    models, default = list_models()
    assert isinstance(models, list) and isinstance(default, (str, type(None)))
    for m in models:
        assert set(m) == {"id", "label", "path"}


def test_list_models_scans_env_dir(tmp_path, monkeypatch):
    run = tmp_path / "my_run" / "artifacts"
    run.mkdir(parents=True)
    ckpt = run / "best.pt"
    ckpt.write_bytes(b"weights")
    monkeypatch.setenv("TINYVISION_MODELS_DIR", str(tmp_path))
    models, _ = list_models()
    paths = {m["path"] for m in models}
    assert str(ckpt.resolve()) in paths
    # label comes from the run dir, not the file stem
    assert any(m["label"] == "my_run" for m in models)


def test_create_server_port_in_use_is_actionable(tmp_path):
    httpd = create_server(port=0, data_dir=tmp_path)
    port = httpd.server_address[1]
    try:
        with pytest.raises(DemoError) as ei:
            create_server(port=port, data_dir=tmp_path)
        assert "already in use" in str(ei.value)
    finally:
        httpd.server_close()


# ---------------------------------------------------------------------------
# Settings coercion
# ---------------------------------------------------------------------------

def test_build_settings_coerces_and_drops_blanks():
    s = build_settings({"sample_fps": "5", "threshold": 0.4, "max_frames": "",
                        "warmup": None, "confirm": "2"})
    assert s == {"sample_fps": 5.0, "threshold": 0.4, "confirm": 2}


def test_build_settings_rejects_non_numeric():
    with pytest.raises(DemoError):
        build_settings({"threshold": "not-a-number"})


# ---------------------------------------------------------------------------
# Error handling (no model run required)
# ---------------------------------------------------------------------------

def test_analyze_missing_video_path_400(server):
    httpd, client = server
    status, body = client.post_json("/api/analyze",
                                    {"checkpoint": str(BASELINE_CKPT)})
    assert status == 400 and "video_path" in body["error"]


def test_analyze_missing_video_file_400(server, tmp_path):
    httpd, client = server
    status, body = client.post_json("/api/analyze", {
        "video_path": str(tmp_path / "nope.mp4"), "checkpoint": str(BASELINE_CKPT)})
    assert status == 400 and "not found" in body["error"]


def test_analyze_missing_checkpoint_400(server, tmp_path):
    httpd, client = server
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    status, body = client.post_json("/api/analyze", {
        "video_path": str(video), "checkpoint": str(tmp_path / "nope.pt")})
    assert status == 400 and "checkpoint not found" in body["error"]


def test_analyze_invalid_setting_400(server, tmp_path):
    httpd, client = server
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    status, body = client.post_json("/api/analyze", {
        "video_path": str(video), "checkpoint": str(BASELINE_CKPT),
        "settings": {"threshold": 5.0}})
    assert status == 400 and "threshold" in body["error"]


def test_analyze_unsupported_content_type_400(server):
    httpd, client = server
    req = urllib.request.Request(client.base + "/api/analyze", data=b"x",
                                 headers={"Content-Type": "text/plain"}, method="POST")
    try:
        urllib.request.urlopen(req)
        raise AssertionError("expected 400")
    except urllib.error.HTTPError as e:
        assert e.code == 400


def test_upload_too_large_400(server, monkeypatch):
    import tv.webdemo as webdemo
    httpd, client = server
    monkeypatch.setattr(webdemo, "MAX_UPLOAD_BYTES", 4)
    status, body = client.post_bytes("/api/analyze", b"0123456789", {
        "Content-Type": "application/octet-stream",
        "X-Filename": "big.mp4",
        "X-Checkpoint": str(BASELINE_CKPT),
    })
    assert status == 400 and "too large" in body["error"]
    # nothing partial is left behind
    assert not list((headless_data_dir(httpd) / "uploads").glob("*"))


def test_upload_empty_400(server):
    httpd, client = server
    status, body = client.post_bytes("/api/analyze", b"", {
        "Content-Type": "application/octet-stream",
        "X-Filename": "empty.mp4",
        "X-Checkpoint": str(BASELINE_CKPT),
    })
    assert status == 400 and "empty" in body["error"]


def test_unknown_job_404(server):
    httpd, client = server
    status, body = client.get("/api/jobs/deadbeef")
    assert status == 404 and "unknown job" in body["error"]


def test_report_before_done_409(server):
    httpd, client = server
    httpd.job_manager._jobs["deadbeef"] = Job(id="deadbeef", source_label="x",
                                              checkpoint="y", settings={})
    status, body = client.get("/api/jobs/deadbeef/report")
    assert status == 409 and "no report" in body["error"]


# ---------------------------------------------------------------------------
# End-to-end jobs (require OpenCV + a checkpoint)
# ---------------------------------------------------------------------------

@E2E
def test_json_path_job_end_to_end(server, tmp_path):
    httpd, client = server
    video = _make_video(tmp_path / "clip.mp4", n_frames=10)
    status, body = client.post_json("/api/analyze", {
        "video_path": str(video), "checkpoint": str(BASELINE_CKPT),
        "settings": {"max_frames": 6}})
    assert status == 202, body
    job_id = body["job"]["id"]
    assert "progress" in body["job"]

    st = _wait(client, job_id)
    assert st["status"] == "done", st.get("error")
    assert st["has_report"] is True
    assert st["progress"]["processed"] == 6
    assert st["progress"]["fraction"] == 1.0
    assert st["progress"]["samples"]

    rstatus, report = client.get(f"/api/jobs/{job_id}/report")
    assert rstatus == 200
    assert report["schema"] == "poc-inference"
    assert report["summary"]["frames"] == 6
    assert report["disclaimer"] == DEMO_DISCLAIMER
    # the exported report is also persisted next to the job
    assert (tmp_path / "reports" / f"{job_id}.poc.json").exists()


@E2E
def test_upload_job_end_to_end(server, tmp_path):
    httpd, client = server
    video = _make_video(tmp_path / "clip.mp4", n_frames=8)
    data = video.read_bytes()
    status, body = client.post_bytes("/api/analyze", data, {
        "Content-Type": "application/octet-stream",
        "X-Filename": "my print.mp4",
        "X-Checkpoint": str(BASELINE_CKPT),
        "X-Settings": json.dumps({"max_frames": 5}),
    })
    assert status == 202, body
    job_id = body["job"]["id"]
    st = _wait(client, job_id)
    assert st["status"] == "done", st.get("error")
    # upload landed in the data dir with a sanitised name
    uploads = list((tmp_path / "uploads").glob("*.mp4"))
    assert len(uploads) == 1
    assert " " not in uploads[0].name


@E2E
def test_job_error_is_readable(server, tmp_path):
    """A file that is not a decodable video fails as a readable job error."""
    httpd, client = server
    bogus = tmp_path / "bogus.mp4"
    bogus.write_bytes(b"this is not a video")
    status, body = client.post_json("/api/analyze", {
        "video_path": str(bogus), "checkpoint": str(BASELINE_CKPT)})
    assert status == 202, body
    st = _wait(client, body["job"]["id"])
    assert st["status"] == "error"
    assert st["error"]


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------

@E2E
def test_synthetic_fixture_is_deterministic(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "make_synthetic_demo_video", REPO / "scripts" / "make_synthetic_demo_video.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    a = mod.generate(tmp_path / "a.mp4", frames=20, size=96)
    b = mod.generate(tmp_path / "b.mp4", frames=20, size=96)
    assert a["synthetic"] is True and "not real" in a["disclaimer"].lower()
    assert a["anomaly_start_frame"] == 10
    assert (tmp_path / "a.fixture.json").exists()

    cv2 = _cv2()
    ca, cb = cv2.VideoCapture(str(tmp_path / "a.mp4")), cv2.VideoCapture(str(tmp_path / "b.mp4"))
    same = True
    while True:
        ra, fa = ca.read()
        rb, fb = cb.read()
        if not ra or not rb:
            same = same and (ra == rb)
            break
        if not (fa == fb).all():
            same = False
            break
    ca.release(); cb.release()
    assert same, "synthetic fixture is not deterministic"
