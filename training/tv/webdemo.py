"""Local web demo for the TinyVision inference POC.

A single-process, dependency-free (Python standard library only) local web UI
around the existing inference pipeline (``tv.poc.run_poc``). It exists so a
person can point the frozen classifier at a local 3D-print recording, watch the
per-frame ``P(failing)`` stream and the temporal failure decision, and export the
JSON report -- without a connected printer, a camera, a cloud account, a hosted
inference service, or a paid API.

Design constraints (deliberate):

* **One inference path.** The demo calls ``tv.poc.run_poc`` directly; it does not
  re-implement preprocessing, sampling, temporal logic, or scoring.
* **Standard library only.** No web framework is added to the training
  environment. This is a local demonstration, not a production service.
* **Localhost by default.** Blocking interfaces should not be exposed by
  accident; the bind address defaults to ``127.0.0.1``.
* **Explicitly experimental.** Every job carries a disclaimer and the report the
  demo exports is labelled as a software-test output, not validated accuracy.

Nothing here is a benchmark or evidence of real-world detection accuracy.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    """Where uploads and reports live by default.

    Overridable with ``TINYVISION_DEMO_DATA``. Otherwise a per-user data
    directory outside the repository (and outside Git), chosen per platform so
    the demo works the same for someone who never saw this repo::

        macOS    ~/Library/Application Support/TinyVision/demo
        Linux    $XDG_DATA_HOME/tinyvision/demo  (or ~/.local/share/...)
        Windows  %LOCALAPPDATA%\\TinyVision\\demo
    """
    env = os.environ.get("TINYVISION_DEMO_DATA")
    if env:
        return Path(env).expanduser()
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "TinyVision" / "demo"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(home)
        return Path(base) / "TinyVision" / "demo"
    base = os.environ.get("XDG_DATA_HOME") or str(home / ".local" / "share")
    return Path(base) / "tinyvision" / "demo"


# Default data root for uploads and exported reports. Kept outside Git (per the
# repository policy that generated artifacts never pollute the working tree).
DEFAULT_DATA_DIR = default_data_dir()

DEMO_DISCLAIMER = (
    "EXPERIMENTAL / NOT VALIDATED FOR PRODUCTION. This output comes from a "
    "research prototype classifier and is intended for inspection and software "
    "testing only. It is not a validated failure-detection product, and it has "
    "not been shown to generalise across printers, cameras, filaments or "
    "lighting. Do not use it to make production or safety decisions."
)

_REPORT_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}
# Refuse browser uploads larger than this; point the user at the "Local path"
# option (which streams from disk) instead. 2 GiB is generous for a demo.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024


class DemoError(Exception):
    """A user-facing demo error (turned into an HTTP 4xx with a message)."""


def _safe_stem(name: str) -> str:
    stem = _REPORT_NAME.sub("_", name).strip("._-") or "upload"
    return stem[:120]


# ---------------------------------------------------------------------------
# Job bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class Job:
    id: str
    source_label: str
    checkpoint: str
    settings: dict[str, Any]
    created: float = field(default_factory=time.time)
    status: str = "queued"  # queued | running | done | error
    error: Optional[str] = None
    progress: dict[str, Any] = field(default_factory=dict)
    report: Optional[dict[str, Any]] = None
    _samples: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def public(self, include_samples: bool = True) -> dict[str, Any]:
        with self._lock:
            progress = dict(self.progress)
            if include_samples:
                progress["samples"] = list(self._samples)
            else:
                progress["sample_count"] = len(self._samples)
            return {
                "id": self.id,
                "status": self.status,
                "source": self.source_label,
                "checkpoint": self.checkpoint,
                "settings": self.settings,
                "created": self.created,
                "error": self.error,
                "progress": progress,
                "disclaimer": DEMO_DISCLAIMER,
                "has_report": self.report is not None,
            }

    def record_frame(self, processed: int, row: dict, total: Optional[int]) -> None:
        with self._lock:
            self.progress = {
                "processed": processed,
                "total": total,
                "fraction": (processed / total) if total else None,
                "last_score": row.get("raw"),
                "last_state": row.get("state"),
                "last_timestamp_s": row.get("timestamp_s"),
            }
            self._samples.append({
                "frame": row.get("frame"),
                "frame_idx": row.get("frame_idx"),
                "t": row.get("timestamp_s"),
                "score": row.get("raw"),
                "signal": row.get("signal"),
                "state": row.get("state"),
                "event": row.get("event"),
            })
            # Bound memory for very long videos; the full per-frame log is still
            # in the exported report.
            if len(self._samples) > 5000:
                del self._samples[: len(self._samples) - 5000]


class JobManager:
    """Runs POC jobs on background threads and tracks their progress."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.uploads_dir = self.data_dir / "uploads"
        self.reports_dir = self.data_dir / "reports"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def start(self, source: str, checkpoint: str, settings: dict[str, Any],
              source_label: str) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            source_label=source_label,
            checkpoint=str(checkpoint),
            settings=dict(settings),
        )
        with self._lock:
            self._jobs[job.id] = job
        t = threading.Thread(target=self._run, args=(job, source, checkpoint, settings),
                             name=f"tv-demo-{job.id}", daemon=True)
        t.start()
        return job

    def _run(self, job: Job, source: str, checkpoint: str,
             settings: dict[str, Any]) -> None:
        from .poc import PocConfig, PocError, run_poc

        job.status = "running"
        try:
            config = PocConfig(source=source, checkpoint=checkpoint, **settings)
            report = run_poc(config, on_frame=job.record_frame)
        except PocError as exc:
            job.status = "error"
            job.error = str(exc)
            return
        except Exception as exc:  # noqa: BLE001 - surface as a readable job error
            job.status = "error"
            job.error = f"unexpected error: {exc}"
            return

        report = dict(report)
        report["disclaimer"] = DEMO_DISCLAIMER
        job.report = report
        job.status = "done"
        with job._lock:
            job.progress = {**job.progress, "fraction": 1.0}
        try:
            out = self.reports_dir / f"{job.id}.poc.json"
            out.write_text(json.dumps(report, indent=2))
        except OSError:
            pass

    def new_upload_path(self, filename: str) -> Path:
        stem = _safe_stem(Path(filename).stem)
        ext = Path(filename).suffix.lower()
        if ext not in _ALLOWED_VIDEO_EXT:
            ext = ".mp4"
        return self.uploads_dir / f"{stem}-{uuid.uuid4().hex[:8]}{ext}"

    def save_upload(self, filename: str, data: bytes) -> Path:
        dest = self.new_upload_path(filename)
        dest.write_bytes(data)
        return dest


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def _iter_checkpoints(root: Path):
    """Yield candidate checkpoint files under a directory (bounded patterns)."""
    if not root.is_dir():
        return
    for pattern in ("*/artifacts/best.pt", "*/best.pt", "*.pt"):
        yield from sorted(root.glob(pattern))


def list_models(repo_root: Path = REPO_ROOT) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Discover local TinyVision checkpoints (``*.pt``) for the model picker.

    Searches the repository ``experiments/`` tree plus any directories listed in
    the ``TINYVISION_MODELS_DIR`` environment variable (``os.pathsep``-separated),
    so a user whose checkpoints live elsewhere is not tied to the repo layout.
    Returns ``(models, default_path)``.
    """
    seen: set[str] = set()
    models: list[dict[str, Any]] = []

    roots: list[Path] = [Path(repo_root) / "experiments"]
    env = os.environ.get("TINYVISION_MODELS_DIR")
    if env:
        roots.extend(Path(p).expanduser() for p in env.split(os.pathsep) if p.strip())

    for root in roots:
        for ckpt in _iter_checkpoints(root):
            resolved = str(ckpt.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            # Label from the experiment dir when it looks like <run>/artifacts/best.pt
            label = ckpt.parent.parent.name if ckpt.parent.name == "artifacts" else ckpt.stem
            models.append({"id": label, "label": label, "path": resolved})

    baseline = (
        Path(repo_root) / "experiments"
        / "shufflenet_v2_x1_0_printing-errors_cdc9e2cad4" / "artifacts" / "best.pt"
    )
    if baseline.exists():
        default: Optional[str] = str(baseline.resolve())
    elif models:
        default = models[0]["path"]
    else:
        default = None
    return models, default


# ---------------------------------------------------------------------------
# Request settings
# ---------------------------------------------------------------------------

_NUMERIC_SETTINGS: dict[str, type] = {
    "sample_fps": float,
    "max_frames": int,
    "threshold": float,
    "image_size": int,
    "warmup": int,
    "confirm": int,
    "alpha": float,
}


def build_settings(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate/coerce the request settings into PocConfig keyword arguments."""
    settings: dict[str, Any] = {}
    for key, caster in _NUMERIC_SETTINGS.items():
        if key not in raw or raw[key] in (None, ""):
            continue
        try:
            value = caster(raw[key])
        except (TypeError, ValueError):
            raise DemoError(f"setting {key!r} must be a number, got {raw[key]!r}")
        settings[key] = value
    return settings


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class DemoHandler(BaseHTTPRequestHandler):
    server_version = "TinyVisionDemo/0.1"
    manager: JobManager = None  # type: ignore[assignment]
    ui_html: str = ""

    # -- helpers ---------------------------------------------------------
    def _send_json(self, obj: Any, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, status: int = 200, ctype: str = "text/plain; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter default logging
        pass

    # -- routing ---------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_text(self.ui_html, ctype="text/html; charset=utf-8")
            return
        if path == "/api/health":
            self._send_json({
                "status": "ok",
                "experimental": True,
                "disclaimer": DEMO_DISCLAIMER,
                "data_dir": str(self.manager.data_dir),
                "retention": ("Uploaded videos and reports stay in the data "
                              "directory until you delete them; nothing is "
                              "uploaded off this machine."),
            })
            return
        if path == "/api/models":
            models, default = list_models()
            self._send_json({"models": models, "default": default})
            return
        if path == "/api/jobs":
            self._send_json({"jobs": [j.public(include_samples=False)
                                      for j in self.manager.list()]})
            return
        m = re.fullmatch(r"/api/jobs/([0-9a-f]+)", path)
        if m:
            job = self.manager.get(m.group(1))
            if job is None:
                self._error(404, f"unknown job id: {m.group(1)}")
                return
            self._send_json(job.public())
            return
        m = re.fullmatch(r"/api/jobs/([0-9a-f]+)/report", path)
        if m:
            job = self.manager.get(m.group(1))
            if job is None:
                self._error(404, f"unknown job id: {m.group(1)}")
                return
            if job.report is None:
                self._error(409, f"job {job.id} has no report yet (status={job.status})")
                return
            body = json.dumps(job.report).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Disposition",
                             f'attachment; filename="{job.id}.demo-report.json"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._error(404, f"not found: {path}")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/analyze":
            self._handle_analyze()
            return
        self._error(404, f"not found: {path}")

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _stream_upload(self, filename: str, length: int) -> Path:
        """Write the request body to the uploads dir in chunks.

        Streaming (instead of reading the whole body into memory) keeps large
        recordings from freezing the server or exhausting RAM.
        """
        if length > MAX_UPLOAD_BYTES:
            raise DemoError(
                f"upload too large ({length / 1e9:.1f} GB > "
                f"{MAX_UPLOAD_BYTES / 1e9:.0f} GB); use the 'Local path' option "
                "instead of uploading"
            )
        dest = self.manager.new_upload_path(filename)
        written = 0
        with open(dest, "wb") as fh:
            remaining = length
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                fh.write(chunk)
                remaining -= len(chunk)
                written += len(chunk)
        if written == 0:
            dest.unlink(missing_ok=True)
            raise DemoError("uploaded file is empty")
        return dest

    def _handle_analyze(self) -> None:
        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        try:
            if ctype == "application/json":
                raw = json.loads(self._read_body() or b"{}")
                video_path = str(raw.get("video_path") or "").strip()
                if not video_path:
                    raise DemoError("provide 'video_path' (or upload a file)")
                source = video_path
                source_label = Path(video_path).name
                checkpoint = str(raw.get("checkpoint") or "").strip()
                settings_raw = raw.get("settings") or {}
            elif ctype == "application/octet-stream":
                filename = (self.headers.get("X-Filename") or "upload.mp4").strip()
                length = int(self.headers.get("Content-Length") or 0)
                dest = self._stream_upload(filename, length)
                source = str(dest)
                source_label = f"{Path(filename).name} (uploaded)"
                checkpoint = (self.headers.get("X-Checkpoint") or "").strip()
                settings_raw = json.loads(self.headers.get("X-Settings") or "{}")
            else:
                raise DemoError(
                    "unsupported Content-Type; use application/json "
                    "(video_path) or application/octet-stream (file upload)"
                )

            if not checkpoint:
                _, default = list_models()
                if not default:
                    raise DemoError("no checkpoint given and no local checkpoint found")
                checkpoint = default

            settings = build_settings(settings_raw)

            # Validate synchronously so obvious mistakes are a 400, not a slow
            # failed job.
            from .poc import PocConfig, PocError

            if not Path(checkpoint).exists():
                raise DemoError(f"checkpoint not found: {checkpoint}")
            if ctype == "application/json" and not Path(source).exists():
                raise DemoError(f"video file not found: {source}")
            try:
                PocConfig(source=source, checkpoint=checkpoint, **settings).validate()
            except PocError as exc:
                raise DemoError(str(exc))

            job = self.manager.start(source, checkpoint, settings, source_label)
            self._send_json({"job": job.public(include_samples=False)}, status=202)
        except DemoError as exc:
            self._error(400, str(exc))
        except json.JSONDecodeError as exc:
            self._error(400, f"invalid JSON body: {exc}")


# ---------------------------------------------------------------------------
# Server construction / entry point
# ---------------------------------------------------------------------------

def load_ui() -> str:
    """Load the single-page UI, whether installed or run from the source tree.

    Uses ``importlib.resources`` first (works for wheels and editable installs
    and regardless of the current working directory), then falls back to the
    file next to this module.
    """
    try:
        import importlib.resources as ir

        resource = ir.files("tv").joinpath("demo_ui.html")
        if resource.is_file():
            return resource.read_text(encoding="utf-8")
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    ui_path = Path(__file__).with_name("demo_ui.html")
    if ui_path.exists():
        return ui_path.read_text(encoding="utf-8")
    return (
        "<!doctype html><h1>TinyVision demo</h1>"
        "<p>UI asset (demo_ui.html) is missing from the installed package. "
        "Reinstall with <code>pip install -e training</code> (or "
        "<code>uv pip install -e training</code>).</p>"
    )


# Backwards-compatible alias.
_load_ui = load_ui


def create_server(host: str = "127.0.0.1", port: int = 0,
                  data_dir: Path = DEFAULT_DATA_DIR) -> ThreadingHTTPServer:
    """Create (but do not start) the demo HTTP server. Port 0 = ephemeral.

    Raises :class:`DemoError` with an actionable message if the address cannot
    be bound (e.g. the port is already in use).
    """
    manager = JobManager(Path(data_dir))

    class _Handler(DemoHandler):
        pass

    _Handler.manager = manager
    _Handler.ui_html = load_ui()
    try:
        httpd = ThreadingHTTPServer((host, port), _Handler)
    except OSError as exc:
        import errno

        if exc.errno in (errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", -1)):
            raise DemoError(
                f"port {port} on {host} is already in use. "
                f"Try a different port, e.g. --port {port + 1}, "
                "or stop the process already using it."
            ) from exc
        raise DemoError(f"cannot bind {host}:{port}: {exc}") from exc
    httpd.daemon_threads = True
    httpd.job_manager = manager  # type: ignore[attr-defined]
    return httpd


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1")


def serve(host: str = "127.0.0.1", port: int = 8000,
          data_dir: Path = DEFAULT_DATA_DIR, open_browser: bool = False) -> None:
    """Run the demo server until interrupted.

    Binds loopback by default. Binding a non-loopback host exposes uploaded
    videos and the inference endpoint to the network, so it is warned about
    explicitly. Raises :class:`DemoError` if the port cannot be bound.
    """
    httpd = create_server(host, port, data_dir)
    bound = httpd.server_address
    bound_host, bound_port = bound[0], bound[1]
    url = f"http://{bound_host}:{bound_port}/"
    models, _ = list_models()
    print("=" * 68, flush=True)
    print("TinyVision local inference demo", flush=True)
    print("=" * 68, flush=True)
    print(f"  Open:        {url}", flush=True)
    print(f"  Data dir:    {Path(data_dir)}", flush=True)
    print(f"  Models:      {len(models)} local checkpoint(s) discovered", flush=True)
    print("  Status:      EXPERIMENTAL — not validated for production use.", flush=True)
    print("  Stop:        press Ctrl+C in this window", flush=True)
    print("=" * 68, flush=True)
    if not _is_loopback(host):
        print("  WARNING: bound to a non-loopback address. Uploaded videos and "
              "the inference\n           endpoint are now reachable by other "
              "machines on this network.", flush=True)
        print("=" * 68, flush=True)
    if open_browser:
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
    finally:
        httpd.shutdown()
        httpd.server_close()
