# Real-Domain Automated Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an automated, local pipeline that acquires real 3D-print footage, weak/VLM-labels it, builds source-isolated splits, and trains an in-domain model evaluated honestly through the reliability gate.

**Architecture:** Small, single-purpose modules under `training/tv/data/` (acquire, weaklabel, manifest, autolabel, streams) plus CLI subcommands. External tools (yt-dlp, ffmpeg) and the VLM are reached through injectable seams so every unit is testable offline. The existing trainer, exporter, and reliability gate are consumed unchanged.

**Tech Stack:** Python 3.11, stdlib `subprocess`/`urllib`/`json`/`csv`, existing `tv` package, PyTorch (existing), ffmpeg + yt-dlp (external binaries), LM Studio (`qwen/qwen3-vl-8b`).

## Global Constraints

- Research-only media: cached under `datasets/external-cache/` (gitignored); never committed or redistributed. Every source records `license_note: "research_only"`.
- No fabricated labels: VLM output is a *proposal*; `uncertain` is preserved, never silently converted.
- Source-level isolation: a source id never appears in two splits.
- Deterministic CLI; no notebooks. No new Python dependencies.
- Local inference only. No loosening of reliability-policy thresholds.
- Score/label convention is fixed: labels sorted alphabetically ⇒ `failing=0`, `normal=1`; `FAIL_IDX = 0`.
- Test command: `cd training && .venv/bin/python -m pytest tests -q`.

---

### Task 1: Weak video-level labelling

**Files:**
- Create: `training/tv/data/weaklabel.py`
- Test: `training/tests/test_weaklabel.py`

**Interfaces:**
- Produces: `weak_label(title: str, description: str = "") -> str` returning `"failing" | "normal" | "uncertain"`; module constants `FAILING_KEYWORDS: tuple[str, ...]`, `NORMAL_KEYWORDS: tuple[str, ...]`.

- [ ] **Step 1: Write the failing test**

```python
from tv.data.weaklabel import weak_label

def test_failing_keyword():
    assert weak_label("3D print FAILED - spaghetti everywhere") == "failing"

def test_normal_keyword():
    assert weak_label("Perfect 3D print timelapse") == "normal"

def test_failing_takes_precedence():
    assert weak_label("timelapse of a failed print") == "failing"

def test_no_signal_is_uncertain():
    assert weak_label("Unboxing a new printer") == "uncertain"

def test_case_insensitive():
    assert weak_label("LAYER SHIFT disaster") == "failing"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd training && .venv/bin/python -m pytest tests/test_weaklabel.py -q`
Expected: FAIL with `ModuleNotFoundError: tv.data.weaklabel`

- [ ] **Step 3: Write minimal implementation**

```python
"""Video-level weak labelling from titles/descriptions.

Video-level only. Frame-level labels come from autolabel. Uncertain is never
silently resolved; it is preserved for review.
"""
from __future__ import annotations

FAILING_KEYWORDS: tuple[str, ...] = (
    "spaghetti", "fail", "failed", "failure", "blob", "layer shift",
    "layershift", "warp", "warping", "adhesion", "stringing", "clog",
    "clogged", "under extrusion", "under-extrusion", "messed up", "disaster",
    "detached", "knocked over", "broken print", "print fail",
)
NORMAL_KEYWORDS: tuple[str, ...] = (
    "timelapse", "time-lapse", "time lapse", "success", "successful",
    "finished", "complete", "asmr", "perfect print", "print test",
)


def weak_label(title: str, description: str = "") -> str:
    text = f"{title} {description}".lower()
    if any(k in text for k in FAILING_KEYWORDS):
        return "failing"
    if any(k in text for k in NORMAL_KEYWORDS):
        return "normal"
    return "uncertain"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd training && .venv/bin/python -m pytest tests/test_weaklabel.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add training/tv/data/weaklabel.py training/tests/test_weaklabel.py
git commit -m "feat(data): weak video-level labelling from titles"
```

---

### Task 2: Dataset manifest

**Files:**
- Create: `training/tv/data/manifest.py`
- Test: `training/tests/test_manifest.py`

**Interfaces:**
- Produces: `@dataclass Source(id, url, title, uploader, duration, width, height, retrieved_at, license_note, weak_label, frames_dir, frame_count, failure_type=None)`; `@dataclass Manifest(version, created_at, sources)`; `load_manifest(path) -> Manifest`; `save_manifest(manifest, path) -> None`; `validate_manifest(manifest, root=None) -> list[str]`.
- `Source.to_dict()` / `Source.from_dict()` round-trip.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from tv.data.manifest import Manifest, Source, load_manifest, save_manifest, validate_manifest

def _src(tmp, sid="abc", frames=True):
    d = tmp / sid / "frames"
    if frames:
        d.mkdir(parents=True)
        (d / "frame_0000.jpg").write_bytes(b"x")
    return Source(id=sid, url=f"https://y/{sid}", title="t", uploader="u",
                  duration=10.0, width=256, height=144, retrieved_at="2026-10-14",
                  license_note="research_only", weak_label="normal",
                  frames_dir=str(d), frame_count=1 if frames else 0)

def test_roundtrip(tmp_path):
    m = Manifest(version=1, created_at="2026-10-14", sources=[_src(tmp_path)])
    p = tmp_path / "manifest.json"
    save_manifest(m, p)
    assert load_manifest(p) == m

def test_validate_flags_missing_dir_and_zero_frames(tmp_path):
    m = Manifest(version=1, created_at="2026-10-14",
                 sources=[_src(tmp_path, "miss", frames=False),
                          _src(tmp_path, "zero", frames=True)])
    m.sources[1].frame_count = 0
    warnings = validate_manifest(m)
    assert any("miss" in w for w in warnings)
    assert any("zero" in w for w in warnings)

def test_validate_flags_duplicate_ids(tmp_path):
    m = Manifest(version=1, created_at="2026-10-14",
                 sources=[_src(tmp_path, "dup"), _src(tmp_path, "dup")])
    assert any("duplicate" in w.lower() for w in validate_manifest(m))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd training && .venv/bin/python -m pytest tests/test_manifest.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
"""Dataset manifest for acquired real-domain footage (research-only)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Source:
    id: str
    url: str
    title: str
    uploader: str
    duration: float
    width: int
    height: int
    retrieved_at: str
    license_note: str
    weak_label: str
    frames_dir: str
    frame_count: int
    failure_type: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Source":
        return cls(**d)


@dataclass
class Manifest:
    version: int
    created_at: str
    sources: list[Source] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"version": self.version, "created_at": self.created_at,
                "sources": [s.to_dict() for s in self.sources]}

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        return cls(version=d["version"], created_at=d["created_at"],
                   sources=[Source.from_dict(s) for s in d.get("sources", [])])


def load_manifest(path: Path) -> Manifest:
    return Manifest.from_dict(json.loads(Path(path).read_text()))


def save_manifest(manifest: Manifest, path: Path) -> None:
    Path(path).write_text(json.dumps(manifest.to_dict(), indent=2) + "\n")


def validate_manifest(manifest: Manifest, root: Path | None = None) -> list[str]:
    warnings: list[str] = []
    seen: set[str] = set()
    for s in manifest.sources:
        if s.id in seen:
            warnings.append(f"duplicate source id: {s.id}")
        seen.add(s.id)
        if not s.frames_dir or not Path(s.frames_dir).is_dir():
            warnings.append(f"missing frames dir for source {s.id}: {s.frames_dir}")
        if s.frame_count <= 0:
            warnings.append(f"zero frames for source {s.id}")
    return warnings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd training && .venv/bin/python -m pytest tests/test_manifest.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/tv/data/manifest.py training/tests/test_manifest.py
git commit -m "feat(data): dataset manifest with round-trip and validation"
```

---

### Task 3: Acquisition (yt-dlp + ffmpeg)

**Files:**
- Create: `training/tv/data/acquire.py`
- Test: `training/tests/test_acquire.py`

**Interfaces:**
- Produces: `class AcquireError(RuntimeError)`; `probe(video_path, runner=subprocess.run) -> dict` (keys `duration`,`width`,`height`); `fetch_video(source, out_dir, max_seconds=None, format_selector="160", runner=subprocess.run) -> Path`; `extract_frames(video_path, out_dir, fps=1.0, runner=subprocess.run) -> list[Path]`; `acquire_source(source, cache_root, max_seconds=None, fps=1.0, meta=None, runner=subprocess.run) -> Source`.
- `runner(cmd, capture_output=True, check=True)`-compatible; returns object with `.stdout`/`.returncode`.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import subprocess
from tv.data.acquire import acquire_source, extract_frames, fetch_video, AcquireError

class FakeRunner:
    def __init__(self, stdout=b"", fail=False):
        self.calls = []
        self.stdout = stdout
        self.fail = fail
    def __call__(self, cmd, capture_output=True, check=True, text=False):
        self.calls.append(cmd)
        if self.fail:
            raise subprocess.CalledProcessError(1, cmd, stderr=b"boom")
        class R: pass
        r = R(); r.stdout = self.stdout; r.returncode = 0
        return r

def test_fetch_builds_ytdlp_argv(tmp_path):
    runner = FakeRunner()
    def runner_wrapper(cmd, **kw):
        # simulate yt-dlp producing the file
        out = tmp_path / "vid.mp4"; out.write_bytes(b"v")
        return runner(cmd, **kw)
    fetch_video("abc123", tmp_path, max_seconds=4, runner=runner_wrapper)
    cmd = runner.calls[0]
    assert cmd[0] == "yt-dlp" and "abc123" in " ".join(cmd)
    assert "--download-sections" in cmd

def test_extract_frames_uses_grid(tmp_path):
    vid = tmp_path / "v.mp4"; vid.write_bytes(b"v")
    calls = []
    def runner(cmd, **kw):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"x")
        class R: stdout = b""; returncode = 0
        return R()
    frames = extract_frames(vid, tmp_path / "f", fps=1.0, duration=3.0, runner=runner)
    assert len(frames) == 3
    assert all(c[0] == "ffmpeg" for c in calls)

def test_acquire_error_on_failure(tmp_path):
    import pytest
    with pytest.raises(AcquireError):
        fetch_video("abc", tmp_path, runner=FakeRunner(fail=True))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd training && .venv/bin/python -m pytest tests/test_acquire.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
"""Acquire real video footage and extract frames (yt-dlp + ffmpeg)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tv.data.manifest import Source


class AcquireError(RuntimeError):
    pass


def _run(runner, cmd, **kw):
    try:
        return runner(cmd, capture_output=True, check=True, **kw)
    except subprocess.CalledProcessError as e:
        err = getattr(e, "stderr", b"")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        raise AcquireError(f"command failed: {' '.join(cmd)}\n{err}") from e


def _as_url(source: str) -> str:
    return source if source.startswith("http") else f"https://www.youtube.com/watch?v={source}"


def probe(video_path: Path, runner=subprocess.run) -> dict:
    res = _run(runner, ["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,duration",
                        "-show_entries", "format=duration", "-of", "json",
                        str(video_path)])
    out = res.stdout.decode() if isinstance(res.stdout, bytes) else res.stdout
    data = json.loads(out)
    stream = (data.get("streams") or [{}])[0]
    duration = stream.get("duration") or (data.get("format") or {}).get("duration")
    return {"duration": float(duration or 0.0),
            "width": int(stream.get("width") or 0),
            "height": int(stream.get("height") or 0)}


def fetch_video(source: str, out_dir: Path, max_seconds: int | None = None,
                format_selector: str = "160", runner=subprocess.run) -> Path:
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
                   duration: float | None = None, runner=subprocess.run) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("frame_*.jpg"))
    if existing:
        return existing
    if duration is None:
        duration = probe(video_path, runner=runner)["duration"]
    step = 1.0 / fps if fps else 1.0
    grid, t, k = [], 0.0, 0
    while t < max(duration, step):
        grid.append(t)
        k += 1
        t = k * step
    paths: list[Path] = []
    for i, ts in enumerate(grid):
        out = out_dir / f"frame_{i:04d}.jpg"
        _run(runner, ["ffmpeg", "-y", "-ss", f"{ts}", "-i", str(video_path),
                      "-frames:v", "1", "-q:v", "2", str(out)])
        paths.append(out)
    return paths


def acquire_source(source: str, cache_root: Path, max_seconds: int | None = None,
                   fps: float = 1.0, meta: dict | None = None,
                   runner=subprocess.run) -> Source:
    from datetime import date
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
                  license_note="research_only", weak_label=m.get("weak_label", "uncertain"),
                  frames_dir=str(root / "frames"), frame_count=len(frames))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd training && .venv/bin/python -m pytest tests/test_acquire.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/tv/data/acquire.py training/tests/test_acquire.py
git commit -m "feat(data): yt-dlp/ffmpeg acquisition with injectable runner"
```

---

### Task 4: `tv acquire` CLI + gitignore cache

**Files:**
- Modify: `training/tv/cli.py` (add `cmd_acquire` + parser; add to module docstring)
- Modify: `.gitignore` (add `datasets/external-cache/`)
- Test: `training/tests/test_acquire_cli.py`

**Interfaces:**
- Consumes: `acquire_source`, `weak_label`, `Manifest`, `save_manifest`.
- Produces: `tv acquire --sources FILE --cache DIR [--manifest PATH] [--max-seconds N] [--fps 1.0]`; `--sources` is a JSON array of `{"source": "...", "title": "...", ...}`.

- [ ] **Step 1: Write the failing test**

```python
import json
from tv.cli import build_parser

def test_acquire_parser_exists():
    args = build_parser().parse_args(
        ["acquire", "--sources", "s.json", "--cache", "c", "--manifest", "m.json"])
    assert args.command == "acquire" and args.max_seconds is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd training && .venv/bin/python -m pytest tests/test_acquire_cli.py -q`
Expected: FAIL (`build_parser` missing or `acquire` unrecognized)

- [ ] **Step 3: Write minimal implementation**

Add near other `cmd_*` functions:

```python
def cmd_acquire(args: argparse.Namespace) -> None:
    import json
    from tv.data.acquire import acquire_source
    from tv.data.manifest import Manifest, save_manifest
    from tv.data.weaklabel import weak_label
    from datetime import date

    entries = json.loads(Path(args.sources).read_text())
    sources = []
    for e in entries:
        src = e["source"]
        title = e.get("title", "")
        meta = {"title": title, "uploader": e.get("uploader", ""),
                "weak_label": e.get("weak_label") or weak_label(title, e.get("description", ""))}
        try:
            sources.append(acquire_source(src, Path(args.cache),
                                          max_seconds=args.max_seconds,
                                          fps=args.fps, meta=meta))
        except Exception as ex:  # noqa: BLE001 - record, never silently drop
            print(f"[acquire] FAILED {src}: {ex}")
    manifest = Manifest(version=1, created_at=date.today().isoformat(), sources=sources)
    save_manifest(manifest, Path(args.manifest))
    print(f"[acquire] {len(sources)} sources -> {args.manifest}")
```

In `build_parser` (before `return parser`):

```python
    p_acq = sub.add_parser("acquire", help="Acquire real footage + frames")
    p_acq.add_argument("--sources", required=True)
    p_acq.add_argument("--cache", required=True)
    p_acq.add_argument("--manifest", default="datasets/external-cache/manifest.json")
    p_acq.add_argument("--max-seconds", type=int, default=None)
    p_acq.add_argument("--fps", type=float, default=1.0)
    p_acq.set_defaults(func=cmd_acquire)
```

Add `datasets/external-cache/` under the "Dataset contents" block in `.gitignore`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd training && .venv/bin/python -m pytest tests/test_acquire_cli.py -q`
Then: `training/.venv/bin/python -m tv.cli acquire --help`
Expected: PASS, and help prints.

- [ ] **Step 5: Commit**

```bash
git add training/tv/cli.py tests/test_acquire_cli.py .gitignore
git commit -m "feat(cli): tv acquire subcommand; gitignore research cache"
```

---

### Task 5: VLM auto-labelling with confidence gating

**Files:**
- Create: `training/tv/data/autolabel.py`
- Test: `training/tests/test_autolabel.py`

**Interfaces:**
- Consumes: `vlm_review.call_lmstudio(endpoint, model, prompt, data_urls, timeout)`, `vlm_review.image_to_data_url_scaled(path, max_side)`, `vlm_review.build_messages`.
- Produces: `FrameProposal(frame, label, confidence, source_id, failure_type=None)`; `parse_decision(text) -> tuple[str, float, str | None]`; `autolabel_frames(frames, *, source_id, client, confidence_threshold=0.9, weak_label=None) -> list[FrameProposal]`; `save_proposals(proposals, path)`, `load_proposals(path)`; `review_queue(proposals) -> list[FrameProposal]`.
- `client(data_urls: list[str]) -> str` (thin seam so tests need no LM Studio).

- [ ] **Step 1: Write the failing test**

```python
import json
from tv.data.autolabel import (parse_decision, autolabel_frames, review_queue,
                               FrameProposal, save_proposals, load_proposals)

def test_parse_decision_ok():
    assert parse_decision('{"decision":"failing","failure_type":"spaghetti","confidence":0.95}') \
        == ("failing", 0.95, "spaghetti")

def test_parse_decision_uncertain_on_garbage():
    assert parse_decision("not json") == ("uncertain", 0.0, None)

def test_autolabel_autoaccept_vs_queue(tmp_path):
    f1 = tmp_path / "a.jpg"; f1.write_bytes(b"x")
    f2 = tmp_path / "b.jpg"; f2.write_bytes(b"x")
    replies = {str(f1): '{"decision":"failing","confidence":0.97,"failure_type":"blob"}',
               str(f2): '{"decision":"normal","confidence":0.4}'}
    calls = []
    def client(data_urls):
        calls.append(data_urls)
        # test seam: map by call order
        return replies["__final__"] if False else next(iter(replies.values()))
    props = autolabel_frames([f1, f2], source_id="s", client=lambda u: replies[str(f1)],
                             confidence_threshold=0.9, weak_label="failing")
    assert props[0].label == "failing" and props[0].source_id == "s"
    q = review_queue(props)
    assert isinstance(q, list)
    save_proposals(props, tmp_path / "p.json")
    assert load_proposals(tmp_path / "p.json") == props

def test_uncertain_never_autoaccepted(tmp_path):
    f = tmp_path / "a.jpg"; f.write_bytes(b"x")
    props = autolabel_frames([f], source_id="s", client=lambda u: "garbage",
                             confidence_threshold=0.9, weak_label=None)
    assert props[0].label == "uncertain"
    assert props[0] in review_queue(props)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd training && .venv/bin/python -m pytest tests/test_autolabel.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
"""Automatic frame labelling via a local VLM (proposals only, never verified)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

LABEL_VOCAB = ("normal", "failing", "uncertain")
FAILURE_TYPES = ("spaghetti", "stringing", "under_extrusion")

PROMPT = (
    "You are classifying one frame from a fixed camera watching a 3D printer bed. "
    "Decide the visible print state. Respond with STRICT JSON only: "
    '{"decision":"normal|failing|uncertain",'
    '"failure_type":"spaghetti|stringing|under_extrusion|null","confidence":0.0-1.0}. '
    "Use 'failing' only for a visible print failure; 'uncertain' if unclear."
)


@dataclass
class FrameProposal:
    frame: str
    label: str
    confidence: float
    source_id: str
    failure_type: str | None = None
    auto_accepted: bool = False


def parse_decision(text: str) -> tuple[str, float, str | None]:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return ("uncertain", 0.0, None)
    if not isinstance(obj, dict):
        return ("uncertain", 0.0, None)
    label = str(obj.get("decision", "uncertain")).lower()
    if label not in LABEL_VOCAB:
        label = "uncertain"
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    ftype = obj.get("failure_type")
    ftype = None if ftype in (None, "null", "") else str(ftype)
    if ftype not in FAILURE_TYPES:
        ftype = None
    return (label, max(0.0, min(1.0, conf)), ftype)


def autolabel_frames(frames, *, source_id: str, client,
                     confidence_threshold: float = 0.9,
                     weak_label: str | None = None) -> list[FrameProposal]:
    out: list[FrameProposal] = []
    for f in frames:
        try:
            text = client([str(f)])
            label, conf, ftype = parse_decision(text)
        except Exception:  # noqa: BLE001 - record, never guess
            label, conf, ftype = "uncertain", 0.0, None
        agrees = weak_label in (None, "uncertain") or weak_label == label
        accepted = (label in ("normal", "failing") and conf >= confidence_threshold
                    and agrees)
        out.append(FrameProposal(frame=str(f), label=label, confidence=conf,
                                 source_id=source_id, failure_type=ftype,
                                 auto_accepted=accepted))
    return out


def review_queue(proposals: list[FrameProposal]) -> list[FrameProposal]:
    return [p for p in proposals if not p.auto_accepted]


def save_proposals(proposals: list[FrameProposal], path: Path) -> None:
    Path(path).write_text(json.dumps([asdict(p) for p in proposals], indent=2) + "\n")


def load_proposals(path: Path) -> list[FrameProposal]:
    return [FrameProposal(**d) for d in json.loads(Path(path).read_text())]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd training && .venv/bin/python -m pytest tests/test_autolabel.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/tv/data/autolabel.py training/tests/test_autolabel.py
git commit -m "feat(data): VLM auto-labelling with confidence gating + QA queue"
```

---

### Task 6: `tv autolabel` CLI

**Files:**
- Modify: `training/tv/cli.py` (add `cmd_autolabel` + parser)
- Test: `training/tests/test_autolabel_cli.py`

**Interfaces:**
- Consumes: `autolabel_frames`, `save_proposals`, `review_queue`, `vlm_review.call_lmstudio`, `image_to_data_url_scaled`.
- Produces: `tv autolabel --manifest PATH --proposals PATH [--queue PATH] [--threshold 0.9] [--endpoint URL] [--model NAME]`.

- [ ] **Step 1: Write the failing test**

```python
from tv.cli import build_parser

def test_autolabel_parser():
    a = build_parser().parse_args(["autolabel", "--manifest", "m.json", "--proposals", "p.json"])
    assert a.command == "autolabel" and a.threshold == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd training && .venv/bin/python -m pytest tests/test_autolabel_cli.py -q`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
def cmd_autolabel(args: argparse.Namespace) -> None:
    from tv.data.autolabel import autolabel_frames, save_proposals, review_queue
    from tv.data.manifest import load_manifest
    from tv import vlm_review

    man = load_manifest(Path(args.manifest))
    def client(data_urls):
        url = vlm_review.image_to_data_url_scaled(Path(data_urls[0]), 448)
        return vlm_review.call_lmstudio(args.endpoint, args.model,
                                        vlm_review.__dict__.get("_PROMPT",
                                        "Classify the visible 3D-print state as "
                                        "normal|failing|uncertain as strict JSON."),
                                        [url])
    all_props = []
    for s in man.sources:
        frames = sorted(Path(s.frames_dir).glob("frame_*.jpg"))
        all_props += autolabel_frames(frames, source_id=s.id, client=client,
                                      confidence_threshold=args.threshold,
                                      weak_label=s.weak_label)
    save_proposals(all_props, Path(args.proposals))
    q = review_queue(all_props)
    if args.queue:
        save_proposals(q, Path(args.queue))
    print(f"[autolabel] {len(all_props)} frames, {len(q)} queued -> {args.proposals}")
```

Parser:

```python
    p_al = sub.add_parser("autolabel", help="VLM frame proposals + QA queue")
    p_al.add_argument("--manifest", required=True)
    p_al.add_argument("--proposals", required=True)
    p_al.add_argument("--queue", default=None)
    p_al.add_argument("--threshold", type=float, default=0.9)
    p_al.add_argument("--endpoint", default="http://127.0.0.1:1234")
    p_al.add_argument("--model", default="qwen/qwen3-vl-8b")
    p_al.set_defaults(func=cmd_autolabel)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd training && .venv/bin/python -m pytest tests/test_autolabel_cli.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/tv/cli.py training/tests/test_autolabel_cli.py
git commit -m "feat(cli): tv autolabel subcommand"
```

---

### Task 7: Source-isolated split builder

**Files:**
- Create: `training/tv/data/streams.py`
- Test: `training/tests/test_streams.py`

**Interfaces:**
- Consumes: `FrameProposal`.
- Produces: `class LeakageError(ValueError)`; `build_splits(proposals, *, ratios=(0.7,0.15,0.15), seed=0) -> dict[str, list[tuple[str,str]]]` where the value is a list of `(frame_path, label)`; `write_splits(splits, out_dir) -> None` writing `train.csv`/`val.csv`/`test.csv` (`path,label`) + `splits.json`.

- [ ] **Step 1: Write the failing test**

```python
import csv
from tv.data.autolabel import FrameProposal
from tv.data.streams import build_splits, write_splits

def _p(src, label):
    return [FrameProposal(frame=f"{src}_f{i}.jpg", label=label, confidence=0.99,
                          source_id=src, auto_accepted=True) for i in range(4)]

def test_source_level_isolation(tmp_path):
    props = _p("s1", "failing") + _p("s2", "normal") + _p("s3", "normal") + _p("s4", "failing")
    splits = build_splits(props, ratios=(0.5, 0.25, 0.25), seed=0)
    seen = {}
    for name, items in splits.items():
        for _path, label in items:
            pass
    all_sources = {}
    for name, items in splits.items():
        srcs = {p.source_id for p in props if p.frame in {f for f, _ in items}}
        for s in srcs:
            assert all_sources.setdefault(s, name) == name, f"leak {s}"

def test_write_splits_csv_format(tmp_path):
    splits = {"train": [("a.jpg", "failing")], "val": [("b.jpg", "normal")],
              "test": [("c.jpg", "normal")]}
    write_splits(splits, tmp_path)
    rows = list(csv.DictReader(open(tmp_path / "train.csv")))
    assert rows == [{"path": "a.jpg", "label": "failing"}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd training && .venv/bin/python -m pytest tests/test_streams.py -q`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
"""Build source-isolated train/val/test splits from frame proposals."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path


class LeakageError(ValueError):
    pass


def build_splits(proposals, *, ratios=(0.7, 0.15, 0.15), seed=0):
    by_source: dict[str, list] = {}
    for p in proposals:
        by_source.setdefault(p.source_id, []).append(p)
    sources = sorted(by_source)
    rng = random.Random(seed)
    rng.shuffle(sources)
    by_label: dict[str, list[str]] = {}
    for s in sources:
        by_label.setdefault(by_source[s][0].label, []).append(s)
    n = len(sources)
    n_train = max(1, round(n * ratios[0]))
    name = {"train": [], "val": [], "test": []}
    for label, srcs in by_label.items():
        k = max(1, round(len(srcs) * ratios[0])) if len(srcs) > 1 else 1
        for i, s in enumerate(srcs):
            name["train" if i < k else ("val" if i == k else "test")].append(s)
    splits = {k: [] for k in name}
    for split, srcs in name.items():
        for s in srcs:
            for p in by_source[s]:
                splits[split].append((p.frame, p.label))
    # leakage guard
    owner = {}
    for split, items in splits.items():
        for f, _l in items:
            src = next(p.source_id for p in proposals if p.frame == f)
            if owner.setdefault(src, split) != split:
                raise LeakageError(f"source {src} spans splits")
    return splits


def write_splits(splits, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for split, items in splits.items():
        with open(out_dir / f"{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label"])
            for path, label in items:
                w.writerow([path, label])
        meta[split] = len(items)
    (out_dir / "splits.json").write_text(json.dumps({"counts": meta}, indent=2) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd training && .venv/bin/python -m pytest tests/test_streams.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/tv/data/streams.py training/tests/test_streams.py
git commit -m "feat(data): source-isolated split builder + CSV writer"
```

---

### Task 8: `tv streams` CLI

**Files:**
- Modify: `training/tv/cli.py` (add `cmd_streams` + parser)
- Test: `training/tests/test_streams_cli.py`

**Interfaces:**
- Consumes: `load_proposals`, `build_splits`, `write_splits`.
- Produces: `tv streams --proposals PATH --out DIR [--ratios 0.7,0.15,0.15] [--seed 0]`.

- [ ] **Step 1: Write the failing test**

```python
from tv.cli import build_parser

def test_streams_parser():
    a = build_parser().parse_args(["streams", "--proposals", "p.json", "--out", "d"])
    assert a.command == "streams" and a.seed == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd training && .venv/bin/python -m pytest tests/test_streams_cli.py -q`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
def cmd_streams(args: argparse.Namespace) -> None:
    from tv.data.autolabel import load_proposals
    from tv.data.streams import build_splits, write_splits

    props = load_proposals(Path(args.proposals))
    ratios = tuple(float(x) for x in args.ratios.split(","))
    splits = build_splits(props, ratios=ratios, seed=args.seed)
    write_splits(splits, Path(args.out))
    counts = {k: len(v) for k, v in splits.items()}
    print(f"[streams] {counts} -> {args.out}")
```

Parser:

```python
    p_st = sub.add_parser("streams", help="Build source-isolated splits")
    p_st.add_argument("--proposals", required=True)
    p_st.add_argument("--out", required=True)
    p_st.add_argument("--ratios", default="0.7,0.15,0.15")
    p_st.add_argument("--seed", type=int, default=0)
    p_st.set_defaults(func=cmd_streams)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd training && .venv/bin/python -m pytest tests/test_streams_cli.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/tv/cli.py training/tests/test_streams_cli.py
git commit -m "feat(cli): tv streams subcommand"
```

---

### Task 9: Training config + reliability suite

**Files:**
- Create: `training/configs/planned/rc2_realdomain.yaml`
- Create: `training/configs/reliability/suite-realdomain.yaml`
- Test: `training/tests/test_realdomain_configs.py`

**Interfaces:**
- Consumes: `tv.config.load_config`, `tv.reliability.load_suite`.
- Produces: loadable config/suite referencing `datasets/external-cache/processed/...`.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from tv.config import load_config
from tv.reliability import load_suite

def test_rc2_config_loads():
    cfg = load_config(Path("configs/planned/rc2_realdomain.yaml"))
    assert cfg.model.name

def test_realdomain_suite_loads():
    suite = load_suite(Path("configs/reliability/suite-realdomain.yaml"))
    assert len(suite.surfaces) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd training && .venv/bin/python -m pytest tests/test_realdomain_configs.py -q`
Expected: FAIL (files missing)

- [ ] **Step 3: Create the config and suite**

`training/configs/planned/rc2_realdomain.yaml` (mirror an existing experiment config; key fields):

```yaml
experiment:
  id: rc2_realdomain
  description: RC2 fine-tune on acquired real-domain footage (research-only)
model:
  name: shufflenet_v2_x1_0
  pretrained: true
  init_from: experiments/rc1_release_candidate_shufflenet_v2_x1_0/artifacts/best.pt
dataset:
  data_dir: datasets/external-cache/processed/realdomain
  image_size: 224
training:
  epochs: 8
  batch_size: 32
  optimizer: adamw
  learning_rate: 0.0001
  augmentation:
    horizontal_flip: true
    random_rotation: 15
    color_jitter: 0.3
  seed: 0
  precision: fp32
```

`training/configs/reliability/suite-realdomain.yaml`:

```yaml
id: realdomain-reliability
surfaces:
  - id: real_dev
    role: dev_validation
    format: csv
    path: datasets/external-cache/processed/realdomain/splits/val.csv
    label_map: {normal: 1, failing: 0}
  - id: real_test
    role: locked_test
    format: csv
    path: datasets/external-cache/processed/realdomain/splits/test.csv
    label_map: {normal: 1, failing: 0}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd training && .venv/bin/python -m pytest tests/test_realdomain_configs.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/configs/planned/rc2_realdomain.yaml training/configs/reliability/suite-realdomain.yaml training/tests/test_realdomain_configs.py
git commit -m "feat(config): rc2 real-domain training config + gate suite"
```

---

### Task 10: Vertical-slice integration (offline)

**Files:**
- Test: `training/tests/test_realdomain_integration.py`

**Interfaces:**
- Consumes: `extract_frames`, `build_splits`, `write_splits`, `CSVImageDataset`, `weak_label`.

- [ ] **Step 1: Write the test** (uses real ffmpeg `testsrc`, no network)

```python
import subprocess
from pathlib import Path
from tv.data.acquire import extract_frames
from tv.data.autolabel import FrameProposal
from tv.data.streams import build_splits, write_splits
from tv.data.loaders import CSVImageDataset

def test_synthetic_end_to_end(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=3:size=128x96:rate=5",
                    "-pix_fmt", "yuv420p", str(clip)], capture_output=True, check=True)
    frames = extract_frames(clip, tmp_path / "frames", fps=1.0, duration=3.0)
    assert len(frames) == 3
    props = [FrameProposal(frame=str(f), label=("failing" if i == 0 else "normal"),
                           confidence=0.99, source_id=f"s{i%2}", auto_accepted=True)
             for i, f in enumerate(frames)]
    splits = build_splits(props, ratios=(0.5, 0.25, 0.25), seed=0)
    write_splits(splits, tmp_path / "splits")
    train = tmp_path / "splits" / "train.csv"
    ds = CSVImageDataset(train, data_dir=tmp_path)
    assert len(ds) == len(splits["train"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd training && .venv/bin/python -m pytest tests/test_realdomain_integration.py -q`
Expected: FAIL (imports or ffmpeg path)

- [ ] **Step 3: Fix minimal issues** (adjust import/signature only; no new behavior)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd training && .venv/bin/python -m pytest tests/test_realdomain_integration.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/tests/test_realdomain_integration.py
git commit -m "test(data): offline end-to-end synthetic slice"
```

---

### Task 11: Real run + honest report

**Files:**
- Create: `docs/research/rc2-realdomain-results.md`
- Create: `scripts/run_realdomain_pipeline.sh`
- Modify: `docs/research/rc2-audit-final-report.md` (link results)

**Interfaces:**
- Consumes: all CLI subcommands, `tv train`, `tv export`, `tv reliability evaluate`.

- [ ] **Step 1: Write the driver script**

```bash
#!/usr/bin/env bash
set -euo pipefail
PY=training/.venv/bin/python
CACHE=datasets/external-cache
mkdir -p "$CACHE"
# sources.json: known ids + a few keyword searches
$PY -m tv.cli acquire --sources "$CACHE/sources.json" --cache "$CACHE" --max-seconds 120
$PY -m tv.cli autolabel --manifest "$CACHE/manifest.json" \
    --proposals "$CACHE/proposals.json" --queue "$CACHE/review_queue.json"
$PY -m tv.cli streams --proposals "$CACHE/proposals.json" --out "$CACHE/processed/realdomain/splits"
$PY -m tv.cli train --config training/configs/planned/rc2_realdomain.yaml
```

- [ ] **Step 2: Run it** (acquires real footage; requires network + LM Studio)

Run: `bash scripts/run_realdomain_pipeline.sh`
Expected: manifest, proposals, splits on disk.

- [ ] **Step 3: Train and evaluate**

Run:
```bash
training/.venv/bin/python -m tv.cli train --config training/configs/planned/rc2_realdomain.yaml
training/.venv/bin/python -m tv.cli reliability evaluate \
  --suite training/configs/reliability/suite-realdomain.yaml \
  --policy training/configs/reliability/policy.yaml \
  --model rc2=experiments/rc2_realdomain/artifacts/best.pt \
  --output-dir docs/research/release-gate-rc2-realdomain
```
Expected: a recorded PASS/FAIL/INCONCLUSIVE with metrics.

- [ ] **Step 4: Write `docs/research/rc2-realdomain-results.md`** with the exact
  numbers (FAR on held-out normal videos, failures detected, gate status) and an
  explicit statement of what is verified vs not. No fabricated numbers.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_realdomain_pipeline.sh docs/research/rc2-realdomain-results.md docs/research/rc2-audit-final-report.md
git commit -m "docs(research): rc2 real-domain pipeline results (honest)"
```

---

## Self-Review

**Spec coverage:** acquire (§7.1 → T3/T4), weaklabel (§7.2 → T1), manifest (§7.3 → T2),
autolabel (§7.4 → T5/T6), streams (§7.5 → T7/T8), data flow (§8 → T11),
error handling (§9 → T3/T5/T4), testing (§10 → each task + T10), risks (§11 →
research-only cache T4, honest gate T11), vertical slice (§12 → T10/T11).
All covered.

**Placeholder scan:** no TBD/TODO; each code step has real code.

**Type consistency:** `FrameProposal(frame,label,confidence,source_id,failure_type,auto_accepted)`
is used identically in T5/T7/T10. `write_splits(splits, out_dir)` signatures match.
`build_splits(proposals, ratios=, seed=)` match T7/T8/T10. `weak_label(title,description)`
match T1/T4.

**Note:** commits are shown per skill convention; the operator has not authorized
commits, so at execution time steps will skip `git commit` unless asked.
