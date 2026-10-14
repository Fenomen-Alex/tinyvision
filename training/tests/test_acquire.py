import subprocess
from pathlib import Path

import pytest

from tv.data.acquire import AcquireError, extract_frames, fetch_video


class FakeRunner:
    def __init__(self, stdout=b"", fail=False):
        self.calls = []
        self.stdout = stdout
        self.fail = fail

    def __call__(self, cmd, capture_output=True, check=True, text=False):
        self.calls.append(cmd)
        if self.fail:
            raise subprocess.CalledProcessError(1, cmd, stderr=b"boom")

        class R:
            pass

        r = R()
        r.stdout = self.stdout
        r.returncode = 0
        return r


def test_fetch_builds_ytdlp_argv(tmp_path):
    runner = FakeRunner()

    def runner_wrapper(cmd, **kw):
        (tmp_path / "vid.mp4").write_bytes(b"v")
        return runner(cmd, **kw)

    path = fetch_video("abc123", tmp_path, max_seconds=4, runner=runner_wrapper)
    cmd = runner.calls[0]
    assert cmd[0] == "yt-dlp" and "abc123" in " ".join(cmd)
    assert "--download-sections" in cmd
    assert path.name == "vid.mp4"


def test_extract_frames_uses_grid(tmp_path):
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"v")
    calls = []

    def runner(cmd, **kw):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"x")

        class R:
            stdout = b""
            returncode = 0

        return R()

    frames = extract_frames(vid, tmp_path / "f", fps=1.0, duration=3.0, runner=runner)
    assert len(frames) == 3
    assert all(c[0] == "ffmpeg" for c in calls)


def test_extract_frames_is_idempotent(tmp_path):
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"v")
    out = tmp_path / "f"
    out.mkdir()
    (out / "frame_0000.jpg").write_bytes(b"x")

    def runner(cmd, **kw):
        raise AssertionError("should not shell out when frames exist")

    frames = extract_frames(vid, out, fps=1.0, duration=3.0, runner=runner)
    assert len(frames) == 1


def test_acquire_error_on_failure(tmp_path):
    with pytest.raises(AcquireError):
        fetch_video("abc", tmp_path, runner=FakeRunner(fail=True))
