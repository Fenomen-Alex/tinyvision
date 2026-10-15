"""Tests for the cross-platform demo launcher (tv/launcher.py)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training"))

import tv.launcher as launcher  # noqa: E402


def test_inference_modules_map_shape():
    for module, package in launcher.INFERENCE_MODULES.items():
        assert isinstance(module, str) and module
        assert isinstance(package, str) and package


def test_missing_inference_modules_is_list_of_pairs():
    missing = launcher.missing_inference_modules()
    assert isinstance(missing, list)
    for item in missing:
        assert isinstance(item, tuple) and len(item) == 2


def test_check_dependencies_ok_when_all_present(monkeypatch):
    monkeypatch.setattr(launcher, "missing_inference_modules", lambda: [])
    assert launcher.check_dependencies(strict=True) == 0


def test_check_dependencies_reports_missing(monkeypatch, capsys):
    monkeypatch.setattr(launcher, "missing_inference_modules",
                        lambda: [("torch", "torch")])
    code = launcher.check_dependencies(strict=True)
    out = capsys.readouterr().out
    assert code == 3
    assert "torch" in out
    assert "pip install -e training" in out  # actionable instructions


def test_run_refuses_without_dependencies(monkeypatch, capsys):
    monkeypatch.setattr(launcher, "missing_inference_modules",
                        lambda: [("cv2", "opencv-python")])
    # Must not attempt to start a server; returns a non-zero code with a reason.
    assert launcher.run(port=0) == 3
    err = capsys.readouterr().err
    assert "opencv-python" in err


def test_build_parser_defaults():
    args = launcher.build_parser().parse_args([])
    assert args.host == "127.0.0.1"  # loopback only by default
    assert args.port == 8000
    assert args.data_dir is None
    assert args.open_browser is False
    assert args.check is False


def test_main_check_subcommand_exit_code(monkeypatch):
    monkeypatch.setattr(launcher, "check_dependencies", lambda strict: 7)
    assert launcher.main(["--check"]) == 7
