"""Packaging tests: the demo UI must ship with the package.

These assert the *declared* packaging config and that the asset is resolvable
through the installed package, so a non-editable install (a wheel) still serves
the UI. They do not require reinstalling the package.
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training"))

import tv  # noqa: E402
from tv.webdemo import load_ui  # noqa: E402

PYPROJECT = REPO / "training" / "pyproject.toml"


def _config() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_demo_ui_is_declared_as_package_data():
    data = _config().get("tool", {}).get("setuptools", {}).get("package-data", {})
    assert "demo_ui.html" in data.get("tv", [])


def test_console_scripts_declared():
    scripts = _config()["project"]["scripts"]
    assert scripts.get("tv") == "tv.cli:main"
    assert scripts.get("tinyvision-demo") == "tv.launcher:main"


def test_pillow_is_a_declared_dependency():
    deps = _config()["project"]["dependencies"]
    assert any(d.lower().startswith("pillow") for d in deps)


def test_demo_ui_resolvable_from_installed_package():
    import importlib.resources as ir

    resource = ir.files("tv").joinpath("demo_ui.html")
    assert resource.is_file()


def test_load_ui_returns_real_ui_not_fallback():
    html = load_ui()
    # The fallback message only appears when the asset is missing.
    assert "is missing from the installed package" not in html
    assert "TinyVision" in html
