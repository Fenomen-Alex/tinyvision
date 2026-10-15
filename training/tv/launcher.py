"""Cross-platform launcher for the TinyVision local inference demo.

A thin, friendly entry point around :mod:`tv.webdemo` (which itself calls the
real inference pipeline in :mod:`tv.poc`). Installable as the ``tinyvision-demo``
console script and also reachable as ``python -m tv.launcher``.

What it adds over ``tv demo``:

* a clear dependency check with actionable install instructions, so someone who
  is not a Python developer gets told exactly what is missing;
* an explicit, honest port-in-use error;
* loopback-by-default binding with a warning if you expose it to the network.

It runs on macOS, Linux and Windows: no shell scripts, no bundled runtime.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Optional, Sequence

from .webdemo import DEFAULT_DATA_DIR, DemoError, serve

# Direct imports used on the inference path, mapped to the pip distribution that
# provides them. These are required to actually run inference (the API/UI still
# load without them, but no analysis can run).
INFERENCE_MODULES: "dict[str, str]" = {
    "numpy": "numpy",
    "cv2": "opencv-python",
    "torch": "torch",
    "torchvision": "torchvision",
    "PIL": "pillow",
}

_INSTALL_HINT = (
    "Install the demo (and its inference dependencies) with one of:\n"
    "    uv pip install -e training        # if you use uv\n"
    "    python -m pip install -e training # if you use a venv with pip"
)


def missing_inference_modules() -> list[tuple[str, str]]:
    """Return ``[(module, pip_package), ...]`` for missing inference imports."""
    missing: list[tuple[str, str]] = []
    for module, package in INFERENCE_MODULES.items():
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append((module, package))
    return missing


def check_dependencies(strict: bool = True) -> int:
    """Report missing inference dependencies. Returns an exit code (0 = ok)."""
    missing = missing_inference_modules()
    if not missing:
        print("All inference dependencies are present "
              f"({', '.join(INFERENCE_MODULES)}).", flush=True)
        return 0
    print("TinyVision demo cannot run inference: missing Python dependencies.",
          flush=True)
    for module, package in missing:
        print(f"  - import {module!r} is unavailable (pip package: {package})",
              flush=True)
    print(_INSTALL_HINT, flush=True)
    return 0 if not strict else 3


def run(host: str = "127.0.0.1", port: int = 8000,
        data_dir: Optional[Path] = None, open_browser: bool = False) -> int:
    """Start the demo server. Returns a process exit code."""
    missing = missing_inference_modules()
    if missing:
        print("error: cannot run inference — missing dependencies:", file=sys.stderr)
        for module, package in missing:
            print(f"  - {module} (pip package: {package})", file=sys.stderr)
        print(_INSTALL_HINT, file=sys.stderr)
        return 3
    try:
        serve(host=host, port=port,
              data_dir=Path(data_dir) if data_dir else DEFAULT_DATA_DIR,
              open_browser=open_browser)
    except DemoError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tinyvision-demo",
        description="Launch the TinyVision local inference demo (experimental).",
    )
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind address (default 127.0.0.1; loopback only). "
                        "Anything else exposes uploads/endpoints to the network.")
    p.add_argument("--port", type=int, default=8000,
                   help="Port (default 8000; use 0 for a free ephemeral port)")
    p.add_argument("--data-dir", default=None,
                   help="Where uploads and reports are stored "
                        "(default: a per-user directory; env TINYVISION_DEMO_DATA)")
    p.add_argument("--open-browser", action="store_true",
                   help="Open the demo in the default browser")
    p.add_argument("--check", action="store_true",
                   help="Only check inference dependencies, then exit")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        return check_dependencies(strict=True)
    return run(host=args.host, port=args.port, data_dir=args.data_dir,
               open_browser=args.open_browser)


if __name__ == "__main__":
    sys.exit(main())
