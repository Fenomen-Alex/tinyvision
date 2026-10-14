"""CLI-level release-gate tests (synthetic fixtures, no external data).

These exercise the real `tv.cli reliability ...` entry points and assert the
documented exit codes, so the release gate is covered end to end in CI without
any large dataset. See docs/reliability-evaluation.md for the exit-code contract.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TRAINING = REPO / "training"


def _run(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(TRAINING) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "tv.cli", *args],
        cwd=REPO, env=env, capture_output=True, text=True,
    )


def _make_checkpoint(path: Path) -> None:
    import torch

    from tv.models.factory import build_classifier

    torch.manual_seed(0)
    model = build_classifier("shufflenet_v2_x1_0", 2, pretrained=False)
    torch.save({"model_state_dict": model.state_dict(), "num_classes": 2,
                "config_model": "shufflenet_v2_x1_0"}, path)


def _write_csv_surface(tmp: Path, name: str, rows: list[tuple[str, str]]) -> Path:
    from PIL import Image

    imgs = tmp / f"{name}_imgs"
    imgs.mkdir(exist_ok=True)
    lines = []
    for i, (label, source) in enumerate(rows):
        p = imgs / f"{name}_{i}.png"
        color = (10, 200, 10) if label == "normal" else (200, 10, 10)
        Image.new("RGB", (64, 64), color).save(p)
        lines.append(f"{p},{label},{source}")
    csv = tmp / f"{name}.csv"
    csv.write_text("path,label,source\n" + "\n".join(lines) + "\n")
    return csv


def _write_suite(tmp: Path, body: str) -> Path:
    suite = tmp / "suite.yaml"
    suite.write_text(body)
    return suite


def test_cli_evaluate_missing_required_surface_exit_3(tmp_path):
    present = _write_csv_surface(tmp_path, "present", [("normal", "s1"), ("failing", "s2")])
    suite = _write_suite(
        tmp_path,
        "name: missing\nversion: 1\npositive_class: 0\nsurfaces:\n"
        "  - id: dev\n    role: dev_validation\n    format: csv\n"
        f"    path: {present}\n    label_map: {{normal: 1, failing: 0}}\n"
        "  - id: nhold\n    role: normal_holdout\n    format: csv\n"
        f"    path: {tmp_path / 'absent.csv'}\n    label_map: {{normal: 1, failing: 0}}\n",
    )
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "id: t\npolicy_version: 1\ngates:\n  normal_holdout:\n"
        "    required: true\n    min_samples: 1\n    max_fpr: 0.1\n")
    ckpt = tmp_path / "best.pt"
    _make_checkpoint(ckpt)

    r = _run("reliability", "evaluate", "--suite", str(suite), "--policy", str(policy),
             "--model", f"m={ckpt}", "--output-dir", str(tmp_path / "out"))
    assert r.returncode == 3, r.stdout + r.stderr
    assert "INCONCLUSIVE" in r.stdout


def test_cli_evaluate_leakage_exit_2(tmp_path):
    shared = _write_csv_surface(tmp_path, "shared", [("normal", "S1"), ("failing", "S1")])
    suite = _write_suite(
        tmp_path,
        "name: leak\nversion: 1\npositive_class: 0\nsurfaces:\n"
        "  - id: locked\n    role: locked_test\n    format: csv\n"
        f"    path: {shared}\n    label_map: {{normal: 1, failing: 0}}\n"
        "    source_key: source\n"
        "  - id: nhold\n    role: normal_holdout\n    format: csv\n"
        f"    path: {shared}\n    label_map: {{normal: 1, failing: 0}}\n"
        "    filter: {field: label, equals: normal}\n    source_key: source\n",
    )
    policy = tmp_path / "policy.yaml"
    policy.write_text("id: t\npolicy_version: 1\ngates: {}\n")
    ckpt = tmp_path / "best.pt"
    _make_checkpoint(ckpt)

    r = _run("reliability", "evaluate", "--suite", str(suite), "--policy", str(policy),
             "--model", f"m={ckpt}", "--output-dir", str(tmp_path / "out"))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "LEAKAGE" in r.stdout


def test_cli_evaluate_pass_exit_0(tmp_path):
    # normal-only surface, max_fpr 1.0 -> FPR <= 1.0 always -> deterministic PASS
    nhold = _write_csv_surface(tmp_path, "nhold", [("normal", "s1"), ("normal", "s2")])
    suite = _write_suite(
        tmp_path,
        "name: pass\nversion: 1\npositive_class: 0\nsurfaces:\n"
        "  - id: nhold\n    role: normal_holdout\n    format: csv\n"
        f"    path: {nhold}\n    label_map: {{normal: 1, failing: 0}}\n"
        "    source_key: source\n",
    )
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "id: t\npolicy_version: 1\ngates:\n  normal_holdout:\n"
        "    required: true\n    min_samples: 2\n    max_fpr: 1.0\n")
    ckpt = tmp_path / "best.pt"
    _make_checkpoint(ckpt)

    r = _run("reliability", "evaluate", "--suite", str(suite), "--policy", str(policy),
             "--model", f"m={ckpt}", "--output-dir", str(tmp_path / "out"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout


def test_cli_evaluate_requires_policy(tmp_path):
    # A release decision without acceptance criteria is a CLI usage error (2).
    nhold = _write_csv_surface(tmp_path, "nhold", [("normal", "s1"), ("normal", "s2")])
    suite = _write_suite(
        tmp_path,
        "name: nopolicy\nversion: 1\npositive_class: 0\nsurfaces:\n"
        "  - id: nhold\n    role: normal_holdout\n    format: csv\n"
        f"    path: {nhold}\n    label_map: {{normal: 1, failing: 0}}\n",
    )
    ckpt = tmp_path / "best.pt"
    _make_checkpoint(ckpt)
    r = _run("reliability", "evaluate", "--suite", str(suite), "--model", f"m={ckpt}")
    assert r.returncode == 2, r.stdout + r.stderr
    assert "--policy" in (r.stdout + r.stderr)


def test_cli_validate_suite_strict_missing_exit_1(tmp_path):
    suite = _write_suite(
        tmp_path,
        "name: missing\nversion: 1\npositive_class: 0\nsurfaces:\n"
        "  - id: nhold\n    role: normal_holdout\n    format: csv\n"
        f"    path: {tmp_path / 'absent.csv'}\n    label_map: {{normal: 1, failing: 0}}\n",
    )
    r = _run("reliability", "validate-suite", "--suite", str(suite), "--strict")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "MISSING" in r.stdout


def test_cli_validate_suite_nonstrict_reports_incomplete(tmp_path):
    suite = _write_suite(
        tmp_path,
        "name: missing\nversion: 1\npositive_class: 0\nsurfaces:\n"
        "  - id: nhold\n    role: normal_holdout\n    format: csv\n"
        f"    path: {tmp_path / 'absent.csv'}\n    label_map: {{normal: 1, failing: 0}}\n",
    )
    r = _run("reliability", "validate-suite", "--suite", str(suite))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "INCOMPLETE" in r.stdout
