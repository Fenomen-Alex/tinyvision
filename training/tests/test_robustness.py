"""Tests for Stage 2B robustness / perturbation-evaluation machinery."""
from __future__ import annotations

import numpy as np
import pytest

from tv.robustness import (
    PERTURBATIONS,
    SEVERITY_NAMES,
    _apply_brightness,
    _apply_combined,
    _apply_contrast,
    _apply_crop,
    _apply_downscale,
    _apply_gaussian_blur,
    _apply_jpeg,
    _apply_noise,
    _apply_perspective,
    _apply_rotation,
    _degradation,
    _pool_by,
    load_resolved_metadata,
    severity_schedule,
)


def _rgb(h=64, w=64, dtype=np.uint8):
    r = np.linspace(0, 255, h, dtype=np.float32)
    g = np.linspace(255, 0, w, dtype=np.float32)
    x = np.tile(r[:, None, None], (1, w, 1))
    y = np.tile(g[None, :, None], (h, 1, 1))
    z = np.full((h, w, 1), 128.0)
    out = np.concatenate([x, y, z], axis=-1)
    return np.clip(out, 0, 255).astype(dtype)


def _assert_valid(out, h, w):
    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_all_perturbations_preserve_shape_and_range():
    img = _rgb()
    rng = np.random.default_rng(0)
    for name in SEVERITY_NAMES:
        _assert_valid(img, 64, 64)
    h, w = img.shape[:2]
    for name, spec in PERTURBATIONS.items():
        for i, level in enumerate(spec["levels"]):
            out = spec["apply"](img, level, rng)
            _assert_valid(out, h, w)


def test_perturbations_area_preserving_stay_same_size():
    img = _rgb()
    rng = np.random.default_rng(0)
    h, w = img.shape[:2]
    for name in PERTURBATIONS:
        spec = PERTURBATIONS[name]
        out = spec["apply"](img, spec["levels"][1], rng)
        assert out.shape == (h, w, 3)


def test_severity_is_monotonic_brightness():
    img = _rgb().astype(np.float32)
    mild = _apply_brightness(img, 1.1)
    sev = _apply_brightness(img, 1.5)
    assert sev.mean() > mild.mean()
    dark = _apply_brightness(img, 0.6)
    assert dark.mean() < img.mean()


def test_blur_reduces_high_frequency_energy():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    blurred = _apply_gaussian_blur(img, 3.0)
    sharp = _apply_gaussian_blur(img, 0.0)
    # High-frequency detail is reduced by blurring.
    diff_orig = np.abs(img.astype(np.float32) - blurred).mean()
    assert diff_orig > 1.0


def test_downscale_restores_original_size():
    img = _rgb(128, 64)
    out = _apply_downscale(img, 0.4)
    assert out.shape == (128, 64, 3)


def test_crop_restores_original_size_and_changes_content():
    img = _rgb(100, 80)
    out = _apply_crop(img, 0.7)
    assert out.shape == (100, 80, 3)
    # A crop + upscale should change pixel content vs identity.
    assert not np.array_equal(out, img)


def test_rotation_changes_content():
    img = _rgb()
    out = _apply_rotation(img, 15.0)
    assert not np.array_equal(out, img)


def test_perspective_deterministic_by_seed():
    img = _rgb()
    a1 = _apply_perspective(img, 2.0, np.random.default_rng(7))
    a2 = _apply_perspective(img, 2.0, np.random.default_rng(7))
    b = _apply_perspective(img, 2.0, np.random.default_rng(8))
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, b)


def test_noise_deterministic_by_seed():
    img = _rgb()
    a1 = _apply_noise(img, 25.0, np.random.default_rng(3))
    a2 = _apply_noise(img, 25.0, np.random.default_rng(3))
    b = _apply_noise(img, 25.0, np.random.default_rng(4))
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, img)
    assert not np.array_equal(a1, b)


def test_combined_preserves_size_and_is_deterministic():
    img = _rgb(48, 48)
    level = {"downscale": 0.7, "brightness": 1.2, "blur": 1.8,
             "noise": 18.0, "jpeg": 60}
    a1 = _apply_combined(img, level, np.random.default_rng(11))
    a2 = _apply_combined(img, level, np.random.default_rng(11))
    h, w = img.shape[:2]
    _assert_valid(a1, h, w)
    assert np.array_equal(a1, a2)


def test_jpeg_changes_content():
    img = _rgb()
    out_low = _apply_jpeg(img, 20)
    out_high = _apply_jpeg(img, 95)
    # Low quality is more destructive; both should differ from the source.
    assert not np.array_equal(out_low, out_high)
    assert not np.array_equal(out_low, img)


def test_severity_schedule():
    assert severity_schedule("identity") == 1
    for name in PERTURBATIONS:
        assert severity_schedule(name) == 3


def test_degradation_signed_deltas():
    base = {"accuracy": 0.8, "f1": 0.7, "recall": 0.9, "precision": 0.6,
            "auroc": 0.9, "fnr": 0.1, "fpr": 0.05}
    pert = {"accuracy": 0.6, "f1": 0.5, "recall": 0.6, "precision": 0.7,
            "auroc": 0.75, "fnr": 0.4, "fpr": 0.3}
    d = _degradation(base, pert)
    assert d["accuracy"] == pytest.approx(-0.2)
    assert d["f1"] == pytest.approx(-0.2)
    assert d["fnr_delta"] == pytest.approx(0.3)
    assert d["fpr_delta"] == pytest.approx(0.25)
    assert "precision" in d


def test_pool_by_groups_and_computes_failing_view(tmp_path):
    samples = [
        {"true_idx": 0, "pred_idx": 0, "failure_type": "under_extrusion"},  # TP
        {"true_idx": 0, "pred_idx": 1, "failure_type": "under_extrusion"},  # FN
        {"true_idx": 1, "pred_idx": 0, "failure_type": "normal"},            # FP
        {"true_idx": 1, "pred_idx": 1, "failure_type": "normal"},            # TN
    ]
    pooled = _pool_by(samples, fail_idx=0, field="failure_type")
    assert pooled["under_extrusion"]["n"] == 2
    assert pooled["under_extrusion"]["recall"] == pytest.approx(0.5)
    assert pooled["under_extrusion"]["fnr"] == pytest.approx(0.5)
    assert pooled["normal"]["n"] == 2
    assert pooled["normal"]["recall"] == pytest.approx(0.0)  # pos=failing view
    assert pooled["normal"]["fnr"] == pytest.approx(0.0)


def test_load_resolved_metadata_joins_raw_csv(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # raw Kaggle-style metadata (semicolon-delimited)
    raw = data_dir / "all_images_no_filter.csv"
    raw.write_text(
        "image;class;shape;recording;brightness;mean_brightness\n"
        "a.png;1;5;Recording_1;0.3;0.5\n"
        "b.png;4;7;Recording_2;0.6;0.7\n"
        "c.png;0;unknown;Recording_3;0.1;0.2\n"
    )
    splits = data_dir / "splits"
    splits.mkdir()
    split_csv = splits / "test.csv"
    split_csv.write_text(
        "path,label\n"
        "failing/Recording_1/a.png,failing\n"
        "failing/Recording_2/b.png,failing\n"
        "normal/Recording_3/c.png,normal\n"
    )
    resolved = load_resolved_metadata(data_dir, "test")
    assert len(resolved) == 3
    assert resolved[0]["recording"] == "Recording_1"
    assert resolved[0]["failure_type"] == "under_extrusion"
    assert resolved[0]["shape"] == "5"
    assert resolved[1]["failure_type"] == "spaghetti"
    assert resolved[1]["shape"] == "7"
    assert resolved[2]["failure_type"] == "normal"
    assert resolved[2]["shape"] == "unknown"
    assert resolved[0]["label"] == "failing"