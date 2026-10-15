"""Regression tests for the rc2 real-domain missed-failure investigation.

These guard the two mechanisms established in
``docs/research/rc2-realdomain-results.md`` ("Missed failure: root cause"):

1. Monotonic (temperature) calibration cannot rescue a fixed 0.5 boundary when
   the model's argmax is wrong for the failure: it preserves argmax for every
   T > 0.
2. Threshold selection on validation data is not identifiable when the only
   validation failing stream sits far above the normals: a wide band of
   thresholds achieves perfect validation performance, and the repository's
   tie-break (higher threshold) selects a value that can miss a lower-scoring
   held-out failure.

Deterministic and fast: synthetic numbers only. These are regression guards, not
real-world evidence; the real-world evidence lives in the research doc.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from tv.calibration import (
    argmax,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    softmax,
)
from tv.reliability import (
    STATUS_PASS,
    select_threshold,
    threshold_sweep,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# ---------------------------------------------------------------------------
# 1. Calibration preserves argmax => cannot flip a below-boundary failure
# ---------------------------------------------------------------------------

def test_temperature_preserves_argmax_for_any_positive_temperature():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 2)) * 3.0
    base = argmax(logits, 1.0)
    for t in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
        assert np.array_equal(argmax(logits, t), base)


def test_temperature_cannot_flip_a_below_half_failing_frame():
    # FAIL_IDX = 0; reproduce a frame whose P(failing) ~= 0.078 (argmax normal).
    logits = np.array([[0.0, 2.47]])  # z_failing < z_normal
    assert softmax(logits, 1.0)[0, 0] < 0.5
    for t in np.linspace(0.01, 20.0, 200):
        p_failing = softmax(logits, float(t))[0, 0]
        assert p_failing < 0.5, f"temperature {t} flipped a below-boundary frame"
        assert argmax(logits, float(t))[0] == 1


def test_fit_temperature_never_worsens_validation_nll():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(300, 2))
    y = rng.integers(0, 2, size=300)
    res = fit_temperature(logits, y)
    assert res["temperature"] > 0
    assert res["nll"] <= res["nll_t1"] + 1e-9
    # argmax is unchanged by the fitted temperature
    assert np.array_equal(argmax(logits, res["temperature"]), argmax(logits, 1.0))


def test_expected_calibration_error_bounds():
    p = np.array([0.9, 0.1, 0.8, 0.2])
    y = np.array([0, 1, 0, 1])  # all correct
    ece = expected_calibration_error(p, y)
    assert 0.0 <= ece <= 1.0
    # Deliberately wrong and overconfident predictions => positive ECE.
    assert expected_calibration_error(np.array([0.99, 0.99]), np.array([1, 1])) > 0.4
    # A perfectly calibrated, fully confident set has ~0 ECE.
    assert expected_calibration_error(np.array([1.0, 0.0]), np.array([0, 1])) < 1e-9


# ---------------------------------------------------------------------------
# 2. Validation threshold selection is not identifiable on the real-like split
# ---------------------------------------------------------------------------

# Medians observed on the rc2 real-domain validation split: two normals and one
# failing stream that sits far above them (see the research doc).
_VAL_Y = np.array([1, 1, 0])            # normal, normal, failing (positive=0)
_VAL_SCORES = np.array([0.0131, 0.0500, 0.8780])


def test_repo_selection_is_most_sensitive_lowest_feasible_threshold():
    """The selector maximises recall then minimises threshold (sensitivity)."""
    sel = select_threshold(_VAL_Y, _VAL_SCORES,
                           objective="max_recall_subject_to_max_fpr",
                           max_fpr=0.0,
                           grid=[round(0.02 * i, 2) for i in range(1, 50)])
    assert sel["status"] == STATUS_PASS
    assert sel["validation_metrics"]["recall"] == 1.0
    assert sel["validation_metrics"]["fpr"] == 0.0
    # Lowest feasible threshold on this val surface (> 0.050 val normal max).
    assert sel["threshold"] == 0.06


def test_validation_band_is_wide_so_threshold_is_not_identifiable():
    """Many thresholds give perfect val performance: the choice is ambiguous.

    The observed held-out failing median is 0.078; the release-policy grid floor
    is 0.3.  A grid that never explores below 0.3 cannot select the ~0.06 that
    detection would need, and even the 0.06 choice has only ~0.01 margin above
    the highest validation normal.
    """
    grid = [round(0.05 * i, 2) for i in range(1, 20)]
    sweep = threshold_sweep(_VAL_Y, _VAL_SCORES, grid)
    perfect = [s["threshold"] for s in sweep
               if s["fpr"] == 0.0 and s["recall"] == 1.0]
    assert len(perfect) >= 15
    assert max(perfect) - min(perfect) >= 0.7  # wide, unidentifiable band
    # Detecting the observed held-out failure (~0.078) needs a threshold below
    # the release grid floor (0.3).
    held_out_failing_median = 0.078
    release_grid_floor = 0.3
    assert held_out_failing_median < release_grid_floor


# ---------------------------------------------------------------------------
# 3. Release-gate policy guard: the weak-label diagnostic can never pass release
# ---------------------------------------------------------------------------

def test_release_policy_requires_normal_holdout_and_onnx_parity():
    from tv.reliability import load_policy

    policy = load_policy(CONFIGS / "reliability" / "policy.yaml")
    assert policy["gates"]["normal_holdout"]["required"] is True
    assert policy["gates"]["normal_holdout"]["max_fpr"] <= 0.1
    assert policy["gates"]["parity"]["require_onnx"] is True
    assert policy["threshold_selection"]["max_fpr"] == 0.2


def test_diagnostic_policy_is_all_optional_and_never_blocks():
    from tv.reliability import load_policy

    policy = load_policy(CONFIGS / "reliability" / "policy-realdomain-diagnostic.yaml")
    assert policy["status"] == "diagnostic"
    assert policy["gates"], "diagnostic policy must declare its gates"
    for name, cfg in policy["gates"].items():
        assert cfg.get("required", False) is False, f"gate {name} must be optional"


def test_realdomain_suite_surfaces_are_unverified():
    from tv.reliability import load_suite

    suite = load_suite(CONFIGS / "reliability" / "suite-realdomain.yaml")
    for s in suite.surfaces:
        assert s.verified is False, (
            f"surface {s.id} is marked verified; weak/VLM labels are not "
            f"human-verified and must never satisfy a release gate"
        )
