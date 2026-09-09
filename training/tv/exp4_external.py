"""Experiment 4: target-domain failure-balancing evaluation (Stage 14).

The frozen TinyVision baseline (binary normal-vs-failing) is compared against
the Experiment-3 model (normal-augmented) and the Experiment-4 model
(normal + small target-domain FAILURE augmentation) on the same surfaces.

Hypothesis under test: adding a small amount of diverse target-domain failure
data (2 independent fixed-cam sources: ZYfXW0JN2k4 spaghetti, 3zXQrw0E-Ag blob)
on top of the Stage-13 normal augmentation restores failure sensitivity (lost
when the model became normal-biased) while retaining the Stage-13 normal-FP
improvement.

Label authority (Strict, Experiment 4):
  * TRAINING augmentation failing -> the 100 VLM-verified failing frames in
    contiguous failure windows from 2 sources; status ``pending_human_review``.
  * TRAINING augmentation normal   -> the 333 VLM-screened normal frames
    (Stage 13), ``pending_human_review``.
  * FROZEN original test          -> byte-identical printing-errors test.csv.
  * HELD-OUT normal eval          -> the 120 normal frames from 2 fully
    held-out sources (LY3jlcO34f4, zkJgLanCSUw).

Conventions (identical to exp3-external):
  * failing-as-positive primary view (threshold on P(class=failing)=P(fail_idx=0));
    no threshold tuning (0.5 everywhere).
  * Held-out normal FPR = fraction of normal frames predicted failing.
"""
from __future__ import annotations

import json
from pathlib import Path

from .exp3_external import (
    SCHEMA,
    SCHEMA_VERSION,
    evaluate_frozen_test,
    evaluate_heldout_normal,
)

REPO = Path(__file__).resolve().parents[2]
SCHEMA_EXP4 = "exp4-external"


def compare_three(baseline_checkpoint, exp3_checkpoint, exp4_checkpoint, config):
    """Compare baseline / Exp3 / Exp4 on frozen original test + held-out normal."""
    return {
        "schema": SCHEMA_EXP4,
        "schema_version": SCHEMA_VERSION,
        "checkpoints": {
            "baseline": str(Path(baseline_checkpoint).resolve()),
            "exp3": str(Path(exp3_checkpoint).resolve()),
            "exp4": str(Path(exp4_checkpoint).resolve()),
        },
        "frozen_original_test": {
            "baseline": evaluate_frozen_test(baseline_checkpoint, config),
            "exp3": evaluate_frozen_test(exp3_checkpoint, config),
            "exp4": evaluate_frozen_test(exp4_checkpoint, config),
        },
        "external_normal_holdout": {
            "baseline": evaluate_heldout_normal(baseline_checkpoint),
            "exp3": evaluate_heldout_normal(exp3_checkpoint),
            "exp4": evaluate_heldout_normal(exp4_checkpoint),
        },
    }
