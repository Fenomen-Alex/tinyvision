"""Experiment 3: target-domain normal augmentation evaluation (Stage 13).

The frozen TinyVision baseline (binary normal-vs-failing) is compared against
the Experiment-3 model (``exp3_target_normal_aug_shufflenet_v2_x1_0``) that was
trained identically but with target-domain NORMAL frames added to training.

Hypothesis under test: persistent normal-print false positives on independent
live footage are caused by insufficient diversity of normal target-domain
training data.

Label authority (Strict, Experiment 3):
  * TRAINING augmentation  -> the 333 VLM-screened ``normal`` frames from 7
    independent sources (decision==normal AND part_visible==True); status
    ``pending_human_review``, never silently treated as human-verified.
  * HELD-OUT normal eval   -> the 120 ``normal`` frames from 2 fully held-out
    sources (LY3jlcO34f4, zkJgLanCSUw). Source-level holdout: no frame from
    these sources appears in any training split.
  * FROZEN original test   -> byte-identical copy of the frozen
    printing-errors test.csv; both checkpoints scored with identical config.

Conventions (identical to deep-eval / exp2 external):
  * failing-as-positive is the primary view (decision threshold on
    P(class=failing)=P(fail_idx=0)); both binary conventions reported.
  * No threshold tuning: threshold 0.5 everywhere, matching the validated
    runtime config.
  * Held-out normal FPR = fraction of normal frames predicted failing.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "datasets/external/stage13-normal/manifest.json"
FROZEN_DIR = REPO / "datasets/processed/printing-errors"
SCHEMA = "exp3-external"
SCHEMA_VERSION = 1
FAIL_IDX = 0  # alphabetical class_to_idx: failing=0, normal=1


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def heldout_normal_entries() -> list[dict]:
    """The 120 held-out normal eval frames (source-level holdout)."""
    return [e for e in load_manifest()["entries"] if e["split"] == "eval_holdout"]


def evaluate_heldout_normal(checkpoint: str | Path, threshold: float = 0.5,
                            image_size: int = 224, batch_size: int = 32) -> dict:
    """Score the held-out normal frames with a frozen checkpoint.

    Every frame is genuinely normal, so FPR = (predicted failing) / total.
    """
    from .exp2_external import _binary_metrics, infer

    rows = [{
        "image_path": e["image_path"],
        "source_id": e["source_id"],
        "binary_label": "normal",
        "binary_idx": 1,
    } for e in heldout_normal_entries()]
    inference = infer(Path(checkpoint), rows, image_size, batch_size, threshold)
    y_true, y_pred, fail_scores = (
        inference["true_idx"], inference["preds"], inference["fail_scores"])

    per_source = {}
    for sid in sorted({r["source_id"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["source_id"] == sid]
        metrics = _binary_metrics(y_true[idx], y_pred[idx], FAIL_IDX)
        per_source[sid] = {
            "frames": len(idx),
            "fpr": metrics["fpr"],
            "predicted_failing": metrics["fp"],
        }

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "checkpoint": str(Path(checkpoint).resolve()),
        "kind": "external_normal_holdout",
        "threshold": threshold,
        "image_size": image_size,
        "evaluated_frames": len(rows),
        "sources": sorted({r["source_id"] for r in rows}),
        "metrics_failing_positive": _binary_metrics(y_true, y_pred, FAIL_IDX),
        "metrics_normal_positive": _binary_metrics(y_true, y_pred, 1),
        "per_source": per_source,
        "normal_frames": {
            "frames": len(rows),
            "predicted_failing": int(np.sum(y_pred == FAIL_IDX)),
            "false_positive_rate": round(
                float(np.mean(y_pred == FAIL_IDX)), 4),
        },
    }


def evaluate_frozen_test(checkpoint: str | Path, config,
                         threshold: float = 0.5) -> dict:
    """Score a frozen checkpoint on the frozen original printing-errors test.

    The derived exp3 test.csv is a byte-identical copy of this one, so the
    comparison is exact regardless of which config's splits are referenced.
    Uses evaluate_classification with the frozen data_dir so both checkpoints
    see identical test rows.
    """
    from .config import ExperimentConfig
    from .evaluate import evaluate_classification

    cfg = config if isinstance(config, ExperimentConfig) else ExperimentConfig(**config)
    import copy
    cfg = copy.deepcopy(cfg)
    cfg.dataset.data_dir = str(FROZEN_DIR)
    cfg.evaluation.decision_threshold = threshold
    result = evaluate_classification(Path(checkpoint), cfg)
    result["schema"] = SCHEMA
    result["schema_version"] = SCHEMA_VERSION
    result["checkpoint"] = str(Path(checkpoint).resolve())
    result["kind"] = "frozen_original_test"
    result["threshold"] = threshold
    return result


def read_csv_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def compare(baseline_checkpoint: str | Path, exp3_checkpoint: str | Path,
            config) -> dict:
    """Run baseline-vs-exp3 on frozen original test + held-out normal eval."""
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "baseline": str(Path(baseline_checkpoint).resolve()),
        "exp3": str(Path(exp3_checkpoint).resolve()),
        "frozen_original_test": {
            "baseline": evaluate_frozen_test(baseline_checkpoint, config),
            "exp3": evaluate_frozen_test(exp3_checkpoint, config),
        },
        "external_normal_holdout": {
            "baseline": evaluate_heldout_normal(baseline_checkpoint),
            "exp3": evaluate_heldout_normal(exp3_checkpoint),
        },
    }