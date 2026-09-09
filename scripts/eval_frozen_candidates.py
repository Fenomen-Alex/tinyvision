"""Uniform failing-positive metrics on the FROZEN original test split.

Runs the identical code path as eval_rc1_frozen_test.py across every candidate
(baseline, exp3, exp4, rc1) so the release comparison is apples-to-apples.
Writes docs/research/rc1-comparison-frozen.json.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

from tv.exp2_external import _binary_metrics, infer

FROZEN = REPO / "datasets/processed/printing-errors"
FAIL_IDX = 0
CANDIDATES = {
    "baseline": "experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/best.pt",
    "exp3": "experiments/exp3_target_normal_aug_shufflenet_v2_x1_0/artifacts/best.pt",
    "exp4": "experiments/exp4_target_balanced_shufflenet_v2_x1_0/artifacts/best.pt",
    "rc1": "experiments/rc1_release_candidate_shufflenet_v2_x1_0/artifacts/best.pt",
}
OUT = REPO / "docs/research/rc1-comparison-frozen.json"


def main() -> None:
    data_dir = FROZEN.resolve()
    rows = []
    with open(FROZEN / "splits/test.csv") as f:
        for r in csv.DictReader(f):
            p = Path(r["path"])
            if not p.is_absolute():
                p = data_dir / p
            rows.append({
                "image_path": str(p),
                "binary_idx": 0 if r["label"] == "failing" else 1,
            })
    result = {}
    for name, ck in CANDIDATES.items():
        inf = infer(Path(ck), rows, 224, 32, 0.5)
        y_true, y_pred, fs = inf["true_idx"], inf["preds"], np.array(inf["fail_scores"])
        result[name] = {
            "metrics_failing_positive": _binary_metrics(y_true, y_pred, FAIL_IDX),
            "auroc_failing": round(float(roc_auc_score((y_true == FAIL_IDX).astype(int), fs)), 6),
        }
        print(name, result[name]["metrics_failing_positive"]["precision"],
              result[name]["metrics_failing_positive"]["recall"],
              result[name]["metrics_failing_positive"]["f1"],
              result[name]["metrics_failing_positive"]["fpr"],
              result[name]["metrics_failing_positive"]["fnr"],
              result[name]["auroc_failing"])
    OUT.write_text(json.dumps({"schema": "frozen-test-failing-positive-uniform", "surfaces": result, "threshold": 0.5}, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()