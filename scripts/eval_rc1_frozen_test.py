"""Failing-positive metrics + AUROC for rc1 on the FROZEN original test split.

Augments rc1-core-eval.json (written by eval_rc1_core.py) with the explicit
failing-as-positive conventions used by the deepeval/exp3/exp4 harness
(precision/recall/F1/FNR/FPR on positive_class=0, AUROC on P(failing)).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

from sklearn.metrics import roc_auc_score
from tv.exp2_external import _binary_metrics, infer

CK = "experiments/rc1_release_candidate_shufflenet_v2_x1_0/artifacts/best.pt"
FROZEN = REPO / "datasets/processed/printing-errors"
OUT = REPO / "docs/research/rc1-core-eval.json"
FAIL_IDX = 0


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
                "binary_label": r["label"],
                "binary_idx": 0 if r["label"] == "failing" else 1,
            })
    inf = infer(Path(CK), rows, 224, 32, 0.5)
    y_true, y_pred, fs = inf["true_idx"], inf["preds"], np.array(inf["fail_scores"])

    core = json.loads(OUT.read_text())
    core["frozen_original_test"]["metrics_failing_positive"] = _binary_metrics(y_true, y_pred, FAIL_IDX)
    core["frozen_original_test"]["metrics_normal_positive"] = _binary_metrics(y_true, y_pred, 1)
    core["frozen_original_test"]["auroc_failing"] = round(
        float(roc_auc_score((y_true == FAIL_IDX).astype(int), fs)), 6)
    OUT.write_text(json.dumps(core, indent=2) + "\n")
    print(json.dumps(core["frozen_original_test"], indent=1))


if __name__ == "__main__":
    main()