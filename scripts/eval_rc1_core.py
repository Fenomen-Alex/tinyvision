"""rc1 core eval: frozen original test + held-out target-domain normal sources.

Mirrors exp3/exp4 harness exactly (failing-as-positive primary view, threshold
0.5, conformant class_to_idx). Writes docs/research/rc1-core-eval.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

from tv.config import load_config
from tv.exp3_external import evaluate_frozen_test, evaluate_heldout_normal

CK = "experiments/rc1_release_candidate_shufflenet_v2_x1_0/artifacts/best.pt"
OUT = REPO / "docs/research/rc1-core-eval.json"


def main() -> None:
    cfg = load_config(REPO / "training/configs/rc1_release_candidate.yaml")
    cfg.data.num_workers = 0
    ft = evaluate_frozen_test(CK, cfg)
    hn = evaluate_heldout_normal(CK)
    result = {
        "schema": "rc1-core-eval",
        "schema_version": 1,
        "checkpoint": CK,
        "frozen_original_test": {
            "metrics_failing_positive": ft.get("metrics_failing_positive"),
            "metrics_normal_positive": ft.get("metrics_normal_positive"),
            "auroc": ft.get("auroc"),
            "sample_count": ft.get("sample_count"),
            "threshold": ft.get("threshold"),
        },
        "external_normal_holdout": {
            "evaluated_frames": hn["evaluated_frames"],
            "sources": hn["sources"],
            "predicted_failing": hn["normal_frames"]["predicted_failing"],
            "false_positive_rate": hn["normal_frames"]["false_positive_rate"],
            "metrics_failing_positive": hn["metrics_failing_positive"],
            "per_source": hn["per_source"],
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()