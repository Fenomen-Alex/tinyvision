#!/usr/bin/env python3
"""Reproduce the rc1 frozen-test evaluation and record FPR=1.0 evidence status.

Uses the reliability framework (single metric definition, failing-as-positive,
CPU-only deterministic scoring) rather than the ad-hoc eval scripts, so the
reproduced numbers are directly comparable to `docs/research/rc1-core-eval.json`.

What this reproduces from files present in a clean checkout:
  * frozen original test metrics (rc1-core-eval.json["frozen_original_test"])

What it cannot reproduce (recorded, never fabricated):
  * the 120-frame independent normal holdout (external media not committed)

Writes docs/research/reliability-repro-rc1.json. Run from the repo root:

    training/.venv/bin/python scripts/reliability_repro_rc1.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

import numpy as np

from tv.reliability import (
    Surface,
    compute_binary_metrics,
    load_samples,
    predictions_from_scores,
    score_framework,
    sha256_file,
)

CK = REPO / "experiments/rc1_release_candidate_shufflenet_v2_x1_0/artifacts/best.pt"
DATA = REPO / "datasets/processed/printing-errors"
CORE = REPO / "docs/research/rc1-core-eval.json"
OUT = REPO / "docs/research/reliability-repro-rc1.json"
HOLDOUT_MANIFEST = REPO / "datasets/external/stage13-normal/manifest.json"
POSITIVE = 0  # failing
THRESHOLD = 0.5
IMAGE_SIZE = 224


def frozen_test_block() -> dict:
    surface = Surface(id="frozen_test", role="locked_test", format="csv",
                      path="splits/test.csv",
                      label_map={"failing": 0, "normal": 1})
    samples = load_samples(surface, DATA)
    scores = score_framework(CK, samples, IMAGE_SIZE, 32)
    y_true = np.asarray([s.label for s in samples])
    y_pred = predictions_from_scores(scores, THRESHOLD, POSITIVE)
    m = compute_binary_metrics(y_true, y_pred, scores, POSITIVE)
    return samples, m


def main() -> None:
    samples, m = frozen_test_block()
    core = json.loads(CORE.read_text()) if CORE.exists() else {}
    ft_block = core.get("frozen_original_test") or {}
    committed = ft_block.get("metrics_failing_positive") or {}

    def near(a, b, ndigits: int) -> bool:
        return a is not None and b is not None and round(float(a), ndigits) == round(float(b), ndigits)

    match = {
        "sample_count": committed == {} or ft_block.get("sample_count") == m["n"],
        "confusion": committed == {} or all(
            committed.get(k) == m[k] for k in ("tp", "fp", "tn", "fn")),
        "precision": near(m["precision"], committed.get("precision"), 4),
        "recall": near(m["recall"], committed.get("recall"), 4),
        "fpr": near(m["fpr"], committed.get("fpr"), 4),
        "fnr": near(m["fnr"], committed.get("fnr"), 4),
        "auroc": near(m["auroc"], ft_block.get("auroc"), 4),
    }

    holdout = {"manifest_present": HOLDOUT_MANIFEST.exists(),
               "frames_on_disk": 0, "eval_holdout_frames": 120}
    if HOLDOUT_MANIFEST.exists():
        d = json.loads(HOLDOUT_MANIFEST.read_text())
        entries = [e for e in d.get("entries", []) if e.get("split") == "eval_holdout"]
        holdout["eval_holdout_frames"] = len(entries)
        holdout["frames_on_disk"] = sum(
            1 for e in entries if Path(e.get("image_path", "")).exists())
    holdout["reproducible_from_clean_checkout"] = holdout["frames_on_disk"] > 0
    holdout["evidence_status"] = (
        "reproducible" if holdout["frames_on_disk"] > 0
        else "confirmed-by-artifact (docs/research/rc1-core-eval.json); "
             "not re-executable offline (frames uncommitted, source "
             "non-redistributable)")

    report = {
        "schema": "reliability-repro-rc1",
        "schema_version": 1,
        "generated": date.today().isoformat(),
        "model": {
            "path": str(CK.relative_to(REPO)),
            "sha256": sha256_file(CK),
        },
        "frozen_original_test": {
            "dataset_csv": "datasets/processed/printing-errors/splits/test.csv",
            "dataset_csv_sha256": sha256_file(DATA / "splits/test.csv"),
            "positive_class": POSITIVE,
            "score_convention": "score=P(failing)=softmax(logits)[0]; failing iff score>=threshold",
            "threshold": THRESHOLD,
            "image_size": IMAGE_SIZE,
            "sample_count": m["n"],
            "class_support": {"failing": m["n_positive"], "normal": m["n_negative"]},
            "confusion": {"tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"]},
            "metrics_failing_positive": {
                "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                "fpr": m["fpr"], "fnr": m["fnr"], "auroc": m["auroc"],
                "fpr_ci95": m["fpr_ci95"], "fnr_ci95": m["fnr_ci95"],
            },
            "matches_committed_rc1_core_eval": match,
            "all_match": all(match.values()),
        },
        "independent_normal_holdout": holdout,
        "evidence_classification": {
            "frozen_test": "verified through a reproducible local test",
            "fpr_1_0_holdout": "confirmed from existing artifact; not yet verified "
                               "through a reproducible local test (data unavailable)",
        },
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["frozen_original_test"]["confusion"], indent=2))
    print("all_match:", report["frozen_original_test"]["all_match"])
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
