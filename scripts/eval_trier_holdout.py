"""Frame-level evaluation of a frozen checkpoint on the Trier holdout.

Surfaces (source_annotation labels, CC BY 4.0; test_data root = external_holdout):
  * Trier FAILURE holdout - the 6 ``test_data`` extrusion_error / layer_shift
    environments (failing-as-positive: recall / FNR, per-env + aggregate).
  * Trier NORMAL holdout - the 3 ``test_data`` no_error_print environments
    (FPR = fraction of truly-normal frames predicted failing).

This is an offline, read-only evaluation. Labels are the dataset-author source
annotations; the environment split is the manifest's external_holdout.
Threshold 0.5, image_size 224, failing = class index 0 (frozen convention),
identical to every other evaluation surface. No threshold tuning.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

MANIFEST = REPO / "datasets/staging/trier/manifests/env_manifest.json"
SAMPLES_DIR = REPO / "datasets/staging/trier/samples"
INDEX = REPO / "datasets/staging/trier/metadata/index.json"
FAIL_IDX = 0

from tv.exp2_external import _binary_metrics, infer  # noqa: E402


def split_rows() -> dict[str, list[dict]]:
    manifest = json.loads(MANIFEST.read_text())["environments"]
    root = Path(json.loads(INDEX.read_text())["root"])
    if not root.is_absolute():
        root = (REPO / root).resolve()
    hold = {e["environment_id"]: e for e in manifest
            if e["split_assignment"] == "external_holdout"}
    rows_by_env: dict[str, list[dict]] = {}
    for f in sorted(SAMPLES_DIR.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            env = hold.get(r["environment_id"])
            if env is None:
                continue
            p = Path(r["metadata"]["file"])
            if not p.is_absolute():
                p = root / p
            if not p.exists():
                raise FileNotFoundError(f"missing frame: {p}")
            rows_by_env.setdefault(r["environment_id"], []).append({
                "image_path": str(p),
                "binary_label": r["source_label"],
                "binary_idx": 1 if r["source_label"] == "normal" else 0,
                "environment_key": env["environment_key"],
                "failure_type": env["metadata"].get("failure_type", "none"),
            })
    return rows_by_env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    rows_by_env = split_rows()
    all_rows = [r for rr in rows_by_env.values() for r in rr]
    inf = infer(Path(args.checkpoint), all_rows, 224, 32, 0.5)
    y_true = np.array(inf["true_idx"]); y_pred = np.array(inf["preds"])

    result = {
        "schema": "trier-holdout-eval",
        "schema_version": 1,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "threshold": 0.5,
        "n_environments": len(rows_by_env),
        "aggregate": {
            "frames": len(all_rows),
            "failing_positive": _binary_metrics(y_true, y_pred, FAIL_IDX),
            "normal_positive": _binary_metrics(y_true, y_pred, 1),
        },
        "by_environment": {},
    }
    i = 0
    for env_id, rows in sorted(rows_by_env.items(), key=lambda kv: kv[1][0]["environment_key"]):
        n = len(rows)
        idx = slice(i, i + n); i += n
        m = _binary_metrics(y_true[idx], y_pred[idx], FAIL_IDX)
        ft = rows[0]["failure_type"]
        is_failure = ft != "none"
        result["by_environment"][rows[0]["environment_key"]] = {
            "split": "external_holdout",
            "failure_type": ft,
            "kind": "failure" if is_failure else "normal",
            "frames": n,
            "metrics_failing_positive": m,
            "predicted_failing": int(np.sum(y_pred[idx] == FAIL_IDX)),
            "fpr_or_recall_note": (
                "FPR (all-normal)" if not is_failure
                else "recall (all-failing)"),
        }

    out = json.dumps(result, indent=2)
    print(out)
    if args.output_json:
        Path(args.output_json).write_text(out + "\n")
        print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    main()