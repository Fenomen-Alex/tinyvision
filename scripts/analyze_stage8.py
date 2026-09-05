#!/usr/bin/env python3
"""Stage 8: threshold & failure-type analysis on the frozen baseline.

Usage:
  python scripts/analyze_stage8.py \
      --cache-dir docs/research/evalcaches \
      --output-json docs/research/stage8-threshold-failure-analysis.json \
      --output-md docs/research/stage8-threshold-failure-analysis.md

Analysis only: no retraining, no data collection, no hardware, test touched
exactly once (validation-selected threshold).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

from tv.stage8 import run_stage8  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=str(REPO / "docs/research/evalcaches"))
    ap.add_argument(
        "--output-json",
        default=str(REPO / "docs/research/stage8-threshold-failure-analysis.json"),
    )
    ap.add_argument(
        "--output-md",
        default=str(REPO / "docs/research/stage8-threshold-failure-analysis.md"),
    )
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)

    print("Stage 8 — Threshold & failure-type analysis")
    print(f"  cache: {cache_dir}")
    print(f"  json:  {out_json}")
    print(f"  md:    {out_md}")

    result = run_stage8(cache_dir)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {out_json.name} ({out_json.stat().st_size} bytes)")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(result))
    print(f"Wrote {out_md.name} ({out_md.stat().st_size} bytes)")

    print("\nDone.")
    return 0


def render_markdown(r: dict) -> str:
    L: list[str] = []
    L.append("# Stage 8 — Threshold and Failure-Type Analysis")
    L.append("")
    L.append(f"**Checkpoint:** `{r['checkpoint']}` (frozen, unmodified)")
    L.append("")
    L.append("Analysis only. No retraining, no new data, no hardware. The test set is")
    L.append("evaluated exactly once, using the threshold selected on validation.")
    L.append("")
    L.append("## 1. Validation threshold sweep")
    L.append("")
    L.append("Positives = failing frames. Metrics at each operating point on the")
    L.append("validation split (never test).")
    L.append("")
    L.append("| Threshold | Acc | Prec | Recall | F1 | FNR | FPR |")
    L.append("|---|---|---|---|---|---|---|")
    for row in r["thresholds_evaluated"]:
        L.append(
            f"| {row['threshold']:.2f} | {row['accuracy']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['fnr']:.4f} | {row['fpr']:.4f} |"
        )
    L.append("")
    v = r["validation"]
    L.append(
        f"**Validation-optimal threshold (F1):** `{v['selected_threshold']:.2f}`"
    )
    L.append(f"- F1={v['selected_metrics']['f1']:.4f}, "
             f"Prec={v['selected_metrics']['precision']:.4f}, "
             f"Rec={v['selected_metrics']['recall']:.4f}")
    L.append(f"- FNR={v['selected_metrics']['fnr']:.4f}, "
             f"FPR={v['selected_metrics']['fpr']:.4f}, "
             f"Acc={v['selected_metrics']['accuracy']:.4f}")
    L.append("")
    L.append("### Precision/recall trade-off")
    L.append("")
    L.append("Lower thresholds favour recall (fewer missed failures) at the cost of")
    L.append("precision (more false alarms); higher thresholds do the opposite. The")
    L.append("F1 optimum balances these two. The full precision/recall curve is")
    L.append("reported in the `.json` artifact (`thresholds_evaluated`).")
    L.append("")
    L.append("## 2. Frozen test evaluation")
    L.append("")
    L.append("| Operating point | Acc | Prec | Recall | F1 | FNR | FPR |")
    L.append("|---|---|---|---|---|---|---|")
    for label, t in (("0.50 baseline", r["test"]["baseline_0_50"]),
                     (f"selected {r['test']['selected_threshold']:.2f}",
                      r["test"]["selected_metrics"])):
        L.append(
            f"| {label} | {t['accuracy']:.4f} | {t['precision']:.4f} | "
            f"{t['recall']:.4f} | {t['f1']:.4f} | {t['fnr']:.4f} | {t['fpr']:.4f} |"
        )
    L.append("")
    L.append("## 3. Failure-type breakdown (test, selected threshold)")
    L.append("")
    L.append("Frame-level recall/precision for each fine class. Groups with few")
    L.append("samples are flagged; do not over-interpret.")
    L.append("")
    L.append("| Failure type | Frames | Failing true | Recall | Prec | F1 | FNR | FPR |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ft in r["failure_type_breakdown"].values():
        L.append(
            f"| {ft['failure_type']} | {ft['n_frames']} | {ft['n_failing_true']} | "
            f"{ft['recall']:.3f} | {ft['precision']:.3f} | "
            f"{ft['f1']:.3f} | {ft['fnr']:.3f} | {ft['fpr']:.3f} |"
        )
    L.append("")
    L.append("## 4. Recording-level error ranking (diagnostic)")
    L.append("")
    rr = r["recording_ranking"]
    L.append("**Highest failing FNR:** `" + ", ".join(rr["highest_fnr"][:8]) + "`")
    L.append("")
    L.append("**Highest FPR on normal frames:** `" + ", ".join(rr["highest_fpr"][:8]) + "`")
    L.append("")
    L.append("**Most false positives:** `" + ", ".join(rr["largest_fp_count"][:8]) + "`")
    L.append("")
    L.append("**Most false negatives:** `" + ", ".join(rr["largest_fn_count"][:8]) + "`")
    L.append("")
    persistent = [k for k, v in r["problematic_recordings"].items() if v["persistent"]]
    L.append("**Persistent problematic (both high FNR and high FP count):**")
    L.append("`" + ("none" if not persistent else ", ".join(persistent)) + "`")
    L.append("")
    L.append("These rankings are diagnostic only. We do not claim recording-level")
    L.append("classification as the task, and we do not invent explanations without")
    L.append("visual/metadata evidence.")
    L.append("")
    L.append("## 5. Hard examples (test, selected threshold)")
    L.append("")
    for idx, cat in enumerate(("false_negatives", "false_positives", "borderline_errors"),
                              start=1):
        title = {"false_negatives": "Highest-confidence false negatives",
                 "false_positives": "Highest-confidence false positives",
                 "borderline_errors": "Borderline errors near the selected threshold"}[cat]
        L.append(f"### 5.{idx} {title}")
        L.append("")
        L.append("| Recording | Frame idx | True | Pred | p(fail) | Failure |")
        L.append("|---|---|---|---|---|---|")
        for ex in r["hard_examples"][cat]:
            L.append(
                f"| {ex['recording']} | {ex['frame_index']} | "
                f"{ex['source_class']} | {ex['pred_class']} | "
                f"{ex['probability']:.4f} | {ex['failure_type']} |"
            )
        L.append("")
    L.append("## 6. Calibration (validation)")
    L.append("")
    c = r["calibration"]
    L.append(f"- **Brier score:** `{c['brier_score']:.4f}` (0 = perfect; "
             f"baseline for a balanced two-class oracle ~0.25)")
    L.append(f"- **ECE:** `{c['ece']:.4f}` (10-bin, weighted by bin occupancy)")
    L.append(
        f"- **Systematic:** overconfident-failing=`{c['systematic_biases']['overconfident_failing']}`, "
        f"underconfident-failing=`{c['systematic_biases']['underconfident_failing']}`"
    )
    L.append("")
    L.append("Bins:")
    L.append("")
    L.append("| Bin | n | mean conf | observed freq | gap |")
    L.append("|---|---|---|---|---|")
    for b in c["bins"]:
        L.append(
            f"| {b['bin']} | {b['n_samples']} | "
            f"{b['mean_confidence']:.3f} | {b['observed_frequency']:.3f} | "
            f"{b['gap']:.3f} |"
        )
    L.append("")
    L.append("## 7. Limitations & recommendation")
    L.append("")
    L.append("| Limitation | Note |")
    L.append("|---|---|")
    for lim in r["limitations"]:
        L.append(f"| {lim} | |")
    L.append("")
    L.append("**Note on calibration vs threshold selection:** if calibration is poor,")
    L.append("hard probability thresholds are less meaningful as calibrated confidences;")
    L.append("they still operate as an effective ranking. No calibration transform was")
    L.append("fitted in this analysis stage; if used later it must be fitted only on")
    L.append("validation and evaluated once on test, to avoid leakage.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## Artifacts")
    L.append("")
    L.append("- `docs/research/stage8-threshold-failure-analysis.json` — machine-readable full results")
    L.append("- `docs/research/stage8-threshold-failure-analysis.md` — this report")
    L.append("- `training/tv/stage8.py` — analysis module")
    L.append("- `training/tests/test_stage8.py` — tests for the new logic")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())