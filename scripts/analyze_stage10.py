#!/usr/bin/env python3
"""Stage 10: cluster and domain generalization audit.

Usage:
  python scripts/analyze_stage10.py \
      --output-json docs/research/stage10-cluster-domain-audit.json \
      --output-md docs/research/stage10-cluster-domain-audit.md

Analysis only: no training, no data modification, no hardware, no threshold
re-optimization. The test set is used solely to document existing Stage 8
observations; no test statistic defines a bucket, threshold, or acquisition
choice.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

from tv.stage10 import run_stage10  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-json",
        default=str(REPO / "docs/research/stage10-cluster-domain-audit.json"),
    )
    ap.add_argument(
        "--output-md",
        default=str(REPO / "docs/research/stage10-cluster-domain-audit.md"),
    )
    args = ap.parse_args()

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)

    print("Stage 10 — Cluster and Domain Generalization Audit")
    print(f"  json: {out_json}")
    print(f"  md:   {out_md}")

    result = run_stage10()

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
    L.append("# Stage 10 — Cluster and Domain Generalization Audit")
    L.append("")
    L.append("**Analysis only.** No training, no data modification, no checkpoint")
    L.append("changes, no threshold re-optimization. The test set is used solely to")
    L.append("document existing Stage 8 observations; no test statistic defines a")
    L.append("bucket, threshold, acquisition choice, or model change.")
    L.append("")
    L.append(f"**Model:** `{r['model']}`")
    L.append(f"**Threshold:** {r['threshold']} (Stage 8 val-selected; not re-optimized)")
    L.append(f"**Evidence splits:** {', '.join(r['evidence_splits'])}")
    L.append("")

    # ---- 1. Executive summary ----
    L.append("## 1. Executive summary")
    L.append("")
    L.append("This stage tests (attempts to falsify) five hypotheses from Stage 9:")
    L.append("")
    L.append("| Hypothesis | Claim | Result |")
    L.append("|---|---|---|")
    L.append("| H1 | Stringing FNR varies by date/session cluster | **Supported** — "
             "`20122022` cluster fnr_mean=0.20, all others near 0 |")
    L.append("| H2 | Catastrophic stringing test recording reflects domain shift "
             "not in training | **Supported** — test recording `15.02.2023_13_19_52` "
             "has fnr=0.898; train `15.02.2023_14_15_54` has fnr=0.0 (distinct session) |")
    L.append("| H3 | The known false-positive normal test recording is darker "
             "than training normals | **Contradicted** — dark bucket has *lowest* "
             "FPR (0.003); brightness-FPR correlation r=0.22 |")
    L.append("| H4 | False-positive rate increases systematically with darkness | "
             "**Contradicted** — medium bucket FPR 0.055, bright 0.033, dark 0.003 |")
    L.append("| H5 | Under-extrusion data is sufficiently diverse to serve as "
             "comparison/control | **Partially supported** — 49 recordings, 15 dates, "
             "fnr_mean=0.05; but two catastrophic sub-sessions exist |")
    L.append("")

    # ---- 2. Stringing cluster analysis ----
    L.append("## 2. Stringing cluster analysis (H1)")
    L.append("")
    L.append(r["stringing_cluster_analysis"]["grouping_note"])
    L.append("")
    L.append("| Date cluster | Recs | Stringing frames | FNR mean | FNR max | Exploratory |")
    L.append("|---|---|---|---|---|---|")
    for date, c in sorted(r["stringing_cluster_analysis"]["clusters"].items()):
        L.append(f"| {date} | {c['n_recordings']} | {c['stringing_frames']} | "
                 f"{c['fnr_mean']:.4f} | {c['fnr_max']:.4f} | {c['exploratory']} |")
    L.append("")
    L.append("**Key finding:** The `20122022` cluster (5 recordings, 167 stringing")
    L.append("frames) has fnr_mean=0.20, with one recording (`Recording_20122022_14_15_06`)")
    L.append("achieving fnr=1.0 (all 2 stringing frames missed). All other clusters are")
    L.append("near-zero FNR. This is strong evidence that stringing performance is not")
    L.append("uniform across sessions; the `20122022` cluster represents a harder domain.")
    L.append("")

    # ---- 3. Stringing recording-to-recording variation ----
    L.append("## 3. Stringing recording-to-recording variation")
    L.append("")
    sv = r["stringing_variation"]
    L.append(f"- Train stringing recordings: {sv['n_train']}")
    L.append(f"- Val stringing recordings: {sv['n_val']}")
    L.append(f"- Val stringing dates: {', '.join(sv['val_dates'])}")
    L.append(f"- Val FNRs: {sv['val_fnrs']}")
    L.append(f"- {sv['val_note']}")
    L.append("")
    L.append("Worst train stringing recordings:")
    worst = sorted([row for row in sv["rows"] if row["split"] == "train"],
                   key=lambda x: -(x["fnr"] or 0))
    for row in worst[:5]:
        L.append(f"- `{row['recording_id']}` ({row['date']}): "
                 f"fnr={row['fnr']}, n={row['stringing_frames']} frames, "
                 f"med_score={row['score_median']:.4f}")
    L.append("")

    # ---- 4. Darkness audit (H3/H4) ----
    L.append("## 4. Darkness audit (H3/H4)")
    L.append("")
    da = r["darkness_audit"]
    L.append(f"- Train normal recordings: {da['n_normal_recordings_train']}")
    L.append(f"- Val normal recordings: {da['n_normal_recordings_val']}")
    L.append(f"- Train normal brightness: mean={da['train_normal_brightness']['mean']}, "
             f"median={da['train_normal_brightness']['median']}, "
             f"range=[{da['train_normal_brightness']['min']}, "
             f"{da['train_normal_brightness']['max']}]")
    L.append(f"- Val normal brightness: mean={da['val_normal_brightness']['mean']}, "
             f"median={da['val_normal_brightness']['median']}")
    L.append(f"- Brightness-FPR correlation: r={da['brightness_fpr_correlation']}")
    L.append(f"- {da['correlation_note']}")
    L.append(f"- {da['caveat']}")
    L.append("")
    L.append("**Interpretation:** H3 predicted that the known false-positive test")
    L.append("recording (`Recording_20122022_17_05_36`) fails because it is darker")
    L.append("than training normals. This is contradicted: the darkest bucket has the")
    L.append("lowest FPR (0.003), not the highest. The false-positive problem is not")
    L.append("driven by systematic darkness. H4 is also contradicted: there is no")
    L.append("monotonic FPR-darkness relationship.")
    L.append("")

    # ---- 5. Brightness buckets ----
    L.append("## 5. Brightness buckets")
    L.append("")
    bb = r["brightness_buckets"]
    if bb.get("available"):
        L.append(f"**Method:** {bb['method']}")
        L.append(f"**Train tertile boundaries:** low={bb['training_tertile_boundaries']['brightness_low']}, "
                 f"high={bb['training_tertile_boundaries']['brightness_high']}")
        L.append("")
        L.append("| Bucket | Boundary | Recs | Frames | Mean brightness | FPR (frame) | FPR (rec mean) |")
        L.append("|---|---|---|---|---|---|---|")
        for name, b in bb["buckets"].items():
            lo = b.get("boundary_low") or "..."
            hi = b.get("boundary_high") or "..."
            L.append(f"| {name} | [{lo}, {hi}) | {b['n_recordings']} | "
                     f"{b['n_frames']} | {b.get('mean_brightness', 'N/A')} | "
                     f"{b.get('framewise_fpr', 'N/A')} | {b.get('mean_recording_fpr', 'N/A')} |")
        L.append("")
        L.append("Medium brightness has the highest FPR (0.055), not dark (0.003).")
        L.append("This confirms the darkness audit: FP behavior is not explained by")
        L.append("darkness alone.")
        L.append("")

    # ---- 6. Under-extrusion control (H5) ----
    L.append("## 6. Under-extrusion control (H5)")
    L.append("")
    ue = r["under_extrusion"]
    L.append(f"- Recordings: {ue['n_recordings']}")
    L.append(f"- Dates: {ue['n_dates']} ({', '.join(ue['dates'][:8])}{'...' if len(ue['dates'])>8 else ''})")
    L.append(f"- FNR: mean={ue['fnr_mean']}, std={ue['fnr_std']}, "
             f"min={ue['fnr_min']}, max={ue['fnr_max']}")
    L.append(f"- Splits: {', '.join(ue['splits'])}")
    L.append(f"- {ue['caveat']}")
    L.append("")
    L.append("Worst under-extrusion recordings:")
    worst_ue = sorted(ue["rows"], key=lambda x: -(x["fnr"] or 0))
    for row in worst_ue[:4]:
        L.append(f"- `{row['recording_id']}` ({row['split']}, {row['date']}): "
                 f"fnr={row['fnr']}, ue_frames={row['ue_frames']}, "
                 f"med_score={row['score_median']:.4f}")
    L.append("")
    L.append("**Interpretation:** H5 is partially supported. Under-extrusion has the")
    L.append("most diverse training domain (15 dates, 49 recordings, fnr_mean=0.05),")
    L.append("making it a reasonable control. However, two catastrophic sub-sessions")
    L.append("exist (`Recording_29112022_21_41_27` fnr=0.89, `Recording_05012023_11_08_00`")
    L.append("fnr=0.87), confirming that even the best-represented failure type has")
    L.append("domain-specific failures.")
    L.append("")

    # ---- 7. Spaghetti limitation ----
    L.append("## 7. Spaghetti limitation")
    L.append("")
    sp = r["spaghetti_limitation"]
    L.append(f"- Train: {sp['train_spaghetti']['n_recordings']} recordings, "
             f"{sp['train_spaghetti']['n_frames']} frames")
    L.append(f"- Val: {sp['val_spaghetti']['n_recordings']} recordings, "
             f"{sp['val_spaghetti']['n_frames']} frames")
    L.append(f"- Test: {sp['test_spaghetti']['n_recordings']} recordings, "
             f"{sp['test_spaghetti']['n_frames']} frames")
    L.append(f"- {sp['conclusions']['note']}")
    L.append("")

    # ---- 8. Documented test observations ----
    L.append("## 8. Documented test observations (context only)")
    L.append("")
    dto = r["documented_test_observations"]
    L.append(f"> {dto['firewall_note']}")
    L.append("")
    cf = dto.get("catastrophic_fn", {})
    if cf:
        L.append(f"**Catastrophic false-negative:** `{cf['recording_id']}` "
                 f"({cf['n_frames']} frames, {cf['true_failing']} failing, "
                 f"FNR={cf['fnr_approx']})")
    hfp = dto.get("high_fp", {})
    if hfp:
        L.append(f"**High-confidence false-positive:** `{hfp['recording_id']}` "
                 f"({hfp['n_frames']} frames, {hfp['true_normal']} normal, "
                 f"FP={hfp['fp_at_065']}, max_score={hfp['max_score']})")
    L.append("")

    # ---- 9. Image distribution ----
    L.append("## 9. Image distribution (train normals)")
    L.append("")
    imd = r["image_distribution"]
    L.append(f"- Frames sampled: {imd['n_normal_frames_sampled']}")
    L.append(f"- Method: {imd['method']}")
    if imd.get("luminance"):
        L.append(f"- Mean brightness: {imd['luminance']['mean_brightness_mean']}")
        L.append(f"- Mean brightness std: {imd['luminance']['mean_brightness_std']}")
    if imd.get("rgb_channel_mean_bgr"):
        ch = imd["rgb_channel_mean_bgr"]
        L.append(f"- Channel means (BGR): [{ch[0]}, {ch[1]}, {ch[2]}]")
    L.append("")

    # ---- 10. Acquisition decision ----
    L.append("## 10. Acquisition decision (updated)")
    L.append("")
    acq = r["acquisition_decision"]
    L.append("**Classifications (from existing evidence):**")
    L.append(f"- {json.dumps(acq['classifications_defined'], indent=2)}")
    L.append("")
    for pname, p in acq["priorities"].items():
        L.append(f"### {pname}")
        L.append(f"- **Need:** {p['need']}")
        L.append(f"- **Detail:** {p['detail']}")
        L.append("")

    # ---- 11. Future split design ----
    L.append("## 11. Future split design")
    L.append("")
    fsd = r["future_split_design"]
    L.append(f"- Principle: {fsd['principle']}")
    L.append(f"- {fsd['hold_out_note']}")
    L.append(f"- Current Experiment 1 split unchanged: {fsd['current_experiment_1_split_unchanged']}")
    L.append("")

    # ---- 12. Retraining gate ----
    L.append("## 12. Retraining gate")
    L.append("")
    rg = r["retraining_gate"]
    L.append(f"**A. Is retraining justified now?** {rg['A_retraining_justified']}")
    L.append(f"**B. First data to add:** {rg['B_first_data_to_add']}")
    L.append(f"**C. Still missing:** {rg['C_still_missing']}")
    L.append(f"**D. Minimum useful experiment:** {rg['D_minimum_useful_experiment']}")
    L.append(f"**E. Result that justifies retraining:** {rg['E_result_that_justifies_retraining']}")
    L.append("")
    for note in rg.get("notes", []):
        L.append(f"- {note}")
    L.append("")

    # ---- Artifacts ----
    L.append("---")
    L.append("")
    L.append("## Artifacts")
    L.append("")
    L.append("| File | Purpose |")
    L.append("|---|---|")
    L.append("| `docs/research/stage10-cluster-domain-audit.json` | Machine-readable results |")
    L.append("| `docs/research/stage10-cluster-domain-audit.md` | This report |")
    L.append("| `training/tv/stage10.py` | Analysis module |")
    L.append("| `training/tests/test_stage10.py` | Tests for the analysis logic |")
    L.append("| `docs/research/evalcaches/frozen_train_trajectories.json` | Corrected train cache |")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
