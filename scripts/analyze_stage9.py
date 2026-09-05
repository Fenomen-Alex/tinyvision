#!/usr/bin/env python3
"""Stage 9: data sufficiency and acquisition design analysis.

Usage:
  python scripts/analyze_stage9.py \
      --output-json docs/research/stage9-data-sufficiency.json \
      --output-md docs/research/stage9-data-sufficiency.md

Analysis only: no training, no data modification, no hardware.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

from tv.stage9 import run_stage9  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-json",
        default=str(REPO / "docs/research/stage9-data-sufficiency.json"),
    )
    ap.add_argument(
        "--output-md",
        default=str(REPO / "docs/research/stage9-data-sufficiency.md"),
    )
    args = ap.parse_args()

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)

    print("Stage 9 — Data Sufficiency & Acquisition Design")
    print(f"  json: {out_json}")
    print(f"  md:   {out_md}")

    result = run_stage9()

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
    L.append("# Stage 9 — Data Sufficiency & Acquisition Design")
    L.append("")
    L.append("**Analysis only.** No training, no data modification, no checkpoint changes.")
    L.append("This report answers: *What independent data is actually missing, why do we")
    L.append("believe it is missing, and how should we acquire it without contaminating")
    L.append("the evaluation?*")
    L.append("")

    # ---- Dataset inventory summary ----
    L.append("## 1. Dataset inventory summary")
    L.append("")
    L.append("| Split | Recordings | Total frames | Normal frames | Failing frames |")
    L.append("|---|---|---|---|---|")
    for split in ("train", "val", "test"):
        fc = r["frame_concentration"][split]
        n_rec = fc["n_recordings"]
        total = fc["total_frames"]
        # Get normal/failing from split_distribution_comparison
        sd = r["split_distribution_comparison"][split]
        L.append(f"| {split} | {n_rec} | {total} | {sd['normal_frames']} | {sd['failing_frames']} |")
    L.append("")

    # ---- Frame concentration ----
    L.append("## 2. Frame-vs-recording concentration")
    L.append("")
    for split in ("train", "val", "test"):
        fc = r["frame_concentration"][split]
        d = fc["frames_per_recording"]
        L.append(f"### {split}")
        L.append(f"- recordings: {fc['n_recordings']}, frames: {fc['total_frames']}")
        L.append(f"- frames/recording: min={d['min']}, max={d['max']}, "
                 f"median={d['median']}, mean={d['mean']}")
        L.append(f"- top 5 recordings: {fc['top5_frame_fraction']:.1%} of all frames")
        for t in fc["top5_recordings"]:
            L.append(f"  - {t['recording']}: {t['frames']} frames ({t['composition']})")
        L.append("")

    # ---- Failure-type diversity ----
    L.append("## 3. Failure-type diversity")
    L.append("")
    L.append("| Failure type | Train rec | Train frames | Val rec | Val frames | Test rec | Test frames |")
    L.append("|---|---|---|---|---|---|---|")
    for ftype in sorted(r["failure_type_diversity"].keys()):
        info = r["failure_type_diversity"][ftype]
        tr = info.get("train", {})
        vr = info.get("val", {})
        te = info.get("test", {})
        L.append(f"| {ftype} | {tr.get('n_recordings', 0)} | {tr.get('frames', 0)} | "
                 f"{vr.get('n_recordings', 0)} | {vr.get('frames', 0)} | "
                 f"{te.get('n_recordings', 0)} | {te.get('frames', 0)} |")
    L.append("")

    # ---- Stringing investigation ----
    L.append("## 4. Stringing investigation")
    L.append("")
    si = r["stringing_investigation"]
    L.append(f"- **Training recordings:** {si['n_recordings']}")
    L.append(f"- **Training frames:** {si['total_stringing_frames']}")
    d = si["frames_per_recording"]
    L.append(f"- Frames/recording: min={d['min']}, max={d['max']}, "
             f"median={d['median']}, mean={d['mean']}")
    L.append(f"- Tiny sessions (<10 frames): {si['n_tiny_sessions_lt10_frames']} — "
             f"`{', '.join(si['tiny_session_ids'])}`")
    L.append(f"- Small sessions (10-50 frames): {si['n_small_sessions_10_50_frames']}")
    L.append(f"- Large sessions (>=50 frames): {len(si['large_session_ids'])}")
    L.append(f"- Date distribution: {si['date_distribution']}")
    L.append(f"- Composition breakdown: {si['composition_breakdown']}")
    L.append("")
    obs = si["stage8_observation"]
    L.append("**Stage 8 observation (evaluation context, not a training-selection signal):**")
    L.append(f"- Recording `{obs['catastrophic_recording']}` is a PURE_FAILING stringing "
             f"recording with 127 frames.")
    L.append(f"- FNR at threshold 0.65: **{obs['fnr']:.3f}** ({int(obs['fnr']*127)}/127 frames missed).")
    L.append(f"- The frozen model produces near-zero fail scores for the majority of its frames.")
    L.append("")

    # ---- Normal-domain investigation ----
    L.append("## 5. Normal-domain diversity")
    L.append("")
    nd = r["normal_domain_investigation"]
    tn = nd["training_normal"]
    L.append("### Training normal distribution")
    L.append(f"- Pure normal recordings: {tn['n_pure_normal_recordings']}")
    L.append(f"- Total normal frames: {tn['total_normal_frames']}")
    d = tn["frames_per_pure_normal_recording"]
    L.append(f"- Frames/recording: min={d['min']}, max={d['max']}, median={d['median']}")
    L.append(f"- Image brightness (sampled): mean={tn['brightness_mean']}, "
             f"stdev={tn['brightness_stdev']}")
    L.append(f"- Image contrast (sampled): mean={tn['contrast_mean']}")
    L.append("")
    fp = nd["problematic_test_recording"]
    L.append("### Problematic test recording (Stage 8 observation)")
    L.append(f"- Recording: `{fp['recording_id']}` ({fp['composition']}, {fp['n_frames']} frames)")
    L.append(f"- Image brightness: mean={fp['brightness_mean']}, stdev={fp['brightness_stdev']}")
    L.append(f"- Image contrast: mean={fp['contrast_mean']}")
    L.append(f"- **Note:** brightness ~76 vs training mean ~156 suggests substantially "
             f"different lighting. This is a proxy observation, not proof of semantic OOD.")
    L.append("")

    # ---- Split distribution comparison ----
    L.append("## 6. Train/val/test distribution comparison")
    L.append("")
    L.append("| Split | Recordings | Total | Normal% | under_ext rec | stringing rec | spaghetti rec |")
    L.append("|---|---|---|---|---|---|---|")
    for split in ("train", "val", "test"):
        sd = r["split_distribution_comparison"][split]
        ue = sd["failure_type_recordings"].get("under_extrusion", 0)
        st = sd["failure_type_recordings"].get("stringing", 0)
        sp = sd["failure_type_recordings"].get("spaghetti", 0)
        L.append(f"| {split} | {sd['n_recordings']} | {sd['total_frames']} | "
                 f"{sd['normal_fraction']:.1%} | {ue} | {st} | {sp} |")
    L.append("")
    L.append("**Observations:**")
    L.append("- Under_extrusion is best represented across all splits (37/12/11 recordings).")
    L.append("- Stringing has reasonable recording count (23) but date concentration is high.")
    L.append("- Spaghetti is severely underrepresented in all splits (5/3/? training sessions).")
    L.append("")

    # ---- Data quality ----
    L.append("## 7. Data quality / label structure")
    L.append("")
    dq = r["data_quality"]
    L.append(f"- Total concerns flagged: {dq['n_concerns']}")
    if dq["concerns"]:
        L.append("")
        for c in dq["concerns"][:10]:
            L.append(f"  - [{c['type']}] {c['split']}/{c['recording']}: {c.get('note', '')}")
    L.append("")

    # ---- Acquisition design ----
    L.append("## 8. Recommended targeted data acquisition")
    L.append("")
    acq = r["acquisition_design"]
    for priority in ["priority_1_stringing", "priority_2_normal_domain",
                     "priority_3_spaghetti", "priority_4_under_extrusion"]:
        p = acq[priority]
        L.append(f"### {p['failure_type'].replace('_', ' ').title()} "
                 f"(Priority {priority.split('_')[1]})")
        L.append(f"**Rationale:** {p['rationale']}")
        if "minimum_exploratory" in p:
            me = p["minimum_exploratory"]
            L.append(f"- Minimum exploratory: {me.get('n_sessions', '')} / "
                     f"{me.get('n_frames_per_session', '')}")
            if "rationale" in me:
                L.append(f"  - {me['rationale']}")
        if "stronger_target" in p:
            st = p["stronger_target"]
            L.append(f"- Stronger target: {st.get('n_sessions', '')} / "
                     f"{st.get('n_frames_per_session', '')}")
        if "ideal_target" in p:
            it = p["ideal_target"]
            L.append(f"- Ideal target: {it.get('n_sessions', '')}")
        if "diversity_requirements" in p:
            L.append("- Diversity requirements:")
            for req in p["diversity_requirements"]:
                L.append(f"  - {req}")
        if "held_out_condition" in p:
            L.append(f"- Held-out: {p['held_out_condition']}")
        L.append("")

    L.append("### General principles")
    for principle in acq["general_principles"]:
        L.append(f"- {principle}")
    L.append("")

    # ---- Future experiment ----
    L.append("## 9. Proposed future retraining experiment")
    L.append("")
    fe = r["future_experiment"]
    L.append(f"**Objective:** {fe['objective']}")
    L.append("")
    L.append(f"**Baseline:** {fe['baseline']['model']}, checkpoint `{fe['baseline']['checkpoint']}`")
    L.append(f"- Test F1 at 0.50: {fe['baseline']['test_performance']['F1_at_0_50']}")
    L.append(f"- Test F1 at 0.65: {fe['baseline']['test_performance']['F1_at_0_65']}")
    L.append(f"- Test FNR at 0.65: {fe['baseline']['test_performance']['FNR_at_0_65']}")
    L.append("")
    L.append(f"**Old test set:** {fe['old_test_set']['policy']}")
    L.append(f"- {fe['old_test_set']['n_recordings']} recordings, "
             f"{fe['old_test_set']['n_frames']} frames")
    L.append("")
    L.append(f"**New data split:** {fe['new_data_split_policy']['unit']}")
    L.append(f"- {fe['new_data_split_policy']['split_ratio']}")
    L.append(f"- {fe['new_data_split_policy']['held_out_set']}")
    L.append("")
    L.append("**Primary metrics:**")
    for m in fe["primary_metrics"]:
        L.append(f"- {m}")
    L.append("")
    L.append("**Success criteria:**")
    sc = fe["success_criteria"]
    L.append(f"- Minimum: {sc['minimum']}")
    L.append(f"- Meaningful: {sc['meaningful']}")
    L.append(f"- Strong: {sc['strong']}")
    L.append("")

    # ---- Retraining gate ----
    L.append("## 10. Retraining gate")
    L.append("")
    rg = r["retraining_gate"]
    L.append(f"**A. Is the evidence sufficient to retrain immediately?** "
             f"{'Yes' if rg['A_is_evidence_sufficient_for_immediate_retraining'] else 'No'}")
    L.append("")
    L.append(f"**B. What evidence/data is missing?**")
    L.append(f"{rg['B_what_evidence_or_data_is_missing']}")
    L.append("")
    L.append(f"**C. What data acquisition unblocks retraining?**")
    L.append(f"{rg['C_what_data_acquisition_unblocks_retraining']}")
    L.append("")
    L.append(f"**D. What constitutes a successful retraining result?**")
    L.append(f"{rg['D_what_would_constitute_success']}")
    L.append("")
    L.append("**Priority order:**")
    for item in rg["priority_order"]:
        L.append(f"- {item}")
    L.append("")

    # ---- Limitations ----
    L.append("## 11. Limitations & methodological caveats")
    L.append("")
    for lim in r["limitations"]:
        L.append(f"- {lim}")
    L.append("")

    # ---- Artifacts ----
    L.append("## Artifacts")
    L.append("")
    L.append("| File | Purpose |")
    L.append("|---|---|")
    L.append("| `docs/research/stage9-data-sufficiency.json` | Machine-readable results |")
    L.append("| `docs/research/stage9-data-sufficiency.md` | This report |")
    L.append("| `training/tv/stage9.py` | Analysis module |")
    L.append("| `training/tests/test_stage9.py` | Tests for the analysis logic |")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())