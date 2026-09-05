#!/usr/bin/env python3
"""Stage 11A: leave-cluster-out generalization experiment design runner.

Analysis only. Produces the experiment design document (JSON + MD).
No training, no checkpoint modification, no dataset modification.

Usage:
  python scripts/analyze_stage11a.py \
      --output-json docs/research/stage11a-cluster-generalization-design.json \
      --output-md docs/research/stage11a-cluster-generalization-design.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))


def _render_json(design: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(design, f, indent=2, ensure_ascii=False)
    print(f"Wrote {path} ({path.stat().st_size:,} bytes)")


def _render_md(design: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w(f"# {design['title']}")
    w()
    w(f"**Status:** {design['status']}")
    w(f"**Objective:** {design['objective']}")
    w()
    w("---")
    w()
    w("## Recording Inventory")
    w()
    inv = design["recording_inventory"]
    w("| Split | Recordings | Frames |")
    w("|-------|-----------|--------|")
    for split in ("train", "val", "test"):
        s = inv[split]
        w(f"| {split} | {s['total_recordings']} | {s['total_frames']} |")
    w()
    ts = inv["train"]
    w(f"**Train stringing recordings:** {ts['stringing_recordings']} ({ts['stringing_frames']} frames)")
    w(f"**Train under-extrusion recordings:** {ts['under_extrusion_recordings']} ({ts['under_extrusion_frames']} frames)")
    if inv["test"].get("note"):
        w(f"**Test note:** {inv['test']['note']}")
    w()

    w("---")
    w()
    w("## Stringing Recording Table")
    w()
    w("| Recording ID | Date | Frames | Stringing Frames | Split | Composition |")
    w("|-------------|------|--------|-----------------|-------|-------------|")
    for row in design["stringing_recording_table"]:
        w(f"| {row['recording_id']} | {row['date']} | {row['frame_count']} | {row['stringing_frames']} | {row['split']} | {row['composition']} |")
    w()

    w("---")
    w()
    w("## Stringing Clusters")
    w()
    for cid, c in design["stringing_clusters"].items():
        w(f"### Cluster `{cid}`")
        w()
        w(f"- **Definition:** {c['definition']}")
        w(f"- **Recordings:** {c['recording_count']}")
        w(f"- **Total frames:** {c['total_frames']}")
        w(f"- **Stringing frames:** {c['stringing_frames']}")
        w(f"- **Rationale:** {c['rationale']}")
        if c["exploratory"]:
            w(f"- **Exploratory:** YES — {c['exploratory_reason']}")
        w()

    w("---")
    w()
    w("## Under-Extrusion Control Clusters")
    w()
    for cid, c in design["under_extrusion_clusters"].items():
        w(f"### Cluster `{cid}`")
        w()
        w(f"- **Definition:** {c['definition']}")
        w(f"- **Recordings:** {c['recording_count']}")
        w(f"- **Total frames:** {c['total_frames']}")
        w(f"- **Rationale:** {c['rationale']}")
        if c["exploratory"]:
            w(f"- **Exploratory:** YES — {c['exploratory_reason']}")
        w()

    w("---")
    w()
    w("## Experimental Runs")
    w()
    for rid, run in design["experimental_runs"].items():
        w(f"### Run `{rid}`")
        w()
        w(f"- **Held-out cluster:** `{run['held_out_cluster_id']}`")
        w(f"- **Held-out recordings:** {len(run['held_out_recordings'])}")
        w(f"- **Held-out frames:** {run['held_out_frame_count']}")
        w(f"- **Held-out stringing frames:** {run['held_out_stringing_frames']}")
        w(f"- **Train recordings (held-in):** {run['train_recording_count']}")
        w(f"- **Train frames (held-in):** {run['train_frame_count']}")
        w(f"- **Train stringing frames (held-in):** {run['train_stringing_frame_count']}")
        w(f"- **Val recordings:** {run['val_recording_count']}")
        w(f"- **Val frames:** {run['val_frame_count']}")
        if run["notes"]:
            for note in run["notes"]:
                w(f"- **Note:** {note}")
        w()

    w("---")
    w()
    w("## Leakage Audit Results")
    w()
    for rid, audit in design["leakage_audits"].items():
        status = "PASS" if audit["passed"] else "FAIL"
        w(f"### `{rid}` — {status}")
        w()
        for chk in audit["checks"]:
            check_status = "PASS" if chk["passed"] else "FAIL"
            w(f"- [{check_status}] **{chk['id']}:** {chk['check']} — {chk['detail']}")
        w()

    w("---")
    w()
    w("## Control Condition")
    w()
    cc = design["control_condition"]
    w(f"**Description:** {cc['description']}")
    w()
    w(f"**Methodology:** {cc['methodology']}")
    w()
    w(f"**Selected clusters:** {', '.join(cc['selected_clusters'])}")
    w()
    w(f"**Interpretation:** {cc['interpretation']}")
    w()

    w("---")
    w()
    w("## Baseline")
    w()
    b = design["baseline"]["experiment_1"]
    w("### Experiment 1 Reference")
    w()
    w(f"- Checkpoint: `{b['checkpoint']}`")
    w(f"- Val F1@0.65: **{b['val_F1_at_0_65']}**")
    w(f"- Test F1@0.65: **{b['test_F1_at_0_65']}**")
    w(f"- Test stringing FNR@0.65: **{b['test_stringing_FNR_at_0_65']}**")
    w(f"- Test under-extrusion FNR@0.65: **{b['test_under_extrusion_FNR_at_0_65']}**")
    w(f"- Test normal FPR@0.65: **{b['test_normal_FPR_at_0_65']}**")
    w(f"- Note: {b['note']}")
    w()
    w(f"**Comparison principle:** {design['baseline']['comparison_principle']}")
    w()

    w("---")
    w()
    w("## Training Configuration")
    w()
    tc = design["training_config"]
    w(f"- Model: `{tc['model']}`")
    w(f"- Image size: {tc['image_size']}")
    w(f"- Optimizer: {tc['optimizer']}")
    w(f"- Learning rate: {tc['learning_rate']}")
    w(f"- Weight decay: {tc['weight_decay']}")
    w(f"- Epochs: {tc['epochs']} (early stopping patience: {tc['early_stopping_patience']})")
    w(f"- Batch size: {tc['batch_size']}")
    w(f"- Precision: {tc['precision']}")
    w(f"- Seed: {tc['seed']}")
    aug = tc["augmentation"]
    w(f"- Augmentation: hflip={aug['horizontal_flip']}, rot={aug['random_rotation']}, cj={aug['color_jitter']}, crop={aug['random_crop']}, normalize={aug['normalize']}")
    w(f"- No HP search: {tc['no_hp_search']}")
    w(f"- No architecture search: {tc['no_architecture_search']}")
    w(f"- No augmentation search: {tc['no_augmentation_search']}")
    w()

    w("---")
    w()
    w("## Metrics")
    w()
    m = design["metrics"]
    w("**Primary:**")
    for k, v in m["primary"].items():
        w(f"- {k}: {v}")
    w()
    w("**Secondary:**")
    for k, v in m["secondary"].items():
        w(f"- {k}: {v}")
    w()
    w(f"**Threshold:** {m['threshold']}")
    w(f"**Frame-level:** {m['frame_level']}")
    w(f"**Recording-level:** {m['recording_level']}")
    w()

    w("---")
    w()
    w("## Run Count & Wall Clock")
    w()
    rc = design["run_count"]
    w(f"- Stringing holdout runs: {rc['stringing_holdout_runs']}")
    w(f"- Under-extrusion control runs: {rc['under_extrusion_control_runs']}")
    w(f"- Total: {rc['total']}")
    per = rc["per_run_estimate"]
    w(f"- Per-run estimate: {per['training_time']} training + {per['evaluation_time']} eval = {per['total_per_run']}")
    w(f"- Total wall clock: {rc['total_wall_clock']}")
    w(f"- Total disk: {rc['total_disk']}")
    w()

    w("---")
    w()
    w("## Success & Interpretation Criteria")
    w()
    sc = design["success_criteria"]
    w("### Strong evidence of domain generalization failure")
    w()
    w(sc["strong_evidence_domain_generalization_failure"])
    w()
    w("### Moderate evidence")
    w()
    w(sc["moderate_evidence"])
    w()
    w("### Weak / inconclusive")
    w()
    w(sc["weak_inconclusive"])
    w()
    w("### No support")
    w()
    w(sc["no_support"])
    w()
    w("### Abort criteria")
    w()
    w(sc["abort_criteria"])
    w()

    w("---")
    w()
    w("## Official Test Set Firewall")
    w()
    fw = design["test_firewall"]
    w(f"- **Frozen:** {fw['frozen']}")
    w(f"- **May reference:** {fw['may_reference']}")
    w()
    w("**Must not influence:**")
    for item in fw["must_not_influence"]:
        w(f"- {item}")
    w()

    w("---")
    w()
    w("## Limitations")
    w()
    for lim in design["limitations"]:
        w(f"1. {lim}")
    w()

    w("---")
    w()
    w("## Stage 11B Execution Plan")
    w()
    ep = design["execution_plan"]
    w(f"**Clusters held out:** {', '.join(ep['clusters_held_out'])}")
    w(f"**Control clusters:** {', '.join(ep['control_clusters'])}")
    w(f"**Total runs:** {ep['total_runs']}")
    w()
    w("**Steps:**")
    for step in ep["steps"]:
        w(f"{step}")
    w()
    w("**Abort conditions:**")
    for ac in ep["abort_conditions"]:
        w(f"- {ac}")
    w()
    w("**Expected artifacts:**")
    for art in ep["expected_artifacts"]:
        w(f"- `{art}`")
    w()

    w("---")
    w()
    w(f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} — analysis only, no training executed.*")
    w()

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {path} ({path.stat().st_size:,} bytes, {len(lines)} lines)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-json",
        default=str(REPO / "docs/research/stage11a-cluster-generalization-design.json"),
    )
    ap.add_argument(
        "--output-md",
        default=str(REPO / "docs/research/stage11a-cluster-generalization-design.md"),
    )
    args = ap.parse_args()

    from tv.stage11a import build_design

    print("Building Stage 11A experiment design...")
    design = build_design()
    print(f"  Runs: {len(design['experimental_runs'])}")
    print(f"  Stringing clusters: {len(design['stringing_clusters'])}")
    print(f"  UE control clusters: {len(design['under_extrusion_clusters'])}")

    # Verify all leakage audits pass
    all_pass = all(a["passed"] for a in design["leakage_audits"].values())
    print(f"  Leakage audits all pass: {all_pass}")
    if not all_pass:
        print("ERROR: Not all leakage audits passed!")
        return 1

    _render_json(design, Path(args.output_json))
    _render_md(design, Path(args.output_md))

    print()
    print("Stage 11A design complete.")
    print("No training executed. No checkpoints modified. No dataset files modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
