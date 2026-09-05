#!/usr/bin/env python3
"""Stage 11: retraining experiment design and runtime estimate.

Analysis only. Produces the experiment design document, runtime estimates,
and configuration template. No training is executed.

Usage:
  python scripts/analyze_stage11.py \
      --output-json docs/research/stage11-retrain-experiment.json \
      --output-md docs/research/stage11-retrain-experiment.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-json",
        default=str(REPO / "docs/research/stage11-retrain-experiment.json"),
    )
    ap.add_argument(
        "--output-md",
        default=str(REPO / "docs/research/stage11-retrain-experiment.md"),
    )
    args = ap.parse_args()

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)

    print("Stage 11 — Retraining Experiment Design")
    print(f"  json: {out_json}")
    print(f"  md:   {out_md}")

    result = build_experiment_design()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {out_json.name} ({out_json.stat().st_size} bytes)")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(result))
    print(f"Wrote {out_md.name} ({out_md.stat().st_size} bytes)")

    print("\nDone.")
    return 0


def build_experiment_design() -> dict:
    return {
        "stage": 11,
        "title": "Retraining Experiment Design",
        "objective": (
            "Test whether adding targeted independent sessions improves "
            "generalization, measured on the frozen Experiment 1 test set "
            "and a new held-out evaluation set."
        ),
        "status": "design_only",
        "no_training": True,

        # --- Baseline ---
        "baseline": {
            "experiment_id": "shufflenet_v2_x1_0_printing-errors_cdc9e2cad4",
            "model": "shufflenet_v2_x1_0",
            "checkpoint": "experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/best.pt",
            "frozen_test": {
                "F1_at_0_50": 0.8763,
                "F1_at_0_65": 0.8799,
                "FNR_at_0_65": 0.1787,
                "FPR_at_0_65": 0.1025,
                "stringing_FNR_at_0_65": 0.213,
                "under_extrusion_FNR_at_0_65": 0.155,
                "normal_FPR_at_0_65": 0.103,
            },
            "train_pool": {
                "recordings": 105,
                "frames": 11254,
                "stringing_recordings": 23,
                "under_extrusion_recordings": 37,
                "spaghetti_recordings": 5,
                "normal_recordings": 40,
            },
        },

        # --- Stage 10 evidence driving this design ---
        "evidence_summary": {
            "H1_stringing_clustered": (
                "20122022 date-cluster has fnr_mean=0.20 vs near-zero elsewhere. "
                "Stringing performance is not uniform across sessions."
            ),
            "H2_catastrophic_test_ood": (
                "Test recording 15.02.2023_13_19_52 (fnr=0.898) is from a distinct "
                "session than any training stringing recording on that date."
            ),
            "H3_dark_not_fp_driver": (
                "Contradicted. Dark bucket has lowest FPR (0.003). "
                "Brightness-FPR correlation r=0.22 (weak positive)."
            ),
            "H4_no_darkness_fnr_gradient": (
                "Contradicted. Medium bucket FPR 0.055, bright 0.033, dark 0.003."
            ),
            "H5_under_extrusion_control_ok": (
                "Partially supported. 49 recordings/15 dates/fnr_mean=0.05, "
                "but two catastrophic sub-sessions exist."
            ),
        },

        # --- Data requirements ---
        "data_requirements": {
            "principle": (
                "INDEPENDENT SESSIONS, not raw frame count. Each session = "
                "a distinct print job, ideally a different date. Session atomicity "
                "preserved: never split frames of one session across sets."
            ),
            "priorities": [
                {
                    "priority": 1,
                    "type": "stringing",
                    "rationale": (
                        "Primary failure mode. Clustered by date/session. "
                        "Stage 8 stringing FNR=0.213 on frozen test."
                    ),
                    "minimum": {
                        "sessions": "5-10",
                        "frames_per_session": "50-200",
                        "total_frames": "250-2000",
                    },
                    "strong": {
                        "sessions": "15-20",
                        "frames_per_session": "80-200",
                        "total_frames": "1200-4000",
                    },
                    "diversity": [
                        "Distinct print objects (if possible)",
                        "Different lighting conditions",
                        "Different print stages where stringing occurs",
                        "Mix of severe and mild stringing",
                        "Dates not already in training set",
                    ],
                    "held_out": (
                        "Hold out 1-2 sessions from a distinct date range or object "
                        "as the new held-out evaluation set."
                    ),
                },
                {
                    "priority": 2,
                    "type": "normal",
                    "rationale": (
                        "Dark-normal FPR is currently low (0.003) but coverage is "
                        "narrow. The known FP test recording (brightness ~76) may "
                        "represent an uncovered condition."
                    ),
                    "minimum": {
                        "sessions": "5-10",
                        "frames_per_session": "50-150",
                        "total_frames": "250-1500",
                    },
                    "strong": {
                        "sessions": "10-15",
                        "frames_per_session": "80-150",
                        "total_frames": "800-2250",
                    },
                    "diversity": [
                        "Varied ambient lighting (bright, dim, mixed)",
                        "Different objects in frame",
                        "Different camera orientations",
                    ],
                },
                {
                    "priority": 3,
                    "type": "spaghetti",
                    "rationale": (
                        "Only 5 training recordings (191 frames), 0 test frames. "
                        "Cannot measure frozen-test spaghetti performance."
                    ),
                    "minimum": {
                        "sessions": "5-8",
                        "frames_per_session": "50-150",
                        "total_frames": "250-1200",
                    },
                    "strong": {
                        "sessions": "10-15",
                        "frames_per_session": "80-150",
                        "total_frames": "800-2250",
                    },
                },
                {
                    "priority": 4,
                    "type": "under_extrusion",
                    "rationale": (
                        "Already well-represented (37 train recordings). "
                        "Maintain or modestly extend."
                    ),
                    "minimum": {
                        "sessions": "3-5",
                        "frames_per_session": "50-150",
                        "total_frames": "150-750",
                    },
                },
            ],
            "labeling_requirements": (
                "Same binary normal/failing + fine failure type as existing. "
                "Metadata: date, object, lighting condition, camera settings."
            ),
        },

        # --- Split strategy ---
        "split_strategy": {
            "unit": "session (recording), never frame",
            "new_data_ratio": "60% train / 20% val / 20% held-out",
            "held_out_set": (
                "Sessions from new data held out at acquisition time. "
                "This becomes the second evaluation set to measure improvement."
            ),
            "frozen_test_set": (
                "The existing Experiment 1 test set remains FROZEN. "
                "Used only for final comparison."
            ),
            "existing_data": (
                "All existing train/val recordings remain in their current splits. "
                "New data is ADDED to the existing train/val pools."
            ),
            "directory_structure": (
                "New sessions follow the existing layout:\n"
                "  datasets/processed/printing-errors/{failing,normal}/"
                "Recording_<date>_<time>/frame.png"
            ),
            "csv_update": (
                "New rows appended to train.csv and val.csv. "
                "splits.json updated with new counts. "
                "metadata.csv updated with new fine-class mappings."
            ),
        },

        # --- Training protocol ---
        "training_protocol": {
            "architecture": "shufflenet_v2_x1_0 (same as Experiment 1)",
            "num_classes": 2,
            "image_size": 224,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "epochs": 30,
            "early_stopping_patience": 10,
            "batch_size": 32,
            "precision": "fp32",
            "seed": 42,
            "augmentation": {
                "horizontal_flip": True,
                "random_rotation": 15,
                "color_jitter": 0.2,
                "random_crop": False,
                "normalize": True,
            },
            "hardware": "MPS (Apple M1 Pro) or CPU",
            "expected_runtime": {
                "training_11k_frames": "~16 minutes (968s, measured)",
                "training_15k_frames": "~22 minutes (estimated, linear scaling)",
                "training_20k_frames": "~29 minutes (estimated)",
                "training_25k_frames": "~36 minutes (estimated)",
            },
            "config_template": {
                "experiment_id": "shufflenet_v2_x1_0_printing-errors_stage11_<hash>",
                "model": "shufflenet_v2_x1_0",
                "task": "classification",
                "dataset": {
                    "name": "printing-errors",
                    "version": "1.1",
                    "data_dir": "datasets/processed/printing-errors",
                    "image_size": 224,
                    "source_mode": "directory",
                },
                "data": {
                    "num_workers": 4,
                    "batch_size": 32,
                    "pin_memory": True,
                },
                "train": {
                    "epochs": 30,
                    "optimizer": "adam",
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "augmentation": {
                        "horizontal_flip": True,
                        "random_rotation": 15,
                        "color_jitter": 0.2,
                        "random_crop": False,
                        "normalize": True,
                    },
                    "precision": "fp32",
                    "seed": 42,
                    "early_stopping_patience": 10,
                },
            },
        },

        # --- Evaluation protocol ---
        "evaluation_protocol": {
            "frozen_test_eval": {
                "description": (
                    "Run Stage 8 evaluation pipeline on the frozen Experiment 1 "
                    "test set with the new checkpoint."
                ),
                "metrics": [
                    "F1 at 0.50",
                    "F1 at 0.65",
                    "FNR at 0.65 (overall)",
                    "FPR at 0.65 (overall)",
                    "Stringing FNR at 0.65",
                    "Under-extrusion FNR at 0.65",
                    "Normal FPR at 0.65",
                    "Calibration (Brier, ECE)",
                ],
                "runtime_estimate": "~30s on MPS",
            },
            "held_out_eval": {
                "description": (
                    "Evaluate on the new held-out evaluation set (sessions "
                    "held out from new data acquisitions)."
                ),
                "metrics": "Same as frozen_test_eval",
                "runtime_estimate": "~15s on MPS (smaller set)",
            },
            "train_cache_rebuild": {
                "description": (
                    "Rebuild frozen_train_trajectories.json with the new checkpoint "
                    "for Stage 10-style analysis."
                ),
                "runtime_estimate": "~40s on MPS",
            },
            "stage10_rerun": {
                "description": (
                    "Re-run Stage 10 cluster/domain audit with new train cache "
                    "to verify stringing cluster FNR improved."
                ),
                "runtime_estimate": "~47s on MPS",
            },
            "threshold_sweep": {
                "description": (
                    "Run Stage 8 threshold sweep on val split to confirm 0.65 "
                    "remains optimal or select new threshold."
                ),
                "runtime_estimate": "~10s on MPS",
            },
        },

        # --- Runtime estimates ---
        "runtime_estimates": {
            "note": (
                "All estimates based on measured runtimes on Apple M1 Pro (MPS). "
                "CPU will be ~2-3x slower. Scaling is roughly linear with frame count."
            ),
            "measured baselines": {
                "experiment_1_training": "968s (11254 train frames, 30 epochs, early stop at ~20)",
                "inference_speed": "~282 fps on MPS",
                "train_cache_rebuild": "~40s (11254 frames)",
                "stage10_analysis": "~47s (image I/O on ~50 normal recordings)",
                "stage8_evaluation": "~30s (val+test, 5033 frames)",
            },
            "projected_totals_by_scenario": {
                "minimum_acquisition": {
                    "new_stringing_sessions": "5-10",
                    "new_normal_sessions": "5-10",
                    "estimated_new_frames": "500-2000",
                    "estimated_total_train_frames": "11754-13254",
                    "training_estimate": "17-20 minutes",
                    "evaluation_estimate": "2-3 minutes",
                    "total_pipeline": "20-25 minutes",
                },
                "strong_acquisition": {
                    "new_stringing_sessions": "15-20",
                    "new_normal_sessions": "10-15",
                    "new_spaghetti_sessions": "5-8",
                    "new_under_extrusion_sessions": "3-5",
                    "estimated_new_frames": "3000-8000",
                    "estimated_total_train_frames": "14254-19254",
                    "training_estimate": "20-28 minutes",
                    "evaluation_estimate": "2-3 minutes",
                    "total_pipeline": "23-33 minutes",
                },
                "ideal_acquisition": {
                    "new_stringing_sessions": "25-30",
                    "new_normal_sessions": "15-20",
                    "new_spaghetti_sessions": "10-15",
                    "new_under_extrusion_sessions": "5-8",
                    "estimated_new_frames": "6000-15000",
                    "estimated_total_train_frames": "17254-26254",
                    "training_estimate": "25-38 minutes",
                    "evaluation_estimate": "3-4 minutes",
                    "total_pipeline": "29-45 minutes",
                },
            },
            "total_wall_time_with_data_prep": {
                "note": (
                    "Data preparation (organizing new sessions into directory layout, "
                    "labeling, updating CSVs) is NOT included in the above estimates. "
                    "This is the dominant time cost and depends on data availability."
                ),
                "data_preparation_estimate": "1-5 hours (depending on data source)",
                "pipeline_after_prep": "20-45 minutes",
            },
        },

        # --- Success criteria ---
        "success_criteria": {
            "minimum": {
                "description": (
                    "Stringing FNR on frozen test improves by >=5pp "
                    "(from 0.213 to <=0.163), without F1 regression."
                ),
                "stringing_FNR_target": "<=0.163",
                "F1_regression_allowed": False,
            },
            "meaningful": {
                "description": (
                    "Stringing FNR improves by >=10pp AND overall F1 "
                    "improves by >=1pp on frozen test."
                ),
                "stringing_FNR_target": "<=0.113",
                "F1_improvement_target": ">=0.8899",
            },
            "strong": {
                "description": (
                    "Stringing FNR on frozen test approaches under-extrusion FNR "
                    "(currently 0.155), and held-out set shows comparable performance."
                ),
                "stringing_FNR_target": "<=0.170",
                "held_out_comparable": True,
            },
            "abort_criteria": {
                "description": (
                    "If F1 on frozen test regresses by >2pp, or if any failure type "
                    "FNR increases by >5pp, the experiment is a regression."
                ),
                "F1_regression_threshold": "<0.8599",
                "FNR_regression_threshold": ">0.2287 (any type)",
            },
        },

        # --- Decision tree ---
        "decision_tree": {
            "minimum_acquisition": {
                "minimum_met": "Proceed to strong acquisition",
                "minimum_not_met": (
                    "Analyze which failure type regressed. Consider: "
                    "(a) more data of the failing type, "
                    "(b) different augmentation, "
                    "(c) learning rate adjustment"
                ),
            },
            "strong_acquisition": {
                "meaningful_met": (
                    "Proceed to ideal acquisition or deployment planning"
                ),
                "meaningful_not_met": (
                    "Evaluate whether the bottleneck is data diversity or model capacity. "
                    "Consider: (a) architecture search, (b) different pretrained backbone, "
                    "(c) more targeted data"
                ),
            },
            "ideal_acquisition": {
                "strong_met": "Ready for deployment evaluation (Stage 12)",
                "strong_not_met": (
                    "Model capacity may be the limiting factor. "
                    "Consider: (a) larger ShuffleNet, (b) MobileNetV3, "
                    "(c) EfficientNet-B0"
                ),
            },
        },

        # --- Gating checklist ---
        "gating_checklist": {
            "description": (
                "All items must be checked before executing the experiment."
            ),
            "items": [
                {
                    "id": "G1",
                    "check": "New data collected and labeled",
                    "required": "5-10 independent stringing sessions minimum",
                },
                {
                    "id": "G2",
                    "check": "Data organized in existing directory layout",
                    "required": "datasets/processed/printing-errors/{failing,normal}/Recording_<date>_<time>/",
                },
                {
                    "id": "G3",
                    "check": "metadata.csv updated with new fine-class mappings",
                    "required": "All new frames have fine class (normal/under_extrusion/stringing/spaghetti)",
                },
                {
                    "id": "G4",
                    "check": "New splits created (train.csv, val.csv updated)",
                    "required": "Session-level split, no frame leakage",
                },
                {
                    "id": "G5",
                    "check": "Held-out sessions identified and reserved",
                    "required": "1-2 sessions from distinct date range",
                },
                {
                    "id": "G6",
                    "check": "Frozen test set untouched",
                    "required": "test.csv and test frames unchanged",
                },
                {
                    "id": "G7",
                    "check": "Config YAML created from template",
                    "required": "experiment_id assigned, all hyperparameters set",
                },
                {
                    "id": "G8",
                    "check": "Training pipeline tested on small subset",
                    "required": "Verify no crashes, correct class mapping",
                },
            ],
        },

        # --- Limitations ---
        "limitations": [
            "Runtime estimates assume MPS; CPU will be ~2-3x slower.",
            "Data preparation time is not included in pipeline estimates.",
            "Success criteria assume the bottleneck is data diversity, not model capacity.",
            "The frozen test set is a single snapshot; held-out set provides a second signal.",
            "No claims of real-world generalization are made.",
            "Session atomicity means some useful within-session frames may be lost at split boundaries.",
        ],
    }


def render_markdown(r: dict) -> str:
    L: list[str] = []
    L.append("# Stage 11 — Retraining Experiment Design")
    L.append("")
    L.append("**Design only.** No training is executed in this stage.")
    L.append(f"**Objective:** {r['objective']}")
    L.append("")

    # ---- 1. Baseline ----
    L.append("## 1. Baseline (Experiment 1)")
    L.append("")
    bl = r["baseline"]
    L.append(f"- **Model:** `{bl['model']}`")
    L.append(f"- **Checkpoint:** `{bl['checkpoint']}`")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    for k, v in bl["frozen_test"].items():
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append(f"**Training pool:** {bl['train_pool']['recordings']} recordings, "
             f"{bl['train_pool']['frames']} frames")
    L.append(f"- Stringing: {bl['train_pool']['stringing_recordings']} recordings")
    L.append(f"- Under-extrusion: {bl['train_pool']['under_extrusion_recordings']} recordings")
    L.append(f"- Spaghetti: {bl['train_pool']['spaghetti_recordings']} recordings")
    L.append(f"- Normal: {bl['train_pool']['normal_recordings']} recordings")
    L.append("")

    # ---- 2. Stage 10 evidence ----
    L.append("## 2. Stage 10 evidence driving this design")
    L.append("")
    for key, val in r["evidence_summary"].items():
        L.append(f"- **{key}:** {val}")
    L.append("")

    # ---- 3. Data requirements ----
    L.append("## 3. Data requirements")
    L.append("")
    L.append(f"**Principle:** {r['data_requirements']['principle']}")
    L.append("")
    for p in r["data_requirements"]["priorities"]:
        L.append(f"### Priority {p['priority']}: {p['type']}")
        L.append(f"**Rationale:** {p['rationale']}")
        L.append("")
        L.append("| Target | Sessions | Frames/session | Total frames |")
        L.append("|---|---|---|---|")
        mn = p["minimum"]
        L.append(f"| Minimum | {mn['sessions']} | {mn['frames_per_session']} | {mn['total_frames']} |")
        if "strong" in p:
            st = p["strong"]
            L.append(f"| Strong | {st['sessions']} | {st['frames_per_session']} | {st['total_frames']} |")
        L.append("")
        if "diversity" in p:
            L.append("**Diversity requirements:**")
            for d in p["diversity"]:
                L.append(f"- {d}")
            L.append("")
        if "held_out" in p:
            L.append(f"**Held-out:** {p['held_out']}")
            L.append("")
    L.append(f"**Labeling:** {r['data_requirements']['labeling_requirements']}")
    L.append("")

    # ---- 4. Split strategy ----
    L.append("## 4. Split strategy")
    L.append("")
    ss = r["split_strategy"]
    L.append(f"- **Unit:** {ss['unit']}")
    L.append(f"- **New data ratio:** {ss['new_data_ratio']}")
    L.append(f"- **Held-out set:** {ss['held_out_set']}")
    L.append(f"- **Frozen test set:** {ss['frozen_test_set']}")
    L.append(f"- **Existing data:** {ss['existing_data']}")
    L.append(f"- **Directory:** {ss['directory_structure']}")
    L.append(f"- **CSV update:** {ss['csv_update']}")
    L.append("")

    # ---- 5. Training protocol ----
    L.append("## 5. Training protocol")
    L.append("")
    tp = r["training_protocol"]
    L.append(f"- Architecture: `{tp['architecture']}`")
    L.append(f"- Image size: {tp['image_size']}")
    L.append(f"- Optimizer: {tp['optimizer']} (lr={tp['learning_rate']}, wd={tp['weight_decay']})")
    L.append(f"- Epochs: {tp['epochs']} (early stopping patience={tp['early_stopping_patience']})")
    L.append(f"- Batch size: {tp['batch_size']}")
    L.append(f"- Precision: {tp['precision']}")
    L.append(f"- Seed: {tp['seed']}")
    L.append(f"- Augmentation: {json.dumps(tp['augmentation'])}")
    L.append(f"- Hardware: {tp['hardware']}")
    L.append("")
    L.append("**Expected training runtimes (measured/estimated):**")
    L.append("")
    L.append("| Train frames | Estimate | Source |")
    L.append("|---|---|---|")
    L.append("| 11,254 | ~16 min | Measured (Experiment 1) |")
    L.append("| 15,000 | ~22 min | Estimated (linear) |")
    L.append("| 20,000 | ~29 min | Estimated |")
    L.append("| 25,000 | ~36 min | Estimated |")
    L.append("")

    # ---- 6. Evaluation protocol ----
    L.append("## 6. Evaluation protocol")
    L.append("")
    for name, proto in r["evaluation_protocol"].items():
        L.append(f"### {name.replace('_', ' ').title()}")
        L.append(f"- {proto['description']}")
        if "metrics" in proto:
            if isinstance(proto["metrics"], list):
                L.append(f"- Metrics: {', '.join(proto['metrics'])}")
            else:
                L.append(f"- Metrics: {proto['metrics']}")
        L.append(f"- Runtime: {proto['runtime_estimate']}")
        L.append("")

    # ---- 7. Runtime estimates ----
    L.append("## 7. Runtime estimates")
    L.append("")
    re = r["runtime_estimates"]
    L.append(f"> {re['note']}")
    L.append("")
    L.append("**Measured baselines:**")
    for k, v in re["measured baselines"].items():
        L.append(f"- {k}: {v}")
    L.append("")
    L.append("**Projected totals by acquisition scenario:**")
    L.append("")
    L.append("| Scenario | New frames | Total train | Training | Eval | Pipeline |")
    L.append("|---|---|---|---|---|---|")
    for name, sc in re["projected_totals_by_scenario"].items():
        L.append(f"| {name.replace('_', ' ').title()} | "
                 f"{sc['estimated_new_frames']} | "
                 f"{sc['estimated_total_train_frames']} | "
                 f"{sc['training_estimate']} | "
                 f"{sc['evaluation_estimate']} | "
                 f"{sc['total_pipeline']} |")
    L.append("")
    L.append(f"**Data preparation:** {re['total_wall_time_with_data_prep']['data_preparation_estimate']}")
    L.append(f"**Pipeline after prep:** {re['total_wall_time_with_data_prep']['pipeline_after_prep']}")
    L.append("")

    # ---- 8. Success criteria ----
    L.append("## 8. Success criteria")
    L.append("")
    for level in ["minimum", "meaningful", "strong"]:
        sc = r["success_criteria"][level]
        L.append(f"### {level.title()}")
        L.append(f"- {sc['description']}")
        for k, v in sc.items():
            if k != "description":
                L.append(f"- {k}: {v}")
        L.append("")
    L.append("### Abort criteria")
    ac = r["success_criteria"]["abort_criteria"]
    L.append(f"- {ac['description']}")
    L.append(f"- F1 regression threshold: {ac['F1_regression_threshold']}")
    L.append(f"- FNR regression threshold: {ac['FNR_regression_threshold']}")
    L.append("")

    # ---- 9. Decision tree ----
    L.append("## 9. Decision tree")
    L.append("")
    for stage_name, options in r["decision_tree"].items():
        L.append(f"### After {stage_name.replace('_', ' ')}")
        for outcome, action in options.items():
            L.append(f"- **{outcome}:** {action}")
        L.append("")

    # ---- 10. Gating checklist ----
    L.append("## 10. Gating checklist")
    L.append("")
    L.append(f"> {r['gating_checklist']['description']}")
    L.append("")
    L.append("| ID | Check | Required |")
    L.append("|---|---|---|")
    for item in r["gating_checklist"]["items"]:
        L.append(f"| {item['id']} | {item['check']} | {item['required']} |")
    L.append("")

    # ---- 11. Limitations ----
    L.append("## 11. Limitations")
    L.append("")
    for lim in r["limitations"]:
        L.append(f"- {lim}")
    L.append("")

    # ---- Artifacts ----
    L.append("---")
    L.append("")
    L.append("## Artifacts")
    L.append("")
    L.append("| File | Purpose |")
    L.append("|---|---|")
    L.append("| `docs/research/stage11-retrain-experiment.json` | Machine-readable design |")
    L.append("| `docs/research/stage11-retrain-experiment.md` | This report |")
    L.append("| `scripts/analyze_stage11.py` | Design generator |")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
