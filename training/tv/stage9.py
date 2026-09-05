"""Stage 9: data sufficiency and acquisition design analysis.

Analysis only. No model training, no data modification, no checkpoint changes.

This module reads the existing processed dataset CSVs and the Stage 4 recording
audit to produce a structured inventory and diversity analysis. All conclusions
are derived from metadata and cached statistics, not from model runs.
"""
from __future__ import annotations

import csv
import json
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Canonical paths
REPO = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO / "datasets" / "processed" / "printing-errors" / "splits"
AUDIT_PATH = REPO / "docs" / "research" / "stage4-recording-audit.json"
DATA_DIR = REPO / "datasets" / "processed" / "printing-errors"

# Known Stage 8 observations (for documentation, not decision-driving)
STAGE8_OBSERVATIONS = {
    "catastrophic_fn_recording": "Recording_15.02.2023_13_19_52",
    "catastrophic_fn_failure_type": "stringing",
    "catastrophic_fn_rate": 0.898,
    "high_fp_recording": "Recording_20122022_17_05_36",
    "high_fp_composition": "PURE_NORMAL",
    "high_fp_note": "p>0.98 on some normal frames",
}

# Failure types that indicate actual printing failures
FAILURE_TYPES = {"under_extrusion", "stringing", "spaghetti"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split_rows(split: str) -> List[dict]:
    csv_path = SPLITS_DIR / f"{split}.csv"
    return list(csv.DictReader(open(csv_path)))


def load_audit() -> dict:
    return json.loads(AUDIT_PATH.read_text())


def _extract_recording_id(path: str) -> Optional[str]:
    parts = path.split("/")
    if len(parts) >= 3:
        return parts[1]
    return None


def _extract_date(recording_id: str) -> Optional[str]:
    if "Recording_" in recording_id:
        return recording_id.split("_")[1]
    return None


# ---------------------------------------------------------------------------
# Recording inventory
# ---------------------------------------------------------------------------

@dataclass
class RecordingInfo:
    recording_id: str
    split: str
    total_frames: int
    normal_frames: int
    failing_frames: int
    composition: str
    fine_counts: Dict[str, int]
    date: Optional[str]
    source_class_dirs: Dict[str, int]  # normal/failing dir -> frame count

    @property
    def has_normal(self) -> bool:
        return self.normal_frames > 0

    @property
    def has_failing(self) -> bool:
        return self.failing_frames > 0

    @property
    def is_pure_normal(self) -> bool:
        return self.failing_frames == 0

    @property
    def is_pure_failing(self) -> bool:
        return self.normal_frames == 0

    @property
    def is_mixed(self) -> bool:
        return self.normal_frames > 0 and self.failing_frames > 0


def build_inventory(splits: Tuple[str, ...] = ("train", "val", "test"),
                    audit: Optional[dict] = None) -> Dict[str, List[RecordingInfo]]:
    """Build per-recording inventory for all splits."""
    if audit is None:
        audit = load_audit()
    audit_recs = audit["recordings"]

    inventory: Dict[str, List[RecordingInfo]] = {}
    for split in splits:
        rows = load_split_rows(split)
        rec_data: Dict[str, dict] = defaultdict(lambda: {
            "total": 0, "normal": 0, "failing": 0,
            "fine_labels": Counter(), "class_dirs": Counter(),
        })
        for row in rows:
            rid = _extract_recording_id(row["path"])
            if rid is None:
                continue
            label = row["label"]
            rec_data[rid]["total"] += 1
            rec_data[rid][label] += 1
            rec_data[rid]["fine_labels"][label] += 1
            parts = row["path"].split("/")
            rec_data[rid]["class_dirs"][parts[0]] += 1

        recs = []
        for rid in sorted(rec_data.keys()):
            d = rec_data[rid]
            audit_info = audit_recs.get(rid, {})
            fine_counts = audit_info.get("fine_counts", {})
            composition = audit_info.get("composition", "UNKNOWN")
            date = _extract_date(rid)
            recs.append(RecordingInfo(
                recording_id=rid,
                split=split,
                total_frames=d["total"],
                normal_frames=d["normal"],
                failing_frames=d["failing"],
                composition=composition,
                fine_counts=dict(fine_counts) if fine_counts else dict(d["fine_labels"]),
                date=date,
                source_class_dirs=dict(d["class_dirs"]),
            ))
        inventory[split] = recs
    return inventory


# ---------------------------------------------------------------------------
# Frame-level statistics
# ---------------------------------------------------------------------------

def frame_concentration(inventory: Dict[str, List[RecordingInfo]]) -> dict:
    """Quantify how evenly frames are distributed across recordings."""
    out = {}
    for split, recs in inventory.items():
        frame_counts = [r.total_frames for r in recs]
        normal_counts = [r.normal_frames for r in recs]
        failing_counts = [r.failing_frames for r in recs]
        total = sum(frame_counts)
        sorted_counts = sorted(frame_counts, reverse=True)
        top5_sum = sum(sorted_counts[:5])
        top10_sum = sum(sorted_counts[:10])

        out[split] = {
            "n_recordings": len(recs),
            "total_frames": total,
            "frames_per_recording": {
                "min": min(frame_counts) if frame_counts else 0,
                "max": max(frame_counts) if frame_counts else 0,
                "median": statistics.median(frame_counts) if frame_counts else 0,
                "mean": round(statistics.mean(frame_counts), 1) if frame_counts else 0,
                "stdev": round(statistics.stdev(frame_counts), 1) if len(frame_counts) > 1 else 0,
            },
            "top5_recordings": [
                {"recording": recs[i].recording_id, "frames": recs[i].total_frames,
                 "composition": recs[i].composition}
                for i in sorted(range(len(recs)),
                                key=lambda i: recs[i].total_frames, reverse=True)[:5]
            ],
            "top5_frame_fraction": round(top5_sum / total, 4) if total > 0 else 0,
            "top10_frame_fraction": round(top10_sum / total, 4) if total > 0 else 0,
        }
    return out


# ---------------------------------------------------------------------------
# Failure-type diversity
# ---------------------------------------------------------------------------

def failure_type_diversity(inventory: Dict[str, List[RecordingInfo]]) -> dict:
    """Per-failure-type recording and frame counts across all splits."""
    all_types = set()
    split_ft: Dict[str, Dict[str, dict]] = {}
    for split, recs in inventory.items():
        by_type: Dict[str, dict] = defaultdict(lambda: {
            "n_recordings": 0, "frames": 0, "recording_ids": [],
        })
        for r in recs:
            for ftype, count in r.fine_counts.items():
                if ftype in FAILURE_TYPES:
                    all_types.add(ftype)
                    by_type[ftype]["n_recordings"] += 1
                    by_type[ftype]["frames"] += count
                    by_type[ftype]["recording_ids"].append(r.recording_id)
        split_ft[split] = dict(by_type)

    # Assemble cross-split summary
    summary = {}
    for ftype in sorted(all_types):
        summary[ftype] = {}
        for split in ("train", "val", "test"):
            info = split_ft.get(split, {}).get(ftype, {"n_recordings": 0, "frames": 0})
            summary[ftype][split] = {
                "n_recordings": info["n_recordings"],
                "frames": info["frames"],
            }
        # Add recording IDs for the training split
        summary[ftype]["train_recording_ids"] = split_ft.get("train", {}).get(
            ftype, {}
        ).get("recording_ids", [])
    return summary


def normal_diversity(inventory: Dict[str, List[RecordingInfo]]) -> dict:
    """Normal-domain recording counts across splits."""
    out = {}
    for split, recs in inventory.items():
        pure_normal = [r for r in recs if r.is_pure_normal]
        mixed_with_normal = [r for r in recs if r.has_normal and not r.is_pure_normal]
        all_normal_frames = sum(r.normal_frames for r in recs)
        out[split] = {
            "pure_normal_recordings": len(pure_normal),
            "mixed_recordings_with_normal": len(mixed_with_normal),
            "total_recordings_with_normal": len(pure_normal) + len(mixed_with_normal),
            "total_normal_frames": all_normal_frames,
            "pure_normal_recording_ids": [r.recording_id for r in pure_normal],
        }
    return out


# ---------------------------------------------------------------------------
# Stringing investigation
# ---------------------------------------------------------------------------

def stringing_investigation(inventory: Dict[str, List[RecordingInfo]]) -> dict:
    """Detailed analysis of stringing training data."""
    train_recs = inventory.get("train", [])
    stringing_recs = [
        r for r in train_recs
        if r.fine_counts.get("stringing", 0) > 0
    ]
    stringing_recs.sort(key=lambda r: r.fine_counts.get("stringing", 0), reverse=True)

    frame_counts = [r.fine_counts.get("stringing", 0) for r in stringing_recs]
    tiny_sessions = [r for r in stringing_recs if r.fine_counts.get("stringing", 0) < 10]
    small_sessions = [r for r in stringing_recs if 10 <= r.fine_counts.get("stringing", 0) < 50]

    # Date diversity
    dates = Counter(r.date for r in stringing_recs if r.date)

    return {
        "n_recordings": len(stringing_recs),
        "total_stringing_frames": sum(frame_counts),
        "frames_per_recording": {
            "min": min(frame_counts) if frame_counts else 0,
            "max": max(frame_counts) if frame_counts else 0,
            "median": statistics.median(frame_counts) if frame_counts else 0,
            "mean": round(statistics.mean(frame_counts), 1) if frame_counts else 0,
        },
        "n_tiny_sessions_lt10_frames": len(tiny_sessions),
        "n_small_sessions_10_50_frames": len(small_sessions),
        "tiny_session_ids": [r.recording_id for r in tiny_sessions],
        "small_session_ids": [r.recording_id for r in small_sessions],
        "large_session_ids": [r.recording_id for r in stringing_recs
                              if r.fine_counts.get("stringing", 0) >= 50],
        "date_distribution": dict(dates),
        "composition_breakdown": dict(Counter(r.composition for r in stringing_recs)),
        "stage8_observation": {
            "catastrophic_recording": STAGE8_OBSERVATIONS["catastrophic_fn_recording"],
            "failure_type": STAGE8_OBSERVATIONS["catastrophic_fn_failure_type"],
            "fnr": STAGE8_OBSERVATIONS["catastrophic_fn_rate"],
            "note": (
                "This test recording has 127 stringing frames with FNR=0.898. "
                "It is a PURE_FAILING stringing recording. The frozen model produces "
                "near-zero fail scores for the majority of its frames."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Normal-domain / false-alarm investigation
# ---------------------------------------------------------------------------

def normal_domain_investigation(inventory: Dict[str, List[RecordingInfo]],
                                sample_n: int = 10) -> dict:
    """Investigate normal-domain diversity and the known false-positive recording."""
    train_recs = inventory.get("train", [])
    pure_normal_train = [r for r in train_recs if r.is_pure_normal]
    normal_frame_counts = [r.normal_frames for r in pure_normal_train]

    # Image statistics for training normal recordings
    brightness_stats = []
    contrast_stats = []
    for r in pure_normal_train[:15]:  # sample first 15
        rec_dir = DATA_DIR / "normal" / r.recording_id
        if not rec_dir.exists():
            continue
        files = sorted(rec_dir.glob("*.png"))[:sample_n]
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightness_stats.append(float(np.mean(gray)))
                contrast_stats.append(float(np.std(gray)))

    # False-positive recording analysis
    fp_recording = STAGE8_OBSERVATIONS["high_fp_recording"]
    fp_brightness = []
    fp_contrast = []
    fp_path = DATA_DIR / "normal" / fp_recording
    if fp_path.exists():
        files = sorted(fp_path.glob("*.png"))[:20]
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                fp_brightness.append(float(np.mean(gray)))
                fp_contrast.append(float(np.std(gray)))

    return {
        "training_normal": {
            "n_pure_normal_recordings": len(pure_normal_train),
            "total_normal_frames": sum(r.normal_frames for r in train_recs),
            "frames_per_pure_normal_recording": {
                "min": min(normal_frame_counts) if normal_frame_counts else 0,
                "max": max(normal_frame_counts) if normal_frame_counts else 0,
                "median": statistics.median(normal_frame_counts) if normal_frame_counts else 0,
            },
            "brightness_mean": round(statistics.mean(brightness_stats), 1) if brightness_stats else None,
            "brightness_stdev": round(statistics.stdev(brightness_stats), 1) if len(brightness_stats) > 1 else None,
            "contrast_mean": round(statistics.mean(contrast_stats), 1) if contrast_stats else None,
            "recording_ids": [r.recording_id for r in pure_normal_train],
        },
        "problematic_test_recording": {
            "recording_id": fp_recording,
            "composition": "PURE_NORMAL",
            "n_frames": 110,
            "brightness_mean": round(statistics.mean(fp_brightness), 1) if fp_brightness else None,
            "brightness_stdev": round(statistics.stdev(fp_brightness), 1) if len(fp_brightness) > 1 else None,
            "contrast_mean": round(statistics.mean(fp_contrast), 1) if fp_contrast else None,
            "stage8_observation": (
                "This PURE_NORMAL recording generates highly confident false positives "
                "(p>0.98). Mean brightness ~76 vs training normal mean ~156 suggests "
                "a substantially different lighting environment."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Distribution comparison across splits
# ---------------------------------------------------------------------------

def split_distribution_comparison(inventory: Dict[str, List[RecordingInfo]]) -> dict:
    """Compare composition and failure-type distributions across splits."""
    comparison = {}
    for split, recs in inventory.items():
        total_frames = sum(r.total_frames for r in recs)
        normal_frames = sum(r.normal_frames for r in recs)
        failing_frames = sum(r.failing_frames for r in recs)
        compositions = Counter(r.composition for r in recs)

        # Failure type breakdown
        type_frames = Counter()
        type_recs = defaultdict(set)
        for r in recs:
            for ftype, count in r.fine_counts.items():
                if ftype in FAILURE_TYPES:
                    type_frames[ftype] += count
                    type_recs[ftype].add(r.recording_id)

        comparison[split] = {
            "n_recordings": len(recs),
            "total_frames": total_frames,
            "normal_frames": normal_frames,
            "failing_frames": failing_frames,
            "normal_fraction": round(normal_frames / total_frames, 4) if total_frames else 0,
            "compositions": dict(compositions),
            "failure_type_frames": dict(type_frames),
            "failure_type_recordings": {k: len(v) for k, v in type_recs.items()},
        }
    return comparison


# ---------------------------------------------------------------------------
# Data quality / label structure
# ---------------------------------------------------------------------------

def data_quality_analysis(inventory: Dict[str, List[RecordingInfo]]) -> dict:
    """Flag potential data quality concerns."""
    concerns = []

    for split, recs in inventory.items():
        for r in recs:
            # Tiny recordings
            if r.total_frames < 5:
                concerns.append({
                    "type": "tiny_recording",
                    "split": split,
                    "recording": r.recording_id,
                    "frames": r.total_frames,
                    "note": "Very few frames; limited diversity from this session.",
                })
            # Mixed recordings with very few frames of one class
            if r.is_mixed:
                if r.normal_frames < 5:
                    concerns.append({
                        "type": "few_normal_frames_in_mixed",
                        "split": split,
                        "recording": r.recording_id,
                        "normal_frames": r.normal_frames,
                    })
                if r.failing_frames < 5:
                    concerns.append({
                        "type": "few_failing_frames_in_mixed",
                        "split": split,
                        "recording": r.recording_id,
                        "failing_frames": r.failing_frames,
                    })
            # Spaghetti scarcity
            if r.fine_counts.get("spaghetti", 0) > 0:
                if r.fine_counts["spaghetti"] < 20:
                    concerns.append({
                        "type": "scarce_spaghetti",
                        "split": split,
                        "recording": r.recording_id,
                        "spaghetti_frames": r.fine_counts["spaghetti"],
                    })

    return {"concerns": concerns, "n_concerns": len(concerns)}


# ---------------------------------------------------------------------------
# Top-level analysis driver
# ---------------------------------------------------------------------------

def run_stage9() -> dict:
    """Run the full Stage 9 analysis. Returns a structured dict."""
    audit = load_audit()
    inventory = build_inventory(audit=audit)

    return {
        "stage": 9,
        "dataset_inventory": {
            split: [
                {
                    "recording_id": r.recording_id,
                    "total_frames": r.total_frames,
                    "normal_frames": r.normal_frames,
                    "failing_frames": r.failing_frames,
                    "composition": r.composition,
                    "fine_counts": r.fine_counts,
                    "date": r.date,
                }
                for r in recs
            ]
            for split, recs in inventory.items()
        },
        "frame_concentration": frame_concentration(inventory),
        "failure_type_diversity": failure_type_diversity(inventory),
        "normal_diversity": normal_diversity(inventory),
        "stringing_investigation": stringing_investigation(inventory),
        "normal_domain_investigation": normal_domain_investigation(inventory),
        "split_distribution_comparison": split_distribution_comparison(inventory),
        "data_quality": data_quality_analysis(inventory),
        "acquisition_design": _design_acquisition(inventory),
        "future_experiment": _design_experiment(),
        "retraining_gate": _retraining_gate(),
        "limitations": [
            "All analyses are metadata/statistical only; no model inference was performed.",
            "Image statistics (brightness, contrast) are non-learned proxies, not proof of semantic OOD.",
            "Frame counts are not independent samples; recording identity is the diversity unit.",
            "Test-set observations are documented for context only; they do not drive data-selection decisions.",
            "No claims of real-world generalization are made.",
        ],
    }


# ---------------------------------------------------------------------------
# Acquisition design
# ---------------------------------------------------------------------------

def _design_acquisition(inventory: Dict[str, List[RecordingInfo]]) -> dict:
    return {
        "priority_1_stringing": {
            "rationale": (
                "Stage 8 found FNR=0.898 on one PURE_FAILING stringing test recording. "
                "The training set has 23 stringing recordings (2341 frames), but 4 have "
                "<10 frames and date diversity is concentrated (22122022, 14.02.2023)."
            ),
            "failure_type": "stringing",
            "minimum_exploratory": {
                "n_sessions": "5-10 independent sessions",
                "n_frames_per_session": "50-200 frames",
                "rationale": (
                    "5-10 new independent sessions would roughly double the session count. "
                    "Each session should come from a distinct print job (different dates if "
                    "possible) to test generalization beyond the existing date clusters."
                ),
            },
            "stronger_target": {
                "n_sessions": "15-20 independent sessions",
                "n_frames_per_session": "80-200 frames",
                "rationale": (
                    "Would bring total stringing sessions to ~40+, approaching the "
                    "diversity level of under_extrusion (37 train sessions). Provides "
                    "meaningful coverage of session-to-session variation."
                ),
            },
            "ideal_target": {
                "n_sessions": "25-30 independent sessions",
                "note": (
                    "Would bring total to ~50+ sessions, matching or exceeding "
                    "under_extrusion diversity. This is a production-grade target, "
                    "not a minimum exploratory target."
                ),
            },
            "diversity_requirements": [
                "Multiple distinct print objects (if possible)",
                "Different lighting conditions (if possible)",
                "Different print stages where stringing occurs",
                "Mix of severe and mild stringing",
            ],
            "held_out_condition": (
                "Hold out 1-2 sessions from a distinct date range or object as a "
                "final held-out set, separate from the existing frozen test set."
            ),
        },
        "priority_2_normal_domain": {
            "rationale": (
                "Recording_20122022_17_05_36 is a PURE_NORMAL test recording with "
                "brightness ~76 vs training normal ~156. This represents a potentially "
                "missing normal condition (darker lighting or different camera setting)."
            ),
            "failure_type": "normal",
            "minimum_exploratory": {
                "n_sessions": "5-10 independent sessions",
                "n_frames_per_session": "50-150 frames",
                "note": (
                    "Capture normal prints under varied lighting: bright, dim, and "
                    "mixed ambient conditions. Include sessions that test the "
                    "model's ability to reject non-failure visual variation."
                ),
            },
            "stronger_target": {
                "n_sessions": "10-15 independent sessions",
                "note": "Cover at least 3 distinct lighting/environment conditions.",
            },
            "diversity_requirements": [
                "Varied ambient lighting (bright, dim, mixed)",
                "Different objects in frame (if applicable)",
                "Different camera orientations/angles if eventually relevant",
            ],
        },
        "priority_3_spaghetti": {
            "rationale": (
                "Spaghetti has only 5 training recordings (191 frames) and "
                "4 validation recordings (49 frames). This is the scarcest failure type."
            ),
            "failure_type": "spaghetti",
            "minimum_exploratory": {
                "n_sessions": "5-8 independent sessions",
                "n_frames_per_session": "50-150 frames",
                "note": "Spaghetti is visually distinctive but severely underrepresented.",
            },
            "stronger_target": {
                "n_sessions": "10-15 independent sessions",
                "note": "Would bring total to ~15-20 sessions, comparable to stringing.",
            },
        },
        "priority_4_under_extrusion": {
            "rationale": (
                "Under_extrusion has 37 training recordings (3331 frames) — "
                "the best-represented failure type. Additional data is lower priority "
                "but could improve generalization at the margins."
            ),
            "failure_type": "under_extrusion",
            "minimum_exploratory": {
                "n_sessions": "3-5 independent sessions",
                "n_frames_per_session": "50-150 frames",
                "note": "Focus on under-represented severity levels or object types.",
            },
        },
        "general_principles": [
            "Prioritize INDEPENDENT SESSIONS over raw frame count.",
            "Each new session should be from a distinct print job, ideally on a different date.",
            "Record metadata: date, object, lighting condition, camera settings if available.",
            "Label requirements: same binary normal/failing + fine failure type as existing.",
            "Split strategy: new sessions split 60/20/20 by session (not frame) into train/val/held-out.",
            "The existing frozen test set remains untouched for evaluation.",
            "Held-out sessions from new data provide a second evaluation set to measure improvement.",
        ],
    }


# ---------------------------------------------------------------------------
# Future experiment design
# ---------------------------------------------------------------------------

def _design_experiment() -> dict:
    return {
        "objective": (
            "Test whether adding targeted independent data improves generalization "
            "to previously unseen recordings."
        ),
        "baseline": {
            "model": "ShuffleNetV2 x1.0",
            "checkpoint": "experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/best.pt",
            "test_performance": {
                "F1_at_0_50": 0.8763,
                "F1_at_0_65": 0.8799,
                "FNR_at_0_65": 0.1787,
            },
        },
        "old_test_set": {
            "policy": "FROZEN. Do not touch. Used only for final comparison.",
            "n_recordings": 24,
            "n_frames": 2859,
        },
        "new_data_split_policy": {
            "unit": "session/recording (never frame)",
            "split_ratio": "60% train / 20% val / 20% held-out evaluation",
            "held_out_set": (
                "Sessions from new data held out at acquisition time; "
                "these become the second evaluation set to measure improvement."
            ),
        },
        "training_setup": {
            "same_architecture": True,
            "same_augmentation": True,
            "same_optimizer": True,
            "same_splits_for_old_data": True,
            "new_data_added_to_train_and_val": True,
        },
        "primary_metrics": [
            "FNR on stringing failure type (test set, at threshold 0.65)",
            "FPR on normal frames (test set, at threshold 0.65)",
            "F1 overall (test set, at threshold 0.50 and 0.65)",
        ],
        "secondary_metrics": [
            "Per-failure-type recall and F1",
            "Calibration (Brier, ECE)",
            "Per-recording error analysis",
            "Performance on the new held-out evaluation set",
        ],
        "success_criteria": {
            "minimum": (
                "Stringing FNR improves by >=5pp on the frozen test set, "
                "without F1 regression on other failure types."
            ),
            "meaningful": (
                "Stringing FNR improves by >=10pp AND overall F1 improves by >=1pp "
                "on the frozen test set."
            ),
            "strong": (
                "Stringing FNR on the frozen test set approaches under_extrusion FNR "
                "(currently 0.155), and the new held-out set shows comparable performance."
            ),
        },
        "deliverables": [
            "Updated frozen checkpoint (retrained with expanded training set)",
            "Same frozen test set evaluation at 0.50 and 0.65",
            "New held-out evaluation set results",
            "Per-failure-type comparison (before vs after)",
            "Calibration comparison",
            "Full comparison report",
        ],
    }


# ---------------------------------------------------------------------------
# Retraining gate
# ---------------------------------------------------------------------------

def _retraining_gate() -> dict:
    return {
        "A_is_evidence_sufficient_for_immediate_retraining": False,
        "B_what_evidence_or_data_is_missing": (
            "1. The current training set has adequate under_extrusion coverage (37 sessions) "
            "but stringing is concentrated in date clusters (22122022, 14.02.2023) with "
            "some tiny sessions (2-24 frames). Adding 5-10 new independent stringing sessions "
            "before retraining would test whether the model can learn stringing from diverse "
            "sessions, not just from the current date clusters.\n"
            "2. The normal-domain false-alarm problem (Recording_20122022_17_05_36, brightness ~76 "
            "vs training ~156) suggests the training normal set may not cover darker lighting "
            "conditions. 5-10 normal sessions under varied lighting would test this.\n"
            "3. Spaghetti is severely underrepresented (5 training sessions, 191 frames). "
            "Adding 5-8 sessions would bring it closer to stringing's diversity."
        ),
        "C_what_data_acquisition_unblocks_retraining": (
            "Minimum: 5-10 new independent stringing sessions (50-200 frames each) from "
            "distinct print jobs and dates. Optionally: 5-10 normal sessions under varied "
            "lighting, and 5-8 spaghetti sessions."
        ),
        "D_what_would_constitute_success": (
            "After retraining with the expanded dataset:\n"
            "- Stringing FNR on the frozen test set improves by >=5pp (ideally >=10pp).\n"
            "- Overall test F1 does not regress.\n"
            "- The new held-out evaluation set shows comparable performance to the frozen test.\n"
            "- Calibration does not materially degrade.\n"
            "- No single recording dominates the error profile."
        ),
        "priority_order": [
            "1. Independent session diversity (not raw frame count)",
            "2. Failure-type diversity (stringing > spaghetti > under_extrusion)",
            "3. Normal-domain diversity (varied lighting conditions)",
            "4. Object/condition diversity (if metadata supports it)",
            "5. Raw frame count (lowest priority — more of the same sessions adds little)",
        ],
    }
