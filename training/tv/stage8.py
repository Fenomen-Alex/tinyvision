"""Stage 8: threshold and failure-type analysis on the frozen baseline.

This is analysis-only. No model weights are modified, no calibration transform is
fitted to be used operationally, and the test set is only touched for ONE frozen
evaluation using the validation-selected threshold.

Methodology (documented to avoid threshold-selection leakage):
* Validation split: threshold sweep. The validation-optimal threshold is selected
  by F1 (with precision/recall tradeoff reported explicitly).
* Test split: exactly one evaluation using the selected threshold. The 0.50 test
  result is also reported as the pre-existing baseline.
* Source per-frame labels remain authoritative. Session/recording is the atomic
  split unit. Recording-level classification is NOT the primary task; we only
  report per-recording error aggregates for diagnosis, never as a performance
  claim.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .metrics import compute_classification_metrics

FAILING = 1
NORMAL = 0
DEFAULT_THRESHOLDS = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]

# Failure-type labels that indicate "good" (i.e. not a failure).
GOOD_FINE = {"normal", "good"}


def load_cached_predictions(cache_dir: Path, split: str) -> Dict[str, list]:
    """Load per-frame frozen predictions from the Stage 4/5 eval cache.

    Returns record_id -> list of frame dicts. Frame dicts have keys:
    name, ts, label (normal/failing), score (float, p_failing), fine.
    """
    path = cache_dir / f"frozen_{split}_trajectories.json"
    data = json_load(path)
    return data["recordings"]


def json_load(path: Path) -> dict:
    import json

    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# Threshold sweep (validation only)
# ---------------------------------------------------------------------------

@dataclass
class ThresholdResult:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    fnr: float
    fpr: float
    confusion: List[List[int]]  # [[TN,FP],[FN,TP]]
    n_positives: int
    n_negatives: int

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "fnr": self.fnr,
            "fpr": self.fpr,
            "confusion_matrix": self.confusion,
            "n_positives": self.n_positives,
            "n_negatives": self.n_negatives,
        }


def decision_from_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Binary decision: 1 (failing) if score >= threshold, else 0 (normal)."""
    return (np.asarray(scores, dtype=float) >= threshold).astype(int)


def evaluate_threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> ThresholdResult:
    """Compute classification metrics at a given operating threshold.

    Positive class = failing (1). Confusion matrix labelled [[TN,FP],[FN,TP]].
    """
    y_pred = decision_from_threshold(scores, threshold)
    m = compute_classification_metrics(y_true, y_pred, np.asarray(scores, dtype=float))
    cm = m.confusion_matrix
    return ThresholdResult(
        threshold=threshold,
        accuracy=m.accuracy,
        precision=m.precision,
        recall=m.recall,
        f1=m.f1,
        fnr=m.fnr,
        fpr=m.fpr,
        confusion=cm,
        n_positives=int((y_true == 1).sum()),
        n_negatives=int((y_true == 0).sum()),
    )


def threshold_sweep(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> List[ThresholdResult]:
    """Evaluate the full threshold grid. Returns results sorted ascending."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    results = []
    for t in thresholds:
        results.append(evaluate_threshold_metrics(y_true, scores, t))
    return results


def select_best_threshold(
    results: List[ThresholdResult],
    metric: str = "f1",
) -> ThresholdResult:
    """Select the threshold optimal on the given metric (default F1).

    Ties broken toward the higher threshold (more conservative, favoring fewer
    false alarms). Called only on VALIDATION data.
    """
    best = max(results, key=lambda r: (getattr(r, metric), r.threshold))
    return best


# ---------------------------------------------------------------------------
# Split-level analysis
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceCache:
    """Cached per-frame results for a whole split, flattened in recording order."""
    y_true: np.ndarray
    scores: np.ndarray
    fine: List[str]
    recording: List[str]
    frame_name: List[str]
    frame_index: List[int]
    n_frames: int

    @classmethod
    def from_recordings(cls, recordings: Dict[str, list]) -> "ConfidenceCache":
        """Flatten cached recordings in deterministic recording order."""
        y_true_list, scores_list = [], []
        fine_list, rec_list, fname_list, fidx_list = [], [], [], []
        for rec in sorted(recordings.keys()):
            for idx, f in enumerate(recordings[rec]["frames"]):
                y_true_list.append(1 if f["label"] == "failing" else 0)
                scores_list.append(float(f["score"]))
                fine_list.append(f.get("fine", "unknown"))
                rec_list.append(rec)
                fname_list.append(f["name"])
                fidx_list.append(idx)
        return cls(
            y_true=np.array(y_true_list, dtype=int),
            scores=np.array(scores_list, dtype=float),
            fine=fine_list,
            recording=rec_list,
            frame_name=fname_list,
            frame_index=fidx_list,
            n_frames=len(y_true_list),
        )

    def decision(self, threshold: float) -> np.ndarray:
        return decision_from_threshold(self.scores, threshold)


def thresholds_on_split(
    cache: ConfidenceCache,
    thresholds: Optional[List[float]] = None,
) -> List[ThresholdResult]:
    return threshold_sweep(cache.y_true, cache.scores, thresholds)


# ---------------------------------------------------------------------------
# Failure-type breakdown (test, frozen, at the selected threshold)
# ---------------------------------------------------------------------------

def failure_type_breakdown(
    cache: ConfidenceCache,
    threshold: float,
) -> Dict[str, dict]:
    """Per-fine-class metrics computed on failing frames of that fine class.

    For each failure type we compute recall-precision at the frame level
    (positive = failing). 'good' frames are excluded from the none-good failure
    groups' own label; they belong to the goodness analysis in recording-level.
    """
    y_pred = cache.decision(threshold)
    out = {}
    for fn in sorted(set(cache.fine)):
        mask = np.array([f == fn for f in cache.fine])
        if mask.sum() == 0:
            continue
        yt = cache.y_true[mask]
        yp = y_pred[mask]
        sc = cache.scores[mask]
        m = compute_classification_metrics(yt, yp, sc)
        out[fn] = {
            "failure_type": fn,
            "n_frames": int(mask.sum()),
            "n_failing_true": int((yt == 1).sum()),
            "n_normal_true_intra": int((yt == 0).sum()),
            **m.to_dict(),
        }
    return out


# ---------------------------------------------------------------------------
# Recording-level error analysis
# ---------------------------------------------------------------------------

def recording_analysis(
    cache: ConfidenceCache,
    threshold: float,
) -> Dict[str, dict]:
    """Per-recording error aggregates at the given threshold."""
    y_pred = cache.decision(threshold)
    by_rec = defaultdict(list)
    for i in range(cache.n_frames):
        by_rec[cache.recording[i]].append(i)

    out = {}
    for rec, idxs in by_rec.items():
        idxs = np.array(idxs)
        yt = cache.y_true[idxs]
        yp = y_pred[idxs]
        sc = cache.scores[idxs]
        m = compute_classification_metrics(yt, yp, sc)
        frames = idxs.tolist()
        n_normal = int((yt == 0).sum())
        n_failing = int((yt == 1).sum())
        # Use explicit labels=[0,1] to guarantee 2x2 confusion matrix
        # even for pure-class recordings
        cm_arr = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp = int(cm_arr[0][0]), int(cm_arr[0][1])
        fn, tp = int(cm_arr[1][0]), int(cm_arr[1][1])
        fn_count = fn
        fp_count = fp
        fine_counts = Counter(cache.fine[i] for i in frames)
        out[rec] = {
            "recording": rec,
            "n_frames": int(len(idxs)),
            "n_normal_frames": n_normal,
            "n_failing_frames": n_failing,
            "n_false_negatives": fn_count,
            "n_false_positives": fp_count,
            "failing_recall": m.recall,
            "failing_precision": m.precision,
            "f1": m.f1,
            "fnr": m.fnr,
            "fpr": m.fpr if n_normal > 0 else None,
            "true_positive": tp,
            "true_negative": tn,
            "fine_counts": dict(fine_counts),
        }
    return out


def rank_recordings_by_error(
    recording_analysis: Dict[str, dict],
) -> Dict[str, list[str]]:
    """Rank test recordings by various error magnitudes (diagnostic only)."""
    recs = list(recording_analysis.values())
    # Smallest helper: sort by a numeric key descending
    by_fnr = sorted(recs, key=lambda r: r["fnr"], reverse=True)
    by_fpr = sorted(
        [r for r in recs if r["fpr"] is not None],
        key=lambda r: r["fpr"],
        reverse=True,
    )
    by_fp_count = sorted(recs, key=lambda r: r["n_false_positives"], reverse=True)
    by_fn_count = sorted(recs, key=lambda r: r["n_false_negatives"], reverse=True)
    return {
        "highest_fnr": [r["recording"] for r in by_fnr if r["n_failing_frames"] > 0],
        "highest_fpr": [r["recording"] for r in by_fpr],
        "largest_fp_count": [r["recording"] for r in by_fp_count],
        "largest_fn_count": [r["recording"] for r in by_fn_count],
    }


def identify_problematic_recordings(
    recording_analysis: Dict[str, dict],
    fnr_threshold: float = 0.5,
    fp_threshold: int = 10,
) -> Dict[str, bool]:
    """Flag recordings that persist across multiple error dimensions."""
    flagged = {}
    for rec, r in recording_analysis.items():
        high_fnr = r["n_failing_frames"] > 0 and r["fnr"] >= fnr_threshold
        many_fp = r["n_false_positives"] >= fp_threshold
        flagged[rec] = {
            "high_fnr": high_fnr,
            "many_fp": many_fp,
            "persistent": high_fnr and many_fp,
        }
    return flagged


# ---------------------------------------------------------------------------
# Hard examples
# ---------------------------------------------------------------------------

def extract_hard_examples(
    cache: ConfidenceCache,
    selected_threshold: float,
    top_n: int = 15,
) -> Dict[str, list[dict]]:
    """Extract representative hard examples.

    Categories:
    * highest-confidence false negatives (predicted normal, true failing)
    * highest-confidence false positives (predicted failing, true normal)
    * borderline errors near the selected threshold
    """
    y_pred = cache.decision(selected_threshold)
    fn_mask = (cache.y_true == 1) & (y_pred == 0)
    fp_mask = (cache.y_true == 0) & (y_pred == 1)

    fn_scores = cache.scores[fn_mask]
    # highest-confidence FN = closest score to the catastrophic fail side but below
    # where true failing pred normal -> as score approaches threshold from below.
    # Most confident FN: lowest score among false negatives (predicted normal strongly).
    ordering_fn = np.argsort(fn_scores)[:top_n]
    fn_idxs = np.where(fn_mask)[0][ordering_fn]

    fp_scores = cache.scores[fp_mask]
    # highest-confidence FP: highest score among false positives (predicted failing strongly)
    ordering_fp = np.argsort(fp_scores)[::-1][:top_n]
    fp_idxs = np.where(fp_mask)[0][ordering_fp]

    # Borderline errors near threshold: |score - threshold| <= 0.15, any error.
    borderline_mask = (np.abs(cache.scores - selected_threshold) <= 0.15) & (
        y_pred != cache.y_true
    )
    borderline_idxs = np.where(borderline_mask)[0]
    borderline_scores = cache.scores[borderline_mask]
    # Rank by closeness to threshold
    closeness = np.abs(borderline_scores - selected_threshold)
    borderline_order = np.argsort(closeness)[:top_n]
    borderline_idxs = borderline_idxs[borderline_order]

    return {
        "false_negatives": _example_list(cache, fn_idxs, selected_threshold),
        "false_positives": _example_list(cache, fp_idxs, selected_threshold),
        "borderline_errors": _example_list(cache, borderline_idxs, selected_threshold),
    }


def _example_list(cache: ConfidenceCache, idxs: np.ndarray, threshold: float) -> List[dict]:
    out = []
    for i in idxs:
        out.append(
            {
                "frame_index": int(cache.frame_index[i]),
                "recording": cache.recording[i],
                "frame_name": cache.frame_name[i],
                "source_class": "failing" if cache.y_true[i] == 1 else "normal",
                "pred_class": "failing" if cache.scores[i] >= threshold else "normal",
                "probability": float(cache.scores[i]),
                "failure_type": cache.fine[i],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Brier score and expected calibration error (ECE) on validation.

    Interpretation helper: Brier 0 = perfect; 0.25 = no-skill for binary balanced.
    ECE = weighted average of |mean confidence - observed frequency| per bin.

    Minimum samples per bin is enforced; bins below the min are merged into the
    total reliability gap once at the end — a documented, standard collapse.
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    brier = float(brier_score_loss(y_true, scores))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_sum, ece_count, bin_reports = 0.0, 0.0, []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (scores >= lo) & (scores <= hi) if i == n_bins - 1 else (
            (scores >= lo) & (scores < hi)
        )
        n = int(mask.sum())
        if n == 0:
            bin_reports.append({"bin": f"{lo:.2f}-{hi:.2f}", "n_samples": 0})
            continue
        mean_conf = float(scores[mask].mean())
        obs_freq = float(y_true[mask].mean())
        gap = abs(mean_conf - obs_freq)
        weight = n / len(scores)
        ece_sum += weight * gap
        ece_count += n
        bin_reports.append(
            {
                "bin": f"{lo:.2f}-{hi:.2f}",
                "n_samples": int(n),
                "mean_confidence": mean_conf,
                "observed_frequency": obs_freq,
                "gap": gap,
            }
        )

    ece = ece_sum / n_bins if n_bins > 0 else 0.0
    return {
        "n_samples": int(len(scores)),
        "n_bins": int(n_bins),
        "brier_score": brier,
        "ece": ece,
        "bins": bin_reports,
    }


def reliability_analysis(
    calibration: dict,
    reliability_floor: float = 0.8,
) -> dict:
    """Lift-out the reliability window: overconfident-failing vs underconfident.

    Summarizes bins where mean confidence deviates meaningfully from observed
    frequency. Diagnostic only.
    """
    over = []
    under = []
    for b in calibration["bins"]:
        if b["n_samples"] == 0:
            continue
        gap = b["gap"]
        if gap >= reliability_floor * abs(b["mean_confidence"] - 0.5) and gap >= 0.05:
            if b["mean_confidence"] > b["observed_frequency"]:
                over.append(b)
            else:
                under.append(b)
    return {
        "overconfident_bins": over,
        "underconfident_bins": under,
        "systematic_biases": {
            "overconfident_failing": bool(over),
            "underconfident_failing": bool(under),
        },
    }


# ---------------------------------------------------------------------------
# Top-level analysis driver
# ---------------------------------------------------------------------------

TOP_LEVEL_KEYS = [
    "stage",
    "date",
    "checkpoint",
    "dataset_basis",
    "thresholds_evaluated",
    "validation",
    "test",
    "failure_type_breakdown",
    "recording_analysis",
    "recording_ranking",
    "problematic_recordings",
    "hard_examples",
    "calibration",
    "limitations",
]


def run_stage8(
    cache_dir: Path,
    thresholds: Optional[List[float]] = None,
    selected_metric: str = "f1",
) -> dict:
    """Run the full Stage-8 structured analysis and return a dict.

    The returned dict follow the schema documented in TOP_LEVEL_KEYS and is the
    direct source for the .json artifact.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS

    # 1. Load caches
    val_recordings = load_cached_predictions(cache_dir, "val")
    test_recordings = load_cached_predictions(cache_dir, "test")

    val_cache = ConfidenceCache.from_recordings(val_recordings)
    test_cache = ConfidenceCache.from_recordings(test_recordings)

    # 2. Validation threshold sweep
    val_sweep = thresholds_on_split(val_cache, thresholds)
    best = select_best_threshold(val_sweep, metric=selected_metric)
    selected_threshold = best.threshold

    # 3. Validation (at best)
    val_best = evaluate_threshold_metrics(val_cache.y_true, val_cache.scores, selected_threshold)

    # 4. Single frozen test evaluation at selected threshold
    test_selected = evaluate_threshold_metrics(
        test_cache.y_true, test_cache.scores, selected_threshold
    )

    # 5. Pre-existing 0.50 test result
    test_baseline = evaluate_threshold_metrics(test_cache.y_true, test_cache.scores, 0.50)

    # 6. Failure-type breakdown (test, selected)
    ftype = failure_type_breakdown(test_cache, selected_threshold)

    # 7. Recording analysis (test, selected)
    rec_analysis = recording_analysis(test_cache, selected_threshold)

    rec_rank = rank_recordings_by_error(rec_analysis)
    flagged = identify_problematic_recordings(rec_analysis)

    # 8. Hard examples (test, selected)
    hard = extract_hard_examples(test_cache, selected_threshold)

    # 9. Calibration (validation only)
    cal = calibration_metrics(val_cache.y_true, val_cache.scores)
    reliab = reliability_analysis(cal)

    return {
        "stage": 8,
        "checkpoint": str(
            Path(
                "experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/best.pt"
            )
        ),
        "dataset_basis": {
            "splits": ["val", "test"],
            "split_unit": "session/recording (solution atomicity preserved)",
            "labels": "source per-frame labels (authoritative)",
        },
        "thresholds_evaluated": [r.to_dict() for r in val_sweep],
        "validation": {
            "n_frames": val_cache.n_frames,
            "n_recordings": len(val_recordings),
            "selected_threshold": selected_threshold,
            "selected_metric": selected_metric,
            "selected_metrics": val_best.to_dict(),
        },
        "test": {
            "n_frames": test_cache.n_frames,
            "n_recordings": len(test_recordings),
            "baseline_0_50": test_baseline.to_dict(),
            "selected_threshold": selected_threshold,
            "selected_metrics": test_selected.to_dict(),
            "selection_made_on": "validation only (never test)",
        },
        "failure_type_breakdown": ftype,
        "recording_analysis": rec_analysis,
        "recording_ranking": rec_rank,
        "problematic_recordings": flagged,
        "hard_examples": hard,
        "calibration": {**cal, **reliab},
        "limitations": [
            "All numbers are frame-level on leaked-free data; recording-level is grouping metadata, not a detection task.",
            "Small failure-type groups (e.g. spaghetti) are not statistically meaningful.",
            "No model retrained, no calibration transform fitted for operational use.",
            "Results are dataset-internal; they do not imply real-world generalization.",
        ],
    }