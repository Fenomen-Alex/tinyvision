"""Stage 5 corrected evaluation: frame-level ground-truth baseline + temporal logic.

The source dataset has **no recording-level label**; the authoritative truth is
each frame's source class (binary normal/failing, plus the fine class). Recording
IDs are grouping/sequence metadata only. This module therefore:

* computes a **frame-level** classification baseline stratified by recording
  composition (PURE_NORMAL / MIXED / PURE_FAILING), failure type, and recording;
* re-evaluates previously-"catastrophic" recordings against per-frame labels;
* evaluates raw / EWMA / EWMA+FSM temporal logic against **per-frame labels**
  (never a synthetic recording label), reporting frame-level and run metrics.

Run/event metrics use a documented contiguous-run definition at the frame level.
Detection "delay" is NOT reported (no verified onset ground truth).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .metrics import compute_classification_metrics
from .temporal import ewma_series, run_temporal_pipeline
from .temporaleval import _ewma_only, _raw_decision

# Composition labels reused from the Stage 4 forensic audit.
PURE_NORMAL = "PURE_NORMAL"
MIXED = "MIXED"
PURE_FAILING = "PURE_FAILING"

# Temporal states that count as a "failing" binary prediction.
FAILING_STATES = {"FAILING"}


@dataclass
class RunMetrics:
    """Contiguous-run statistics over a per-frame binary decision sequence.

    Run definition (documented): a maximal contiguous block of frames sharing the
    same *flag* (false-alarm = predicted failing while true label is normal;
    missed = predicted normal while true label is failing). A flag run of length
    1 is a singleton run; every frame belongs to at most one run of each kind.
    """
    false_alarm_frames: int = 0
    false_alarm_runs: int = 0
    longest_false_alarm_run: int = 0
    missed_failing_frames: int = 0
    missed_runs: int = 0
    longest_missed_run: int = 0
    n_state_transitions: int = 0


def _binary_pred_from_states(states: List[str]) -> np.ndarray:
    """Map per-frame temporal states to binary predictions (1=failing)."""
    return np.array([1 if s in FAILING_STATES else 0 for s in states], dtype=int)


def _frame_labels(frames: List[dict]) -> tuple:
    """Authoritative per-frame labels from the cache (never the recording label)."""
    y = np.array([1 if f["label"] == "failing" else 0 for f in frames], dtype=int)
    fine = [f.get("fine", "unknown") for f in frames]
    scores = np.array([f["score"] for f in frames], dtype=float)
    return y, fine, scores


def _run_stats(flags: List[bool]) -> dict:
    """Contiguous-run stats for a boolean flag sequence.

    A run is a maximal contiguous block of True values. Tracks count of runs,
    total True frames, and the longest run.
    """
    runs = 0
    cur = 0
    longest = 0
    total = 0
    for flag in flags:
        if flag:
            total += 1
            cur += 1
            longest = max(longest, cur)
        else:
            if cur > 0:
                runs += 1
            cur = 0
    if cur > 0:
        runs += 1
    return {"runs": runs, "frames": total, "longest": longest}


def _state_transitions(states: List[str]) -> int:
    n = 0
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            n += 1
    return n


def _variant_decisions(scores, variant: str, params: dict) -> Dict[str, list]:
    """Produce per-frame {state, event, signal, raw} for a variant."""
    sc = list(scores)
    if variant == "raw":
        thr = params.get("threshold", 0.5)
        return {"states": [d["state"] for d in _raw_decision(sc, thr)],
                "signal": sc, "events": [d["event"] for d in _raw_decision(sc, thr)]}
    if variant == "ewma":
        alpha = params.get("alpha", 0.15)
        thr = params.get("ewma_threshold", 0.5)
        d = _ewma_only(sc, alpha, thr)
        return {"states": [x["state"] for x in d], "signal": [x["signal"] for x in d],
                "events": [x["event"] for x in d]}
    # fsm
    d = run_temporal_pipeline(
        sc,
        alpha=params.get("alpha", 0.15),
        baseline_window=params.get("baseline_window", 60),
        warmup=params.get("warmup", 30),
        suspect_threshold=params.get("suspect_threshold", 0.05),
        fail_threshold=params.get("fail_threshold", 0.20),
        recover_threshold=params.get("recover_threshold", 0.05),
        fail_threshold_abs=params.get("fail_threshold_abs"),
        confirm=params.get("confirm", 3),
        use_delta=params.get("use_delta", True),
    )
    return {"states": [x["state"] for x in d], "signal": [x["signal"] for x in d],
            "events": [x["event"] for x in d]}


def compose_labels() -> Dict[str, str]:
    """Load recording compositions from the Stage 4 audit (authoritative)."""
    path = Path(__file__).resolve().parents[2] / "docs" / "research" / "stage4-recording-audit.json"
    audit = json.loads(path.read_text())
    out = {}
    for rec, info in audit["recordings"].items():
        c = info["composition"]
        if c in ("PURE_NORMAL", "PURE_FAILING"):
            out[rec] = c
        else:  # MIXED_NORMAL_LABEL / MIXED_FAILING_LABEL / UNKNOWN
            out[rec] = MIXED
    return out


def _load_recordings(split: str, cache_dir: Path) -> Dict[str, dict]:
    path = cache_dir / f"frozen_{split}_trajectories.json"
    data = json.loads(path.read_text())
    return data["recordings"]


@dataclass
class FrameEval:
    variant: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_scores: Optional[np.ndarray]
    labels: List[str]          # composition per frame
    fine: List[str]            # fine class per frame
    recording: List[str]       # recording per frame
    runs: RunMetrics = field(default_factory=RunMetrics)
    n_recording_state_transitions: int = 0

    def _cm(self, yt, yp):
        m = compute_classification_metrics(yt, yp, y_scores=None)
        cm = m.confusion_matrix
        # confusion matrix with labels=[0,1] -> [[TN,FP],[FN,TP]]
        return m, cm

    def metric_dict(self, subset_mask=None) -> dict:
        yt = self.y_true if subset_mask is None else self.y_true[subset_mask]
        yp = self.y_pred if subset_mask is None else self.y_pred[subset_mask]
        if len(yt) == 0:
            return {"sample_count": 0, "note": "no frames in subset"}
        sc = None
        if self.y_scores is not None and subset_mask is not None:
            # AUROC needs the subset scores
            sc = self.y_scores[subset_mask]
        m = compute_classification_metrics(yt, yp, sc)
        return m.to_dict()

    def full_metrics(self) -> dict:
        m = compute_classification_metrics(self.y_true, self.y_pred, self.y_scores)
        d = m.to_dict()
        d["runs"] = {
            "false_alarm_frames": self.runs.false_alarm_frames,
            "false_alarm_runs": self.runs.false_alarm_runs,
            "longest_false_alarm_run": self.runs.longest_false_alarm_run,
            "missed_failing_frames": self.runs.missed_failing_frames,
            "missed_runs": self.runs.missed_runs,
            "longest_missed_run": self.runs.longest_missed_run,
        }
        return d


def _compute_runs_for_mask(frames, y_true, y_pred, states):
    """Compute run statistics per recording, summed, plus recording transitions.

    FALSE-ALARM RUN = predicted FAILING while true label is normal.
    MISSED RUN      = predicted NORMAL while true label is failing.
    State transitions use the **actual temporal state sequence** (raw/EWMA have
    binary states; FSM has OBSERVING/SUSPECT/FAILING/NORMAL) so FSM's richer
    state machinery is reflected in the transition count.
    """
    runs = RunMetrics()
    total_transitions = 0
    # group frames by recording (they arrive grouped already per record, but be safe)
    by_rec = defaultdict(list)
    for i, (f, yt, yp, st) in enumerate(zip(frames, y_true, y_pred, states)):
        by_rec[f["recording"]].append((i, yt, yp, st))
    for rec, items in by_rec.items():
        items.sort(key=lambda x: x[0])
        fa_flags = []
        miss_flags = []
        st_seq = []
        for _i, yt, yp, st in items:
            fa_flags.append(bool(yt == 0 and yp == 1))
            miss_flags.append(bool(yt == 1 and yp == 0))
            st_seq.append(st)
        fa = _run_stats(fa_flags)
        ms = _run_stats(miss_flags)
        runs.false_alarm_frames += fa["frames"]
        runs.false_alarm_runs += fa["runs"]
        runs.longest_false_alarm_run = max(runs.longest_false_alarm_run, fa["longest"])
        runs.missed_failing_frames += ms["frames"]
        runs.missed_runs += ms["runs"]
        runs.longest_missed_run = max(runs.longest_missed_run, ms["longest"])
        total_transitions += _state_transitions(st_seq)
    return runs, total_transitions


def evaluate_split_corrected(
    split: str,
    cache_dir: Path,
    params: dict,
    variant: str,
    compositions: Optional[Dict[str, str]] = None,
) -> dict:
    """Evaluate one split against per-frame labels for a decision variant.

    Returns a structured dict with overall metrics, composition-stratified
    metrics, failure-type metrics, and per-recording metrics.
    """
    recordings = _load_recordings(split, cache_dir)
    if compositions is None:
        compositions = compose_labels()

    # Flatten per-frame arrays (ordered by recording, then original frame order).
    y_true_list, fine_list, rec_list, comp_list, score_list = [], [], [], [], []
    order = sorted(recordings.keys())
    for rec in order:
        frames = recordings[rec]["frames"]
        y, fine, scores = _frame_labels(frames)
        comp = compositions.get(rec, MIXED)
        for j in range(len(frames)):
            # capture recording into frame dict for run computation
            frames[j]["recording"] = rec
            y_true_list.append(y[j])
            fine_list.append(fine[j])
            rec_list.append(rec)
            comp_list.append(comp)
            score_list.append(scores[j])

    y_true = np.array(y_true_list, dtype=int)
    comp = np.array(comp_list)
    fine_arr = np.array(fine_list)
    rec_arr = np.array(rec_list)

    dec = _variant_decisions(score_list, variant, params)
    states = dec["states"]
    y_pred = _binary_pred_from_states(states)
    # For raw/ewma, y_scores = signal score (for AUROC). For fsm, no continuous score.
    if variant == "raw":
        y_scores = np.array(score_list, dtype=float)
    elif variant == "ewma":
        y_scores = np.array(dec["signal"], dtype=float)
    else:
        y_scores = None

    frames_flat = [f for rec in order for f in recordings[rec]["frames"]]
    runs, n_transitions = _compute_runs_for_mask(frames_flat, y_true, y_pred, states)

    eval_obj = FrameEval(
        variant=variant,
        y_true=y_true,
        y_pred=y_pred,
        y_scores=y_scores,
        labels=comp_list,
        fine=fine_list,
        recording=rec_list,
        runs=runs,
        n_recording_state_transitions=n_transitions,
    )

    overall = eval_obj.full_metrics()

    # Composition stratification.
    comp_metrics = {}
    for c in (PURE_NORMAL, MIXED, PURE_FAILING):
        mask = comp == c
        comp_metrics[c] = eval_obj.metric_dict(mask)

    # Failure-type stratification (per-frame fine class).
    ftype = {}
    for fn in sorted(set(fine_arr)):
        mask = fine_arr == fn
        ftype[fn] = eval_obj.metric_dict(mask)

    # Recording-level metrics.
    per_recording = {}
    for rec in order:
        mask = rec_arr == rec
        per_recording[rec] = eval_obj.metric_dict(mask)

    return {
        "split": split,
        "variant": variant,
        "n_frames": int(len(y_true)),
        "n_recordings": len(order),
        "negatives": int((y_true == 0).sum()),
        "positives": int((y_true == 1).sum()),
        "overall": overall,
        "by_composition": comp_metrics,
        "by_failure_type": ftype,
        "by_recording": per_recording,
        "state_transitions": n_transitions,
    }


def evaluate_catastrophic_recording(rec: str, split: str, cache_dir: Path, params: dict,
                                    variant: str) -> dict:
    """Evaluate a single recording against per-frame labels (Part B)."""
    recordings = _load_recordings(split, cache_dir)
    if rec not in recordings:
        # allow prefix match
        candidates = [k for k in recordings if k == rec or k.endswith(rec)]
        if not candidates:
            raise KeyError(f"recording {rec!r} not found in {split}")
        rec = candidates[0]
    frames = recordings[rec]["frames"]
    y, fine, scores = _frame_labels(frames)
    dec = _variant_decisions(scores.tolist(), variant, params)
    y_pred = _binary_pred_from_states(dec["states"])
    return {
        "recording": rec,
        "split": split,
        "variant": variant,
        "n_frames": int(len(frames)),
        "true_failing_frames": int((y == 1).sum()),
        "true_normal_frames": int((y == 0).sum()),
        "metrics": compute_classification_metrics(y, y_pred, scores if variant in ("raw", "ewma") else None).to_dict(),
        "fine_counts": dict(Counter(fine)),
    }
