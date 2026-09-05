"""Stage 10: cluster and domain generalization audit.

Analysis only. No model training, no checkpoint changes, no dataset or split
changes, no threshold optimization, no acquisition or hardware access.

This stage tests (attempts to falsify) the domain-diversity hypotheses raised in
Stage 9 for the frozen Experiment 1 binary classifier:

  H1  Stringing failures are hard because available stringing training data is
      clustered by date/session/domain.
  H2  The catastrophic stringing test recording reflects a domain shift not
      represented in the existing stringing training distribution.
  H3  The known PURE_NORMAL false-positive test recording is darker than the
      training normal distribution.
  H4  False-positive behavior increases systematically with image darkness.
  H5  Current under-extrusion data has enough diversity to serve as a
      comparison/control for stringing.

Methodological constraints honoured here:
  * Primary development-domain evidence comes from TRAIN and VAL only. The
    official TEST set is FROZEN and used solely to document existing Stage 8
    observations; no test statistic defines a bucket, threshold, acquisition
    choice, session count, or model change.
  * Image statistics are deterministic, non-learned (mean brightness, brightness
    median, luminance percentiles, per-channel means/std, std as contrast proxy).
    No classifier is trained on these features.
  * Brightness bucket boundaries are derived from the TRAIN normal distribution
    only.
  * Date is treated as a provenance/session label, never a causal factor.
  * Per-recording metrics use the authoritative per-frame labels (never a
    synthetic recording label). Within-train/val recall is reported
    descriptively; VAL is the primary development-domain generalization signal.
  * No estimate for spaghetti test performance is invented (test contains no
    spaghetti frames).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

# Canonical paths
REPO = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO / "datasets" / "processed" / "printing-errors" / "splits"
DATA_DIR = REPO / "datasets" / "processed" / "printing-errors"
CACHE_DIR = REPO / "docs" / "research" / "evalcaches"

THRESHOLD = 0.65  # primary operational threshold (Stage 8 val-selected); not re-optimized

FAILURE_TYPES = {"under_extrusion", "stringing", "spaghetti"}

# Known Stage 8 observations (documentation only; never used to set analysis params)
STAGE8_OBSERVATIONS = {
    "catastrophic_fn_recording": "Recording_15.02.2023_13_19_52",
    "catastrophic_fn_failure_type": "stringing",
    "catastrophic_fn_rate": 0.898,  # FNR at 0.65; ~114/127 failing frames missed
    "high_fp_recording": "Recording_20122022_17_05_36",
    "high_fp_composition": "PURE_NORMAL",
    "high_fp_note": "high-confidence false positives (p>0.98) despite pure normal label",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cached_predictions(split: str) -> Dict[str, dict]:
    """Load per-frame frozen predictions for a split.

    Returns recording_id -> {label, frames:[{name, ts, label, score, fine}]}.
    """
    path = CACHE_DIR / f"frozen_{split}_trajectories.json"
    if not path.exists():
        raise FileNotFoundError(f"cache not found: {path}")
    data = json.loads(Path(path).read_text())
    return data["recordings"]


def load_split_rows(split: str) -> List[dict]:
    csv_path = SPLITS_DIR / f"{split}.csv"
    return list(csv.DictReader(open(csv_path)))


_PATH_INDEX: Dict[str, Dict[str, Path]] = {}
_STATS_CACHE: Dict[str, Optional[dict]] = {}


def _build_path_index(split: str) -> Dict[str, Path]:
    """Build recording/name -> disk path index once per split."""
    if split in _PATH_INDEX:
        return _PATH_INDEX[split]
    idx: Dict[str, Path] = {}
    for row in load_split_rows(split):
        parts = row["path"].split("/")
        if len(parts) >= 3:
            idx[(parts[1], parts[2])] = DATA_DIR / row["path"]
    _PATH_INDEX[split] = idx
    return idx


def _get_image_path(split: str, recording: str, name: str) -> Optional[Path]:
    return _build_path_index(split).get((recording, name))


@dataclass
class Frame:
    name: str
    label: str
    fine: str
    score: float


@dataclass
class Recording:
    recording_id: str
    split: str
    frames: List[Frame] = field(default_factory=list)

    @property
    def composition(self) -> str:
        labels = {f.label for f in self.frames}
        if labels == {"normal"}:
            return "PURE_NORMAL"
        if labels == {"failing"}:
            return "PURE_FAILING"
        return "MIXED"

    @property
    def date(self) -> Optional[str]:
        return extract_date(self.recording_id)

    def frames_with_fine(self, fine: str) -> List[Frame]:
        return [f for f in self.frames if f.fine == fine]

    @property
    def n_frames(self) -> int:
        return len(self.frames)


def extract_date(recording_id: str) -> Optional[str]:
    """Extract the date token from a recording id.

    Naming convention (documented): recording ids follow
        ``Recording_<date>_<time>``
    where ``<date>`` is either ``ddMMyyyy`` (e.g. ``03012023``) or
    ``dd.MM.yyyy`` (e.g. ``14.02.2023``). The date token is the second
    underscore-delimited component.
    """
    if "Recording_" not in recording_id:
        return None
    parts = recording_id.split("_")
    if len(parts) < 2:
        return None
    return parts[1]


def load_recordings(splits: Sequence[str] = ("train", "val")) -> Dict[str, List[Recording]]:
    """Load cached predictions into Recording objects for the given splits."""
    out: Dict[str, List[Recording]] = {}
    for split in splits:
        recs = load_cached_predictions(split)
        rs = []
        for rec_id, r in recs.items():
            frames = [
                Frame(name=f["name"], label=f["label"], fine=f.get("fine", "unknown"),
                      score=float(f["score"]))
                for f in r["frames"]
            ]
            rs.append(Recording(recording_id=rec_id, split=split, frames=frames))
        out[split] = rs
    return out


# ---------------------------------------------------------------------------
# Recording-level metrics helpers
# ---------------------------------------------------------------------------

def recording_metrics(rec: Recording, fine: Optional[str] = None,
                      threshold: float = THRESHOLD) -> dict:
    """Per-recording metrics over (optionally fine-filtered) frames at threshold.

    Uses authoritative per-frame labels. Failing recall/FNR are only meaningful
    when the recording contains failing frames of the filtered type.
    """
    fr = rec.frames_with_fine(fine) if fine else rec.frames
    if not fr:
        return {"n_frames": 0, "present": False}
    scores = np.array([f.score for f in fr])
    lab = np.array([1 if f.label == "failing" else 0 for f in fr], dtype=int)
    pred = (scores >= threshold).astype(int)
    tp = int(((lab == 1) & (pred == 1)).sum())
    fn = int(((lab == 1) & (pred == 0)).sum())
    tn = int(((lab == 0) & (pred == 0)).sum())
    fp = int(((lab == 0) & (pred == 1)).sum())
    rec_val = tp / (tp + fn) if (tp + fn) else None
    fnr = fn / (fn + tp) if (fn + tp) else None
    fpr = fp / (fp + tn) if (fp + tn) else None
    return {
        "n_frames": len(fr),
        "present": True,
        "true_failing": tp + fn,
        "true_normal": tn + fp,
        "recall": round(float(rec_val), 4) if rec_val is not None else None,
        "fnr": round(float(fnr), 4) if fnr is not None else None,
        "fpr": round(float(fpr), 4) if fpr is not None else None,
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "score_q05": float(np.quantile(scores, 0.05)),
        "score_q95": float(np.quantile(scores, 0.95)),
    }


# ---------------------------------------------------------------------------
# Image statistics (deterministic, non-learned)
# ---------------------------------------------------------------------------

def image_stats(path: Path) -> Optional[dict]:
    """Compute deterministic brightness/contrast statistics for a single image."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    bgr = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    return {
        "brightness_mean": float(np.mean(gray)),
        "brightness_median": float(np.median(gray)),
        "brightness_std": float(np.std(gray)),
        "brightness_p05": float(np.percentile(gray, 5)),
        "brightness_p95": float(np.percentile(gray, 95)),
        "chan_mean": [float(np.mean(b)), float(np.mean(g)), float(np.mean(r))],
        "chan_std": [float(np.std(b)), float(np.std(g)), float(np.std(r))],
    }


def recording_image_stats(rec: Recording, normal_only: bool = True,
                          max_frames: int = 120) -> Optional[dict]:
    """Aggregate per-frame image statistics for a recording (normal frames by
    default), sampling at most ``max_frames`` frames for tractability."""
    key = f"{rec.split}/{rec.recording_id}/{normal_only}"
    if key in _STATS_CACHE:
        return _STATS_CACHE[key]
    frames = [f for f in rec.frames if f.label == "normal"] if normal_only else rec.frames
    if not frames:
        return None
    vals = []
    for f in frames[:max_frames]:
        p = _get_image_path(rec.split, rec.recording_id, f.name)
        if p is None or not p.exists():
            continue
        s = image_stats(p)
        if s is not None:
            vals.append(s)
    if not vals:
        return None
    agg = {}
    for key in ("brightness_mean", "brightness_median", "brightness_std",
                "brightness_p05", "brightness_p95"):
        agg[key] = round(float(np.mean([v[key] for v in vals])), 1)
    agg["n_frames_sampled"] = len(vals)
    _STATS_CACHE[key] = agg
    return agg


# ---------------------------------------------------------------------------
# 3/4. Stringing cluster + recording-to-recording analysis
# ---------------------------------------------------------------------------

def stringing_recordings(recordings: Dict[str, List[Recording]]) -> List[Recording]:
    rs = []
    for split in recordings:
        for rec in recordings[split]:
            if rec.frames_with_fine("stringing"):
                rs.append(rec)
    return rs


def stringing_cluster_analysis(recordings: Dict[str, List[Recording]]) -> dict:
    """Group stringing recordings by date cluster; report per-cluster behavior."""
    clusters: Dict[str, List[dict]] = defaultdict(list)
    for rec in stringing_recordings(recordings):
        fr = rec.frames_with_fine("stringing")
        m = recording_metrics(rec, fine="stringing")
        clusters[rec.date or "unknown"].append(
            {
                "recording_id": rec.recording_id,
                "split": rec.split,
                "date": rec.date,
                "composition": rec.composition,
                "stringing_frames": len(fr),
                "total_frames": rec.n_frames,
                **{k: m[k] for k in ("n_frames", "recall", "fnr", "true_failing",
                                     "score_mean", "score_median",
                                     "score_q05", "score_q95")},
            }
        )

    cluster_summary = {}
    for date, recs in sorted(clusters.items()):
        n_recs = len(recs)
        n_frames = sum(r["stringing_frames"] for r in recs)
        scores = [r["score_median"] for r in recs]
        fnrs = [r["fnr"] for r in recs if r["fnr"] is not None]
        exploratory = n_recs < 3 or n_frames < 50
        cluster_summary[date] = {
            "n_recordings": n_recs,
            "stringing_frames": n_frames,
            "splits": sorted({r["split"] for r in recs}),
            "score_median_mean": round(float(np.mean(scores)), 4) if scores else None,
            "score_median_min": round(float(np.min(scores)), 4) if scores else None,
            "score_median_max": round(float(np.max(scores)), 4) if scores else None,
            "fnr_mean": round(float(np.mean(fnrs)), 4) if fnrs else None,
            "fnr_min": round(float(np.min(fnrs)), 4) if fnrs else None,
            "fnr_max": round(float(np.max(fnrs)), 4) if fnrs else None,
            "exploratory": exploratory,
            "exploratory_note": (
                "Too few recordings/frames to draw a robust cluster conclusion."
                if exploratory else "Sufficient recordings/frames for a cluster comparison."
            ),
            "recordings": recs,
        }

    return {
        "grouping_note": (
            "Grouping is by the date token embedded in the recording id "
            "(Recording_<date>_<time>; date = ddMMyyyy or dd.MM.yyyy). "
            "Date is a provenance/session cluster label, NOT a causal factor."
        ),
        "clusters": cluster_summary,
    }


def stringing_variation(recordings: Dict[str, List[Recording]]) -> dict:
    """Recording-to-recording variation in stringing behavior (train + val)."""
    recs = [r for r in stringing_recordings(recordings)
            if r.frames_with_fine("stringing")]
    rows = []
    for rec in recs:
        fr = rec.frames_with_fine("stringing")
        m = recording_metrics(rec, fine="stringing")
        rows.append({
            "recording_id": rec.recording_id,
            "split": rec.split,
            "date": rec.date,
            "stringing_frames": len(fr),
            "recall": m["recall"], "fnr": m["fnr"],
            "score_mean": m["score_mean"], "score_median": m["score_median"],
            "score_q05": m["score_q05"], "score_q95": m["score_q95"],
        })
    # class A/B/C/D determination from VAL (primary dev-domain evidence)
    val_rows = [r for r in rows if r["split"] == "val"]
    val_fnrs = [r["fnr"] for r in val_rows if r["fnr"] is not None]
    val_split = {r["date"] for r in val_rows}
    interpretation = {
        "A": (
            "most stringing recordings handled reasonably, one/few domains fail badly"
        ),
        "B": "stringing broadly weak across recordings",
        "C": "performance differs strongly by date/provenance cluster",
        "D": "insufficient evidence",
    }
    return {
        "rows": rows,
        "n_train": sum(1 for r in rows if r["split"] == "train"),
        "n_val": sum(1 for r in rows if r["split"] == "val"),
        "val_note": (
            "VAL is the primary development-domain generalization evidence; "
            "training-set performance is descriptive only, not generalization proof."
        ),
        "val_fnrs": val_fnrs,
        "val_dates": sorted(val_split),
        "caveat": (
            "Determining which class (A/B/C/D) is best supported is a judgment "
            "based on the reported numbers; no automated claim is made."
        ),
    }


# ---------------------------------------------------------------------------
# 5/6. Darkness audit + brightness buckets
# ---------------------------------------------------------------------------

def normal_recording_stats(recordings: Dict[str, List[Recording]]) -> List[dict]:
    """Per-recording image-stat + false-positive summary for normal recordings."""
    rows = []
    for split in ("train", "val"):
        for rec in recordings.get(split, []):
            if not any(f.label == "normal" for f in rec.frames):
                continue
            m = recording_metrics(rec, fine=None)  # full recording metrics
            stats = recording_image_stats(rec, normal_only=True)
            if stats is None:
                continue
            rows.append({
                "recording_id": rec.recording_id,
                "split": split,
                "date": rec.date,
                "n_normal_frames": sum(1 for f in rec.frames if f.label == "normal"),
                "fp": m["fp"], "fpr": m["fpr"],
                "score_mean": m["score_mean"], "score_median": m["score_median"],
                "brightness_mean": stats["brightness_mean"],
                "brightness_std": stats["brightness_std"],
                "n_frames_sampled": stats["n_frames_sampled"],
            })
    return rows


def darkness_audit(recordings: Dict[str, List[Recording]]) -> dict:
    """Compare training vs validation normal brightness and FP behavior vs darkness."""
    rows = normal_recording_stats(recordings)
    train_norm = [r["brightness_mean"] for r in rows if r["split"] == "train"]
    val_norm = [r["brightness_mean"] for r in rows if r["split"] == "val"]

    # Correlation of brightness -> FPR among all normal recordings (train+val)
    br = [r["brightness_mean"] for r in rows]
    fprs = [r["fpr"] if r["fpr"] is not None else 0.0 for r in rows]
    corr = float(np.corrcoef(br, fprs)[0, 1]) if len(br) > 1 else None

    # Darkest normal recordings (train+val) and their FP behavior
    dark_sorted = sorted(rows, key=lambda r: r["brightness_mean"])
    darkest = dark_sorted[: min(10, len(dark_sorted))]

    return {
        "n_normal_recordings_train": len(train_norm),
        "n_normal_recordings_val": len(val_norm),
        "train_normal_brightness": {
            "mean": round(float(np.mean(train_norm)), 1) if train_norm else None,
            "median": round(float(np.median(train_norm)), 1) if train_norm else None,
            "min": round(float(np.min(train_norm)), 1) if train_norm else None,
            "max": round(float(np.max(train_norm)), 1) if train_norm else None,
        },
        "val_normal_brightness": {
            "mean": round(float(np.mean(val_norm)), 1) if val_norm else None,
            "median": round(float(np.median(val_norm)), 1) if val_norm else None,
            "min": round(float(np.min(val_norm)), 1) if val_norm else None,
            "max": round(float(np.max(val_norm)), 1) if val_norm else None,
        },
        "brightness_fpr_correlation": round(corr, 4) if corr is not None else None,
        "correlation_note": (
            "Negative correlation would indicate darker normal recordings trend "
            "toward higher FPR. Correlation over ~recordings is descriptive and "
            "does NOT establish causality or semantic OOD."
        ),
        "darkest_normal_recordings": darkest,
        "caveat": (
            "Brightness is a proxy, not proof of semantic domain shift; no "
            "causal claim is made."
        ),
    }


def brightness_buckets(recordings: Dict[str, List[Recording]]) -> dict:
    """Bucket normal recordings by TRAIN-derived brightness boundaries only."""
    rows = normal_recording_stats(recordings)
    train_br = [r["brightness_mean"] for r in rows if r["split"] == "train"]
    n = len(train_br)
    if n < 6:
        return {
            "available": False,
            "note": f"Insufficient train normal recordings (n={n}); not forcing buckets.",
        }
    q = np.quantile(train_br, [1 / 3, 2 / 3])
    lo, hi = float(q[0]), float(q[1])
    buckets = {"dark": {"low": None, "high": lo},
               "medium": {"low": lo, "high": hi},
               "bright": {"low": hi, "high": None}}
    out = {}
    for name, bnd in buckets.items():
        in_bucket = [r for r in rows
                     if (bnd["low"] is None or r["brightness_mean"] >= bnd["low"])
                     and (bnd["high"] is None or r["brightness_mean"] < bnd["high"])]
        n_recs = len(in_bucket)
        n_frames = sum(r["n_normal_frames"] for r in in_bucket)
        fps = [r["fp"] for r in in_bucket]
        fprs = [r["fpr"] if r["fpr"] is not None else 0.0 for r in in_bucket]
        scores = [r["score_mean"] for r in in_bucket]
        out[name] = {
            "boundary_low": bnd["low"], "boundary_high": bnd["high"],
            "n_recordings": n_recs,
            "n_frames": n_frames,
            "mean_brightness": round(float(np.mean([r["brightness_mean"] for r in in_bucket])), 1)
            if in_bucket else None,
            "total_fp": int(sum(fps)),
            "framewise_fpr": round(float(sum(fps) / n_frames), 4) if n_frames else None,
            "mean_recording_fpr": round(float(np.mean(fprs)), 4) if fprs else None,
            "mean_score": round(float(np.mean(scores)), 4) if scores else None,
            "recordings": [r["recording_id"] for r in in_bucket],
        }
    return {
        "available": True,
        "method": (
            "Bucket boundaries derived ONLY from the TRAIN normal brightness "
            "distribution (tertiles). No official TEST statistic was used. "
            "Explicitly avoiding tiny buckets; if a bucket is empty it is reported "
            "as empty rather than forcing a split."
        ),
        "training_tertile_boundaries": {"brightness_low": round(lo, 1), "brightness_high": round(hi, 1)},
        "buckets": out,
    }


# ---------------------------------------------------------------------------
# 7. Under-extrusion control
# ---------------------------------------------------------------------------

def under_extrusion_analysis(recordings: Dict[str, List[Recording]]) -> dict:
    """Recording-level under-extrusion metrics as a comparison/control."""
    ue_recs = []
    for split in recordings:
        for rec in recordings[split]:
            if rec.frames_with_fine("under_extrusion"):
                ue_recs.append(rec)
    row_data = []
    dates = []
    for rec in ue_recs:
        fr = rec.frames_with_fine("under_extrusion")
        m = recording_metrics(rec, fine="under_extrusion")
        dates.append(rec.date or "unknown")
        row_data.append({
            "recording_id": rec.recording_id,
            "split": rec.split,
            "date": rec.date,
            "ue_frames": len(fr),
            "recall": m["recall"], "fnr": m["fnr"],
            "score_median": m["score_median"],
            "score_q05": m["score_q05"], "score_q95": m["score_q95"],
        })
    fnrs = [r["fnr"] for r in row_data if r["fnr"] is not None]
    ue_splits = sorted(set(r["split"] for r in row_data))
    return {
        "n_recordings": len(row_data),
        "n_dates": len({d for d in dates}),
        "dates": sorted({d for d in dates}),
        "fnr_mean": round(float(np.mean(fnrs)), 4) if fnrs else None,
        "fnr_std": round(float(np.std(fnrs)), 4) if fnrs else None,
        "fnr_min": round(float(np.min(fnrs)), 4) if fnrs else None,
        "fnr_max": round(float(np.max(fnrs)), 4) if fnrs else None,
        "splits": ue_splits,
        "rows": row_data,
        "caveat": (
            "Comparison to stringing is descriptive; session count is NOT claimed "
            "as the sole explanation for any performance difference."
        ),
    }


# ---------------------------------------------------------------------------
# 8. Spaghetti limitation
# ---------------------------------------------------------------------------

def spaghetti_limitation(recordings: Dict[str, List[Recording]]) -> dict:
    train = recordings.get("train", [])
    val = recordings.get("val", [])
    test = load_cached_predictions("test") if (CACHE_DIR / "frozen_test_trajectories.json").exists() else {}
    train_sp = [r for r in train if r.frames_with_fine("spaghetti")]
    val_sp = [r for r in val if r.frames_with_fine("spaghetti")]
    test_rows = [r for r in test.values() if any(f.get("fine") == "spaghetti" for f in r["frames"])]
    return {
        "train_spaghetti": {
            "n_recordings": len(train_sp),
            "n_frames": sum(r.n_frames for r in train_sp),
        },
        "val_spaghetti": {
            "n_recordings": len(val_sp),
            "n_frames": sum(r.n_frames for r in val_sp),
        },
        "test_spaghetti": {
            "n_recordings": len(test_rows),
            "n_frames": sum(len(r["frames"]) for r in test_rows),
        },
        "conclusions": {
            "frozen_test_spaghetti_performance_measurable": False,
            "valid_spaghetti_generalization_conclusion": False,
            "note": (
                "Official test set contains no spaghetti frames, so frozen test "
                "spaghetti performance cannot be measured and no generalization "
                "conclusion is possible. No spaghetti performance estimate is invented. "
                "Spaghetti remains a future-data priority."
            ),
        },
    }


# ---------------------------------------------------------------------------
# 9/10. Documented test observations
# ---------------------------------------------------------------------------

def documented_test_observations() -> dict:
    """Re-document the two known Stage 8 final-test observations.

    These are EXISTING final-test evaluation observations, surfaced here purely
    for context. They must not be used to set any analysis parameter. Nothing in
    this stage optimizes against the test set.
    """
    test = load_cached_predictions("test")
    out = {}
    # Catastrophic FN
    cat = STAGE8_OBSERVATIONS["catastrophic_fn_recording"]
    if cat in test:
        fr = test[cat]["frames"]
        n = len(fr)
        failing = sum(1 for f in fr if f["label"] == "failing")
        missed = sum(1 for f in fr if f["label"] == "failing" and f["score"] < THRESHOLD)
        out["catastrophic_fn"] = {
            "recording_id": cat,
            "composition": test[cat]["label"],
            "n_frames": n,
            "true_failing": failing,
            "missed_at_065": missed,
            "fnr_approx": round(missed / failing, 3) if failing else None,
            "from_stage8": STAGE8_OBSERVATIONS["catastrophic_fn_rate"],
            "note": "Existing final-test observation; documented for context only.",
        }
    # High-FP normal
    hfp = STAGE8_OBSERVATIONS["high_fp_recording"]
    if hfp in test:
        fr = test[hfp]["frames"]
        n = len(fr)
        normal = sum(1 for f in fr if f["label"] == "normal")
        fp = sum(1 for f in fr if f["label"] == "normal" and f["score"] >= THRESHOLD)
        maxp = max((f["score"] for f in fr), default=None)
        out["high_fp"] = {
            "recording_id": hfp,
            "composition": test[hfp]["label"],
            "n_frames": n,
            "true_normal": normal,
            "fp_at_065": fp,
            "max_score": round(float(maxp), 4) if maxp is not None else None,
            "from_stage8": STAGE8_OBSERVATIONS["high_fp_note"],
            "note": "Existing final-test observation; documented for context only.",
        }
    out["firewall_note"] = (
        "Official test statistics are displayed for context ONLY. No test "
        "statistic defines a brightness bucket, selects a threshold, determines "
        "acquisition, sets session counts, or triggers a model change."
    )
    return out


# ---------------------------------------------------------------------------
# 13. Acquisition decision update
# ---------------------------------------------------------------------------

def acquisition_decision() -> dict:
    """Update of the Stage 9 acquisition recommendation, TRAIN/VAL-based."""
    return {
        "priorities": {
            "stringing": {
                "classification": "see stringing evidence summary",
                "need": "independent sessions",
                "detail": (
                    "Prefer genuinely independent stringing sessions (distinct "
                    "dates, objects, recording conditions, optical/camera "
                    "conditions) over additional frames from existing recordings, "
                    "because independent sessions add provenance/domain coverage "
                    "whereas extra frames from one session only add within-session "
                    "redundancy."
                ),
            },
            "dark_normal": {
                "classification": "see darkness summary",
                "need": "normal images across lighting conditions",
                "detail": (
                    "Add normal-domain images spanning the brightness range, "
                    "especially darker conditions, to broaden coverage."
                ),
            },
            "spaghetti": {
                "classification": "see spaghetti limitation",
                "need": "any independent spaghetti sessions",
                "detail": (
                    "Spaghetti has only 5 training recordings / 191 train frames "
                    "and no test frames; it remains a future-data priority with no "
                    "measurable frozen-test baseline."
                ),
            },
            "under_extrusion": {
                "classification": "see under-extrusion summary",
                "need": "maintain / modestly extend",
                "detail": (
                    "Under-extrusion has the most diverse training domain; treat it "
                    "as the comparison/control for the stringing decision."
                ),
            },
        },
        "classifications_defined": {
            "strongly_supported": "clear TRAIN/VAL evidence",
            "moderately_supported": "suggestive TRAIN/VAL evidence",
            "weakly_supported": "limited/ambiguous TRAIN/VAL evidence",
            "insufficient_evidence": "cannot conclude from available TRAIN/VAL data",
        },
    }


# ---------------------------------------------------------------------------
# 14. Future data split design
# ---------------------------------------------------------------------------

def future_split_design() -> dict:
    return {
        "principle": "Session atomicity preserved; never split frames of one session across sets.",
        "dimensions_to_hold_out": [
            "sessions",
            "dates",
            "objects",
            "lighting conditions",
            "environmental/recording conditions",
        ],
        "splits": {
            "train": "sessions for training",
            "validation": "sessions for dev-domain threshold/selection",
            "held_out": "sessions for final generalization testing",
        },
        "hold_out_note": (
            "The official Experiment 1 test set remains FROZEN. A future targeted "
            "experiment should use sessions reserved across dates/objects/lighting "
            "for its held-out generalization set; do not reuse Experiment 1 frozen "
            "test frames as training or validation data."
        ),
        "current_experiment_1_split_unchanged": True,
    }


# ---------------------------------------------------------------------------
# 15. Retraining gate
# ---------------------------------------------------------------------------

def retraining_gate() -> dict:
    return {
        "A_retraining_justified": False,
        "B_first_data_to_add": (
            "Independent stringing sessions across new dates/objects/conditions "
            "('5-10 minimum, 15-20 strong' from Stage 9); plus targeted dark-normal "
            "sessions and any available spaghetti sessions."
        ),
        "C_still_missing": (
            "Evidence that added stringing diversity reduces stringing FNR and "
            "that added dark-normal coverage reduces normal FPR, without regressing "
            "overall F1, as measured on FROZEN independent evaluation sets."
        ),
        "D_minimum_useful_experiment": (
            "Collect proposed new sessions; freeze a new split by session with a "
            "held-out set; retrain the same architecture on the expanded training "
            "set; evaluate on the FROZEN Experiment 1 test at 0.50 and 0.65."
        ),
        "E_result_that_justifies_retraining": (
            "Stringing FNR improves by >=5pp (ideally >=10pp) with no F1 regression, "
            "dark-normal FPR falls, and the held-out set performs comparably to the "
            "frozen test."
        ),
        "notes": [
            "No architecture change is recommended unless the evidence specifically supports it.",
            "Temporal/FSM logic is NOT proposed as a solution to the underlying frame-classification problem.",
        ],
    }


# ---------------------------------------------------------------------------
# 11. Optional image-distribution analysis
# ---------------------------------------------------------------------------

def image_distribution_analysis(recordings: Dict[str, List[Recording]]) -> dict:
    """Deterministic per-channel/luminance statistics across normal frames.

    Aggregates from the cached per-recording image stats to avoid redundant I/O.
    Returns channel-mean/std and luminance summaries for the training normal set.
    """
    train_normal = [r for r in recordings.get("train", [])
                    if any(f.label == "normal" for f in r.frames)]
    chan_acc = []
    lum_acc = []
    sampled = 0
    for rec in train_normal:
        s = recording_image_stats(rec, normal_only=True)
        if s is None:
            continue
        # recompute channel stats by sampling frames of this recording
        frames = [f for f in rec.frames if f.label == "normal"][:120]
        cvals = []
        for f in frames:
            p = _get_image_path(rec.split, rec.recording_id, f.name)
            st = image_stats(p) if (p is not None and p.exists()) else None
            if st:
                cvals.append(st)
        for st in cvals:
            chan_acc.append(st["chan_mean"])
            lum_acc.append((st["brightness_mean"], st["brightness_median"], st["brightness_std"]))
            sampled += 1
    if sampled == 0:
        return {
            "n_normal_frames_sampled": 0,
            "method": "Deterministic non-learned statistics only.",
            "luminance": None,
            "rgb_channel_mean_bgr": None,
            "rgb_channel_std_bgr": None,
        }
    cm = np.array(chan_acc)  # (n,3)
    lum = np.array(lum_acc)
    return {
        "n_normal_frames_sampled": sampled,
        "method": (
            "Deterministic non-learned statistics only; no PCA, clustering, or "
            "learned embeddings are used. Analysis is intentionally kept simple."
        ),
        "luminance": {
            "mean_brightness_mean": round(float(np.mean(lum[:, 0])), 1),
            "mean_brightness_std": round(float(np.mean(lum[:, 2])), 1),
        },
        "rgb_channel_mean_bgr": [round(float(np.mean(cm[:, i])), 1) for i in range(3)],
        "rgb_channel_std_bgr": [round(float(np.mean(np.std(cm, axis=0)[i])), 1) for i in range(3)],
    }


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_stage10() -> dict:
    recordings = load_recordings(("train", "val"))
    report = {
        "stage": 10,
        "title": "Cluster and Domain Generalization Audit",
        "model": "shufflenet_v2_x1_0_printing-errors_cdc9e2cad4 (frozen, Experiment 1)",
        "threshold": THRESHOLD,
        "evidence_splits": ["train", "val"],
        "test_frozen": True,
        "stringing_cluster_analysis": stringing_cluster_analysis(recordings),
        "stringing_variation": stringing_variation(recordings),
        "darkness_audit": darkness_audit(recordings),
        "brightness_buckets": brightness_buckets(recordings),
        "under_extrusion": under_extrusion_analysis(recordings),
        "spaghetti_limitation": spaghetti_limitation(recordings),
        "documented_test_observations": documented_test_observations(),
        "image_distribution": image_distribution_analysis(recordings),
        "acquisition_decision": acquisition_decision(),
        "future_split_design": future_split_design(),
        "retraining_gate": retraining_gate(),
    }
    return report
