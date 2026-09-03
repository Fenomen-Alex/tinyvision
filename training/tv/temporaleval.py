"""Stage 3 offline temporal-decision evaluation on a frozen frame classifier.

Runs a frozen checkpoint once over a split, orders each recording's frames by
capture timestamp, and evaluates raw vs. EWMA vs. EWMA+FSM decision logic with
sequence-level and frame-level metrics.

The inference results are cached to a JSON file so parameter sweeps reuse the
same frozen scores and never retrain or alter the frozen test set.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .temporal import FAILING_EVENT, run_temporal_pipeline

TS_RE = re.compile(r"_(\d+)\.(?:png|jpg|jpeg)$")

# Raw-CSV fine-class meanings (dataset's own per-image label).
FINE_CLASS_NAMES = {"0": "good", "1": "under_extrusion", "2": "stringing", "4": "spaghetti"}


def _timestamp_from_name(name: str) -> Optional[int]:
    m = TS_RE.search(name)
    return int(m.group(1)) if m else None


def load_split_rows(split_csv: Path) -> List[dict]:
    with open(split_csv) as f:
        return list(csv.DictReader(f))


def build_proxy_meta(data_dir, split: str):
    """Join the frozen split to the raw metadata CSV (by image filename).

    Returns {image_name: {"class":..., "recording":...}} for every split image.
    Used ONLY for analysis of proxy onsets, not as ground-truth labels.
    """
    from tv.robustness import load_resolved_metadata

    resolved = load_resolved_metadata(data_dir, split)
    return {r["image"]: r for r in resolved}


# ---------------------------------------------------------------------------
# Inference (runs the frozen checkpoint once).
# ---------------------------------------------------------------------------

def run_frozen_inference(config, checkpoint_path, split, output_json=None):
    """Run the frozen model over a split and cache per-frame fail-scores + meta.

    Returns a dict: {recording: {"label":..., "frames": [{ts, name, score, fine}], ...}}
    ordered by capture timestamp.
    """
    import torch
    from .models.factory import build_classifier
    from .data.loaders import get_classification_loaders
    from .deepeval import _get_device

    device = _get_device()
    split_dir = Path(config.dataset.data_dir) / "splits"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_to_idx = ckpt.get("class_to_idx", {})
    fail_idx = None
    for k, v in class_to_idx.items():
        if k.lower() == "failing" or v == 0:
            fail_idx = int(v)
            break
    if fail_idx is None:
        fail_idx = 0
    model = build_classifier(ckpt.get("config_model", config.model), 2, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    loader, _ = get_classification_loaders(
        split_dir,
        split,
        image_size=config.dataset.image_size,
        batch_size=config.data.batch_size,
        num_workers=0,
        pin_memory=False,
        augmentation={},
    )

    rows = load_split_rows(Path(config.dataset.data_dir) / "splits" / f"{split}.csv")
    proxy = build_proxy_meta(config.dataset.data_dir, split)

    rec_frames: Dict[str, list] = {}
    with torch.no_grad():
        idx = 0
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            for i in range(inputs.size(0)):
                row = rows[idx]
                idx += 1
                parts = row["path"].split("/")
                rec = parts[1] if len(parts) >= 3 else "unknown"
                name = Path(row["path"]).name
                meta = proxy.get(name, {})
                rec_frames.setdefault(rec, []).append(
                    {
                        "name": name,
                        "ts": _timestamp_from_name(name),
                        "label": row["label"],
                        "score": float(probs[i][fail_idx]),
                        "fine": meta.get("failure_type", "normal"),
                    }
                )
    # sort each recording by capture timestamp
    for rec in rec_frames:
        rec_frames[rec] = sorted(rec_frames[rec], key=lambda f: (f["ts"] is None, f["ts"]))
    rec_labels = {rec: frames[0]["label"] for rec, frames in rec_frames.items()}
    result = {
        "split": split,
        "checkpoint": str(checkpoint_path),
        "recordings": {
            rec: {"label": rec_labels[rec], "frames": frames}
            for rec, frames in rec_frames.items()
        },
    }
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
    return result


# ---------------------------------------------------------------------------
# Decision variants.
# ---------------------------------------------------------------------------

def _raw_decision(scores, threshold: float) -> List[dict]:
    out = []
    for i, s in enumerate(scores):
        out.append(
            {
                "frame": i,
                "raw": s,
                "state": "FAILING" if s >= threshold else "NORMAL",
                "event": FAILING_EVENT if s >= threshold else None,
                "signal": s,
                "delta": s,
            }
        )
    return out


def _ewma_only(scores, alpha: float, threshold: float) -> List[dict]:
    from .temporal import ewma_series

    sig = ewma_series(scores, alpha)
    out = []
    for i, s in enumerate(sig):
        out.append(
            {
                "frame": i,
                "raw": scores[i],
                "signal": s,
                "delta": s,
                "state": "FAILING" if s >= threshold else "NORMAL",
                "event": FAILING_EVENT if s >= threshold else None,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------

def _order_frames(rec):
    return rec["frames"]


def classify_recording(rec) -> str:
    """Classify a recording's ground-truth reliability.

    Returns one of:
      * "failing"           - folder label failing (reliable positive truth).
      * "normal_confirmed"  - folder normal AND every frame fine=='good'
                              (reliable negative truth).
      * "normal_ambiguous"  - folder normal but contains non-good fine frames
                              (NOT usable as a false-alarm ground truth).
      * "unknown"           - no usable class signal.
    """
    label = rec.get("label")
    frames = _order_frames(rec)
    has_defect = any(
        f.get("fine") not in (None, "normal", "good", "unknown") for f in frames
    )
    if label == "failing":
        return "failing"
    if label == "normal":
        return "normal_ambiguous" if has_defect else "normal_confirmed"
    return "unknown"


def _proxy_onset_index(rec) -> Optional[int]:
    """First frame index (time-ordered) of a genuine good->defect transition.

    Returns the index of the first non-'good' frame ONLY if the recording
    starts in the 'good' state (i.e. there is a real normal->defect boundary).
    Returns None for recordings that are constantly defect (no onset) or have
    no usable class signal. Uses the dataset's per-frame class label as a
    PROXY only -- it is NOT treated as verified temporal ground truth.
    """
    frames = _order_frames(rec)
    if not frames:
        return None
    first_fine = frames[0].get("fine")
    if first_fine not in (None, "normal", "good"):
        return None  # already failing at start: no boundary in this recording
    for i, f in enumerate(frames):
        if f.get("fine") not in (None, "normal", "good", "unknown"):
            return i
    return None


def _event_frames(decisions, state="FAILING"):
    """Frame indices where the pipeline is in the given state."""
    return [d["frame"] for d in decisions if d["state"] == state]


def _first_event_frame(decisions):
    for d in decisions:
        if d.get("event") == FAILING_EVENT:
            return d["frame"]
    return None


def evaluate_recording(rec, params: dict, variant: str) -> dict:
    """Evaluate one recording's decision logic. Returns structured results."""
    scores = [f["score"] for f in _order_frames(rec)]
    label = rec["label"]
    reliability = classify_recording(rec)
    truth = 1 if reliability == "failing" else 0  # reliable positive truth only

    if variant == "raw":
        decisions = _raw_decision(scores, params.get("threshold", 0.5))
    elif variant == "ewma":
        decisions = _ewma_only(scores, params.get("alpha", 0.15), params.get("threshold", 0.5))
    else:
        decisions = run_temporal_pipeline(
            scores,
            alpha=params.get("alpha", 0.15),
            baseline_window=params.get("baseline_window", 60),
            warmup=params.get("warmup", 30),
            suspect_threshold=params.get("suspect_threshold", 0.05),
            fail_threshold=params.get("fail_threshold", 0.2),
            recover_threshold=params.get("recover_threshold", 0.05),
            fail_threshold_abs=params.get("fail_threshold_abs"),
            confirm=params.get("confirm", 3),
            use_delta=params.get("use_delta", True),
        )

    failing_frames = _event_frames(decisions, "FAILING")
    first_event = _first_event_frame(decisions)
    nf = len(scores)

    seq_detected = first_event is not None
    # frame-level FNR/FPR relative to folder label per frame
    if truth == 1:
        fn_frames = nf - len(failing_frames)
        fnr = fn_frames / nf if nf else 0.0
        fpr = None
    else:
        fp_frames = len(failing_frames)
        fpr = fp_frames / nf if nf else 0.0
        fnr = None

    proxy_on = _proxy_onset_index(rec)
    delay = None
    if proxy_on is not None and first_event is not None and truth == 1:
        delay = first_event - proxy_on  # proxy-relative, frames

    return {
        "recording": None,
        "label": label,
        "reliability": reliability,
        "n": nf,
        "truth": truth,
        "seq_detected": seq_detected,
        "first_event_frame": first_event,
        "failing_frames": len(failing_frames),
        "fnr": fnr,
        "fpr": fpr,
        "proxy_onset": proxy_on,
        "delay": delay,
    }


def evaluate_split(result: dict, params: dict, variant: str) -> dict:
    """Run a decision variant over all recordings in a cached inference result.

    Sequence-level detection rate is computed only over ``failing`` recordings
    (reliable positive truth); false-alarm rate only over ``normal_confirmed``
    recordings (reliable negative truth). ``normal_ambiguous`` recordings are
    excluded from both because their folder 'normal' label conflicts with the
    source dataset's own per-frame defect labels.
    """
    per_rec = {}
    for rec, payload in result["recordings"].items():
        r = evaluate_recording(payload, params, variant)
        r["recording"] = rec
        per_rec[rec] = r

    n = len(per_rec)

    # Reliable positive: folder-failing recordings (all concordant).
    fail_seq = [v for v in per_rec.values() if v["reliability"] == "failing"]
    detected = [v for v in fail_seq if v["seq_detected"]]

    # Reliable negative: folder-normal with all-good fine labels.
    norm_conf = [v for v in per_rec.values() if v["reliability"] == "normal_confirmed"]
    false_alarm = [v for v in norm_conf if v["failing_frames"] > 0]

    ambigu = [v for v in per_rec.values() if v["reliability"] == "normal_ambiguous"]

    # frame-level aggregate over reliable subsets
    fail_fnr = [v["fnr"] for v in per_rec.values() if v["reliability"] == "failing" and v["fnr"] is not None]
    norm_fpr = [v["fpr"] for v in per_rec.values() if v["reliability"] == "normal_confirmed" and v["fpr"] is not None]
    delays = [v["delay"] for v in per_rec.values() if v["delay"] is not None]

    def _mean(x):
        return round(float(np.mean(x)), 4) if x else None

    return {
        "split": result.get("split"),
        "variant": variant,
        "params": params,
        "n_recordings": n,
        "n_failing_seq": len(fail_seq),
        "n_normal_confirmed_seq": len(norm_conf),
        "n_normal_ambiguous_seq": len(ambigu),
        "sequence_detection_rate": round(len(detected) / len(fail_seq), 4) if fail_seq else None,
        "n_detected_seq": len(detected),
        "missed_failure_events": len(fail_seq) - len(detected),
        "sequence_false_alarm_rate": round(len(false_alarm) / len(norm_conf), 4) if norm_conf else None,
        "n_false_trigger_seq": len(false_alarm),
        "mean_seq_fnr": _mean(fail_fnr),
        "mean_seq_fpr": _mean(norm_fpr),
        "proxy_delay_frames_mean": round(float(np.mean(delays)), 2) if delays else None,
        "proxy_delay_frames_median": round(float(np.median(delays)), 2) if delays else None,
        "n_delay_measurable": len(delays),
    }
