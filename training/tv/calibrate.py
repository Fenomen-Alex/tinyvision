"""Per-printer threshold calibration from normal + failing target clips.

Selects a per-frame ``P(failing)`` decision threshold using **complete
print/session clips** captured with one target printer/camera setup. Frames are
never split: each clip is one session and is assigned atomically to either the
``calibration`` or the ``holdout`` split.

The command answers, for the target setup only:

  * at the selected threshold, how many **normal prints** raise a false alarm
    (false-alarm rate per print, with a Wilson interval),
  * what fraction of **failing prints** are detected (detection rate, Wilson),
  * the **detection latency** (seconds from clip start to alarm),
  * per-print detail so every decision is auditable.

Selection uses **calibration sessions only**; holdout sessions are reported but
never influence the chosen threshold. The report always carries an explicit
warning when the data is insufficient or single-class, and an explicit
disclaimer that per-printer calibration does **not** establish cross-printer
generalization.

This module does not train anything and never touches a locked test set.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .reliability import wilson_interval

CALIBRATION_SPLITS = ("calibration", "holdout")
VALID_LABELS = ("normal", "failing")


class CalibrationError(Exception):
    """Actionable, user-facing calibration error."""


@dataclass
class CalibrationClip:
    """One complete print/session clip with a print-level label."""

    id: str
    label: str
    source: str
    printer: str | None = None
    camera: str | None = None
    split: str = "calibration"
    # Optional precomputed per-frame scores: [{"timestamp_s": float, "score": float}]
    # or a flat list of scores. When present, no model/video is required.
    scores: list | None = None


def load_clip_spec(path: str | Path) -> dict:
    """Load a calibration spec (YAML or JSON).

    Expected shape::

        printer: printer-A
        camera: cam-0
        checkpoint: /path/to/best.pt     # optional; --checkpoint overrides
        clips:
          - {id: n01, label: normal, source: /path/n01.mp4, split: calibration}
          - {id: f01, label: failing, source: /path/f01.mp4, split: holdout}

    ``source`` may be a video file/glob or a directory of ordered frames; a clip
    may instead carry precomputed ``scores``. Raises :class:`CalibrationError`
    on malformed input.
    """
    p = Path(path)
    if not p.exists():
        raise CalibrationError(f"calibration spec not found: {p}")
    text = p.read_text()
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise CalibrationError(f"could not parse YAML spec {p}: {exc}") from exc
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CalibrationError(f"could not parse JSON spec {p}: {exc}") from exc

    if not isinstance(data, dict) or "clips" not in data:
        raise CalibrationError(
            f"spec {p} must be a mapping with a 'clips' list of "
            "{id, label, source, split}"
        )
    clips = []
    for i, raw in enumerate(data["clips"]):
        if not isinstance(raw, dict):
            raise CalibrationError(f"clip #{i} in {p} is not a mapping")
        if "id" not in raw or "label" not in raw:
            raise CalibrationError(f"clip #{i} in {p} must have 'id' and 'label'")
        if "source" not in raw and "scores" not in raw:
            raise CalibrationError(
                f"clip {raw.get('id', i)!r} in {p} needs 'source' or 'scores'"
            )
        label = str(raw["label"]).lower()
        if label not in VALID_LABELS:
            raise CalibrationError(
                f"clip {raw.get('id', i)!r}: label must be one of {VALID_LABELS}, "
                f"got {raw['label']!r}"
            )
        split = str(raw.get("split", "calibration")).lower()
        if split not in CALIBRATION_SPLITS:
            raise CalibrationError(
                f"clip {raw.get('id', i)!r}: split must be one of "
                f"{CALIBRATION_SPLITS} (never 'test'/'locked'), got {raw['split']!r}"
            )
        clips.append(CalibrationClip(
            id=str(raw["id"]),
            label=label,
            source=str(raw.get("source", "")),
            printer=raw.get("printer") or data.get("printer"),
            camera=raw.get("camera") or data.get("camera"),
            split=split,
            scores=raw.get("scores"),
        ))
    if not clips:
        raise CalibrationError(f"spec {p} contains no clips")
    ids = [c.id for c in clips]
    if len(set(ids)) != len(ids):
        raise CalibrationError(f"spec {p} has duplicate clip ids: {ids}")
    return {
        "printer": data.get("printer"),
        "camera": data.get("camera"),
        "checkpoint": data.get("checkpoint"),
        "image_size": data.get("image_size"),
        "clips": clips,
    }


def _normalise_scores(raw: list) -> list[dict]:
    """Normalise a clip's precomputed scores to [{"timestamp_s","score"}]."""
    out: list[dict] = []
    for i, item in enumerate(raw):
        if isinstance(item, dict):
            ts = float(item.get("timestamp_s", i))
            score = float(item.get("score"))
        else:
            ts = float(i)
            score = float(item)
        out.append({"timestamp_s": ts, "score": score})
    return out


def _frame_scores_from_dir(source: str, sample_fps: float) -> list[tuple[float, str]]:
    """Ordered (timestamp_s, image_path) for a directory of frames."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted(p for p in Path(source).iterdir() if p.suffix.lower() in exts)
    if not files:
        raise CalibrationError(f"no image frames found in directory: {source}")
    step = 1.0 / (sample_fps if sample_fps and sample_fps > 0 else 1.0)
    return [(i * step, str(f)) for i, f in enumerate(files)]


def score_clip(
    clip: CalibrationClip,
    model,
    transform,
    device,
    fail_idx: int,
    sample_fps: float | None = None,
    max_frames: int | None = None,
) -> list[dict]:
    """Return [{"timestamp_s", "score"}] (P(failing)) for one clip.

    Uses the canonical POC inference path (identical preprocessing). Source may
    be a video file/glob or a directory of ordered frames. Raises
    :class:`CalibrationError` for unreadable/empty sources.
    """
    import cv2
    from PIL import Image

    from .poc import (
        PocError,
        _iter_sampled_frames,
        _open_capture,
        _resolve_source,
        predict_pil,
    )

    if clip.scores is not None:
        return _normalise_scores(clip.scores)

    source = _resolve_source(clip.source)
    rows: list[dict] = []
    if isinstance(source, str) and Path(source).is_dir():
        for ts, img_path in _frame_scores_from_dir(source, sample_fps or 1.0):
            if max_frames is not None and len(rows) >= max_frames:
                break
            pil = Image.open(img_path).convert("RGB")
            pred = predict_pil(model, transform, device, pil, 0.5, fail_idx=fail_idx)
            rows.append({"timestamp_s": round(float(ts), 6), "score": pred["fail_score"]})
        if not rows:
            raise CalibrationError(f"no frames scored for clip {clip.id!r} ({source})")
        return rows

    try:
        cap = _open_capture(source)
    except PocError as exc:
        raise CalibrationError(f"clip {clip.id!r}: {exc}") from exc
    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    effective_fps = native_fps if native_fps > 0 else (sample_fps or 1.0)
    if sample_fps is not None and native_fps > 0:
        stride = max(1, int(round(native_fps / sample_fps)))
    else:
        stride = 1
    try:
        for frame_idx, frame, _ in _iter_sampled_frames(cap, stride, max_frames):
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pred = predict_pil(model, transform, device, pil, 0.5, fail_idx=fail_idx)
            rows.append({
                "timestamp_s": round(frame_idx / effective_fps, 6),
                "score": pred["fail_score"],
            })
    finally:
        cap.release()
    if not rows:
        raise CalibrationError(f"clip {clip.id!r}: no decodable frames in {source!r}")
    return rows


def _alarm(
    frames: list[dict], threshold: float, confirm: int
) -> tuple[bool, float | None, int]:
    """Session decision for one clip at ``threshold``.

    Alarmed iff ``confirm`` consecutive frames have ``score >= threshold``.
    Returns ``(alarmed, alarm_timestamp_s, frames_above_threshold)``.
    """
    run = 0
    frames_above = 0
    for f in frames:
        if f["score"] >= threshold:
            frames_above += 1
            run += 1
            if run >= confirm:
                return True, float(f["timestamp_s"]), frames_above
        else:
            run = 0
    return False, None, frames_above


def _aggregate(clips: list[CalibrationClip],
               scored: dict[str, list[dict]],
               threshold: float, confirm: int) -> dict:
    """Per-clip decisions + aggregate metrics at ``threshold``."""
    per_clip = []
    n_normal = n_failing = 0
    detected = false_alarms = 0
    latencies: list[float] = []
    frames_flagged_fraction: list[float] = []
    for clip in clips:
        frames = scored[clip.id]
        alarmed, ts, frames_above = _alarm(frames, threshold, confirm)
        n_frames = len(frames)
        frac = round(frames_above / n_frames, 6) if n_frames else 0.0
        per_clip.append({
            "id": clip.id,
            "label": clip.label,
            "split": clip.split,
            "frames": n_frames,
            "score_min": round(min((f["score"] for f in frames), default=0.0), 6),
            "score_max": round(max((f["score"] for f in frames), default=0.0), 6),
            "score_mean": round(sum(f["score"] for f in frames) / n_frames, 6) if n_frames else 0.0,
            "frames_above_threshold": frames_above,
            "fraction_frames_flagged": frac,
            "alarmed": alarmed,
            "alarm_timestamp_s": ts,
            "latency_s": ts if (alarmed and clip.label == "failing") else None,
        })
        if clip.label == "normal":
            n_normal += 1
            if alarmed:
                false_alarms += 1
            frames_flagged_fraction.append(frac)
        else:
            n_failing += 1
            if alarmed:
                detected += 1
                if ts is not None:
                    latencies.append(ts)

    det_rate = round(detected / n_failing, 6) if n_failing else None
    far = round(false_alarms / n_normal, 6) if n_normal else None
    lat_sorted = sorted(latencies)
    latency = None
    if lat_sorted:
        n = len(lat_sorted)
        latency = {
            "count": n,
            "mean_s": round(sum(lat_sorted) / n, 6),
            "median_s": round(lat_sorted[n // 2], 6),
            "min_s": round(lat_sorted[0], 6),
            "max_s": round(lat_sorted[-1], 6),
        }
    return {
        "threshold": round(float(threshold), 6),
        "confirm": confirm,
        "n_normal": n_normal,
        "n_failing": n_failing,
        "detected": detected,
        "false_alarms": false_alarms,
        "detection_rate": det_rate,
        "detection_rate_ci": wilson_interval(detected, n_failing) if n_failing else None,
        "false_alarm_rate_per_print": far,
        "false_alarm_rate_ci": wilson_interval(false_alarms, n_normal) if n_normal else None,
        "mean_false_alarms_per_normal_print": far,
        "mean_flagged_fraction_normal": (
            round(sum(frames_flagged_fraction) / n_normal, 6) if n_normal else None
        ),
        "detection_latency_s": latency,
        "per_clip": per_clip,
    }


def _select_threshold(
    clips: list[CalibrationClip],
    scored: dict[str, list[dict]],
    thresholds: list[float],
    confirm: int,
    fpr_ceiling: float,
) -> dict:
    """Select a threshold from calibration clips only.

    Maximises detection rate subject to false-alarm-rate-per-print <= ceiling;
    ties break on lower false-alarm rate then lower threshold (more sensitive).
    """
    cal = [c for c in clips if c.split == "calibration"]
    n_normal = sum(1 for c in cal if c.label == "normal")
    n_failing = sum(1 for c in cal if c.label == "failing")

    curve = []
    for t in thresholds:
        agg = _aggregate(cal, scored, t, confirm)
        curve.append({
            "threshold": round(float(t), 6),
            "detection_rate": agg["detection_rate"],
            "false_alarm_rate_per_print": agg["false_alarm_rate_per_print"],
            "detected": agg["detected"],
            "false_alarms": agg["false_alarms"],
        })

    if n_normal == 0 or n_failing == 0:
        return {
            "status": "insufficient_data",
            "selected_threshold": None,
            "reason": (
                "calibration split needs at least one normal and one failing "
                f"print (have normal={n_normal}, failing={n_failing})"
            ),
            "curve": curve,
        }

    feasible = [c for c in curve if (c["false_alarm_rate_per_print"] or 0.0) <= fpr_ceiling]
    if feasible:
        best = max(
            feasible,
            key=lambda c: (c["detection_rate"], -c["false_alarm_rate_per_print"], -c["threshold"]),
        )
        return {
            "status": "ok",
            "selected_threshold": best["threshold"],
            "reason": f"max detection rate subject to FPR/print <= {fpr_ceiling}",
            "curve": curve,
        }
    best = min(
        curve,
        key=lambda c: (c["false_alarm_rate_per_print"], -c["detection_rate"], c["threshold"]),
    )
    return {
        "status": "no_threshold_meets_fpr_ceiling",
        "selected_threshold": best["threshold"],
        "reason": (
            f"no threshold reaches FPR/print <= {fpr_ceiling}; "
            "returned the lowest-FPR point but it is NOT an accepted operating point"
        ),
        "curve": curve,
    }


def calibrate(
    spec: dict,
    checkpoint: str | Path | None = None,
    image_size: int | None = None,
    sample_fps: float | None = None,
    thresholds: list[float] | None = None,
    confirm: int = 1,
    fpr_ceiling: float = 0.1,
    max_frames: int | None = None,
    score_fn: Optional[Callable[[CalibrationClip], list[dict]]] = None,
) -> dict:
    """Calibrate a per-frame threshold from labelled target-printer clips.

    ``spec`` is the dict from :func:`load_clip_spec`. ``score_fn`` overrides the
    default model scoring (used by tests); otherwise ``checkpoint`` is required.
    """
    clips: list[CalibrationClip] = spec["clips"]
    if not 0.0 <= fpr_ceiling <= 1.0:
        raise CalibrationError(f"fpr_ceiling must be in [0, 1], got {fpr_ceiling}")
    if confirm < 1:
        raise CalibrationError(f"confirm must be >= 1, got {confirm}")
    thresholds = thresholds or [round(0.05 * i, 2) for i in range(1, 20)]
    if not thresholds:
        raise CalibrationError("thresholds grid is empty")

    printer = spec.get("printer")
    camera = spec.get("camera")
    warnings: list[str] = []

    resolved_score_fn = score_fn
    model_info = None
    all_precomputed = all(c.scores is not None for c in clips)
    if resolved_score_fn is None and all_precomputed:
        resolved_score_fn = lambda clip: _normalise_scores(clip.scores)  # noqa: E731
    elif resolved_score_fn is None:
        ckpt = checkpoint or spec.get("checkpoint")
        if not ckpt:
            raise CalibrationError(
                "no checkpoint: pass --checkpoint or set 'checkpoint' in the spec "
                "(or give every clip precomputed 'scores')"
            )
        from .poc import PocError, load_model

        try:
            model, transform, device, info = load_model(ckpt, image_size, "auto")
        except PocError as exc:
            raise CalibrationError(str(exc)) from exc

        def _model_score(clip: CalibrationClip) -> list[dict]:
            return score_clip(clip, model, transform, device, info.fail_idx,
                              sample_fps, max_frames)

        resolved_score_fn = _model_score
        model_info = {
            "checkpoint": info.checkpoint,
            "model_name": info.model_name,
            "image_size": info.image_size,
            "normalization": {
                "mean": info.normalize_mean,
                "std": info.normalize_std,
                "source": "checkpoint" if info.normalize_mean else "imagenet",
            },
        }

    scored: dict[str, list[dict]] = {}
    for clip in clips:
        rows = resolved_score_fn(clip)
        if not rows:
            raise CalibrationError(f"clip {clip.id!r} produced no scored frames")
        scored[clip.id] = rows

    cal = [c for c in clips if c.split == "calibration"]
    hold = [c for c in clips if c.split == "holdout"]
    cal_normal = sum(1 for c in cal if c.label == "normal")
    cal_failing = sum(1 for c in cal if c.label == "failing")
    if cal_normal == 0 or cal_failing == 0:
        warnings.append(
            "calibration split has a single class "
            f"(normal={cal_normal}, failing={cal_failing}); no threshold can be selected"
        )
    if 0 < cal_normal < 2:
        warnings.append(f"only {cal_normal} normal calibration print(s); "
                        "false-alarm estimate is unreliable")
    if 0 < cal_failing < 2:
        warnings.append(f"only {cal_failing} failing calibration print(s); "
                        "detection-rate estimate is unreliable")
    if not hold:
        warnings.append(
            "no holdout clips: the selected threshold is not validated on "
            "independent sessions"
        )
    if not printer:
        warnings.append("no 'printer' declared in the spec; the report cannot "
                        "assert which printer this calibration applies to")

    selection = _select_threshold(clips, scored, thresholds, confirm, fpr_ceiling)
    if selection["status"] == "no_threshold_meets_fpr_ceiling":
        warnings.append(selection["reason"])
    selected = selection.get("selected_threshold")

    calibration_metrics = (
        _aggregate(cal, scored, selected, confirm) if selected is not None else None
    )
    holdout_metrics = (
        _aggregate(hold, scored, selected, confirm)
        if (selected is not None and hold) else None
    )

    return {
        "schema": "tinyvision-calibration",
        "schema_version": 1,
        "printer": printer,
        "camera": camera,
        "model": model_info,
        "config": {
            "sample_fps": sample_fps,
            "thresholds": [round(float(t), 6) for t in thresholds],
            "confirm": confirm,
            "fpr_ceiling": fpr_ceiling,
            "objective": "max_detection_subject_to_fpr_ceiling",
            "select_on": "calibration",
        },
        "data": {
            "n_clips": len(clips),
            "n_calibration": {"normal": cal_normal, "failing": cal_failing},
            "n_holdout": {
                "normal": sum(1 for c in hold if c.label == "normal"),
                "failing": sum(1 for c in hold if c.label == "failing"),
            },
            "split_by": "complete_print_session",
        },
        "status": selection["status"],
        "selection": selection,
        "calibration": calibration_metrics,
        "holdout": holdout_metrics,
        "warnings": warnings,
        "disclaimer": (
            "Calibrated for the declared printer/camera setup only. A per-printer "
            "threshold does not establish cross-printer generalization."
        ),
    }


def format_summary(report: dict) -> str:
    """Render a concise human-readable calibration summary."""
    cfg = report["config"]
    lines = [
        "TinyVision threshold calibration",
        f"  printer/camera : {report.get('printer')} / {report.get('camera')}",
        f"  status         : {report['status']}",
        f"  clips          : {report['data']['n_clips']} "
        f"(calibration {report['data']['n_calibration']}, holdout {report['data']['n_holdout']})",
        f"  objective      : {cfg['objective']} (FPR/print ceiling {cfg['fpr_ceiling']})",
    ]
    sel = report["selection"]
    lines.append(f"  selected thresh: {sel.get('selected_threshold')}  ({sel.get('reason')})")
    for name in ("calibration", "holdout"):
        m = report.get(name)
        if not m:
            continue
        lines.append(
            f"  [{name}] n_normal={m['n_normal']} n_failing={m['n_failing']} | "
            f"detection={m['detection_rate']} far/print={m['false_alarm_rate_per_print']}"
        )
        lat = m.get("detection_latency_s")
        if lat:
            lines.append(
                f"           latency mean={lat['mean_s']}s median={lat['median_s']}s "
                f"min={lat['min_s']}s max={lat['max_s']}s (n={lat['count']})"
            )
    for w in report.get("warnings", []):
        lines.append(f"  WARNING: {w}")
    lines.append(f"  {report['disclaimer']}")
    return "\n".join(lines)
