"""End-to-end inference proof-of-concept (POC) for TinyVision.

Runs the frozen binary frame classifier in a real-time-style loop over a video
file or webcam source, feeds the per-frame P(failing) stream through the
existing Stage-3 temporal decision logic (EWMA + Failure-FSM), and produces:

  * a timestamped per-frame decision log (raw threshold, EWMA signal, FSM
    state, event) and a timestamped ``detections`` list,
  * state/event summaries (did the stream enter FAILING? when? how often?),
  * process timing (decode / preprocess / inference per-frame, cadence FPS),
  * an optional annotated MP4/AVI (overlaid state) for eyeballing.

Everything here reuses the canonical inference path as-is:
  * checkpoint loading + ``build_classifier`` (models.factory),
  * the eval-time transform from ``_build_transforms`` built with the exact
    normalisation stored in the checkpoint (custom train-time statistics when
    present, otherwise ImageNet), so train/deploy preprocessing never drifts,
  * ``EwmaBaseline`` / ``FailureFsm`` (temporal.py) so the POC uses exactly the
    decision logic already validated in Stage 3 offline eval.

The path is deterministic: identical input video + checkpoint + config yields
an identical report (timing fields aside).

Errors are surfaced as :class:`PocError` with an actionable message for
unreadable/empty videos, invalid checkpoints and invalid configuration.

This is architecture/plumbing verification, NOT a benchmark: POC smoke numbers
are qualitative and labeled accordingly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from .temporal import EwmaBaseline, FailureFsm


class PocError(Exception):
    """Actionable, user-facing error on the end-to-end inference path."""


@dataclass
class PocConfig:
    """Small, explicit POC configuration (all knobs that change behavior)."""

    source: str | int = ""
    checkpoint: str | Path = (
        "experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/best.pt"
    )
    # ``None`` means "use the image size stored in the checkpoint" (falls back
    # to 224 when the checkpoint predates that field).
    image_size: Optional[int] = None
    device: str = "auto"
    sample_fps: float | None = None
    max_frames: int | None = None
    threshold: float = 0.5  # raw per-frame decision threshold on P(failing)
    annotated_output: str | Path | None = None
    # Temporal parameters; defaults mirror the Stage-3 "fsm" variant.
    alpha: float = 0.15
    baseline_window: int = 60
    warmup: int = 30
    suspect_threshold: float = 0.05
    fail_threshold: float = 0.30
    recover_threshold: float = 0.01
    fail_threshold_abs: float = 0.65
    confirm: int = 3
    use_delta: bool = False

    def validate(self) -> None:
        """Raise :class:`PocError` for invalid configuration with a fix hint."""
        errors: list[str] = []
        if not isinstance(self.threshold, (int, float)) or not 0.0 <= float(self.threshold) <= 1.0:
            errors.append(f"threshold must be in [0, 1], got {self.threshold!r}")
        if self.sample_fps is not None and not (
            isinstance(self.sample_fps, (int, float)) and float(self.sample_fps) > 0
        ):
            errors.append(f"sample_fps must be > 0 or null, got {self.sample_fps!r}")
        if self.max_frames is not None and not (
            isinstance(self.max_frames, int) and self.max_frames > 0
        ):
            errors.append(f"max_frames must be a positive integer or null, got {self.max_frames!r}")
        if self.image_size is not None and not (
            isinstance(self.image_size, int) and self.image_size > 0
        ):
            errors.append(f"image_size must be a positive integer or null, got {self.image_size!r}")
        if not 0.0 < float(self.alpha) <= 1.0:
            errors.append(f"alpha must be in (0, 1], got {self.alpha!r}")
        if not (isinstance(self.baseline_window, int) and self.baseline_window >= 1):
            errors.append(f"baseline_window must be >= 1, got {self.baseline_window!r}")
        if not (isinstance(self.warmup, int) and self.warmup >= 0):
            errors.append(f"warmup must be >= 0, got {self.warmup!r}")
        if not (isinstance(self.confirm, int) and self.confirm >= 1):
            errors.append(f"confirm must be >= 1, got {self.confirm!r}")
        for name in ("suspect_threshold", "fail_threshold", "recover_threshold"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0):
                errors.append(f"{name} must be in [0, 1], got {v!r}")
        if float(self.recover_threshold) >= float(self.fail_threshold):
            errors.append(
                "recover_threshold must be < fail_threshold (hysteresis): "
                f"got recover={self.recover_threshold}, fail={self.fail_threshold}"
            )
        if self.fail_threshold_abs is not None and not (
            isinstance(self.fail_threshold_abs, (int, float))
            and 0.0 <= float(self.fail_threshold_abs) <= 1.0
        ):
            errors.append(
                f"fail_threshold_abs must be in [0, 1] or null, got {self.fail_threshold_abs!r}"
            )
        if errors:
            raise PocError("invalid configuration:\n  - " + "\n  - ".join(errors))


@dataclass
class ModelInfo:
    model_name: str
    num_classes: int
    fail_idx: int
    checkpoint: str
    image_size: int = 224
    normalize_mean: list[float] | None = None
    normalize_std: list[float] | None = None


def load_model(
    checkpoint: str | Path,
    image_size: Optional[int] = None,
    device: str = "auto",
    threshold: float = 0.5,
):
    """Load the frozen binary classifier + its eval-time transform.

    Returns ``(model, transform, device, info)``. The transform is the exact
    eval path used by ``tv eval-deep`` / ``tv reliability`` built with the
    normalisation stored in the checkpoint, so the POC's preprocessing is
    identical to offline evaluation and to training.

    ``image_size=None`` uses the size stored in the checkpoint (224 fallback).
    Raises :class:`PocError` for a missing or malformed checkpoint.
    """
    import torch

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise PocError(f"checkpoint not found: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - surface any load failure actionably
        raise PocError(
            f"could not read checkpoint {ckpt_path}: {exc} "
            "(expected a TinyVision best.pt saved by `tv train`)"
        ) from exc
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise PocError(
            f"checkpoint {ckpt_path} is not a TinyVision checkpoint "
            "(missing 'model_state_dict'); pass a best.pt produced by `tv train`"
        )

    if device == "auto":
        from .deepeval import _get_device

        device = _get_device()

    num_classes = int(ckpt.get("num_classes", 2))
    model_name = ckpt.get("config_model", "shufflenet_v2_x1_0")

    class_to_idx = ckpt.get("class_to_idx", {})
    fail_idx: Optional[int] = None
    for k, v in class_to_idx.items():
        if k.lower() == "failing":
            fail_idx = int(v)
            break
    if fail_idx is None:
        fail_idx = 0  # frozen binary convention: failing=0, normal=1

    resolved_size = int(image_size) if image_size else int(ckpt.get("image_size", 224) or 224)

    # Reuse the exact normalisation the checkpoint was trained with; ``None``
    # falls back to ImageNet statistics (keeps legacy checkpoints identical).
    normalize_mean = ckpt.get("normalize_mean")
    normalize_std = ckpt.get("normalize_std")

    from .models.factory import build_classifier

    model = build_classifier(model_name, num_classes, pretrained=False)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except Exception as exc:  # noqa: BLE001
        raise PocError(
            f"checkpoint {ckpt_path} state_dict does not match model "
            f"{model_name!r} ({num_classes} classes): {exc}"
        ) from exc
    model = model.to(device).eval()

    from .data.loaders import _build_transforms

    aug: dict[str, Any] = {}
    if normalize_mean and normalize_std:
        aug["normalize_mean"] = list(normalize_mean)
        aug["normalize_std"] = list(normalize_std)
    transform = _build_transforms(resolved_size, augmentation=aug, is_train=False)

    info = ModelInfo(
        model_name=model_name,
        num_classes=num_classes,
        fail_idx=fail_idx,
        checkpoint=str(ckpt_path.resolve()),
        image_size=resolved_size,
        normalize_mean=list(normalize_mean) if normalize_mean else None,
        normalize_std=list(normalize_std) if normalize_std else None,
    )
    return model, transform, device, info


def predict_pil(model, transform, device, pil_image, threshold: float,
                fail_idx: int = 0) -> dict:
    """Run one PIL image through preprocess + inference.

    Returns the per-frame dict with ``fail_score`` (P(failing)), the raw index
    prediction, and the raw binarized decision (``"failing"``/``"normal"`` at
    ``threshold`` on P(failing)).
    """
    import torch

    tensor = transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor.to(device)), dim=1).cpu().numpy()[0]
    fail_score = float(probs[fail_idx])
    pred_idx = fail_idx if fail_score >= threshold else 1 - fail_idx
    return {
        "fail_score": round(fail_score, 6),
        "pred_idx": int(pred_idx),
        "raw_decision": "failing" if pred_idx == fail_idx else "normal",
    }


class DecisionTracker:
    """Incremental EWMA + Failure-FSM over a stream of P(failing) values.

    Wraps the canonical ``EwmaBaseline`` / ``FailureFsm`` and records a
    per-frame decision row every step, so the POC can (a) overlay the live state
    while capturing and (b) emit a deterministic per-frame log afterwards.
    Kept a thin, pure wrapper for unit-testing on synthetic score sequences.
    """

    def __init__(self, config: PocConfig):
        self.ew = EwmaBaseline(alpha=config.alpha,
                               baseline_window=config.baseline_window)
        self.fsm = FailureFsm(
            warmup=config.warmup,
            suspect_threshold=config.suspect_threshold,
            fail_threshold=config.fail_threshold,
            recover_threshold=config.recover_threshold,
            fail_threshold_abs=config.fail_threshold_abs,
            confirm=config.confirm,
            use_delta=config.use_delta,
        )
        self.threshold = config.threshold
        self._rows: list[dict] = []

    def step(self, fail_score: float, frame_idx: int | None = None,
             timestamp_s: float | None = None) -> dict:
        st = self.ew.step(fail_score)
        d = self.fsm.step(st["signal"], st["baseline"])
        row = {
            "frame": len(self._rows),
            "raw": round(float(fail_score), 6),
            "signal": round(float(st["signal"]), 6),
            "baseline": round(float(st["baseline"]), 6),
            "delta": round(float(st["delta"]), 6),
            "state": d.state,
            "raw_decision": "failing" if fail_score >= self.threshold else "normal",
            "event": d.events[-1] if d.events else None,
        }
        if frame_idx is not None:
            row["frame_idx"] = frame_idx
        if timestamp_s is not None:
            row["timestamp_s"] = round(float(timestamp_s), 6)
        self._rows.append(row)
        return row

    def results(self) -> list[dict]:
        return list(self._rows)

    def summary(self) -> dict:
        rows = self._rows
        states: dict[str, int] = {}
        events = []
        first_failing = None
        first_failing_ts = None
        failing_frames = 0
        for r in rows:
            states[r["state"]] = states.get(r["state"], 0) + 1
            if r["event"] == "failing_event" and first_failing is None:
                first_failing = r["frame"]
                first_failing_ts = r.get("timestamp_s")
            if r["state"] == "FAILING":
                failing_frames += 1
            if r["event"]:
                events.append({"frame": r["frame"], "event": r["event"]})
        return {
            "frames": len(rows),
            "state_counts": states,
            "events": events,
            "first_failing_event_frame": first_failing,
            "first_failing_event_timestamp_s": first_failing_ts,
            "failing_frames": failing_frames,
            "last_state": rows[-1]["state"] if rows else "NONE",
        }


def _iter_sampled_frames(
    cap, stride: int, max_frames: int | None
) -> Iterator[tuple[int, Any, float]]:
    """Yield ``(frame_index, frame_bgr, decode_ms)`` for 1 of every ``stride`` frames.

    ``stride=1`` processes every frame; the frame index is the true 0-based
    index in the source video (so ``index / native_fps`` is the video
    timestamp). Frames are decoded sequentially to keep indices exact, and
    ``decode_ms`` is the wall time spent decoding the frames consumed since the
    previous yield (including any skipped frames).
    """
    idx = 0
    produced = 0
    pending_ms = 0.0
    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        pending_ms += (time.perf_counter() - t0) * 1000
        if not ret or frame is None:
            return
        if idx % stride == 0:
            if max_frames is not None and produced >= max_frames:
                return
            yield idx, frame, pending_ms
            produced += 1
            pending_ms = 0.0
        idx += 1


def _stats_ms(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    sv = sorted(values)
    n = len(sv)
    return {
        "count": n,
        "mean_ms": round(sum(sv) / n, 4),
        "p50_ms": round(sv[int(n * 0.5)], 4),
        "p95_ms": round(sv[min(int(n * 0.95), n - 1)], 4),
        "max_ms": round(sv[-1], 4),
    }


def _resolve_source(source: str | int | None) -> str | int:
    """Resolve a path/glob/int source. Raises :class:`PocError` if unmatched."""
    import glob
    import os

    if source is None or source == "":
        raise PocError("no input source: pass --input FILE or --camera INDEX")
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        return int(source)
    if os.path.exists(source):
        return source
    matches = sorted(glob.glob(source))
    if not matches:
        raise PocError(f"video source not found: {source}")
    return matches[0]


def _open_capture(source: str | int):
    import cv2

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise PocError(
            f"cannot open video source: {source!r} "
            "(unreadable file, unsupported codec, or empty/corrupt video)"
        )
    return cap


def _detections_from_rows(rows: list[dict]) -> list[dict]:
    """Timestamped detections (temporal FAILING events + raw threshold crossings)."""
    detections: list[dict] = []
    for r in rows:
        if r["event"]:
            detections.append({
                "frame": r["frame"],
                "frame_idx": r.get("frame_idx"),
                "timestamp_s": r.get("timestamp_s"),
                "kind": r["event"],
                "score": r["raw"],
                "state": r["state"],
            })
    return detections


def run_poc(config: PocConfig, on_frame: "Callable[[int, dict, int | None], None] | None" = None) -> dict:
    """Run the end-to-end POC over a video file or webcam source.

    Returns a report dict (timestamped per-frame log, detections, summary,
    timing). ``config.source`` is a path (file/glob) or an int (camera index).

    ``on_frame`` (optional) is called after each processed frame as
    ``on_frame(processed_count, row, total_estimate)`` so a caller (e.g. the
    local demo) can report progress without a second inference implementation.
    ``total_estimate`` is the number of frames expected to be processed
    (``None`` for live sources). Exceptions from ``on_frame`` propagate.

    Raises :class:`PocError` for invalid configuration, missing/invalid
    checkpoints, and unreadable or empty video sources.
    """
    import math

    import cv2
    from PIL import Image

    config.validate()
    source = _resolve_source(config.source)

    model, transform, device, info = load_model(
        config.checkpoint, config.image_size, config.device, config.threshold
    )

    track = DecisionTracker(config)
    cap = _open_capture(source)

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ``stride`` = number of frames between processed samples (1 = every frame).
    if config.sample_fps is not None and native_fps > 0:
        stride = max(1, int(round(native_fps / config.sample_fps)))
    else:
        stride = 1
    effective_fps = native_fps if native_fps > 0 else (config.sample_fps or 1.0)

    total_estimate: int | None = None
    if isinstance(source, str) and total_frames > 0:
        total_estimate = int(math.ceil(total_frames / stride))
        if config.max_frames is not None:
            total_estimate = min(total_estimate, config.max_frames)

    writer = None
    if config.annotated_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = Path(config.annotated_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer_fps = native_fps if native_fps > 0 else 1.0
        writer = cv2.VideoWriter(str(out_path), fourcc, writer_fps, (width, height))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out_path = out_path.with_suffix(".avi")
            writer = cv2.VideoWriter(str(out_path), fourcc, writer_fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise PocError(f"cannot open VideoWriter for {config.annotated_output}")
            config.annotated_output = out_path

    decode_ms: list[float] = []
    preprocess_ms: list[float] = []
    infer_ms: list[float] = []
    t_stream_start = time.perf_counter()
    annotated_path: Optional[str] = None

    try:
        for frame_idx, frame, frame_decode_ms in _iter_sampled_frames(
            cap, stride, config.max_frames
        ):
            decode_ms.append(frame_decode_ms)

            t_pre0 = time.perf_counter()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb)
            t_pre1 = time.perf_counter()
            preprocess_ms.append((t_pre1 - t_pre0) * 1000)

            t_inf0 = time.perf_counter()
            pred = predict_pil(model, transform, device, pil, config.threshold,
                               fail_idx=info.fail_idx)
            t_inf1 = time.perf_counter()
            infer_ms.append((t_inf1 - t_inf0) * 1000)

            timestamp_s = frame_idx / effective_fps
            row = track.step(pred["fail_score"], frame_idx=frame_idx,
                             timestamp_s=timestamp_s)

            if on_frame is not None:
                on_frame(int(row["frame"]) + 1, row, total_estimate)

            if writer is not None:
                state = row["state"]
                txt = (
                    f"f{frame_idx}  P(fail)={pred['fail_score']:.3f}  "
                    f"sig={row['signal']:.3f}  {state}"
                )
                if row["event"]:
                    txt += f"  <-- {row['event']}"
                cv2.putText(
                    frame, txt, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if state == "FAILING" else (255, 255, 255), 2,
                    cv2.LINE_AA,
                )
                writer.write(frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            annotated_path = str(config.annotated_output)

    rows = track.results()
    if not rows:
        raise PocError(
            f"no decodable frames in video source: {source!r} "
            "(the file opened but yielded zero frames; is it empty or truncated?)"
        )

    stream_seconds = max(time.perf_counter() - t_stream_start, 1e-9)
    summary = track.summary()
    # NOTE: timing-derived metrics live under ``timing`` only, so ``summary``
    # (and the whole report apart from ``timing``) is deterministic.
    summary["video_duration_s"] = round(rows[-1].get("timestamp_s") or 0.0, 6)
    summary["detections"] = len(_detections_from_rows(rows))

    detections = _detections_from_rows(rows)

    norm_source = "checkpoint" if info.normalize_mean and info.normalize_std else "imagenet"
    from .data.loaders import IMAGENET_MEAN, IMAGENET_STD

    norm_mean = info.normalize_mean or list(IMAGENET_MEAN)
    norm_std = info.normalize_std or list(IMAGENET_STD)

    return {
        "schema": "poc-inference",
        "schema_version": 1,
        "model": {
            "checkpoint": info.checkpoint,
            "model_name": info.model_name,
            "num_classes": info.num_classes,
            "fail_idx": info.fail_idx,
            "image_size": info.image_size,
            "threshold": config.threshold,
            "normalization": {
                "mean": [round(float(v), 6) for v in norm_mean],
                "std": [round(float(v), 6) for v in norm_std],
                "source": norm_source,
            },
        },
        "source": {
            "input": str(source),
            "resolution": f"{width}x{height}",
            "native_fps": round(native_fps, 3),
            "total_frames_in_video": total_frames if isinstance(source, str) else None,
            "sample_fps": config.sample_fps,
            "stride": stride,
            "effective_fps": round(effective_fps, 6),
        },
        "temporal": {
            "alpha": config.alpha,
            "baseline_window": config.baseline_window,
            "warmup": config.warmup,
            "suspect_threshold": config.suspect_threshold,
            "fail_threshold": config.fail_threshold,
            "recover_threshold": config.recover_threshold,
            "fail_threshold_abs": config.fail_threshold_abs,
            "confirm": config.confirm,
            "use_delta": config.use_delta,
        },
        "summary": summary,
        "detections": detections,
        "timing": {
            "processed_frames": len(rows),
            "stream_seconds": round(stream_seconds, 3),
            "cadence_fps": round(len(rows) / stream_seconds, 3),
            "decode_ms": _stats_ms(decode_ms),
            "preprocess_ms": _stats_ms(preprocess_ms),
            "inference_ms": _stats_ms(infer_ms),
        },
        "annotated_output": annotated_path,
        "per_frame": rows,
    }


def format_summary(report: dict) -> str:
    """Render a concise human-readable summary of a POC report."""
    s = report["summary"]
    m = report["model"]
    src = report["source"]
    lines = [
        "TinyVision POC inference summary",
        f"  checkpoint : {m['checkpoint']}",
        f"  model      : {m['model_name']}  classes={m['num_classes']}  P(failing) idx={m['fail_idx']}",
        f"  input      : {src['input']}  ({src['resolution']}, {src['native_fps']} fps"
        + (f", sampled at {src['sample_fps']} fps (stride {src['stride']})" if src.get("sample_fps") else "")
        + ")",
        f"  threshold  : {m['threshold']}   normalization: {m['normalization']['source']}",
        f"  frames     : {s['frames']} processed"
        + (f" over {s.get('video_duration_s')}s" if s.get("video_duration_s") else ""),
        f"  states     : {s['state_counts']}",
        f"  final      : {s['last_state']}",
    ]
    first = s.get("first_failing_event_timestamp_s")
    if first is not None:
        lines.append(
            f"  DETECTED failing at t={first}s (frame {s['first_failing_event_frame']}); "
            f"{s['failing_frames']} FAILING frames"
        )
    else:
        lines.append("  no FAILING event (stream stayed normal/observing)")
    if report.get("annotated_output"):
        lines.append(f"  annotated  : {report['annotated_output']}")
    return "\n".join(lines)
