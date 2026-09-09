"""End-to-end inference proof-of-concept (POC) for TinyVision.

Runs the frozen binary frame classifier in a real-time-style loop over a video
file or webcam source, feeds the per-frame P(failing) stream through the
existing Stage-3 temporal decision logic (EWMA + Failure-FSM), and produces:

  * a per-frame decision log (raw threshold, EWMA signal, FSM state, event),
  * state/event summaries (did the stream enter FAILING? when? how often?),
  * process timing (decode / preprocess / inference per-frame, cadence FPS),
  * an optional annotated MP4/AVI (overlaid state) for eyeballing.

Everything here reuses the canonical inference path as-is:
  * checkpoint loading + ``build_classifier`` (models.factory),
  * the eval-time transform from ``_build_transforms(image_size, {}, False)``
    so train/deploy preprocessing never drifts,
  * ``EwmaBaseline`` / ``FailureFsm`` (temporal.py) so the POC uses exactly the
    decision logic already validated in Stage 3 offline eval.

This is architecture/plumbing verification, NOT a benchmark: POC smoke numbers
are qualitative and labeled accordingly.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .temporal import EwmaBaseline, FailureFsm


@dataclass
class PocConfig:
    """Small, explicit POC configuration (all knobs that change behavior)."""

    source: str | int = ""
    checkpoint: str | Path = (
        "experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/best.pt"
    )
    image_size: int = 224
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


def _eta(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


@dataclass
class ModelInfo:
    model_name: str
    num_classes: int
    fail_idx: int
    checkpoint: str


def load_model(
    checkpoint: str | Path,
    image_size: int = 224,
    device: str = "auto",
    threshold: float = 0.5,
):
    """Load the frozen binary classifier + its eval-time transform.

    Returns ``(model, transform, device, info)``. The transform is the exact
    eval path used by ``tv eval-deep`` (``_build_transforms(..., is_train=False)``),
    so the POC's preprocessing is byte-identical to offline evaluation.
    """
    import torch

    if device == "auto":
        from .deepeval import _get_device

        device = _get_device()

    ckpt = torch.load(Path(checkpoint), map_location=device, weights_only=False)
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

    from .models.factory import build_classifier

    model = build_classifier(model_name, num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    from .data.loaders import _build_transforms

    transform = _build_transforms(image_size, augmentation={}, is_train=False)

    info = ModelInfo(
        model_name=model_name,
        num_classes=num_classes,
        fail_idx=fail_idx,
        checkpoint=str(Path(checkpoint).resolve()),
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

    def step(self, fail_score: float) -> dict:
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
        self._rows.append(row)
        return row

    def results(self) -> list[dict]:
        return list(self._rows)

    def summary(self) -> dict:
        rows = self._rows
        states: dict[str, int] = {}
        events = []
        first_failing = None
        failing_frames = 0
        for r in rows:
            states[r["state"]] = states.get(r["state"], 0) + 1
            if r["event"] == "failing_event" and first_failing is None:
                first_failing = r["frame"]
            if r["state"] == "FAILING":
                failing_frames += 1
            if r["event"]:
                events.append({"frame": r["frame"], "event": r["event"]})
        return {
            "frames": len(rows),
            "state_counts": states,
            "events": events,
            "first_failing_event_frame": first_failing,
            "failing_frames": failing_frames,
            "last_state": rows[-1]["state"] if rows else "NONE",
        }


def _open_capture(source: str | int):
    import cv2

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return cap


def _next_frame(cap, skip: int, frame_idx: int):
    """Read frames from ``cap``, skipping to honour the sampling cadence."""
    import cv2

    for _ in range(skip):
        ret, frame = cap.read()
        if not ret or frame is None:
            return None, frame_idx
        frame_idx += 1
    ret, frame = cap.read()
    if not ret or frame is None:
        return None, frame_idx
    frame_idx += 1
    return frame, frame_idx


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


def run_poc(config: PocConfig) -> dict:
    """Run the end-to-end POC over a video file or webcam source.

    Returns a report dict (per-frame log, summary, timing). ``config.source``
    is a path (file) or an int (camera index). Raises ``FileNotFoundError`` /
    ``RuntimeError`` for missing checkpoints or unopenable sources.
    """
    import glob
    import os

    import cv2
    import numpy as np
    from PIL import Image

    source = config.source
    if isinstance(source, str) and not os.path.exists(source) and not str(source).isdigit():
        matches = glob.glob(source)
        if not matches:
            raise FileNotFoundError(f"Video source not found: {source}")
        source = matches[0]

    model, transform, device, info = load_model(
        config.checkpoint, config.image_size, config.device, config.threshold
    )

    track = DecisionTracker(config)
    cap = _open_capture(source)

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    skip = 1
    if config.sample_fps is not None and native_fps > 0:
        skip = max(1, round(native_fps / config.sample_fps))

    writer = None
    if config.annotated_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = Path(config.annotated_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), fourcc, native_fps, (width, height))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out_path = out_path.with_suffix(".avi")
            writer = cv2.VideoWriter(str(out_path), fourcc, native_fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open VideoWriter for {config.annotated_output}")
            config.annotated_output = out_path

    decode_ms: list[float] = []
    preprocess_ms: list[float] = []
    infer_ms: list[float] = []
    frame_idx = 0
    t_stream_start = time.perf_counter()
    annotated_path: Optional[str] = None

    try:
        while True:
            if config.max_frames is not None and len(track.results()) >= config.max_frames:
                break

            t0 = time.perf_counter()
            frame, frame_idx = _next_frame(cap, skip, frame_idx)
            t1 = time.perf_counter()
            if frame is None:
                break
            decode_ms.append((t1 - t0) * 1000)

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

            row = track.step(pred["fail_score"])
            row["frame_idx"] = frame_idx

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

    stream_seconds = max(time.perf_counter() - t_stream_start, 1e-9)
    rows = track.results()
    summary = track.summary()
    summary["cadence_fps"] = round(len(rows) / stream_seconds, 3)

    return {
        "schema": "poc-inference",
        "schema_version": 1,
        "model": {
            "checkpoint": info.checkpoint,
            "model_name": info.model_name,
            "num_classes": info.num_classes,
            "fail_idx": info.fail_idx,
            "image_size": config.image_size,
            "threshold": config.threshold,
        },
        "source": {
            "input": str(source),
            "resolution": f"{width}x{height}",
            "native_fps": round(native_fps, 3),
            "total_frames_in_video": total_frames if isinstance(source, str) else None,
            "sample_fps": config.sample_fps,
            "skip": skip,
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