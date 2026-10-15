"""Tests for tv/poc.py (end-to-end inference proof of concept)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training"))

from tv.data.loaders import _build_transforms  # noqa: E402
from tv.poc import (  # noqa: E402
    DecisionTracker,
    PocConfig,
    PocError,
    format_summary,
    load_model,
    predict_pil,
    run_poc,
)

BASELINE_CKPT = (
    REPO / "experiments" / "shufflenet_v2_x1_0_printing-errors_cdc9e2cad4"
    / "artifacts" / "best.pt"
)

FAILING_EVENT = "failing_event"


def _default_config(**overrides) -> PocConfig:
    kwargs = dict(
        source="unused.mp4",
        alpha=0.15,
        baseline_window=60,
        warmup=30,
        suspect_threshold=0.05,
        fail_threshold=0.30,
        recover_threshold=0.01,
        fail_threshold_abs=0.65,
        confirm=3,
        use_delta=False,
    )
    kwargs.update(overrides)
    return PocConfig(**kwargs)


def _tracker(config: PocConfig) -> DecisionTracker:
    return DecisionTracker(config)


def _quiet_then_spike(n_quiet: int = 35, n_spike: int = 20, low: float = 0.02,
                      high: float = 0.8):
    return [low] * n_quiet + [high] * n_spike


def test_transform_no_augmentation_matches_manual_path():
    """The eval-time transform used by load_model is Resize+ToTensor+ImageNet."""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms

    tfm = _build_transforms(224, augmentation={}, is_train=False)
    arr = (np.random.RandomState(0).rand(240, 320, 3) * 255).astype("uint8")
    pil = Image.fromarray(arr)

    expected = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])(pil)

    assert torch.equal(tfm(pil), expected)
    assert tuple(tfm(pil).shape) == (3, 224, 224)


def test_decision_tracker_quiet_sequence_never_fails():
    cfg = _default_config()
    tracker = _tracker(cfg)
    for s in [0.02] * 80:
        tracker.step(s)
    summary = tracker.summary()
    assert summary["frames"] == 80
    assert summary["events"] == []
    assert summary["first_failing_event_frame"] is None
    assert summary["state_counts"].get("FAILING", 0) == 0
    # warm-up OBSERVING then NORMAL
    states = tracker.results()[0]["state"], tracker.results()[-1]["state"]
    assert states[0] == "OBSERVING"
    assert states[1] == "NORMAL"


def test_decision_tracker_short_blip_does_not_fail():
    cfg = _default_config()
    tracker = _tracker(cfg)
    scores = [0.02] * 35 + [0.6, 0.6] + [0.02] * 40
    for s in scores:
        tracker.step(s)
    summary = tracker.summary()
    assert summary["first_failing_event_frame"] is None
    assert summary["events"] == []
    assert summary["state_counts"].get("FAILING", 0) == 0
    assert summary["last_state"] == "NORMAL"


def test_decision_tracker_sustained_spike_fires_once():
    cfg = _default_config()
    tracker = _tracker(cfg)
    for s in _quiet_then_spike():
        tracker.step(s)
    summary = tracker.summary()
    assert summary["first_failing_event_frame"] is not None
    # fire-once semantics: exactly one failing event on the episode
    assert summary["events"] == [
        {"frame": summary["first_failing_event_frame"], "event": FAILING_EVENT}
    ]
    assert summary["state_counts"].get("FAILING", 0) > 0
    # event frame lies within the spike region (after warm-up + confirm)
    assert summary["first_failing_event_frame"] >= 32


def test_decision_tracker_sustained_spike_recovers_to_paused():
    cfg = _default_config()
    tracker = _tracker(cfg)
    scores = _quiet_then_spike() + [0.0] * 40
    for s in scores:
        tracker.step(s)
    summary = tracker.summary()
    assert summary["first_failing_event_frame"] is not None
    assert summary["last_state"] == "PAUSED"


def test_decision_tracker_absolute_latch_fires_immediately():
    # Lower the absolute latch below the sustained threshold so a very high
    # couple of frames trip the hard latch even though confirm is not met.
    cfg = _default_config(fail_threshold_abs=0.2)
    tracker = _tracker(cfg)
    scores = [0.02] * 32 + [0.9, 0.9] + [0.02] * 30
    for s in scores:
        tracker.step(s)
    summary = tracker.summary()
    assert summary["first_failing_event_frame"] is not None
    # fired on the 2nd of the two high frames (abs latch <= confirm counter)
    assert summary["first_failing_event_frame"] <= 33


def test_decision_tracker_raw_decision_uses_threshold():
    cfg = _default_config(threshold=0.5)
    tracker = _tracker(cfg)
    tracker.step(0.9)
    tracker.step(0.1)
    rows = tracker.results()
    assert rows[0]["raw_decision"] == "failing"
    assert rows[1]["raw_decision"] == "normal"


@pytest.mark.skipif(not BASELINE_CKPT.exists(), reason="baseline checkpoint not present")
def test_load_model_roundtrip_and_deterministic_predict():
    import numpy as np
    from PIL import Image

    model, transform, device, info = load_model(BASELINE_CKPT, image_size=224)
    assert info.model_name == "shufflenet_v2_x1_0"
    assert info.num_classes == 2
    assert info.fail_idx == 0

    arr = (np.random.RandomState(7).rand(300, 400, 3) * 255).astype("uint8")
    pil = Image.fromarray(arr)

    a = predict_pil(model, transform, device, pil, threshold=0.5, fail_idx=info.fail_idx)
    b = predict_pil(model, transform, device, pil, threshold=0.5, fail_idx=info.fail_idx)
    assert a["fail_score"] == b["fail_score"]
    assert a["raw_decision"] in ("failing", "normal")
    assert 0.0 <= a["fail_score"] <= 1.0


@pytest.mark.skipif(not BASELINE_CKPT.exists(), reason="baseline checkpoint not present")
def test_poc_transform_matches_model_checkpoint_eval_path():
    """The POC preprocessing must be byte-identical to the offline eval path."""
    import numpy as np
    import torch
    from PIL import Image

    _, transform, _, _ = load_model(BASELINE_CKPT, image_size=224)
    expected = _build_transforms(224, augmentation={}, is_train=False)

    arr = (np.random.RandomState(3).rand(256, 256, 3) * 255).astype("uint8")
    pil = Image.fromarray(arr)
    assert torch.equal(transform(pil), expected(pil))


@pytest.mark.skipif(not BASELINE_CKPT.exists(), reason="baseline checkpoint not present")
def test_predict_failing_score_is_well_defined_range():
    """The failing score is P(class=failing); it must be a probability in [0,1]."""
    import numpy as np
    from PIL import Image

    model, transform, device, info = load_model(BASELINE_CKPT, image_size=224)
    arr = (np.random.RandomState(1).rand(224, 224, 3) * 255).astype("uint8")
    pil = Image.fromarray(arr)
    result = predict_pil(model, transform, device, pil, threshold=0.5, fail_idx=info.fail_idx)
    assert result["fail_score"] >= 0.0 and result["fail_score"] <= 1.0


# ---------------------------------------------------------------------------
# Configuration validation (regression: invalid config was silently accepted)
# ---------------------------------------------------------------------------

def test_config_validate_accepts_defaults():
    PocConfig(source="x.mp4").validate()  # must not raise


@pytest.mark.parametrize("override", [
    {"threshold": 5.0},
    {"threshold": -0.1},
    {"sample_fps": 0.0},
    {"sample_fps": -2.0},
    {"alpha": 0.0},
    {"alpha": 1.5},
    {"baseline_window": 0},
    {"warmup": -1},
    {"confirm": 0},
    {"suspect_threshold": 2.0},
    {"fail_threshold": 1.5},
    {"recover_threshold": 0.9, "fail_threshold": 0.3},  # hysteresis violated
    {"fail_threshold_abs": 1.2},
    {"image_size": 0},
])
def test_config_validate_rejects_invalid(override):
    cfg = PocConfig(source="x.mp4", **override)
    with pytest.raises(PocError):
        cfg.validate()


# ---------------------------------------------------------------------------
# Checkpoint handling
# ---------------------------------------------------------------------------

def test_missing_checkpoint_raises_pocerror(tmp_path):
    with pytest.raises(PocError):
        load_model(tmp_path / "does_not_exist.pt", image_size=224)


def test_garbage_checkpoint_raises_pocerror(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"this is not a torch checkpoint")
    with pytest.raises(PocError):
        load_model(bad, image_size=224)


def test_checkpoint_without_state_dict_raises_pocerror(tmp_path):
    import torch

    ckpt = tmp_path / "nostate.pt"
    torch.save({"not_a_model": 1}, ckpt)
    with pytest.raises(PocError):
        load_model(ckpt, image_size=224)


def _write_tiny_checkpoint(path, normalize_mean=None, normalize_std=None,
                           image_size=160):
    import torch

    from tv.models.factory import build_classifier

    model = build_classifier("shufflenet_v2_x1_0", 2, pretrained=False)
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": 2,
        "config_model": "shufflenet_v2_x1_0",
        "class_to_idx": {"failing": 0, "normal": 1},
        "normalize_mean": normalize_mean,
        "normalize_std": normalize_std,
        "image_size": image_size,
    }, path)


def test_custom_normalization_and_image_size_read_from_checkpoint(tmp_path):
    """Regression: inference must reuse the checkpoint's stored normalization.

    The old poc hardcoded ImageNet stats and size 224, so a checkpoint trained
    with custom statistics was silently scored with the wrong preprocessing.
    """
    import numpy as np
    import torch
    from PIL import Image

    mean = [0.44555, 0.445957, 0.434777]
    std = [0.308143, 0.313864, 0.335357]
    ckpt = tmp_path / "custom.pt"
    _write_tiny_checkpoint(ckpt, normalize_mean=mean, normalize_std=std, image_size=160)

    _, transform, _, info = load_model(ckpt, image_size=None)

    assert info.image_size == 160
    assert info.normalize_mean == mean
    assert info.normalize_std == std

    expected = _build_transforms(
        160,
        augmentation={"normalize_mean": mean, "normalize_std": std},
        is_train=False,
    )
    arr = (np.random.RandomState(11).rand(200, 200, 3) * 255).astype("uint8")
    pil = Image.fromarray(arr)
    assert torch.equal(transform(pil), expected(pil))
    # And it must NOT equal the ImageNet-normalised path.
    imagenet = _build_transforms(160, augmentation={}, is_train=False)
    assert not torch.equal(transform(pil), imagenet(pil))


def test_imagenet_normalization_when_checkpoint_has_none(tmp_path):
    """Legacy checkpoints without stats fall back to ImageNet (unchanged)."""
    import numpy as np
    import torch
    from PIL import Image

    ckpt = tmp_path / "legacy.pt"
    _write_tiny_checkpoint(ckpt, normalize_mean=None, normalize_std=None, image_size=224)
    _, transform, _, info = load_model(ckpt, image_size=None)
    assert info.normalize_mean is None
    assert info.image_size == 224
    arr = (np.random.RandomState(5).rand(200, 200, 3) * 255).astype("uint8")
    pil = Image.fromarray(arr)
    assert torch.equal(transform(pil), _build_transforms(224, {}, is_train=False)(pil))


# ---------------------------------------------------------------------------
# End-to-end run over a synthetic video (frame sampling + timestamps + report)
# ---------------------------------------------------------------------------

def _make_video(path: Path, n_frames: int, fps: float = 10.0, size: int = 64) -> Path:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size)
    )
    assert writer.isOpened(), "cannot open VideoWriter for test fixture"
    for i in range(n_frames):
        frame = np.full((size, size, 3), 20 + (i * 7) % 200, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _e2e_available() -> bool:
    if not BASELINE_CKPT.exists():
        return False
    try:
        import cv2  # noqa: F401
    except Exception:
        return False
    return True


E2E = pytest.mark.skipif(not _e2e_available(),
                         reason="baseline checkpoint or OpenCV not available")


@E2E
def test_run_poc_processes_every_frame_with_monotonic_timestamps(tmp_path):
    video = _make_video(tmp_path / "clip.mp4", n_frames=20, fps=10.0)
    report = run_poc(PocConfig(source=str(video), checkpoint=BASELINE_CKPT,
                               device="cpu"))
    assert report["source"]["stride"] == 1
    assert report["timing"]["processed_frames"] == 20
    idx = [r["frame_idx"] for r in report["per_frame"]]
    assert idx == list(range(20))
    ts = [r["timestamp_s"] for r in report["per_frame"]]
    assert ts == [round(i / 10.0, 6) for i in range(20)]
    assert all(b > a for a, b in zip(ts, ts[1:]))


@E2E
def test_run_poc_half_cadence_uses_stride(tmp_path):
    """Regression: sample_fps==native must NOT drop every other frame."""
    video = _make_video(tmp_path / "clip.mp4", n_frames=20, fps=10.0)
    report = run_poc(PocConfig(source=str(video), checkpoint=BASELINE_CKPT,
                               sample_fps=5.0, device="cpu"))
    assert report["source"]["stride"] == 2
    assert [r["frame_idx"] for r in report["per_frame"]] == list(range(0, 20, 2))


@E2E
def test_run_poc_report_schema_and_json_serializable(tmp_path):
    import json

    video = _make_video(tmp_path / "clip.mp4", n_frames=6, fps=3.0)
    report = run_poc(PocConfig(source=str(video), checkpoint=BASELINE_CKPT,
                               device="cpu"))
    assert report["schema"] == "poc-inference"
    assert report["schema_version"] == 1
    for key in ("model", "source", "temporal", "summary", "detections",
                "timing", "per_frame"):
        assert key in report, f"missing report key: {key}"
    # model normalization is recorded and is a real per-channel triple
    norm = report["model"]["normalization"]
    assert len(norm["mean"]) == 3 and len(norm["std"]) == 3
    assert norm["source"] in ("checkpoint", "imagenet")
    # summary counts are consistent with the per-frame log
    assert report["summary"]["frames"] == len(report["per_frame"])
    assert "state_counts" in report["summary"]
    assert isinstance(report["detections"], list)
    # fully JSON-serialisable
    assert json.loads(json.dumps(report))["schema"] == "poc-inference"


@E2E
def test_run_poc_is_deterministic(tmp_path):
    import json

    video = _make_video(tmp_path / "clip.mp4", n_frames=12, fps=6.0)
    cfg = lambda: PocConfig(source=str(video), checkpoint=BASELINE_CKPT, device="cpu")
    a, b = run_poc(cfg()), run_poc(cfg())
    for key in ("per_frame", "detections", "summary", "model", "source"):
        assert json.dumps(a[key], sort_keys=True) == json.dumps(b[key], sort_keys=True)


@E2E
def test_run_poc_empty_or_unreadable_video_raises(tmp_path):
    bogus = tmp_path / "not_a_video.mp4"
    bogus.write_bytes(b"nope")
    with pytest.raises(PocError):
        run_poc(PocConfig(source=str(bogus), checkpoint=BASELINE_CKPT, device="cpu"))


@E2E
def test_run_poc_on_frame_callback_reports_progress(tmp_path):
    """The optional progress callback sees every frame + a total estimate."""
    video = _make_video(tmp_path / "clip.mp4", n_frames=10, fps=5.0)
    seen: list[tuple] = []

    def on_frame(processed, row, total):
        seen.append((processed, row["frame"], total))

    report = run_poc(PocConfig(source=str(video), checkpoint=BASELINE_CKPT,
                               device="cpu"), on_frame=on_frame)
    assert len(seen) == report["timing"]["processed_frames"] == 10
    assert [p for p, _, _ in seen] == list(range(1, 11))
    assert [f for _, f, _ in seen] == list(range(10))
    assert all(t == 10 for _, _, t in seen)


@E2E
def test_run_poc_on_frame_total_respects_max_frames(tmp_path):
    video = _make_video(tmp_path / "clip.mp4", n_frames=40, fps=4.0)
    totals: list = []
    run_poc(PocConfig(source=str(video), checkpoint=BASELINE_CKPT, device="cpu",
                      max_frames=6, sample_fps=2.0),
            on_frame=lambda p, r, t: totals.append(t))
    assert totals and all(t == 6 for t in totals)


def test_run_poc_missing_source_raises_pocerror():
    with pytest.raises(PocError):
        run_poc(PocConfig(source="/no/such/video_xyz.mp4", device="cpu"))


def test_run_poc_invalid_config_raises_pocerror():
    with pytest.raises(PocError):
        run_poc(PocConfig(source="x.mp4", threshold=9.0, device="cpu"))


@E2E
def test_format_summary_reports_key_fields(tmp_path):
    video = _make_video(tmp_path / "clip.mp4", n_frames=4, fps=2.0)
    report = run_poc(PocConfig(source=str(video), checkpoint=BASELINE_CKPT,
                               device="cpu"))
    text = format_summary(report)
    assert "POC inference summary" in text
    assert "threshold" in text
    assert "frames" in text