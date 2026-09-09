"""Tests for tv/poc.py (end-to-end inference proof of concept)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training"))

from tv.data.loaders import _build_transforms  # noqa: E402
from tv.poc import DecisionTracker, PocConfig, load_model, predict_pil  # noqa: E402

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