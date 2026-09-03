"""Tests for Stage 3 temporal decision logic (EWMA + baseline + FSM)."""
from __future__ import annotations

import numpy as np
import pytest

from tv.temporal import (
    FAILING_EVENT,
    EwmaBaseline,
    FailureFsm,
    ewma_series,
    ewma_update,
    rolling_mean_series,
    run_temporal_pipeline,
)


def test_ewma_update_math():
    assert ewma_update(0.0, 1.0, 0.15) == pytest.approx(0.15)
    # E2 = 0.15*0.5 + 0.85*0.15
    assert ewma_update(0.15, 0.5, 0.15) == pytest.approx(0.15 * 0.5 + 0.85 * 0.15)


def test_ewma_series_starts_at_init():
    s = ewma_series([0.5, 0.5], alpha=0.5, init=0.0)
    assert s[0] == pytest.approx(0.25)
    assert s[1] == pytest.approx(0.375)


def test_ewma_alpha_invalid():
    from tv.temporal import EwmaBaseline

    with pytest.raises(ValueError):
        EwmaBaseline(alpha=0.0)
    with pytest.raises(ValueError):
        EwmaBaseline(alpha=1.5)


def test_rolling_mean_window():
    out = rolling_mean_series([1, 2, 3, 4, 5], window=3)
    # [1/1, 3/2, 6/3, 9/3, 12/3]
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(1.5)
    assert out[2] == pytest.approx(2.0)
    assert out[3] == pytest.approx(3.0)
    assert out[4] == pytest.approx(4.0)


def test_ewma_baseline_delta():
    eb = EwmaBaseline(alpha=0.5, baseline_window=3)
    # constant input -> delta ~ 0
    for _ in range(20):
        st = eb.step(0.5)
    assert st["delta"] == pytest.approx(0.0, abs=0.01)
    # a jump in score raises the signal above baseline
    eb.reset()
    for _ in range(10):
        eb.step(0.2)
    st = eb.step(0.9)
    assert st["delta"] > 0.05


def test_fsm_warmup_suppresses():
    fsm = FailureFsm(warmup=30)
    # extreme scores during warm-up must NOT fire
    fired = None
    for _ in range(40):
        d = fsm.step(0.99, 0.0)
        if d.events:
            fired = d.events[-1]
    assert d.state == "FAILING"
    # since warm-up ended, eventual firing is at/beyond warmup frame, but no
    # event should appear before warmup frames elapsed
    assert fired == FAILING_EVENT


def test_fsm_fire_once():
    fsm = FailureFsm(warmup=5, confirm=1, fail_threshold=0.2, recover_threshold=0.0)
    n_events = 0
    for _ in range(5):
        fsm.step(0.2, 0.0)
    for _ in range(200):
        d = fsm.step(0.9, 0.0)
        n_events += len(d.events)
    assert n_events == 1


def test_fsm_isolated_blip_does_not_fire():
    # confirm=3: a single-frame blip must not trigger FAILING
    fsm = FailureFsm(warmup=5, confirm=3, fail_threshold=0.2, recover_threshold=0.0)
    scores = [0.1] * 10 + [0.95] + [0.1] * 20
    any_event = any(
        FAILING_EVENT in d.events for d in fsm.run(scores, [0.0] * len(scores))
    )
    assert any_event is False


def test_fsm_persistent_failure_fires():
    fsm = FailureFsm(warmup=5, confirm=3, fail_threshold=0.2, recover_threshold=0.0)
    scores = [0.1] * 10 + [0.95] * 20
    dlist = list(fsm.run(scores, [0.0] * len(scores)))
    fired = any(FAILING_EVENT in d.events for d in dlist)
    assert fired is True


def test_fsm_hysteresis_state_after_recover():
    fsm = FailureFsm(
        warmup=5,
        confirm=1,
        fail_threshold=0.8,
        recover_threshold=0.3,
        fail_threshold_abs=0.95,
    )
    # warm up as NORMAL
    for _ in range(6):
        fsm.step(0.1, 0.0)
    # strong failure -> FAILING
    d = fsm.step(0.96, 0.0)
    assert d.state == "FAILING"
    # dip but stays above recover -> still FAILING (hysteresis)
    d = fsm.step(0.5, 0.0)
    assert d.state == "FAILING"
    # drop below recover -> PAUSED
    d = fsm.step(0.1, 0.0)
    assert d.state == "PAUSED"
    # stays paused (latched, no re-fire)
    d = fsm.step(0.9, 0.0)
    assert d.state == "PAUSED"
    assert len(d.events) == 0


def test_hysteresis_recover_must_be_lower():
    with pytest.raises(ValueError):
        FailureFsm(
            warmup=5, confirm=3, fail_threshold=0.3, recover_threshold=0.3
        )


def test_run_temporal_pipeline_shape_and_keys():
    scores = [0.1] * 50 + [0.9] * 30
    out = run_temporal_pipeline(scores, confirm=3, warmup=10)
    assert len(out) == len(scores)
    keys = {"frame", "raw", "signal", "baseline", "delta", "state", "event"}
    assert keys <= set(out[0].keys())
    assert any(o["event"] == FAILING_EVENT for o in out)
    states = {o["state"] for o in out}
    assert states <= {"OBSERVING", "NORMAL", "SUSPECT", "FAILING", "PAUSED"}


def test_fsm_title_case_no_event_before_warmup():
    fsm = FailureFsm(warmup=100, confirm=1, fail_threshold=0.1, recover_threshold=0.0)
    first_event_frame = None
    for i, d in enumerate(fsm.run([0.9] * 120, [0.0] * 120)):
        if FAILING_EVENT in d.events:
            first_event_frame = i
            break
    assert first_event_frame is not None and first_event_frame >= 100
