"""Temporal decision logic (Stage 3).

Implements the decision layer that operates on a stream of per-frame failure
probabilities produced by a frozen frame classifier:

    frame p_t --> EWMA smoothing --> baseline-relative signal --> FSM.

The FSM states are OBSERVING (warm-up) -> NORMAL -> SUSPECT -> FAILING -> PAUSED,
with hysteresis (entry threshold > exit threshold) and fire-once action
semantics (the FAILING event / pause is emitted at most once per episode).

This module is deliberately pure-DSL (no torch / no data loading) so the logic
is unit-testable and independent of any model. See
``docs/research/temporal-decision-logic.md`` for the design rationale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional


def ewma_update(prev: float, value: float, alpha: float) -> float:
    """Single EWMA update: E_t = alpha*value + (1-alpha)*E_{t-1}."""
    return alpha * value + (1.0 - alpha) * prev


def ewma_series(values, alpha: float, init: float = 0.0):
    """Return the EWMA of ``values`` with smoothing factor ``alpha``.

    ``init`` is E_0 (the value used for the very first frame). The returned
    list has the same length as ``values``.
    """
    out = []
    e = init
    for v in values:
        e = ewma_update(e, v, alpha)
        out.append(e)
    return out


def rolling_mean_series(values, window: int):
    """Running (causal) window mean; shorter windows at the start."""
    out = []
    acc = 0.0
    n = 0
    for i, v in enumerate(values):
        acc += v
        n += 1
        if n > window:
            acc -= values[i - window]
            n = window
        out.append(acc / n)
    return out


class EwmaBaseline:
    """EWMA smoothed signal plus a long causal baseline reference.

    ``signal`` is the EWMA of the input scores; ``delta`` = signal - baseline
    measures "this moment is worse than the print's rolling normal baseline",
    which is the baseline-relative trigger signal (mirrors Obico's
    ewm_mean - rolling_mean_long).
    """

    def __init__(self, alpha: float = 0.15, baseline_window: int = 60):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0,1], got {alpha}")
        self.alpha = alpha
        self.baseline_window = baseline_window
        self._sig = 0.0
        self._buf: List[float] = []

    def reset(self) -> None:
        self._sig = 0.0
        self._buf = []

    def step(self, p: float) -> dict:
        self._sig = ewma_update(self._sig, p, self.alpha)
        self._buf.append(self._sig)
        if len(self._buf) > self.baseline_window:
            self._buf.pop(0)
        baseline = sum(self._buf) / len(self._buf)
        return {
            "signal": self._sig,
            "baseline": baseline,
            "delta": self._sig - baseline,
        }


FAILING_EVENT = "failing_event"


@dataclass
class FsmDecision:
    state: str = "OBSERVING"
    signal: float = 0.0
    delta: float = 0.0
    suspect_count: int = 0
    events: List[str] = field(default_factory=list)
    frame: int = 0


class FailureFsm:
    """Minimal FSM over a smoothed failure signal with hysteresis + fire-once.

    States: OBSERVING -> NORMAL -> SUSPECT -> FAILING -> PAUSED.

    Transitions (all O(1) per frame):
      * OBSERVING -> NORMAL after ``warmup`` frames.
      * NORMAL -> SUSPECT when ``delta > suspend_threshold`` (or signal-based
        threshold when ``use_delta`` is False).
      * SUSPECT -> FAILING after ``confirm`` consecutive SUSPECT frames AND
        ``delta`` above ``fail_threshold``.
      * NORMAL/SUSPECT -> FAILING immediately when the absolute score exceeds
        ``fail_threshold_abs`` (hard latch guard).
      * FAILING -> PAUSED when ``delta`` falls below ``recover_threshold``
        (hysteresis: recover_threshold < fail_threshold).
      * SUSPECT -> NORMAL when ``delta`` falls below ``suspect_threshold``.

    The FAILING event is emitted exactly once (on the first NORMAL/SUSPECT ->
    FAILING transition); later frames in FAILING/PAUSED do not re-fire.
    """

    def __init__(
        self,
        warmup: int = 30,
        suspect_threshold: float = 0.05,
        fail_threshold: float = 0.20,
        recover_threshold: float = 0.05,
        fail_threshold_abs: Optional[float] = None,
        confirm: int = 3,
        use_delta: bool = True,
    ):
        if fail_threshold_abs is not None and not 0 <= fail_threshold_abs <= 1:
            raise ValueError("fail_threshold_abs must be in [0,1] or None")
        if recover_threshold >= fail_threshold:
            raise ValueError("recover_threshold must be < fail_threshold (hysteresis)")
        self.warmup = warmup
        self.suspect_threshold = suspect_threshold
        self.fail_threshold = fail_threshold
        self.recover_threshold = recover_threshold
        self.fail_threshold_abs = fail_threshold_abs
        self.confirm = confirm
        self.use_delta = use_delta
        self.reset()

    def reset(self) -> None:
        self.state = "OBSERVING"
        self.frame = 0
        self.suspect_count = 0
        self._fired = False

    def _guard(self, d: FsmDecision) -> float:
        return d.delta if self.use_delta else d.signal

    def step(self, signal: float, baseline: Optional[float] = None) -> FsmDecision:
        self.frame += 1
        delta = signal - baseline if baseline is not None else signal
        g = self._guard(FsmDecision(signal=signal, delta=delta, frame=self.frame))
        d = FsmDecision(
            state=self.state,
            signal=signal,
            delta=delta,
            suspect_count=self.suspect_count,
            frame=self.frame,
        )

        if self.state != "PAUSED":
            if self.state == "OBSERVING":
                if d.frame >= self.warmup:
                    self.state = "NORMAL"
            elif self.state == "FAILING":
                if g <= self.recover_threshold:
                    self.state = "PAUSED"
            else:
                # NORMAL / SUSPECT
                abs_hit = (
                    self.fail_threshold_abs is not None
                    and signal >= self.fail_threshold_abs
                )
                if abs_hit:
                    self._enter_failing(d)
                elif g >= self.fail_threshold:
                    self.suspect_count += 1
                    if self.suspect_count >= self.confirm:
                        self._enter_failing(d)
                    elif self.state != "SUSPECT":
                        self.state = "SUSPECT"
                elif self.state == "SUSPECT":
                    if g <= self.suspect_threshold:
                        self.suspect_count = 0
                        self.state = "NORMAL"
                elif g >= self.suspect_threshold:
                    self.state = "SUSPECT"
                    self.suspect_count = 1

        d.state = self.state
        d.suspect_count = self.suspect_count
        return d

    def _enter_failing(self, d: FsmDecision) -> None:
        transitioning = self.state in ("NORMAL", "SUSPECT")
        self.state = "FAILING"
        if transitioning and not self._fired:
            d.events.append(FAILING_EVENT)
            self._fired = True

    def run(self, signals, baselines=None) -> Iterator[FsmDecision]:
        for i, s in enumerate(signals):
            b = baselines[i] if baselines is not None else None
            yield self.step(s, b)


def run_temporal_pipeline(
    scores,
    alpha: float = 0.15,
    baseline_window: int = 60,
    warmup: int = 30,
    suspect_threshold: float = 0.05,
    fail_threshold: float = 0.20,
        recover_threshold: float = 0.05,
        fail_threshold_abs: Optional[float] = None,
        confirm: int = 3,
        use_delta: bool = True,
    ):
    """End-to-end temporal pipeline: scores -> decisions (one dict per frame).

    Returns a list of per-frame dicts with keys: frame, raw, signal, baseline,
    delta, state, event.
    """
    ew = EwmaBaseline(alpha=alpha, baseline_window=baseline_window)
    fsm = FailureFsm(
        warmup=warmup,
        suspect_threshold=suspect_threshold,
        fail_threshold=fail_threshold,
        recover_threshold=recover_threshold,
        fail_threshold_abs=fail_threshold_abs,
        confirm=confirm,
        use_delta=use_delta,
    )
    out = []
    for i, p in enumerate(scores):
        st = ew.step(p)
        d = fsm.step(st["signal"], st["baseline"])
        out.append(
            {
                "frame": i,
                "raw": float(p),
                "signal": st["signal"],
                "baseline": st["baseline"],
                "delta": st["delta"],
                "state": d.state,
                "event": d.events[-1] if d.events else None,
            }
        )
    return out
