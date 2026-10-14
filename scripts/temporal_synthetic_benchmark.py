#!/usr/bin/env python3
"""Synthetic-stream evaluation of the temporal failure detector.

The frame classifier is the noisy part; the temporal decision layer (EWMA +
baseline + FSM in ``tv.temporal``) must turn a stream of per-frame failure
probabilities into a small number of reliable, fire-once events. This script
drives that layer with hand-built streams whose ground truth is known, so the
layer can be validated without any model, dataset or hardware.

Scenarios:
  * steady_normal    - no failure; must NOT fire.
  * sustained_fail   - clean normal->failure step; must fire once, low latency.
  * single_blip      - one isolated high frame; must NOT fire.
  * spike_train      - repeated short spikes below confirmation; must NOT fire.
  * gradual_drift    - slow ramp into failure; should fire (higher latency ok).
  * recovery         - failure then return to normal; fires once then pauses.

Output: docs/research/temporal-synthetic-benchmark.json

Usage:
    python scripts/temporal_synthetic_benchmark.py
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from tv.temporal import FAILING_EVENT, run_temporal_pipeline


def _stream_normal(n: int, value: float = 0.05) -> list[float]:
    return [value] * n


def scenarios() -> dict[str, dict]:
    steady = _stream_normal(200, 0.05)

    sustained = _stream_normal(60, 0.05) + [0.92] * 140

    blip = list(steady)
    blip[100] = 0.97

    spikes = list(steady)
    for i in range(60, 200, 10):  # a spike every 10 frames, never long enough
        spikes[i] = 0.9

    drift = [0.05 + (0.9 * max(0, i - 60) / 140) for i in range(200)]
    drift = [min(v, 0.95) for v in drift]

    recovery = _stream_normal(60, 0.05) + [0.92] * 40 + _stream_normal(100, 0.05)

    return {
        "steady_normal": {"scores": steady, "onset": None, "expect_fire": False},
        "sustained_fail": {"scores": sustained, "onset": 60, "expect_fire": True},
        "single_blip": {"scores": blip, "onset": None, "expect_fire": False},
        "spike_train": {"scores": spikes, "onset": None, "expect_fire": False},
        "gradual_drift": {"scores": drift, "onset": 60, "expect_fire": True},
        "recovery": {"scores": recovery, "onset": 60, "expect_fire": True},
    }


def evaluate(name: str, spec: dict, params: dict) -> dict:
    out = run_temporal_pipeline(spec["scores"], **params)
    events = [o["frame"] for o in out if o.get("event") == FAILING_EVENT]
    fired = bool(events)
    first = events[0] if events else None
    onset = spec["onset"]
    latency = (first - onset) if (first is not None and onset is not None) else None
    return {
        "scenario": name,
        "n_frames": len(spec["scores"]),
        "expect_fire": spec["expect_fire"],
        "fired": fired,
        "n_events": len(events),
        "first_event_frame": first,
        "onset_frame": onset,
        "latency_frames": latency,
        "correct": fired == spec["expect_fire"] and len(events) <= 1,
    }


def main() -> int:
    base = dict(
        alpha=0.15,
        baseline_window=60,
        warmup=30,
        suspect_threshold=0.05,
        fail_threshold=0.2,
        recover_threshold=0.05,
        confirm=3,
        use_delta=True,
    )
    # Two configurations are reported: the shipped default (delta/baseline only)
    # and one that also honours an absolute signal floor. The comparison exists
    # because a purely baseline-relative detector can miss a *gradual* drift.
    configs = {
        "default_delta_only": dict(base),
        "with_abs_floor": {**base, "fail_threshold_abs": 0.6},
    }

    report_configs = {}
    for cname, params in configs.items():
        results = [evaluate(name, spec, params) for name, spec in scenarios().items()]
        report_configs[cname] = {
            "params": params,
            "results": results,
            "all_correct": all(r["correct"] for r in results),
        }
        print(f"\n== {cname} ==")
        print(f"{'scenario':16} {'expect':>6} {'fired':>6} {'evts':>5} {'latency':>8}  ok")
        for r in results:
            lat = "-" if r["latency_frames"] is None else str(r["latency_frames"])
            print(
                f"{r['scenario']:16} {str(r['expect_fire']):>6} {str(r['fired']):>6} "
                f"{r['n_events']:>5} {lat:>8}  {r['correct']}"
            )

    report = {
        "schema": "tinyvision-temporal-synthetic-benchmark",
        "generated": date.today().isoformat(),
        "configs": report_configs,
        "note": (
            "Synthetic ground truth for the decision layer only. A 'fire' means "
            "the FSM emitted FAILING_EVENT at least once; correct scenarios fire "
            "exactly once when failure is expected and never when it is not."
        ),
    }
    out = Path("docs/research/temporal-synthetic-benchmark.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out}")
    print("default all_correct:", report_configs["default_delta_only"]["all_correct"])
    print("abs-floor all_correct:", report_configs["with_abs_floor"]["all_correct"])
    # Success = the absolute-floor configuration is fully correct.
    return 0 if report_configs["with_abs_floor"]["all_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
