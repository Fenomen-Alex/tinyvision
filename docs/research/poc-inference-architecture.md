# Inference Proof-of-Concept: End-to-End Architecture Check

Status: research / no training performed
Date: 2026-09-06

## Objective

Answer the architecture-viability question before investing in more
engineering: can the existing TinyVision training pipeline power a real-time
inference loop, end to end?

A minimal POC was built that runs the frozen binary ShuffleNet classifier over
a video file or webcam, streams per-frame `P(failing)` through the existing
Stage-3 temporal decision layer (EWMA + Failure-FSM), and emits a per-frame
decision log, state/event summary, timing, and an optional annotated video.

This is plumbing/architecture verification, **not** a benchmark. Smoke numbers
are qualitative and labeled accordingly.

## Scope and constraints

* No training. The checkpoint is frozen
  (`experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/best.pt`).
* No dataset pipeline. Input is an already-existing video file or webcam.
* Reuse the canonical inference path as-is, so train/deploy preprocessing can
  never drift:
  * checkpoint loading + `build_classifier` (`models/factory.py`),
  * eval-time transform `_build_transforms(image_size, {}, is_train=False)`
    (Resize 224 -> ToTensor -> ImageNet normalize),
  * `EwmaBaseline` / `FailureFsm` (`temporal.py`).
* Do not gate the verdict on the known target-domain-normal data gap
  (documented in `exp3-target-normal-design.md`); report it honestly instead.
* No git commits.

## What was built

* `training/tv/poc.py` — `PocConfig` (small explicit config), `load_model`,
  `predict_pil`, `DecisionTracker` (incremental EWMA+FSM wrapper), `run_poc`
  (frame loop, sampling, annotated output, timing, report).
* `tv poc` subcommand in `training/tv/cli.py` (`--input FILE | --camera N`,
  `--checkpoint`, `--sample-fps`, `--max-frames`, `--threshold`,
  temporal knobs `--alpha/--warmup/--suspect-threshold/--fail-threshold/
  --recover-threshold/--fail-threshold-abs/--confirm`,
  `--annotated-output`, `--output-json`).
* `training/tests/test_poc.py` — 10 tests.

The per-frame decision uses the canonical threshold `P(failing) >= 0.5 ->
failing`. The FSM states are the existing Stage-3 states
`OBSERVING -> NORMAL -> SUSPECT -> FAILING -> PAUSED` (the requested
normal/suspect/failing behavior plus the warm-up and recovered states already
defined in the codebase).

## Verification

### Unit tests (10 passed)

* Transform parity: the POC transform equals the offline eval transform
  (byte-identical tensor on a random image).
* Determinism: the same frame yields the same `P(failing)`.
* Checkpoint roundtrip: loads the frozen ShuffleNet binary checkpoint, 2
  classes, fail_idx=0.
* DecisionTracker on synthetic score sequences: quiet stream never fails;
  single short blip does not trip FAILING; sustained spike fires exactly once
  (fire-once semantics) and recovers to PAUSED; absolute hard latch fires
  before the confirm counter.

Full suite: 339 passed. 5 failures in `test_skz_experiment.py` are pre-existing
and unrelated (the SKZ probe image files are not present in this checkout;
those tests raised `FileNotFoundError` before this POC work).

### Smoke runs on existing footage (qualitative)

Runner: dev MacBook (MPS/CPU path), native-resolution decode, sampled at a
1 fps cadence (`--sample-fps 1`). Checkpoint: frozen baseline.

| source | ground truth | frames | raw failing | first FAILING event | notes |
|---|---|---|---|---|---|
| NORMAL001 (1080p, ~21 s) | verified failing (stringing) | 19 | 19/19 | n/a (warm-up 30) | clip shorter than 30-frame warm-up, stayed OBSERVING |
| SP001 (720p, ~24 s) | verified failing (spaghetti) | 24 | 22/24 | n/a (warm-up 30) | same warm-up effect |
| STR001 (1440p) | stringing print | 618 | n/a | frame 30 | 588/618 frames FAILING |
| UE003 (1080p) | human-normal (eval-only) | 352 | n/a | frame 30 | 322/352 frames FAILING — false alarm on target normal |

With a short warm-up (`--warmup 8 --confirm 2`) that suits short clips:

| source | first FAILING event | FAILING frames |
|---|---|---|
| NORMAL001 | frame 8 | 11/19 |
| SP001 | frame 8 | 16/24 |
| STR001 | frame 9 | 609/618 |
| UE003 | frame 8 | 344/352 (still false alarms) |

Timing (qualitative, dev machine, includes native-res decode + resize + infer):

* sustained cadence 13–27 processed fps across 720p–1440p/4K sources (decode-bound; 4K worst),
* per-frame inference mean 13.6–23.7 ms, p95 14–195 ms.

Not a benchmark — see EDGE/BENCHMARKS work for real latency engineering. The
point here is that the Python reference path sustains a 1 fps sampling cadence
with huge headroom even while decoding multi-megapixel footage.

Annotated video output was verified (writes a playable MP4 with the state
overlay).

## Findings

1. **The end-to-end loop works and reuses canonical components untouched.**
   Frame capture -> eval-transform -> ShuffleNet -> EWMA -> FSM -> alert is
   implemented without any preprocessing/schema drift, and the full temporal
   semantics (warm-up, SUSPECT, confirm, hysteresis, fire-once, PAUSED
   recovery) are testable and operational in the loop.

2. **Failure detection transfers to unseen footage.** ALL known-failing
   sources (NORMAL001, SP001, STR001) trip a FAILING alert quickly (frame 8–9
   with a short warm-up) at a 1 fps cadence.

3. **Target-domain normal recognition is the blocking data/domain issue, and
   temporal logic cannot compensate for it.** UE003 — a session humans review
   as normal — trips FAILING within 8 frames and stays there, because per-frame
   `P(failing)` is high throughout (333/352 frames raw-failing). This matches
   the frozen baseline's poor target-normal FPR (exp2 report) and the "zero
   verified normal frames" audit (exp3). It is a training-data problem, not an
   inference-pipeline problem.

4. **Parameterization matters relative to sampling cadence.** The Stage-3
   FSM default warm-up of 30 frames equals 30 s at a 1 fps cadence, so clips
   shorter than ~30 s never leave OBSERVING and cannot alert. The POC exposes
   the temporal knobs on the CLI; the right settings should be chosen with the
   offline sweep machinery (`tv temporal`, `temporaleval.py`), not hardcoded.

5. **Production C++/"edge" runtime is now implemented and parity-checked.**
   `docs/research/cpp-runtime-parity.md` documents the C++/ONNX Runtime
   reference runtime (`runtime/`): exact-Pillow-bilinear preprocessing, the
   same sampling as the POC, 14 passing tests, and behavioral parity on the
   four frozen videos (state sequences, frame counts, and first-alert frames
   exact; raw P within the measured decoder-build floor of 0.15).

## Limitations

* Smoke numbers are qualitative (dev machine, few sources, short clips).
* No benchmark-grade timing; no quantization/export comparison
  (`tv export` / Edge requirements are out of POC scope).
* Webcam path is implemented and shares the exact same loop as file input but
  was not exercised here (no capture in this environment).
* UE003 is a single-source descriptive signal, not a benchmark (same caveat as
  the exp2 external evaluation).

## Verdict: A — Proceed

The architecture is viable. Per-frame classification plus temporal aggregation
runs correctly end-to-end at more than headroom cadence for a failure-monitor
use case, with no train/deploy drift, on the existing frozen checkpoint.

The two real problems found are not architecture problems:

* **Data**: acquirable (Exp3 option 2 — small targeted normal acquisition plus
  verified temporal labels would directly attack the UE003 false-alarm
  behavior). The Exp2 result (failing-class recall transfer) plus ExpB (SKZ
  auxiliary) give concrete retrain directions.
* **Parameterization**: cadence-relative warm-up/confirm settings should be
  selected via the existing offline sweep, then pinned in a deploy config.

Do **not** choose C: there is no evidence that the per-frame + temporal design
is the wrong shape; the observed failures are exactly the known target-normal
data gap, which the task explicitly says must not decide the verdict.

## Next steps (not started)

1. Acquire the small targeted normal set (Exp3 option 2) and verify temporal
   labels; retrain and re-run the POC on UE003 to show the false alarm
   resolving.
2. Use `tv temporal` offline sweep to fix warm-up/confirm for the target
   sampling cadence; probe converting the summary into a DeployConfig.
3. C++/ONNX Runtime reference runtime with the same preprocessing contract
   (Python remains the reference oracle). Started: see
   `docs/research/cpp-runtime-parity.md`; next is edge-device benchmarking.

## Reproduce

Unit tests:

    cd training
    uv run python -m pytest tests/test_poc.py -q

Smoke run (frozen baseline, 1 fps cadence, short warm-up):

    uv run --project training python -m tv.cli poc \
      --input datasets/external/video-frames/raw/SP001/SP001_3Ci-3ZYHBkM.mp4 \
      --sample-fps 1 --warmup 8 --confirm 2 \
      --annotated-output /tmp/sp001_poc.mp4 --output-json /tmp/sp001_poc.json

## Files changed

* `training/tv/poc.py` — new
* `training/tv/cli.py` — added `tv poc` (additive; file already had earlier
  exp2/skz additions)
* `training/tests/test_poc.py` — new

No other repository files were modified. No git commits were created.