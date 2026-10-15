# Inference Proof-of-Concept: End-to-End Architecture Check

Status: research / no training performed
Date: 2026-09-06
Updated: 2026-10-15 (inference hardening + per-printer threshold calibration;
see "Update 2026-10-15" below)

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
    (Resize -> ToTensor -> checkpoint normalization; ImageNet only when the
    checkpoint does not record its own statistics, see Update 2026-10-15),
  * `EwmaBaseline` / `FailureFsm` (`temporal.py`).
* Do not gate the verdict on the known target-domain-normal data gap
  (documented in `exp3-target-normal-design.md`); report it honestly instead.
* No git commits.

## What was built

* `training/tv/poc.py` — `PocConfig` (small explicit config), `load_model`,
  `predict_pil`, `DecisionTracker` (incremental EWMA+FSM wrapper), `run_poc`
  (frame loop, sampling, annotated output, timing, report).
* `tv poc` subcommand in `training/tv/cli.py` (`--input FILE | --camera N`,
  `--checkpoint`, `--sample-fps`, `--max-frames`, `--threshold`, `--image-size`
  (default: taken from the checkpoint), temporal knobs `--alpha/--warmup/
  --suspect-threshold/--fail-threshold/--recover-threshold/--fail-threshold-abs/
  --confirm`, `--annotated-output`, `--output-json`).
* `training/tv/calibrate.py` + `tv calibrate` subcommand — per-printer
  threshold calibration (see Update 2026-10-15).
* `training/tests/test_poc.py` — 38 tests; `tests/test_calibrate.py` — 18;
  `tests/test_cli_inference.py` — 9.

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
    uv run python -m pytest tests/test_poc.py tests/test_calibrate.py \
      tests/test_cli_inference.py -q

Smoke run (frozen baseline, 1 fps cadence, short warm-up). The external
SP001 footage is research-only and not committed; substitute any local video:

    uv run --project training python -m tv.cli poc \
      --input /path/to/clip.mp4 \
      --sample-fps 1 --warmup 8 --confirm 2 \
      --annotated-output /tmp/poc.mp4 --output-json /tmp/poc.json

## Update 2026-10-15: inference hardening + per-printer threshold calibration

This update makes the file -> frames -> inference -> temporal decision ->
timestamped report path usable end to end, fixes four defects found by
inspection, adds a reproducible per-printer threshold-calibration workflow, and
proves both with tests plus synthetic-video end-to-end runs. No new datasets and
no training were used.

### Fixes to `training/tv/poc.py`

1. **Checkpoint normalization was ignored.** `load_model` hardcoded ImageNet
   statistics and `image_size=224`, so a checkpoint trained with custom
   normalization was scored with the wrong transform (the same defect already
   fixed in `reliability.score_framework`). `load_model` now reads
   `normalize_mean`/`normalize_std`/`image_size` from the checkpoint, records
   the source in the report (`model.normalization.source` = `checkpoint` or
   `imagenet_default`), and `PocConfig.image_size` defaults to the checkpoint's
   stored size.
2. **Frame-sampling off-by-one/stride bug.** The old `_next_frame` with the
   default `skip=1` dropped every other frame and mislabelled indices (a 10 fps
   / 20-frame clip processed 10 frames at indices `2,4,...,20`). Replaced by
   `_iter_sampled_frames(cap, stride, max_frames)`: `stride=1` means every
   frame, `frame_idx` is the true 0-based index, and
   `timestamp_s = frame_idx / effective_fps`.
3. **No timestamps.** Per-frame rows and the `detections` list now carry
   `timestamp_s` and `frame_idx`, so alerts are time-referenced.
4. **Unactionable errors.** Empty/unreadable video, an invalid checkpoint, and
   an invalid config now raise `PocError` with a specific message (the CLI
   prints it and exits 2) instead of a silent empty report or a raw
   `UnpicklingError`. `PocConfig.validate()` rejects nonsense
   (threshold outside [0,1], `sample_fps <= 0`, bad FSM knobs, `recover < fail`,
   etc.).

### Report schema

`tv poc` writes a deterministic report (`schema: "poc-inference"`,
`schema_version: 1`); only the `timing` block is non-deterministic. Top-level
keys: `model`, `source`, `temporal`, `summary`, `detections`, `timing`,
`annotated_output`, `per_frame`. `summary` includes
`first_failing_event_frame` and `first_failing_event_timestamp_s`.

### Per-printer threshold calibration (`tv calibrate`)

The domain-shift work (`docs/research/rc2-kaggle-experiment.md`) showed that a
threshold selected on one printer does not transfer to another, so a deployable
operating point must be calibrated per printer/camera. `training/tv/calibrate.py`
implements that as a deterministic CLI:

* Input is a clip **spec** (YAML or JSON) listing clips with `id`, `label`
  (`normal`|`failing`), `split` (`calibration`|`holdout`), and either a `source`
  (video file, frame glob, or frames dir) or precomputed `scores`. `test`/`locked`
  splits are rejected so a locked set can never be used for calibration.
* Each clip is reduced to a single alarm by `confirm` consecutive frames >=
  threshold (default `confirm=1`), reusing the POC scoring/temporal path.
* The threshold is selected **on the calibration split only**: maximise
  detection rate subject to a per-print false-alarm-rate ceiling
  (`--fpr-ceiling`, default 0.1). A clip is one complete print/session, so
  `FPR/print` = fraction of normal prints alarmed. Per-clip and per-print rates
  carry Wilson confidence intervals.
* Writes a report (`schema: "tinyvision-calibration"`, `schema_version: 1`) with
  the selection curve, calibration and holdout breakdowns, warnings, and an
  explicit disclaimer that a per-printer operating point does not establish
  cross-printer generalization.

Exact commands (from the repository root):

    # one-off inference on a video file (default report: <input>.poc.json)
    training/.venv/bin/python -m tv.cli poc \
      --input /path/to/clip.mp4 \
      --checkpoint experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/best.pt \
      --sample-fps 5 --threshold 0.5 --output-json /path/to/clip.poc.json

    # calibrate a threshold for one printer/camera
    training/.venv/bin/python -m tv.cli calibrate \
      --clips /path/to/clips.yaml \
      --checkpoint experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/best.pt \
      --sample-fps 5 --confirm 2 --fpr-ceiling 0.1 \
      --output-json /path/to/calibration.json

### Verification

* `tests/test_poc.py` (38), `tests/test_calibrate.py` (18),
  `tests/test_cli_inference.py` (9) pass; full suite 507 passed / 17 skipped.
* Synthetic-video end-to-end run (generated, not committed): a 10 fps / 20-frame
  clip at `--sample-fps 10` processes 20 frames at indices 0..19 with timestamps
  0.0..1.9 s; at `--sample-fps 5` it strides to indices 0,2,...,18.
* Error paths verified: missing source, unreadable/empty video, invalid
  checkpoint, and invalid config all exit 2 with a specific `error:` message.

### Footage still required

Calibration is implemented and tested with synthetic/precomputed scores, but it
has not been run on real target data because none exists. To calibrate and
validate a real operating point we need, from **one** printer/camera, a set of
matched clips: several `normal` clips and several `failing` clips, **each clip a
complete print/session**, each tagged with `printer`, `camera`, and a
`calibration`/`holdout` split. The holdout clips must come from prints/sessions
disjoint from the calibration clips.

## Local demo (2026-10-15): inspect a recording without a printer or camera

The smallest complete interface over the existing pipeline is a **local web UI
served by the Python backend**, so the demo needs no extra runtime (no Node, no
extra Python packages) and no cloud/hosted/paid service. It is deliberately not
a parallel application: every analysis calls `tv.poc.run_poc` directly.

* `training/tv/webdemo.py` — stdlib `http.server` server + background job
  manager. Routes: `GET /` (UI), `GET /api/health`, `GET /api/models`,
  `POST /api/analyze` (JSON `video_path`, or `application/octet-stream` upload
  with an `X-Filename` header), `GET /api/jobs/<id>` (status + live progress +
  rolling samples), `GET /api/jobs/<id>/report` (download the JSON report).
* `training/tv/demo_ui.html` — the single-page UI (model picker, upload-or-path
  video source, sample FPS / threshold / warm-up / confirm / max-frames, a live
  `P(failing)` + EWMA chart, a detections/events table, a per-frame table, and a
  JSON download). An **EXPERIMENTAL / not validated** banner is always shown, and
  the exported report carries a `disclaimer` field.
* `tv demo` subcommand in `training/tv/cli.py` (`--host/--port/--data-dir/
  --open-browser/--check`).
* `training/tv/launcher.py` — cross-platform launcher behind the installable
  `tinyvision-demo` console script: dependency check with actionable install
  instructions, loopback-by-default binding with a network-exposure warning, and
  actionable port-in-use errors. `tv demo` delegates to it, so both entry points
  behave identically.
* `scripts/make_synthetic_demo_video.py` — deterministic (fixed seed) synthetic
  test clip, explicitly labelled a **software test fixture**, not real data and
  not evidence of detection accuracy.

Uploads and exported reports are written under a per-user data directory outside
Git (`TINYVISION_DEMO_DATA`, default per platform — see the top-level README).
The model picker lists the local `experiments/*/artifacts/best.pt` checkpoints
and any directories in `TINYVISION_MODELS_DIR`; the UI also accepts a typed
checkpoint path. No weights are committed and nothing is downloaded.

### Install without being a Python developer

The UI asset ships with the package
(`[tool.setuptools.package-data] tv = ["demo_ui.html"]`), so a normal install
serves it without the source tree. Install once, launch with one command:

    uv pip install -e training          # or: python -m pip install -e training
    training/.venv/bin/tinyvision-demo --check          # report missing deps
    training/.venv/bin/tinyvision-demo --open-browser   # launch the UI

On Windows the entry point is `training\.venv\Scripts\tinyvision-demo.exe`.

### Exact commands

Launch the demo (one command; binds to localhost by default):

    training/.venv/bin/tinyvision-demo --open-browser
    # or: training/.venv/bin/tv demo --host 127.0.0.1 --port 8000

Run inference on a local video, either through the browser UI or headless:

    # headless (same pipeline the UI uses)
    training/.venv/bin/tv poc \
      --input /path/to/clip.mp4 \
      --checkpoint experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/best.pt \
      --sample-fps 5 --max-frames 300 --threshold 0.5 \
      --output-json /path/to/clip.poc.json

Generate the synthetic test clip:

    training/.venv/bin/python scripts/make_synthetic_demo_video.py \
      --out /Users/alex/tinyvision-data/tinyvision-demo/fixtures/synthetic_print.mp4

Whatever the demo shows on the synthetic clip proves only that the pipeline
runs; it is **not** a statement about real-world detection accuracy.

### Verification

* `tests/test_webdemo.py`: health/models/index (incl. `data_dir` + retention in
  health), settings coercion, the 400 error paths (missing path/file/checkpoint,
  invalid setting, unsupported content type, oversized upload, empty upload),
  unknown-job 404, report-before-done 409, actionable port-in-use error,
  `TINYVISION_MODELS_DIR` discovery, packaged-UI loading, and end-to-end
  JSON-path and upload jobs (progress → done → report → persisted file), plus
  synthetic-fixture determinism. `tests/test_poc.py` adds two `on_frame`
  progress-callback tests.
* `tests/test_launcher.py`: dependency map shape, `--check` exit codes, the
  refuse-to-run-without-dependencies path, parser defaults (loopback), and the
  `--check` subcommand exit code.
* `tests/test_packaging.py`: `demo_ui.html` is declared package data and
  resolvable via `importlib.resources`, the `tv` / `tinyvision-demo` console
  scripts are declared, `pillow` is a declared dependency, and `load_ui` returns
  the real UI (not the missing-asset fallback).

## Files changed

* `training/tv/poc.py` — new
* `training/tv/cli.py` — added `tv poc`, `tv demo` (additive; file already had
  earlier exp2/skz additions)
* `training/tests/test_poc.py` — new
* `training/tv/webdemo.py`, `training/tv/demo_ui.html`,
  `training/tv/launcher.py`, `scripts/make_synthetic_demo_video.py` — the local
  demo
* `training/pyproject.toml` — package-data for the UI asset, the
  `tinyvision-demo` console script, and a declared `pillow` dependency
* `training/tests/test_webdemo.py`, `tests/test_launcher.py`,
  `tests/test_packaging.py` — demo/launcher/packaging tests

No git commits were created.