# C++ Edge Runtime: Architecture and Python-Reference Parity

Status: implemented / measured
Date: 2026-09-06

## Purpose

The smallest practical C++/ONNX Runtime edge inference runtime that reproduces
the Python POC's behavior for the TinyVision failure-monitoring loop, so that
production inference can move off the Python training stack without changing
observed semantics. It runs the frozen ShuffleNet binary classifier
(`experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/model.onnx`)
through the same preprocessing contract and the same temporal decision logic as
`tv poc`.

Build requirements / stack: C++17, OpenCV (`find_package(OpenCV)`), ONNX Runtime
CPU EP (`find_package(onnxruntime)`), CMake + Ninja. The runtime has no
dependency on the Python environment.

## Components

* `runtime/src/preprocess.cpp` — BGR decode frame -> RGB -> resize(224) ->
  `(p/255 - mean)/std` -> NCHW float. Resize defaults to a faithful port of
  Pillow's two-pass BILINEAR resampler (`src/libImaging/Resample.c`, HPND):
  separable convolution, truncating `(int)(center +/- support + 0.5)` bounds,
  `PRECISION_BITS = 22` fixed-point coefficient quantization, horizontal then
  vertical passes. Output pixels are bit-identical to `Image.resize(..., BILINEAR)`
  (verified against real PIL at 16x32->8x8 and 320x180->224 downscales).
  `cv::INTER_AREA` / `INTER_LINEAR` remain selectable for comparison only.
* `runtime/src/model.cpp` — load ONNX, run CPU session, P(failing) from softmax
  logits [N,2], fail_idx=0. Clear errors on missing/malformed model or session
  not loaded.
* `runtime/src/temporal.cpp` — direct port of `EwmaBaseline` + `FailureFsm`
  (OBSERVING -> NORMAL -> SUSPECT -> FAILING -> PAUSED; warm-up, confirm,
  hysteresis, fire-once, PAUSED recovery; `delta = signal - baseline` computed
  exactly as the reference). Bit-exact on synthetic golden traces.
* `runtime/src/main.cpp` — sampling mirrors the POC `_next_frame` exactly
  (discard `skip` frames, keep the next), identical JSON report schema
  (`schema: "poc-inference"`), per-frame `frame` (0-based sample index) and
  `frame_idx` (raw video frame index) matching the POC.

## Tests (14, all green via ctest)

* Preprocess layout/normalization hand-computed; empty-frame throws.
* `PilBilinearMatchesPillowSmall` / `PilBilinearMatchesPillowLarge`: bit-exact
  (tolerance 0) vs actual Pillow output.
* Temporal: matches Python golden traces bit-exact; quiet sequence never fails;
  spike fires once + recovery; abs-hard-latch before confirm.
* Model: logits match Python ONNX Runtime (<1e-6); clear failures on missing
  file / bad input shape / session-not-loaded.
* `ResizeAreaMatchesCv2Golden`: keeps the cv2 path checked for comparison
  purposes.

## Parity methodology

`scripts/edge_parity.py` runs `tinyvision` and `tv poc` on the same frozen
videos (`--sample-fps 1 --warmup 8 --confirm 2`, threshold 0.5) and compares
per-frame P(failing), raw decisions, FSM state sequences, and first FAILING
event, plus length.

### Parity result (all four videos)

| source | frames | state diffs | flips | first event py/cpp | max \|dP\| |
|---|---|---|---|---|---|
| NORMAL001 | 19 | 0/19 | 0/19 | 8 / 8 | 0.036457 |
| SP001 | 24 | 0/24 | 0/24 | 8 / 8 | 0.039991 |
| UE003 | 352 | 0/352 | 1/352 | 8 / 8 | 0.057462 |
| STR001 | 618 | 0/618 | 7/618 | 9 / 9 | 0.137408 |

State sequences, sampled-frame counts, and first-alert frames are **exact** on
all four sources. UE003's false alarm reproduces on the C++ runtime exactly as
in the POC (expected; it is a target-normal data problem, not a runtime bug).

### Implementation notes that made parity exact

* **Sampling skew was the dominant raw-P divergence.** A one-frame offset in
  sampled frames produces P(failing) differences up to ~0.3 on these streams.
  After aligning the C++ loop to the POC `_next_frame` (discard `skip`, then
  read one more), the per-sampled video frame is identical and raw-P diffs
  collapsed from ~0.99 to <=0.04 on three of four videos.
* **The Pillow-bilinear port closes the resize path.** Naive `cv::INTER_AREA`
  alone measured up to ~0.99 raw-P difference; the exact-PIL port matches
  PIL/torchvision bit-for-bit.

### Remaining floor: decoder build differences

Even with identical sampling and bit-exact resize, STR001 shows max |dP| =
0.137 (mean diff 0.006, std 0.022). Root cause measured directly: Homebrew
FFmpeg (linked by the C++ build) and the wheel-bundled FFmpeg (used by the
Python `opencv-python`) decode the same source frame with up to 8/255 per-pixel
differences (mean ~2/255), which the classifier amplifies. This is a platform
property of the decode stack, not a runtime logic difference: the C++ runtime
decodes and processes consistently with its own stack. Parity tolerance is set
to this empirical floor (0.15); 0.10 would flag only decoder noise.

## Timing (dev machine, qualitative)

C++ per-frame: inference mean ~2.2 ms (p95 2.3-4.4 ms); decode mean 14-44 ms
(decode-bound, 1440p worst); preprocess mean 2.1-4.2 ms; sustained cadence
19-53 processed fps. Python reference: inference mean 13.6-43 ms (p95 up to
554 ms warm-up spikes); cadence 13-27 fps.

## Rebuild / run

    cmake -S runtime -B runtime/build -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build runtime/build
    ctest --test-dir runtime/build --output-on-failure
    python3 scripts/edge_parity.py          # runs from repo root

## Deliberately not done

No model retraining, no dataset collection, no threshold/FSM tuning to
suppress UE003, no quantization yet. The `cv::INTER_AREA`/`INTER_LINEAR` paths
exist only for comparison. Frozen artifacts (`model.onnx`, `best.pt`, videos)
were read-only.