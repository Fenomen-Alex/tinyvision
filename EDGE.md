# TinyVision Edge Runtime: Deployment Guide

The C++ inference runtime (`runtime/`) is the production path for TinyVision.
It runs a frozen ShuffleNetV2 binary classifier through the same
preprocessing contract and temporal decision logic as the Python POC, with no
dependency on the training environment. Behavioral parity methodology,
results, and the decoder-build tolerance floor are documented in
[docs/research/cpp-runtime-parity.md](docs/research/cpp-runtime-parity.md);
baseline timing/memory in
[docs/research/edge-runtime-benchmark.md](docs/research/edge-runtime-benchmark.md).

## Architecture

```
video file or webcam
        │  cv::VideoCapture (FFmpeg backend)
        ▼
BGR frame ── runtime/src/preprocess.cpp ──► RGB 224x224 NCHW fp32
        │     (Pillow-bilinear port, bit-exact with PIL on resize)
        ▼
runtime/src/model.cpp  (ONNX Runtime CPU EP, fp32)
        │     P(failing) = softmax(logits)[fail_idx=0]
        ▼
runtime/src/temporal.cpp  (EwmaBaseline + FailureFsm port)
        │     raw P ─► signal ─► baseline/delta ─► FSM state
        ▼
stdout summary + JSON report (schema "poc-inference")
```

Directory layout: `runtime/include/tinyvision/` headers, `runtime/src/` sources,
`runtime/tests/` (CTest unit tests + integration smoke).

## Deployment environment

* Build: CMake >= 3.16, Ninja, C++17.
* Libraries: OpenCV (>= 4.x; developed against 5.0.0 via Homebrew),
  ONNX Runtime C++ (developed against 1.29.0, CPU EP).
* Target hypothesis (not yet measured): Raspberry Pi 4/5 + USB/CSI webcam,
  CPU-only inference. Quantization/int8 is deferred.
* The runtime has **no** Python/PyTorch dependency.

## Model requirements

* ONNX format, fp32, single frozen classifier exported from
  `tv.models.factory.build_classifier("shufflenet_v2_x1_0", 2)`.
* Input `[1, 3, 224, 224]` float, mean/std ImageNet normalization applied in
  preprocessing. Output `[1, 2]` logits; `fail_idx = 0`.
* Deployed artifact: `experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/model.onnx`
  (default path; overridable with `--model`). The runtime refuses to start if
  the configured `--image-size` does not match the model input shape.

## Input sources

* `--input FILE.mp4` — offline video file.
* `--camera N` — live webcam via OpenCV (AVFoundation on macOS). Unavailable
  devices and macOS privacy denials produce a clear error and exit 1.
* Exactly one source is required; `--input` and `--camera` are mutually exclusive.
* The camera capture loop is the same shared frame/sampling/FSM/JSON path as
  file input, which is the validated surface (see `runtime/tests/`). A physical
  device open has **not** been validated in a privacy-permitted environment;
  the device-open failure path (unavailable device, macOS permission denial)
  is verified and exits cleanly with a diagnostic.

## Sampling

* `--sample-fps F` selects frames from the native stream exactly as the POC
  `_next_frame`: discard `skip` frames, then keep the next (skip =
  `round(native_fps / F)`). `0` (default) = every frame.
* The deployed baseline is `--sample-fps 1` (the POC configuration).

## Configuration parameters

All options have validated defaults equal to the POC config; the defaults are
behavioral commitments and are not tuned per-deployment.

| Flag | Default | Meaning |
| --- | --- | --- |
| `-m, --model` | `experiments/.../model.onnx` | ONNX model path (must be non-empty) |
| `--image-size` | 224 | Resize target; must equal model input |
| `--sample-fps` | 0 | Sampling cadence (0 = every frame) |
| `--max-frames` | 0 | Stop after N sampled frames (0 = unlimited) |
| `--threshold` | 0.5 | Raw binary decision threshold on P(failing) |
| `--output-json` | (none) | Write JSON report to this file |
| `--alpha` | 0.15 | EWMA alpha |
| `--baseline-window` | 60 | Baseline tracking window |
| `--warmup` | 30 | FSM warm-up samples |
| `--suspect-threshold` | 0.05 | Enter SUSPECT |
| `--fail-threshold` | 0.30 | Enter FAILING (hysteresis) |
| `--recover-threshold` | 0.01 | Leave FAILING/PAUSED after confirmation |
| `--fail-threshold-abs` | 0.65 | Absolute hard-latch level |
| `--confirm` | 3 | Consecutive samples to confirm transitions |
| `--use-delta` | false | Use `signal - baseline` as the FSM input (POC default is signal) |

CLI validation rejects out-of-range values, unknown options, missing values,
mutually exclusive sources, empty model/input paths (exit 2 with usage);
runtime failures exit 1 with a diagnostic (`error: ...`).

## Output

* stdout: processed frame count, FSM state counts, event count, first failing
  event, cadence, per-stage mean timing.
* Optional JSON (`--output-json`): schema `"poc-inference"`, version 1,
  runtime `"cpp-onnxruntime"`, model/source/temporal fingerprints, summary
  (`frames`, `interrupted`, state counts, first failing event, cadence),
  per-stage timing statistics, and a `per_frame` array with `frame` (sampled
  index), `frame_idx` (raw video frame index), `raw`, `signal`, `baseline`,
  `delta`, `state`, `raw_decision`, `event` — matching the POC semantics.

## Failure handling

Missing model, unreadable/invalid video, unavailable/privacy-blocked camera,
model/shape mismatch: exit 1, clear message, no partial report. Malformed CLI:
exit 2, message + usage. End of stream: final report then exit 0. SIGINT /
SIGTERM: stop sampling gracefully, write the final report with
`"interrupted": true`, exit 0.

## Known limitations and caveats

* **UE003 false alarm reproduces** on the C++ runtime exactly as in the Python
  POC. It is a target-normal data/decision problem, not a runtime bug, and is
  intentionally not "fixed" here.
* **Decoder/platform tolerance.** Per-pixel decode output differs between FFmpeg
  builds (up to 8/255 on STR001). Parity tolerance is set to that empirical
  floor (0.15 raw-P). This is a measurement condition of the decode stack, not
  a model or runtime correctness threshold.
* **Benchmark numbers are macOS-workstation baselines**, not edge-device
  targets. Cross-compile and re-measure on the target before deployment.
* **Live-camera validation is deferred.** The processing loop is identical for
  camera and file sources (file path fully validated); a physical camera open
  requires a macOS-privacy-permitted environment and was not exercised.
* **No quantization yet.** fp32 ONNX Runtime CPU EP; peak RSS on macOS 150-335 MB
  includes decoder + ORT arena.

## Build / test / run

```
cmake -S runtime -B runtime/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build runtime/build
ctest --test-dir runtime/build --output-on-failure   # unit + integration
python3 scripts/edge_parity.py                       # behavioral parity gate
./runtime/build/tinyvision --input video.mp4 --output-json report.json
./runtime/build/tinyvision --camera 0 --max-frames 200
```