# Raspberry Pi edge-benchmark procedure

**Status:** procedure only — **no Raspberry Pi results have been produced yet.**
Every number in `edge-runtime-benchmark.md` is from an Apple M1 Pro workstation.
Nothing here may be reported as an edge result until it has actually been run on
the target and recorded.

This document exists so that the edge measurement is reproducible and so that a
claim of "runs on ordinary hardware" can later be backed by a real artifact
rather than an assumption.

## Why this is required

TinyVision's first principle is local inference on ordinary hardware. The C++
runtime's parity and the Apple-M1 timing establish that the *design* works; they
do **not** establish that it meets a latency/FPS/RAM budget on a Pi. Per the
project's edge requirements, a model that performs well on a workstation is not
automatically successful.

## Target hardware (record exactly what you used)

| Field | Example value | Record actual |
| --- | --- | --- |
| Board | Raspberry Pi 5 (8 GB) | |
| CPU | BCM2712, 4× Cortex-A76 @ 2.4 GHz | |
| RAM | 8 GB | |
| Storage | microSD (A2) or USB SSD | |
| Cooling | active cooler / passive | |
| OS | Raspberry Pi OS (64-bit) Bookworm | |
| Kernel | `uname -r` | |

## Software prerequisites

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build libopencv-dev
# ONNX Runtime for linux-aarch64 (match the version used locally, e.g. 1.20.1):
ORT=onnxruntime-linux-aarch64-1.20.1
curl -fsSL -o ort.tgz \
  "https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/${ORT}.tgz"
tar -xzf ort.tgz && export ORT_DIR="$PWD/${ORT}"
```

## Build

```bash
cmake -S runtime -B runtime/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -Donnxruntime_DIR="$ORT_DIR/cmake" \
  -DOpenCV_DIR="$(dirname "$(find /usr -name OpenCVConfig.cmake | head -1)")"
cmake --build runtime/build
ctest --test-dir runtime/build --output-on-failure   # requires a frozen model.onnx
```

Confirm the model artifact is present locally (it is git-ignored):
`experiments/<id>/artifacts/model.onnx`. Both fp32 and INT8 exports should be
measured.

## Measure

Run the existing benchmark driver, which records per-stage timing, process CPU%
and peak RSS via `psutil`:

```bash
python3 -m pip install psutil
python3 scripts/benchmark_edge_runtime.py \
  --model experiments/rc1_release_candidate_shufflenet_v2_x1_0/artifacts/model.onnx \
  --outdir docs/research/edge-bench-rpi
```

It expects the four validation MP4s under
`datasets/external/video-frames/raw/` (not committed). If they are unavailable,
record the procedure as **not yet run** rather than substituting other footage.

Also measure a **live-camera** path (not just offline MP4), since that is the
deployment shape: point the C++ binary at a `v4l2`/USB camera device and record
end-to-end cadence and drop rate.

## Metrics to record (per model: fp32 and INT8)

| Metric | Unit | fp32 | INT8 |
| --- | --- | --- | --- |
| Model file size | MB | | |
| Startup (load + first frame) | ms | | |
| Inference latency p50 / p95 | ms | | |
| Preprocess latency p50 | ms | | |
| Decode latency p50 (1 fps config) | ms | | |
| End-to-end cadence | processed fps | | |
| Peak RSS | MB | | |
| Process CPU% (aggregate) | % | | |
| Live-camera sustained fps | fps | | |
| Thermal throttling observed | yes/no | | |

## Acceptance framing

Do not invent thresholds. State the deployment target first (e.g. "1 decision per
second, whole print visible"), then check whether the measured cadence and RSS
meet it, and record accuracy on the same device ("accuracy degradation after
export" — compare INT8 vs fp32 outputs on the same frames, which the parity tool
already quantifies:

```bash
training/.venv/bin/python -m tv.cli reliability parity \
  --suite training/configs/reliability/suite.yaml \
  --framework experiments/rc1_release_candidate_shufflenet_v2_x1_0/artifacts/best.pt \
  --other experiments/rc1_release_candidate_shufflenet_v2_x1_0/artifacts/model_int8.onnx
```

## Recording template

Append a `docs/research/edge-bench-rpi/results.json` and a short markdown report
following the experiment policy: hardware, OS, software versions, model hashes,
command line, raw per-stage numbers, and any thermal notes. Until that artifact
exists, all edge claims remain **not yet verified**.
