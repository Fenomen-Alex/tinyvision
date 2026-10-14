# Benchmarks

Benchmarks record **methodology and measured numbers**, never aspirations.
A model that scores well on a workstation but cannot run on the target edge
device is not automatically successful.

## Categories

| Category | Where | Command |
|---|---|---|
| Reliability / release gate | [`docs/reliability-evaluation.md`](docs/reliability-evaluation.md) | `tv reliability evaluate --suite ... --policy ... --model ...` |
| Deployment parity (FW→ONNX→INT8→C++) | [`docs/research/deployment-parity.md`](docs/research/deployment-parity.md) | `tv reliability parity ...` |
| Edge runtime (C++) | [`docs/research/edge-runtime-benchmark.md`](docs/research/edge-runtime-benchmark.md), `runtime/EDGE.md` | `python scripts/benchmark_edge_runtime.py` |
| Temporal decision layer (synthetic) | [`docs/research/temporal-synthetic-benchmark.json`](docs/research/temporal-synthetic-benchmark.json) | `python scripts/temporal_synthetic_benchmark.py` |
| Temporal validation (real footage) | [`docs/research/real-footage-analysis.json`](docs/research/real-footage-analysis.json) / [`.md`](docs/research/real-footage-analysis.md) | `python scripts/temporal_real_footage_analysis.py` |
| Real failure-frame diagnostic (Stage 14) | [`docs/research/stage14-real-frames-eval.json`](docs/research/stage14-real-frames-eval.json) | `python scripts/eval_real_failure_frames.py --output ...` |
| Robustness / perturbations | [`docs/research/stage2-robustness.md`](docs/research/stage2-robustness.md) | `tv robustness ...` |

## Metrics that must be tracked

precision, recall, F1, FNR, **FPR**, AUROC (classification); detection latency
and false alarms per hour (temporal); model size, RAM, CPU latency, FPS,
image resolution, quantization impact, accuracy after export (edge).

For failure detection, **false negatives and false alarms both matter**: FNR is
the missed-failure cost; FPR is the operator-trust cost (RC1's FPR 1.000 is the
documented blocker).

## Rules

* Report the host and toolchain (CPU/GPU, RAM, OS, OpenCV/ONNX Runtime versions,
  build type). See the environment table in the edge benchmark.
* Separate **development-hardware baselines** from **target-hardware** results.
  No Raspberry Pi result exists yet; do not claim edge readiness.
* Benchmarks are reproducible: keep the script, the exact inputs, and the
  sampling configuration.
* Do not compare numbers produced under different methodology.

## Current state

* Edge runtime: C++ baseline on Apple M1 Pro — inference ~2.1–2.4 ms/frame;
  93–179 fps offline at native cadence; peak RSS 150–335 MB. Decode dominates at
  1 fps sampling. (Not target hardware.)
* Export fidelity: fp32 ONNX is exact (0 disagreement, ≤4e-6 score drift); INT8
  raises dev_val FPR by ~4 points.
* Real-footage temporal validation: over the 7 committed run videos
  (VLM-labelled), all five normal videos fire (video-level FAR = 1.0) and RC1's
  median P(failing) is 0.74–0.99 on normal footage; on the 11 Stage-14 idle
  frames every score is above 0.5 (median 0.92). P(failing) is uncalibrated on
  real frames — the RC1 blocker.
* Target hardware (Raspberry Pi 4/5 + USB/CSI camera): **not benchmarked.**
