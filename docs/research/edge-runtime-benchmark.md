# Edge Runtime Benchmark (C++ / ONNX Runtime)

Baseline performance of the deployed `tinyvision` binary on the four frozen
validation videos. **This is the reference, unoptimized measurement.** No
throughput or correctness tuning was performed; the validated POC defaults are
used unchanged.

Structured results: `edge-runtime-benchmark.json`
Reproduction: `python3 scripts/benchmark_edge_runtime.py` (run from repo root)

## Environment

| Item | Value |
| --- | --- |
| Host | Apple M1 Pro (macOS 26.6.2, arm64), 10 cores, 32 GB RAM |
| Runtime | C++17, OpenCV (Homebrew 5.0.0), ONNX Runtime CPU EP 1.29.0, Release build (Ninja) |
| Model | `shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/model.onnx` (fp32, 224x224 in / 2 logits out) |
| Input | 4 offline validation MP4s (NORMAL001, SP001, UE003, STR001) |
| Measure | per-stage timing (decode/preprocess/inference, JSON report), process CPU%, peak RSS, end-to-end cadence |

Two sampling configurations per video:

* `sample1` — the POC configuration `--sample-fps 1` (`skip=30` on a 31 fps
  video), which is the deployment baseline.
* `native` — `--sample-fps 0` (every frame), the POC `_next_frame` semantics
  remain (one frame dropped per read window), the max-throughput reference.

## Results

| Video | Config | Frames | Total ms/frame | Decode ms | Preprocess ms | Inference ms | Cadence fps | CPU% | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NORMAL001 | sample1 1fps | 19 | 38.09 | 32.67 | 3.14 | 2.28 | 25.8 | 369% | 200 MB |
| NORMAL001 | native | 301 | 7.51 | 2.47 | 2.91 | 2.13 | 133.1 | 572% | 200 MB |
| SP001 | sample1 1fps | 24 | 20.76 | 15.77 | 2.76 | 2.23 | 47.6 | 442% | 153 MB |
| SP001 | native | 382 | 5.58 | 0.74 | 2.71 | 2.14 | 179.0 | 520% | 153 MB |
| UE003 | sample1 1fps | 352 | 23.96 | 19.41 | 2.17 | 2.37 | 41.7 | 675% | 205 MB |
| UE003 | native | 4407 | 5.50 | 1.13 | 2.17 | 2.20 | 181.6 | 630% | 205 MB |
| STR001 | sample1 1fps | 618 | 53.23 | 46.59 | 4.21 | 2.43 | 18.8 | 739% | 318 MB |
| STR001 | native | 7733 | 10.72 | 4.35 | 4.14 | 2.23 | 93.3 | 683% | 335 MB |

Startup (model load + source open + first processed frame + shutdown), SP001:
~297 ms.

## Interpretation

* **Inference is not the bottleneck.** CPU inference is ~2.1–2.4 ms per frame
  (p50~2.2 ms, p95≤3.7 ms). The preprocess (Pillow-bilinear port) is ~2.1–4.2 ms.
* **Decode dominates at 1 fps sampling.** At `sample1`, the runtime decodes the
  30 discarded frames per window, so decode (15–47 ms) outweighs inference. The
  runtime processes a 31 fps stream at 19–52 processed frames/s end to end,
  comfortably above the 1 fps decision cadence.
* **At native cadence** the runtime sustains 93–179 frames/s offline. This is
  the ceiling; live-camera throughput will differ.
* **CPU% is aggregate across cores** (FFmpeg decode + ONNX Runtime threadpool),
  not a single-core figure. Peak RSS 150–335 MB reflects the ONNX Runtime arena
  plus decoder buffering; it is a baseline, not a target.
* **No claim of edge-device readiness from this host.** Latency/RSS will differ
  on the target (e.g. Raspberry Pi 4/5); this report establishes the C++
  baseline before any optimization.

## Where the time goes (1 fps POC config)

| Stage | Typical | Dominant cost |
| --- | --- | --- |
| Decode (incl. discarded frames) | 15–47 ms | FFmpeg (OpenCV `VideoCapture`) |
| Preprocess | 2.1–4.2 ms | Pillow two-pass bilinear + normalize |
| Inference | 2.1–2.4 ms | ShuffleNetV2, ONNX Runtime CPU |
| Total | 21–53 ms | decode-bound |

## Limitations

* macOS workstation numbers; not yet measured on the edge target.
* Peak RSS includes decoder and ORT arena growth and is single-process only.
* Not a correctness benchmark — behavioral parity is separately tracked in
  `cpp-runtime-parity.md`.
* Per prior conclusion, decoder output differs slightly between FFmpeg builds
  (up to 8/255 per-pixel); this is a measurement condition, not a runtime defect.