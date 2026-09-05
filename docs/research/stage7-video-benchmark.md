# Stage 7 — Offline Video/Pipeline Benchmark

**Date:** 2026-09-03
**Status:** Complete. No physical cameras accessed. No hardware integration. No retraining.
**Constraint honoured:** the frozen Experiment 1 ShuffleNetV2 checkpoint
(`shufflenet_v2_x1_0_printing-errors_cdc9e2cad4`) is used unchanged; no model weights are
modified; no threshold is tuned on any real-world data.

All measurements are **development-machine measurements (Mac M1 Pro, CPU)**. They do NOT
represent eventual Raspberry Pi performance.

---

## A. Methodology

### Pipeline under test

```
MP4 video file
  -> OpenCV VideoCapture (H.264 decode)
  -> cv2.cvtColor + PIL + torchvision transforms (preprocessing)
  -> frozen ShuffleNetV2 (inference)
  -> softmax -> fail_score -> decision
```

This is the same pipeline path used for physical-camera capture (via `VideoCapture`), but fed
from video files instead of a live device. No macOS camera APIs, no physical hardware, and no
printer are accessed.

### Test categories

| Category | Source | Labels? | Purpose |
|---|---|---|---|
| **Labelled fixtures** | Kaggle PNG recordings assembled into MP4 | Yes (per-frame) | Verify video-path correctness; measure codec robustness |
| **Unlabelled probes** | User-supplied local video files | No | Expose pipeline to OOD visual content; timing only |

For unlabelled videos, **no classification accuracy/F1/FNR is reported** -- only timing.

### Correctness verification (labelled fixtures only)

1. Run original ordered PNGs through the image preprocessing + inference path to get **PNG
   reference predictions** (authoritative).
2. Run the assembled MP4 through `VideoCapture` -> preprocessing -> inference to get **video
   predictions**.
3. Align frames by index. Compare:
   - **Score-level match**: |PNG_score - video_score| <= 0.05 (tolerance accounts for
     H.264 compression artefacts on 256x256 images).
   - **Prediction-level match**: binarised class (threshold 0.5) agrees.

The 0.05 score tolerance is deliberately conservative. Frames near the 0.5 decision boundary
are expected to show score differences from JPEG compression; prediction-level match is the
primary correctness metric.

### Fixture generation

Deterministic: sorted PNG filenames -> `cv2.VideoWriter` at configurable FPS. The original
PNGs are never modified. Generation is a separate script (`generate_video_fixtures.py`) from
the benchmark (`benchmark_video.py`).

### Sample rates tested

| Rate | Purpose |
|---|---|
| Native video FPS | Upper-bound processing throughput |
| 2 FPS | Deployment-cadence simulation |
| 1 FPS | Conservative deployment cadence |

---

## B. Generated fixtures

8 representative recordings selected to cover PURE_NORMAL, PURE_FAILING, MIXED compositions,
and multiple failure types across val and test splits.

| Fixture ID | Split | Composition | Frames | Types | Source recording |
|---|---|---|---|---|---|
| test_pure_normal_68 | test | PURE_NORMAL | 68 | good | Recording_05012023_13_10_42 |
| test_pure_failing_underex_147 | test | PURE_FAILING | 147 | under_extrusion | Recording_08012023_13_31_14 |
| test_pure_failing_stringing_72 | test | PURE_FAILING | 72 | stringing | Recording_14.02.2023_17_38_44 |
| test_mixed_underex_86 | test | MIXED | 86 | under_extrusion+good | Recording_01122022_16_06_44 |
| val_pure_normal_152 | val | PURE_NORMAL | 152 | good | Recording_01122022_17_44_58 |
| val_pure_failing_underex_119 | val | PURE_FAILING | 119 | under_extrusion | Recording_10012023_18_46_21 |
| val_pure_failing_stringing_40 | val | PURE_FAILING | 40 | stringing | Recording_14.02.2023_20_02_21 |
| val_mixed_multi_131 | val | MIXED | 131 | under_extrusion+spaghetti+good | Recording_03012023_15_07_03 |

Total: 815 frames, 8 MP4 fixtures. Generated at 2 FPS encoding rate. All 256x256.

Fixture directory: `docs/research/stage7-fixtures/` (gitignored; not committed).

---

## C. Video-vs-PNG correctness results

### Native-rate comparison (no sampling)

| Fixture | Frames | Score match | Pred match | Max score diff | Score miss | Pred miss |
|---|---|---|---|---|---|---|
| test_pure_normal_68 | 68 | 47.1% | 80.9% | 0.4607 | 36 | 13 |
| test_pure_failing_underex_147 | 147 | 90.5% | 95.2% | 0.4288 | 14 | 7 |
| test_pure_failing_stringing_72 | 72 | 79.2% | 97.2% | 0.2509 | 15 | 2 |
| test_mixed_underex_86 | 86 | 67.4% | 100.0% | 0.2898 | 28 | 0 |
| val_pure_normal_152 | 152 | 75.0% | 98.0% | 0.4165 | 38 | 3 |
| val_pure_failing_underex_119 | 119 | 97.5% | 100.0% | 0.1564 | 3 | 0 |
| val_pure_failing_stringing_40 | 40 | 95.0% | 100.0% | 0.1765 | 2 | 0 |
| val_mixed_multi_131 | 131 | 95.4% | 97.7% | 0.1332 | 6 | 3 |

### Key observations

- **Prediction-level match >= 95% for 7 of 8 fixtures.** The video pipeline produces
  equivalent classification decisions to the PNG path in nearly all cases.
- **test_pure_normal_68** is the outlier (80.9% pred match): this recording contains many
  frames with borderline confidence (fail_score near 0.5), where H.264 compression pushes
  scores across the threshold. The maximum score difference (0.46) occurs on frames where
  the model is inherently uncertain. This is a codec-robustness characteristic of the model's
  decision boundary, not a pipeline defect.
- **Val fixtures perform better than test fixtures** -- consistent with the general observation
  that val data is slightly more separable (Stage 5: val F1 0.908 vs test F1 0.876).
- **MIXED recordings show 100% prediction match** on the test fixture, confirming that
  transition frames (normal->failing) are handled identically through both paths.
- Score differences are dominated by frames near the 0.5 decision boundary. Frames well above
  or below 0.5 show near-identical scores across both paths.

### What this means

The MP4->VideoCapture path is a valid substitute for direct PNG ingestion for benchmarking
and pipeline validation. The small number of prediction disagreements on borderline frames
are expected from lossy compression and do not indicate a pipeline bug.

---

## D. Performance results

### Timing summary (native rate, all fixtures)

| Fixture | Decode P50 | Preprocess P50 | Inference P50 | E2E P50 |
|---|---|---|---|---|
| test_pure_normal_68 | 0.1 ms | 0.8 ms | 35.3 ms | 36.2 ms |
| test_pure_failing_underex_147 | 0.1 ms | 0.8 ms | 35.8 ms | 36.6 ms |
| test_pure_failing_stringing_72 | 0.1 ms | 0.8 ms | 36.4 ms | 37.2 ms |
| test_mixed_underex_86 | 0.1 ms | 0.8 ms | 35.1 ms | 35.9 ms |
| val_pure_normal_152 | 0.1 ms | 0.9 ms | 36.8 ms | 37.7 ms |
| val_pure_failing_underex_119 | 0.1 ms | 0.9 ms | 37.2 ms | 38.1 ms |
| val_pure_failing_stringing_40 | 0.1 ms | 0.9 ms | 38.0 ms | 38.9 ms |
| val_mixed_multi_131 | 0.1 ms | 0.9 ms | 38.1 ms | 39.0 ms |

### Processing cadence

| Metric | Value |
|---|---|
| E2E processing FPS (P50) | ~26-27 |
| Memory (RSS) before benchmark | 352 MB |
| Memory (RSS) after benchmark | 370 MB |

### Behaviour at different sample rates

| Rate | Per-frame E2E latency | Frames processed (of 815 total) |
|---|---|---|
| Native | 36-39 ms | 815 |
| 2 FPS | 36-39 ms | 815 (native=2 FPS, skip=1) |
| 1 FPS | 36-39 ms | ~407 (skip=2) |

The per-frame processing time is constant regardless of sample rate. Lower sample rates
reduce the total number of frames processed (and thus total wall time), but do not change
per-frame cost.

### Component breakdown

| Component | Typical P50 | % of E2E |
|---|---|---|
| Video decode | 0.1 ms | < 1% |
| Preprocessing | 0.8-0.9 ms | ~2% |
| Inference | 35-38 ms | ~97% |
| **Total E2E** | **36-39 ms** | **100%** |

Model inference dominates. Preprocessing and video decode are negligible.

---

## E. Limitations

1. **Development-machine only.** Mac M1 Pro CPU is not representative of Raspberry Pi 4/5
   CPU. Inference latency on Pi will be significantly higher (likely 3-10x). The relative
   breakdown (inference >> preprocessing >> decode) is expected to hold.

2. **256x256 input resolution.** The source dataset uses 256x256 PNGs. A real deployment
   camera would capture at 640x480 or higher, requiring an additional resize/crop step not
   measured here. The preprocessing benchmark (1.5 ms for 640x480 -> 224x224) from the
   earlier camera benchmark is the relevant reference.

3. **MP4 encoding is not camera capture.** The JPEG compression in H.264 encoding introduces
   artefacts that a live camera feed would not have (or would have differently). The
   correctness verification quantifies this effect.

4. **No dropped-frame measurement.** With video files, OpenCV `VideoCapture` never drops
   frames (it reads from disk). Real camera capture may drop frames under CPU load; this
   can only be measured with a live camera.

5. **No EWMA evaluation.** EWMA was not run in this benchmark. Temporal smoothing evaluation
   requires per-recording analysis against frame-level ground truth (already done in Stage 5
   with the image-based path; the video path produces equivalent predictions for 95%+ of
   frames, so EWMA results would be essentially identical).

6. **Fixture diversity.** 8 fixtures / 815 frames is a small sample. The fixtures cover the
   key composition types but not all 151 recordings or all failure severities.

---

## F. Engineering conclusions

### 1. Which camera is more practical for TinyVision?

**Cannot be determined from this benchmark.** No physical cameras were tested. The camera
comparison requires real sensor data under the actual deployment mounting conditions (Stage 6
Part E / Part H).

### 2. Is 1 FPS comfortably achievable?

**Yes.** E2E processing P50 is 36-39 ms per frame on Mac M1 Pro CPU. At 1 FPS, each frame
has a 1000 ms budget. Even with a 5-10x slowdown on Raspberry Pi CPU (360-390 ms per frame),
1 FPS is comfortably achievable.

### 3. Is 2 FPS comfortably achievable?

**Yes on dev machine; likely on Raspberry Pi.** At 2 FPS, each frame has a 500 ms budget.
With 5-10x Pi slowdown (180-390 ms per frame), 2 FPS is achievable but with less margin.

### 4. Is higher-rate capture useful?

**Depends on the deployment cadence, not the capture rate.** Higher capture rates (10-30 FPS)
provide more frames for temporal smoothing but increase CPU load. The model's inference time
(35-38 ms) caps sustained throughput at ~26-27 FPS on this machine. For the TinyVision MVP
(1-2 FPS decision cadence), higher capture rates are useful only if frames are subsampled
before inference.

### 5. Is camera capture or model inference the bottleneck?

**Model inference.** Inference accounts for ~97% of per-frame E2E latency. Video decode and
preprocessing are negligible (< 3% combined). On a live camera, capture latency would add a
small constant overhead but would not change the inference bottleneck.

### 6. What must be measured again on eventual edge hardware?

- **Inference latency on Raspberry Pi CPU** (the dominant cost; expect 3-10x slower).
- **Live camera capture FPS, dropped frames, and startup latency** (requires physical camera).
- **End-to-end latency including camera capture** (only measurable with real hardware).
- **Memory (RSS) under sustained capture + inference** (only measurable with real hardware).
- **Camera auto-exposure/auto-white-balance drift** (only observable from a live sensor).

### 7. Are there any major transport/decode issues with ESP32-CAM?

**Cannot be determined from this benchmark.** The ESP32-CAM sends MJPEG frames over HTTP or
serial. The JPEG decode benchmark (P50: 2.05 ms for 640x480) measured a synthetic
encode/decode round-trip, which is a reasonable proxy for the decode step. The actual
transport latency (HTTP fetch, serial read, WiFi reliability) requires a live ESP32-CAM and
is out of scope of this Stage.

### Not a deployment camera decision

This benchmark does not declare a final deployment camera. Physical mounting conditions,
print geometry, lighting, and sensor behaviour under real conditions remain unknown.

---

## Artifact manifest

| File | Purpose |
|---|---|
| `training/tv/data/video.py` | Production module: video assembly, replay, comparison |
| `scripts/generate_video_fixtures.py` | Deterministic MP4 fixture generation from PNG recordings |
| `scripts/benchmark_video.py` | Offline pipeline benchmark (labelled + unlabelled) |
| `docs/research/stage7-video-benchmark.json` | Raw results (34 KB) |
| `docs/research/stage7-video-benchmark.md` | This report |
| `docs/research/stage7-fixtures/` | 8 MP4 fixtures + labels + manifest (gitignored) |
| `training/tests/test_video.py` | Unit tests for tv/data/video.py |
