# Model Comparison for 3D-Print Failure Detection

**Status**: Research only.
**Date**: 2026-09-02

This document compares candidate model architectures for the TinyVision MVP, covering
classification, detection, and anomaly detection.

---

## 1. Classification models (primary path)

### Architecture comparison

| Model | Params | FLOPs | Top-1 (ImageNet) | RPi5 latency | Size (INT8) | ONNX | INT8 | License |
|---|---|---|---|---|---|---|---|---|
| **EfficientNet-Lite0** | 4.7M | 407M | 75.1% | ~49ms | 5.1MB | Yes | Yes | **Apache 2.0** |
| **GhostNetV2** | 5.2M | 142M | 76.2% | ~25ms | 5.2MB | Yes | Yes | **Apache 2.0** |
| **MobileNetV3-Small** | 2.5M | 56M | 67.9% | ~61ms | 2.9MB | Yes | Yes | **Apache 2.0** |
| **MobileNetV2** | 3.5M | 300M | 72.0% | ~92ms | 3.6MB | Yes | Yes | **Apache 2.0** |
| **ShuffleNetV2** | 1.37M | 146M | 69.4% | ~30ms | 1.47MB | Yes | Yes | **MIT** |
| YOLOv11n-cls | 1.6M | ~3.2B | 70.0% | ~10ms | 1.6MB | Yes | Yes | AGPL-3.0 |
| YOLOv8n-cls | 2.7M | 4.3B | 69.0% | ~13ms | 2.7MB | Yes | Yes | AGPL-3.0 |

### Licensing note

Ultralytics YOLO models are AGPL-3.0. This means:
- For open-source TinyVision distribution: AGPL weights are fine (copyleft applies).
- For commercial use: requires an Ultralytics commercial license.
- For training only (weights not distributed): AGPL does not apply.

TinyVision's open-source project can use YOLO for training and distribute AGPL weights.
If a permissive license is preferred for the weights, use EfficientNet-Lite0, GhostNetV2,
or ShuffleNetV2.

### Recommendation

**Primary**: EfficientNet-Lite0 (best accuracy/latency ratio among permissive-licensed
models, 49ms on RPi4, Apache 2.0).

**Alternative**: YOLOv11n-cls (fastest at ~10ms, best ecosystem support, but AGPL).

**Tertiary**: GhostNetV2 (highest accuracy among Apache 2.0 models at 76.2%, 25ms).

### Existing project results with these architectures

| Project | Architecture | Result | Hardware |
|---|---|---|---|
| PrintGuard | ShuffleNetV2 + Prototypical Networks | F1=0.937 | Pi 4B |
| YOLO-3DPrint-Monitor | YOLOv8n-cls | 99.7% accuracy | Desktop CPU |
| klipper-pfd | YOLOv8n (detection) | mAP=0.502 | Pi (TFLite) |

PrintGuard's result is the most relevant: ShuffleNetV2 achieves F1=0.937 on Pi 4B
with 43x throughput advantage over Obico. EfficientNet-Lite0 has higher ImageNet
accuracy than ShuffleNetV2 (75.1% vs 69.4%) and should achieve comparable or better
fine-tuning performance.

---

## 2. Detection models (secondary path)

### Architecture comparison

| Model | Params | mAP50-95 (COCO) | RPi5 latency | Size (INT8) | ONNX | License |
|---|---|---|---|---|---|---|
| **YOLOv11n** | 2.6M | 39.5 | 56ms (640), 45ms (320) | 2.6MB | Yes | AGPL-3.0 |
| **YOLOv8n** | 3.2M | 37.3 | 80ms (640) | 3.2MB | Yes | AGPL-3.0 |

### Resolution impact

| Input size | YOLOv11n latency | mAP impact |
|---|---|---|
| 640×640 | 56ms | Baseline |
| 320×320 | 45ms | ~2% mAP loss |

Reducing input from 640→320 gives ~3× speedup with <2% mAP loss. For 3D-print
monitoring, 320×320 may be sufficient since failures are visually prominent.

### Existing project results

| Project | Model | mAP50 | Classes | Notes |
|---|---|---|---|---|
| klipper-pfd | YOLOv8n | 0.502 | 4 | TFLite on Pi |
| Obico | Darknet YOLO | Not published | 2 | Server-based |

### Recommendation

**YOLOv11n** if detection is pursued. Better mAP/speed tradeoff than YOLOv8n. Only
option for detection with nano-class models.

---

## 3. Anomaly detection models (complementary path)

### Anomalib models (Apache 2.0)

| Model | AUROC (MVTec) | GPU latency | RPi4B CPU latency | Model size |
|---|---|---|---|---|
| **EfficientAD-S** | 99.8% | <1ms | **Untested** (~20–40ms est.) | ~5MB |
| EfficientAD-M | 96.0% | ~2ms | ~40–60ms est. | ~10MB |
| PatchCore | 99.1% | ~50ms | ~52s | Large (embeddings) |
| PaDiM | 97.9% | ~30ms | ~45s | Medium (Gaussians) |
| SimpleNet | 99.6% | ~13ms | ~5–10s | Large |

### Key finding

PatchCore, PaDiM, and SimpleNet are **far too slow for Pi CPU** (>30s per inference).
Only EfficientAD has a chance of achieving acceptable speed on Pi. Its architecture
(4-layer Patch Description Network) is much lighter than the backbone-heavy alternatives.

**EfficientAD on Pi CPU is the single most important missing data point.** It must be
benchmarked empirically during Experiment 3.

### Training data requirements

| Model | Min normal images | Recommended | Training time |
|---|---|---|---|
| PatchCore | 10–30 | 100–300 | Seconds |
| PaDiM | 10–30 | 100–300 | Seconds |
| EfficientAD | 50–100 | 200–500 | Minutes |

### Recommendation

**EfficientAD-S** as the anomaly detection candidate. Apache 2.0 license. Must benchmark
on Pi 5 CPU before committing. If <1 FPS, run at reduced frequency (every 5th frame)
alongside the supervised classifier.

---

## 4. Models already trained for 3D-print failure detection

### Obico model

- **Architecture**: Darknet YOLO (exact variant undisclosed)
- **Weights**: Bundled inside Docker images, AGPL-3.0
- **Training data**: Proprietary (not published)
- **Standalone usability**: No — weights are not published as standalone downloads.
  The Print Guardian (nlaha) project uses them via Darknet C bindings.
- **Edge support**: Not supported on Raspberry Pi (too slow per original docs)

### PrintGuard model

- **Architecture**: ShuffleNetV2 encoder + Prototypical Networks (few-shot)
- **Weights**: Available in the PrintGuard repo (GPL-2.0)
- **Training data**: 7 public datasets (Roboflow + Kaggle)
- **Standalone usability**: Yes, but GPL-2.0 license (copyleft)
- **Edge support**: Pi 4/5 (Docker), macOS, Windows, browser (WASM)

### klipper-pfd model

- **Architecture**: YOLOv8n → TFLite
- **Weights**: model.tflite committed in repo (no license specified)
- **Training data**: Two Roboflow datasets
- **Edge support**: Pi (Klipper host)

### HuggingFace 3D-Print-Failure-Detector

- **Architecture**: YOLOv11s (small, not nano)
- **Weights**: MIT license
- **Training data**: 9,000+ images
- **Classes**: 3 (spaghetti, stringing, zits)
- **Edge support**: Would need export to nano variant for Pi

### Can any existing weights be used as a baseline?

**Yes:**
- **PrintGuard** (GPL-2.0): Can be used for transfer learning or as a baseline. The
  GPL-2.0 license means derivative works must also be GPL-2.0. Compatible with
  TinyVision's open-source distribution.
- **HuggingFace YOLOv11s** (MIT): Can be used for transfer learning. MIT license is
  compatible with any TinyVision license. The model is larger than nano — would need
  fine-tuning or architecture distillation.
- **klipper-pfd** (no license): Cannot assume commercial compatibility. Use only if
  the author clarifies licensing.

**Not usable:**
- **Obico** (AGPL-3.0, not standalone): Weights are embedded in the AGPL server. While
  AGPL permits use, the weights are not published as standalone downloadable artifacts
  in standard formats. Would need to extract from Docker.

---

## 5. Practical considerations for edge deployment

### Quantization impact

INT8 quantization typically reduces model size 3–4× and improves CPU inference 2–3×.
The accuracy degradation is usually <1% for classification and <2% mAP for detection.

### Thermal throttling

RPi4/5 throttles after ~60s without active cooling. Add heatsink+fan for sustained
inference. At 49ms/frame (EfficientNet-Lite0), the Pi can sustain 20 FPS without
throttling with cooling.

### Camera latency

CSI cameras add ~8ms latency. USB cameras add ~25ms. For 3D-print monitoring
(1–5 FPS target), this is negligible.

### Memory budget

| Component | RAM estimate |
|---|---|
| OS + services | ~200MB |
| Model loading | ~50–100MB |
| Frame buffer | ~5MB |
| Inference workspace | ~50–100MB |
| **Total** | ~300–400MB |

All candidate models fit within a 512MB RAM budget (RPi4/5).

---

## 6. Final recommendations

### For Experiment 1 (binary classification MVP)

| Choice | Architecture | Why |
|---|---|---|
| **Primary** | EfficientNet-Lite0 | Best accuracy/latency among Apache 2.0 models |
| **Alternative** | YOLOv11n-cls | Fastest, best ecosystem, but AGPL |
| **Baseline** | PrintGuard (ShuffleNetV2) | Published F1=0.937 on Pi, GPL-2.0 |

### For Experiment 2 (detection)

| Choice | Architecture | Why |
|---|---|---|
| **Primary** | YOLOv11n | Best mAP/speed, AGPL |
| **Input size** | 320×320 | ~3× faster than 640, <2% mAP loss |

### For Experiment 3 (anomaly detection)

| Choice | Architecture | Why |
|---|---|---|
| **Primary** | EfficientAD-S | Only fast-enough candidate, Apache 2.0 |
| **Fallback** | Autoencoder (TFLite) | 3MB, 45ms on Pi, lower accuracy but functional |
