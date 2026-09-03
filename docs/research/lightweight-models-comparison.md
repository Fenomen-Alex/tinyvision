# Lightweight Models for Edge Deployment

## Model Comparison for Raspberry Pi / ARM SBC Deployment

### 1. YOLO Classification Models

#### YOLOv8n-cls
| Attribute | Value |
|-----------|-------|
| Architecture | YOLOv8-Nano Classification |
| Parameters | 2.7M |
| FLOPs | 4.3B (at 224×224) |
| Input Resolution | 224×224 (fixed) |
| Top-1 Accuracy (ImageNet) | 69.0% |
| Top-5 Accuracy (ImageNet) | 88.3% |
| CPU ONNX Latency | 12.9ms |
| A100 TensorRT Latency | 0.31ms |
| Model Size (FP32) | ~10.8MB |
| Model Size (INT8) | ~2.7MB |
| ONNX Export | Yes, natively supported |
| INT8 Quantization | Yes (ONNX, TensorRT) |
| License | AGPL-3.0 (Ultralytics) |
| Commercial Use | Requires Ultralytics license for commercial use |

#### YOLOv11n-cls
| Attribute | Value |
|-----------|-------|
| Architecture | YOLO11-Nano Classification |
| Parameters | 1.6M |
| FLOPs | ~3.2B (estimated) |
| Input Resolution | 224×224 (fixed) |
| Top-1 Accuracy (ImageNet) | 70.0% |
| CPU ONNX Latency | ~10ms (estimated) |
| Model Size (FP32) | ~6.4MB |
| Model Size (INT8) | ~1.6MB |
| ONNX Export | Yes, natively supported |
| INT8 Quantization | Yes (ONNX, TensorRT) |
| License | AGPL-3.0 (Ultralytics) |
| Commercial Use | Requires Ultralytics license for commercial use |

---

### 2. YOLO Detection Models

#### YOLOv8n
| Attribute | Value |
|-----------|-------|
| Architecture | YOLOv8-Nano Detection |
| Parameters | 3.2M |
| FLOPs | 8.7B |
| Input Resolution | 640×640 (default) |
| mAP50-95 (COCO) | 37.3% |
| CPU ONNX Latency | 80.4ms |
| A100 TensorRT Latency | 0.99ms |
| Model Size (FP32) | ~12.8MB |
| Model Size (INT8) | ~3.2MB |
| ONNX Export | Yes, natively supported |
| INT8 Quantization | Yes (ONNX, TensorRT) |
| License | AGPL-3.0 (Ultralytics) |
| Commercial Use | Requires Ultralytics license for commercial use |

#### YOLOv11n
| Attribute | Value |
|-----------|-------|
| Architecture | YOLO11-Nano Detection |
| Parameters | 2.6M |
| FLOPs | 6.5B |
| Input Resolution | 640×640 (default) |
| mAP50-95 (COCO) | 39.5% |
| CPU ONNX Latency | 56.1ms |
| T4 TensorRT Latency | 1.5ms |
| Model Size (FP32) | ~10.4MB |
| Model Size (INT8) | ~2.6MB |
| ONNX Export | Yes, natively supported |
| INT8 Quantization | Yes (ONNX, TensorRT) |
| License | AGPL-3.0 (Ultralytics) |
| Commercial Use | Requires Ultralytics license for commercial use |

**RPi5 Real-World Performance (YOLOv11n INT8):**
- 320×320 input: ~22 FPS (44.8ms latency)
- 640×640 input: ~6 FPS (166ms latency)

---

### 3. MobileNet Family

#### MobileNetV2
| Attribute | Value |
|-----------|-------|
| Architecture | MobileNetV2 (Inverted Residuals) |
| Parameters | 3.5M |
| FLOPs | 300M |
| Input Resolution | 224×224 (default) |
| Top-1 Accuracy (ImageNet) | 72.0% |
| Top-5 Accuracy (ImageNet) | 91.0% |
| RPi4 Latency (INT8 TFLite) | 92ms |
| Model Size (FP32) | 13.3MB |
| Model Size (INT8) | 3.6MB |
| ONNX Export | Yes (via tf2onnx or PyTorch) |
| INT8 Quantization | Yes (TFLite, ONNX Runtime) |
| License | Apache 2.0 |
| Commercial Use | Yes |

#### MobileNetV3-Small
| Attribute | Value |
|-----------|-------|
| Architecture | MobileNetV3-Small (NAS optimized) |
| Parameters | 2.5M |
| FLOPs | 56M |
| Input Resolution | 224×224 (default) |
| Top-1 Accuracy (ImageNet) | 67.88% |
| Top-5 Accuracy (ImageNet) | 87.84% |
| RPi4 Latency (INT8 TFLite) | 61ms |
| Model Size (FP32) | ~10MB |
| Model Size (INT8) | 2.9MB |
| ONNX Export | Yes (via tf2onnx or PyTorch) |
| INT8 Quantization | Yes (TFLite, ONNX Runtime) |
| License | Apache 2.0 |
| Commercial Use | Yes |

---

### 4. EfficientNet-Lite

#### EfficientNet-Lite0
| Attribute | Value |
|-----------|-------|
| Architecture | EfficientNet-Lite0 (Compound Scaling) |
| Parameters | 4.7M |
| FLOPs | 407M |
| Input Resolution | 224×224 (default) |
| Top-1 Accuracy (FP32) | 75.1% |
| Top-1 Accuracy (INT8) | 74.4% |
| Pixel4 CPU Latency (FP32) | 12ms |
| Pixel4 CPU Latency (INT8) | 6.5ms |
| RPi4 Latency (INT8 TFLite) | 49ms |
| Model Size (FP32) | ~18.8MB |
| Model Size (INT8) | 5.1MB |
| ONNX Export | Yes (via tf2onnx) |
| INT8 Quantization | Yes (TFLite PTQ) |
| License | Apache 2.0 (Google) |
| Commercial Use | Yes |

#### EfficientNet-Lite1
| Attribute | Value |
|-----------|-------|
| Architecture | EfficientNet-Lite1 |
| Parameters | 5.4M |
| FLOPs | 631M |
| Input Resolution | 240×240 (default) |
| Top-1 Accuracy (FP32) | 76.7% |
| Top-1 Accuracy (INT8) | 75.9% |
| Pixel4 CPU Latency (FP32) | 18ms |
| Pixel4 CPU Latency (INT8) | 9.1ms |
| RPi4 Latency (INT8 TFLite) | 73ms |
| Model Size (FP32) | ~21.6MB |
| Model Size (INT8) | 6.2MB |
| ONNX Export | Yes (via tf2onnx) |
| INT8 Quantization | Yes (TFLite PTQ) |
| License | Apache 2.0 (Google) |
| Commercial Use | Yes |

---

### 5. Other Tiny Classifiers

#### ShuffleNetV2 (1.0×)
| Attribute | Value |
|-----------|-------|
| Architecture | ShuffleNetV2 (Channel Split + Shuffle) |
| Parameters | 1.37M |
| FLOPs | 146M |
| Input Resolution | 224×224 |
| Top-1 Accuracy (ImageNet) | 69.4% (1.0× variant) |
| Mobile Latency | ~32ms |
| Model Size (FP32) | 5.24MB |
| Model Size (INT8) | 1.47MB |
| ONNX Export | Yes (ONNX Model Zoo) |
| INT8 Quantization | Yes |
| License | MIT |
| Commercial Use | Yes |

#### SqueezeNet v1.1
| Attribute | Value |
|-----------|-------|
| Architecture | SqueezeNet (Fire Module) |
| Parameters | 1.25M |
| FLOPs | ~352M |
| Input Resolution | 224×224 |
| Top-1 Accuracy (ImageNet) | 57.5% |
| Top-5 Accuracy (ImageNet) | 80.3% |
| Model Size (FP32) | 5MB |
| Model Size (INT8) | ~1.25MB |
| ONNX Export | Yes (ONNX Model Zoo) |
| INT8 Quantization | Yes |
| License | BSD 3-Clause |
| Commercial Use | Yes |

**Note:** Very old architecture (2016). Low accuracy makes it less suitable for modern applications.

#### GhostNetV2
| Attribute | Value |
|-----------|-------|
| Architecture | GhostNetV2 (Cheap Operations) |
| Parameters | 5.2M |
| FLOPs | 142M |
| Input Resolution | 224×224 |
| Top-1 Accuracy (ImageNet) | 76.2% |
| Mobile Latency | ~25ms |
| Model Size (FP32) | ~20.8MB |
| Model Size (INT8) | ~5.2MB |
| ONNX Export | Yes |
| INT8 Quantization | Yes |
| License | Apache 2.0 |
| Commercial Use | Yes |

---

### 6. Anomalib Models (Anomaly Detection)

#### Anomalib Overview
- **License:** Apache 2.0
- **Commercial Use:** Yes
- **Focus:** Unsupervised anomaly detection (one-class learning)
- **Export:** OpenVINO, ONNX supported

#### EfficientAD-S (Small)
| Attribute | Value |
|-----------|-------|
| Architecture | Student-Teacher with EfficientNet backbone |
| Teacher | Pre-trained EfficientNet |
| Parameters | ~4M (student only) |
| Input Resolution | 256×256 |
| AU-ROC (MVTec AD) | 95.4% |
| AU-PRO (Localization) | 92.5% |
| GPU Latency | 2.2ms |
| CPU Latency | ~20-40ms (estimated on RPi5) |
| Model Size | ~15-20MB (FP32) |
| ONNX Export | Yes |
| INT8 Quantization | Limited (requires careful calibration) |
| License | Apache 2.0 |
| Commercial Use | Yes |

#### EfficientAD-M (Medium)
| Attribute | Value |
|-----------|-------|
| Architecture | Student-Teacher with larger backbone |
| Parameters | ~6M |
| Input Resolution | 256×256 |
| AU-ROC (MVTec AD) | 96.0% |
| AU-PRO (Localization) | 93.3% |
| GPU Latency | 4.5ms |
| CPU Latency | ~40-60ms (estimated on RPi5) |
| Model Size | ~25-30MB (FP32) |
| License | Apache 2.0 |
| Commercial Use | Yes |

#### PatchCore
| Attribute | Value |
|-----------|-------|
| Architecture | CNN backbone + Memory Bank (coreset subsampling) |
| Backbone | WideResNet-50 (default) or EfficientNet |
| Input Resolution | 256×256 |
| AU-ROC (MVTec AD) | 99.1% (with WideResNet-50) |
| Inference Latency | ~100-200ms (memory bank lookup) |
| Model Size | Large (backbone + feature bank ~50-100MB) |
| ONNX Export | Partial (backbone only, bank requires custom runtime) |
| License | Apache 2.0 |
| Commercial Use | Yes |

**Note:** PatchCore achieves highest accuracy but requires significant memory for feature bank, making it less suitable for resource-constrained edge devices.

#### SimpleNet
| Attribute | Value |
|-----------|-------|
| Architecture | Student-Teacher with feature alignment |
| Input Resolution | 256×256 |
| AU-ROC (MVTec AD) | ~95-97% |
| GPU Latency | ~5-10ms |
| CPU Latency | ~50-100ms (estimated) |
| License | Apache 2.0 |
| Commercial Use | Yes |

---

### Summary Comparison Table

| Model | Params | FLOPs | Top-1 Acc | RPi4/5 Latency | Size (INT8) | ONNX | INT8 | License |
|-------|--------|-------|-----------|-----------------|-------------|------|------|---------|
| **Classification** |||||||||
| YOLOv8n-cls | 2.7M | 4.3B | 69.0% | ~12ms (CPU ONNX) | 2.7MB | ✅ | ✅ | AGPL* |
| YOLOv11n-cls | 1.6M | ~3.2B | 70.0% | ~10ms | 1.6MB | ✅ | ✅ | AGPL* |
| MobileNetV2 | 3.5M | 300M | 72.0% | 92ms | 3.6MB | ✅ | ✅ | Apache 2.0 |
| MobileNetV3-Small | 2.5M | 56M | 67.9% | 61ms | 2.9MB | ✅ | ✅ | Apache 2.0 |
| EfficientNet-Lite0 | 4.7M | 407M | 75.1% | 49ms | 5.1MB | ✅ | ✅ | Apache 2.0 |
| ShuffleNetV2 | 1.37M | 146M | 69.4% | ~30ms | 1.47MB | ✅ | ✅ | MIT |
| GhostNetV2 | 5.2M | 142M | 76.2% | ~25ms | 5.2MB | ✅ | ✅ | Apache 2.0 |
| **Detection** |||||||||
| YOLOv8n | 3.2M | 8.7B | 37.3 mAP | 80ms | 3.2MB | ✅ | ✅ | AGPL* |
| YOLOv11n | 2.6M | 6.5B | 39.5 mAP | 56ms | 2.6MB | ✅ | ✅ | AGPL* |
| **Anomaly** |||||||||
| EfficientAD-S | ~4M | - | 95.4% AUROC | ~30ms | ~5MB | ✅ | ⚠️ | Apache 2.0 |

*AGPL-3.0 requires source code distribution for commercial use; Ultralytics offers commercial license.*

---

### Recommendations for 3D-Print Failure Detection

#### For Classification (Good vs Bad Print):
1. **EfficientNet-Lite0** - Best accuracy/latency ratio (75.1% acc, 49ms, Apache 2.0)
2. **YOLOv11n-cls** - Modern architecture, excellent support (70% acc, ~10ms, but AGPL)
3. **MobileNetV3-Small** - Fastest inference (61ms, 67.9% acc, Apache 2.0)

#### For Detection (Locate Failures):
1. **YOLOv11n** - Best mAP/speed tradeoff (39.5 mAP, 56ms, but AGPL)
2. **YOLOv8n** - Mature ecosystem (37.3 mAP, 80ms, but AGPL)

#### For Anomaly Detection (Unsupervised):
1. **EfficientAD-S** - Best for CPU edge deployment (2.2ms GPU, ~30ms CPU, Apache 2.0)

---

### Key Considerations

1. **Licensing**: YOLO models require AGPL license or commercial license from Ultralytics. MobileNet/EfficientNet/GhostNet are Apache 2.0.

2. **INT8 Quantization**: Critical for RPi deployment. All models show 2-4× speedup with INT8.

3. **Resolution Impact**: Reducing input from 640→320 for detection gives ~3× speedup with <2% mAP loss.

4. **Thermal Throttling**: RPi4/5 throttles after 60s without cooling. Add heatsink+fan for sustained inference.

5. **Camera Latency**: CSI cameras add ~8ms vs USB cameras adding ~25ms latency.

---

### Sources

- Ultralytics YOLOv8/v11 Documentation (2024)
- TensorFlow Lite Model Zoo
- Anomalib Documentation (Intel OpenVINO)
- Edge AI Benchmark Studies (2024-2026)
- ONNX Model Zoo
