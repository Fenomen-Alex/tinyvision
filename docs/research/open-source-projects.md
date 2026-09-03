# Open-Source 3D Print Failure Detection Projects — Research Report

**Date:** 2026-09-02
**Status:** Research compilation

---

## 1. Obico (formerly The Spaghetti Detective)

| Field | Details |
|---|---|
| **GitHub URL** | https://github.com/TheSpaghettiDetective/obico-server |
| **License** | AGPL-3.0 |
| **Stars** | ~1,902 |
| **Forks** | 360 |
| **Last commit** | Active (5,272 commits, release branch) |
| **Created** | 2019-01-02 |

### Model Architecture
- **Detection approach:** Binary classification (print_ok vs print_failure) using a **YOLO-family object detector** — specifically a **Darknet-based YOLO model** (exact variant not publicly disclosed; referred to as a "Deep Learning model" throughout docs).
- The model outputs bounding boxes with class labels (`spaghetti`, `print_failure`) and confidence scores.
- This is **object detection, not classification** — it localizes failure regions in the frame.

### Detection Pipeline
- ML API is a containerized **Flask** service that receives image URLs from the Django backend.
- Supports two runtimes with automatic fallback:
  1. **Darknet** (C, from YOLO author AlexeyAB's fork) — primary/stable
  2. **ONNX Runtime** — beta
- Priority: Darknet+GPU → ONNX+GPU → ONNX+CPU → Darknet+CPU
- Model weights are **downloaded at Docker build time** from URLs stored in `ml_api/model/model-weights.darknet.url` and `model-weights.onnx.url`.

### Training Data
- **Not publicly disclosed.** The model is trained internally by the Obico team on proprietary data.
- The cloud service has watched >80M hours of printing and caught ~1M failures (per marketing).
- No dataset is published. No training code is published.

### Available Weights
- Weights are bundled inside Docker images, not published as standalone downloads.
- URL files in the repo point to hosted weight files (the actual URLs are in the git history).
- **Weights license:** Same as repo (AGPL-3.0).

### Inference Runtime
- Python (Flask) + Darknet C binding or ONNX Runtime Python binding.
- Docker-based deployment. No standalone C++ runtime.

### Supported Hardware
- x86_64 CPU, NVIDIA GPU (CUDA), NVIDIA Jetson (JetPack).
- Explicitly **not supported on Raspberry Pi** (too slow per original TSD docs).
- RK3588 NPU support via a separate docker-compose file.

### Implementation Quality
- **High.** Production-grade system with:
  - Full Django backend with user management, notifications, printer integration
  - Mobile apps (iOS, Android)
  - OctoPrint and Klipper (Moonraker) plugin support
  - WebRTC streaming, timelapse analysis
  - First-layer AI analysis
  - Sentry error tracking, health checks
  - Extensive documentation and Docker deployment
- AGPL-3.0 license means any distributed modifications must be source-available.

### AGPL Licensing Scope
- **Server-side only.** The server code is AGPL-3.0. The OctoPrint plugin and mobile apps are separate.
- Self-hosted is fully open source. The cloud service is a separate commercial offering.
- The ML model weights inherit AGPL since they are distributed with the AGPL server.

### Unique/Useful Ideas
- Multi-sensitivity threshold system (normalized probability p-value: <0.33 normal, 0.33-0.66 warning, ≥0.66 failure)
- Timelapse-based post-print analysis
- First-layer adhesion detection
- Mature notification system (email, push, SMS, Telegram, Discord)
- Plugin architecture for printer integration
- The most battle-tested system in this space

---

## 2. klipper-print-failure-detection (CookiezRGood)

| Field | Details |
|---|---|
| **GitHub URL** | https://github.com/CookiezRGood/klipper-print-failure-detection |
| **License** | Not specified in repo (no LICENSE file found) |
| **Stars** | ~30 |
| **Forks** | 1 |
| **Last commit** | ~Jan 2026 (24 commits) |
| **Created** | ~2025 |

### Model Architecture
- **YOLOv8n** (nano variant) trained via Ultralytics, then **exported to TensorFlow Lite (.tflite)**.
- **Object detection** (bounding boxes), not classification.
- 4 classes: **spaghetti, blob, cracks, warping**
- Image size: 640x640

### Training Data
- Combined from two Roboflow datasets:
  1. https://universe.roboflow.com/3d-printer-failure/all_anomalies-rfe61
  2. https://universe.roboflow.com/3d-printer-failure/fdm-w6znp-qnbzd
- Two-phase training:
  - Phase 1: 70 epochs, lr=0.002, heavy augmentation (mosaic 0.8, mixup 0.05)
  - Phase 2: 110 epochs, lr=0.0007, reduced augmentation, strong class separation (cls=2.0)

### Reported Results (mAP@50)
- All classes: 0.502
- Spaghetti: 0.556
- Blob: 0.626
- Warping: 0.341
- Cracks: 0.485

### Available Weights
- `model.tflite` is committed directly in the repo root.
- Weights trained by the author, no separate license specified.

### Inference Runtime
- **Python + TensorFlow Lite** via tflite-runtime.
- Runs on the Klipper host (typically a Pi or SBC).

### Supported Hardware
- **Raspberry Pi** (primary target — runs on Klipper host)
- Any Linux SBC running Klipper + Moonraker + Crowsnest

### Implementation Quality
- **Good for a hobby project.** Well-structured Klipper plugin with:
  - Web dashboard with live camera feeds and bounding box overlays
  - Multi-zone masking system
  - Dual thresholds (detection vs trigger)
  - Configurable actions (warn, pause, cancel)
  - Mobileraker notification integration
  - Smart idle mode
  - Auto-start macros
  - Detection history table
  - Custom dashboard themes
  - Moonraker update manager integration
- Install script for Klipper extras directory

### Unique/Useful Ideas
- **Multi-zone masking** — user draws rectangles on camera feed to mask irrelevant regions
- **Dual threshold system** — separate detection and trigger thresholds with configurable retries
- **Auto-start macro** — G-code macro to start monitoring at PRINT_START
- **Mobileraker integration** — phone notifications via existing Klipper ecosystem
- **TFLite on Pi** — explicitly optimized for edge deployment on Klipper hosts
- **4-class detection** including warping and cracks (most projects only detect spaghetti)

---

## 3. EmilyMyaskovski/3d-print-defect-detect

| Field | Details |
|---|---|
| **GitHub URL** | https://github.com/EmilyMyaskovski/3d-print-defect-detect |
| **License** | Academic project (no formal open-source license specified) |
| **Stars** | 0 |
| **Forks** | 0 |
| **Last commit** | ~May 2026 (51 commits) |
| **Created** | 2026-01-30 |

### Model Architecture
- **YOLOv11 ONNX** model optimized for ARM CPU inference.
- **Object detection** (bounding boxes).
- 3 classes: **spaghetti, stringing, camera occlusion/poor lighting**
- Custom-trained for embedded ARM architectures.

### Training Data
- Custom dataset (not publicly released or linked).
- Trained as a Bachelor's Degree Final Project (CS, 2025-2026).

### Available Weights
- ONNX model included in the `model/` directory.
- No separate weight license specified.

### Inference Runtime
- **Python + ONNX Runtime** (onnxruntime on ARM)
- FastAPI backend + OpenCV frame processing
- WebSocket-based dashboard
- SQLite database for detection history

### Supported Hardware
- **Raspberry Pi 4 (4GB/8GB) and Pi 5** — primary target
- Camera: Pi Camera Module (CSI) or USB webcam
- 640x480 @ 15 FPS capture

### Implementation Quality
- **Decent for an academic project.** Includes:
  - FastAPI server with WebSocket live dashboard
  - Telegram alerts with interactive emergency-stop
  - Brightness guard (skips inference when too dark)
  - 30-day data retention with auto-purge
  - systemd service configuration
  - Environment variable configuration
- Lacks formal license, not production-ready.

### Unique/Useful Ideas
- **Luminance guard** — skips YOLO inference when brightness μ < 15 to save CPU
- **Telegram emergency-stop** — interactive buttons in Telegram notifications
- **Brightness-based camera health detection** — identifies occlusion/failure without ML
- Explicit 640x480@15FPS for Pi performance budget

---

## 4. YOLO-3DPrint-Monitor (none-none-n)

| Field | Details |
|---|---|
| **GitHub URL** | https://github.com/none-none-n/YOLO-3DPrint-Monitor |
| **License** | MIT |
| **Stars** | 1 |
| **Forks** | 0 |
| **Last commit** | ~2026 (18 commits) |
| **Created** | 2026-06-23 |

### Model Architecture
- **YOLOv8n-cls** (nano classifier) — **image classification, NOT object detection**.
- 2.8M parameters, transfer learning from ImageNet.
- **Binary classification** (good vs defective print).
- Input preprocessing: center square crop → grayscale → RGB duplication.
- Skips even frames for throughput.

### Training Data
- Custom dataset, 1,884 test images.
- 20 epochs training, reached plateau at epoch 10.
- Final accuracy: 99.71%, Precision: 99.82%
- Confusion matrix: 941 TP, 938 TN, 2 FP, 3 FN

### Available Weights
- Weights are part of the codebase (not separately distributed).
- Trained with Ultralytics + PyTorch 2.0.

### Inference Runtime
- **Python + PyTorch** (CPU inference, no ONNX export documented).
- PyQt6 GUI with real-time video feed.
- UART serial communication for G-code control.

### Supported Hardware
- Tested on AMD Ryzen 5 5600U (training on RTX 4060).
- Future plans: Raspberry Pi, Jetson Nano.
- Printer: Anycubic Kobra 2 Neo (Marlin firmware).

### Implementation Quality
- **Academic project (B.Sc. Thesis, ETI LETI, 2026).** Includes:
  - PyQt6 GUI with confidence threshold slider, digital zoom
  - Closed-loop G-code control (M25 pause, M24 resume via UART)
  - Async UART polling (100ms interval)
  - Event-driven multi-threaded architecture
  - Bilingual documentation (English/Russian)
  - Full thesis PDF included
- Claims <250ms latency, ~300MB RAM.

### Unique/Useful Ideas
- **Closed-loop hardware control via UART** — directly sends G-code commands to pause/resume
- **Classifier approach** (simpler than detection) — sufficient for binary pass/fail monitoring
- **Frame skipping** — processes every other frame for performance
- **Center crop preprocessing** — focuses on print area, reduces noise
- Full thesis with architecture diagrams and evaluation

---

## 5. Other Notable Repos

### 5a. PrintGuard (oliverbravery) — **Most Active/Mature Alternative**

| Field | Details |
|---|---|
| **GitHub URL** | https://github.com/oliverbravery/PrintGuard |
| **License** | GPL-2.0 |
| **Stars** | ~322 |
| **Forks** | 31 |
| **Last commit** | Active (838 commits) |
| **Created** | 2025-05-10 |

**The strongest open-source competitor to Obico.**

- **Model:** ShuffleNetV2 encoder with nearest-prototype classification (Prototypical Networks / few-shot learning). NOT a standard YOLO model.
- **Training:** Custom training pipeline using 7 public datasets (Roboflow + Kaggle). See companion repo: https://github.com/oliverbravery/Edge-FDM-Fault-Detection (38 stars, GPL-2.0, with published dissertation).
- **Detection approach:** Binary (good vs failure), prototype-distance based. Sensitivity/threshold sliders map to prototype distances.
- **Runtime:** LiteRT (TensorFlow Lite) and ONNX Runtime — benchmarks both at startup, keeps the faster one.
- **Hardware acceleration:** Core ML (macOS), Windows ML, OpenVINO (Intel), TensorRT (NVIDIA).
- **Supported hardware:** Pi 4/5 (Docker images for arm64), macOS, Windows, Intel NUC, NVIDIA GPU.
- **Printer support:** OctoPrint, Klipper/Moonraker, Elegoo, Prusa (PrusaLink), Bambu Lab.
- **Alerts:** ntfy, Telegram, Discord, native push notifications.
- **Deployment:** pip install, Docker, native macOS/Windows apps, runs in browser (WASM/Pyodide).
- **Benchmarks vs Obico (Pi 4B):** 43x throughput (15.1 vs 0.35 img/s), 2x F1 (0.937 vs 0.411).
- **Quality:** Production-grade. Tests, CI, CHANGELOG, API, MCP integration, Home Assistant MQTT, plugin system, theming, remote access via Tailscale/Cloudflare Tunnel.
- **Unique:** Browser-based demo (no install), few-shot learning approach, dissertation-backed claims, MCP/REST API.

### 5b. Edge-FDM-Fault-Detection (oliverbravery) — Training Research

| Field | Details |
|---|---|
| **GitHub URL** | https://github.com/oliverbravery/Edge-FDM-Fault-Detection |
| **License** | GPL-2.0 |
| **Stars** | 38 |

Companion repo to PrintGuard. Contains:
- Prototypical Network training code
- CNN fine-tuning (ShuffleNetV2 feature extractor)
- 7 public dataset preparation scripts
- Obico model evaluation code
- Published dissertation (PDF)

**Useful for:** Understanding the few-shot training approach, dataset curation, and comparative evaluation methodology.

### 5c. Print Guardian (nlaha)

| Field | Details |
|---|---|
| **GitHub URL** | https://github.com/nlaha/print-guardian |
| **License** | AGPL-3.0 |
| **Stars** | 3 |
| **Forks** | 1 |
| **Commits** | 77 |

- **Model:** Uses the Obico Darknet model weights directly.
- **Runtime:** **Rust** — built with Cargo, uses Darknet C bindings.
- **Integration:** Discord webhooks + Moonraker printer control.
- **Unique:** Only Rust implementation in this space. Lightweight, Docker-friendly.

### 5d. TinyPrintGuard (TahielGiordana)

| Field | Details |
|---|---|
| **GitHub URL** | https://github.com/TahielGiordana/TinyPrintGuard |
| **License** | Apache-2.0 |
| **Stars** | 1 |

- **Platform:** ESP32-Cam microcontroller.
- **Model:** TFLite quantized model running on ESP32.
- **Unique:** Smallest deployment target — runs failure detection on a $5 microcontroller.

### 5e. 3DPrintSaviour (Manicben)

| Field | Details |
|---|---|
| **GitHub URL** | https://github.com/Manicben/3DPrintSaviour |
| **License** | Not specified |
| **Stars** | 74 |

- **Approach:** Image comparison (NRMSE between Octolapse timelapse frames) + Obico spaghetti detection model.
- **Runtime:** Python + OpenCV on separate 64-bit computer.
- **Unique:** Hybrid approach combining classical CV with ML. Timelapse-based (not real-time).
- **Status:** Last updated ~2020, largely unmaintained.

### 5f. failure_detection_for_3d_printing (alexgrigoras)

| Field | Details |
|---|---|
| **GitHub URL** | https://github.com/alexgrigoras/failure_detection_for_3d_printing |
| **License** | MIT |
| **Stars** | 22 |

- **Approach:** Three methods: (1) Image matching (slicer vs real), (2) Object tracking (print head position), (3) Spaghetti detection (inspired by TSD).
- **Unique:** Multi-method approach including slicer comparison and print head tracking.

---

## Summary Comparison Table

| Project | Stars | License | Model | Task | Edge? | Pi? | Active? |
|---|---|---|---|---|---|---|---|
| **Obico** | 1,902 | AGPL-3.0 | YOLO (Darknet) | Detection | Server | No | Yes |
| **PrintGuard** | 322 | GPL-2.0 | ShuffleNetV2+Proto | Classification | Yes | Yes | Very |
| **klipper-pfd** | 30 | None | YOLOv8n→TFLite | Detection | Yes | Yes | Somewhat |
| **Edge-FDM-FD** | 38 | GPL-2.0 | ShuffleNetV2 | Training only | N/A | N/A | Somewhat |
| **3DPrintSaviour** | 74 | None | CV+TSD model | Detection | Separate PC | No | No |
| **EmilyM PrintGuard** | 0 | Academic | YOLOv11 ONNX | Detection | Yes | Yes | Brief |
| **YOLO-3DPrint-Monitor** | 1 | MIT | YOLOv8n-cls | Classification | Future | No | Brief |
| **Print Guardian** | 3 | AGPL-3.0 | Obico Darknet | Detection | Yes | Yes | Active |
| **TinyPrintGuard** | 1 | Apache-2.0 | TFLite quant | Classification | ESP32 | N/A | Low |

---

## Key Observations for TinyVision

1. **PrintGuard (oliverbravery)** is the most technically interesting open-source project. Its few-shot learning approach with prototype classification is novel and its published dissertation provides rigorous evaluation. The 43x throughput advantage over Obico on Pi 4B is significant.

2. **Obico** is the most mature production system but AGPL-3.0 is restrictive and the model is not portable (bundled in Docker, no standalone weights published in standard formats).

3. **klipper-print-failure-detection** demonstrates that YOLOv8n→TFLite is viable on Pi for detection with 4 classes. The two-phase training approach and multi-zone masking are useful patterns.

4. **4-class detection** (spaghetti, blob, warping, cracks) from CookiezRGood is the most granular open classification. Most projects only do binary or 2-3 classes.

5. **Classifier vs Detection:** YOLO-3DPrint-Monitor shows a binary classifier can achieve 99.7% accuracy, but lacks localization. Detection (bounding boxes) is more useful for understanding *what* failed and *where*.

6. **No project publishes a standalone C++ inference runtime.** All use Python or Rust bindings. This is an opportunity for TinyVision.

7. **The HuggingFace model** (ApatheticWithoutTheA/3D-Print-Failure-Detector) with YOLOv11s and 3 classes (spaghetti, stringing, zits) trained on 9,000+ images (MIT license) is worth noting as a potential starting point for transfer learning.

8. **Dataset availability:** Several Roboflow and Kaggle datasets exist. The Edge-FDM-Fault-Detection repo catalogues 7 of them with download links. No single large, well-curated, multi-class dataset dominates.
