# Anomaly Detection for 3D-Print Failure Detection

**Status**: Research only — no training, no dataset downloads, no implementation.
**Date**: 2026-09-02
**Question**: Can we train mainly on "normal" prints and detect failures as anomalies, given that real failures are rare?

---

## Table of contents

1. [Executive summary](#executive-summary)
2. [Anomalib library (Intel, Apache 2.0)](#1-anomalib-library-intel-apache-20)
3. [Anomaly detection for manufacturing](#2-anomaly-detection-for-manufacturing)
4. [Specific to 3D printing](#3-specific-to-3d-printing)
5. [Practical considerations](#4-practical-considerations)
6. [Alternative: one-class classification](#5-alternative-one-class-classification)
7. [Anomaly detection vs supervised classification](#6-anomaly-detection-vs-supervised-classification)
8. [Can anomaly detection and supervised classification be combined?](#7-can-anomaly-detection-and-supervised-classification-be-combined)
9. [Implications for TinyVision](#8-implications-for-tinyvision)
10. [Recommendations](#9-recommendations)
11. [Source references](#source-references)

---

## Executive summary

**Short answer**: Anomaly detection is technically viable for 3D-print failure detection, but
with significant caveats that make it **complementary to, not a replacement for, supervised
classification** in TinyVision's context.

**Key findings**:

1. **Anomalib provides mature, well-benchmarked tools** — PatchCore, EfficientAD, PaDiM,
   SimpleNet all achieve >98% AUROC on MVTec AD using only normal training images.
2. **Raspberry Pi deployment is feasible but slow** — PaDiM/PatchCore achieve ~0.02–0.03 FPS
   on Pi 4B (~45–52 seconds per inference). EfficientAD may be faster but is not yet
   benchmarked on Pi CPU.
3. **Anomaly detection excels when**: defect types are unknown, defects are rare, and the
   "normal" class is visually consistent.
4. **Anomaly detection struggles when**: normal variation is high (different colors, geometries,
   lighting), requiring retraining per print/geometry/setup.
5. **For 3D printing specifically**: normal-print variation (filament colors, part geometries,
   printer models, lighting conditions) makes anomaly detection harder than on MVTec AD's
   controlled manufacturing images.
6. **Hybrid approaches show promise** — combining anomaly detection (general "something is
   wrong") with supervised classification (specific failure type) could be optimal.

---

## 1. Anomalib library (Intel, Apache 2.0)

### Overview

Anomalib is Intel's open-source deep learning library for visual anomaly detection.
GitHub: `open-edge-platform/anomalib` (~6k stars, Apache 2.0 license).

Key features:
- Largest public collection of ready-to-use deep learning anomaly detection algorithms
- Lightning-based model implementations
- OpenVINO export for Intel hardware acceleration
- CLI training, inference, benchmarking, and hyperparameter optimization
- MVTec AD and VisA dataset integration

### Models provided

| Model | Type | Architecture | MVTec AD I-AUROC | Key characteristics |
|-------|------|-------------|-----------------|---------------------|
| **PatchCore** | Embedding/memory bank | WideResNet50 backbone + coreset sampling | ~99.1% | No training required — extracts features, stores in memory bank, uses nearest-neighbor search |
| **EfficientAD** | Student-teacher | Lightweight Patch Description Network (PDN) | ~99.8% (with early stopping) | Fastest — <2ms latency on GPU, 600+ FPS |
| **PaDiM** | Embedding/Gaussian | ResNet18 or WideResNet50 + multivariate Gaussian | ~97.9% | Stores patch distributions, Mahalanobis distance scoring |
| **SimpleNet** | Feature space synthesis | WideResNet50 + discriminator | ~99.6% | Single-stream inference, synthesizes anomalies in feature space |
| **FastFlow** | Normalizing flow | WideResNet50 + flow layers | ~95–97% | Good for logical anomalies, memory-intensive |
| **CFA** | Contextual feature | Custom | ~95% | Focuses on contextual anomalies |
| **DFM** | Autoencoder-based | MobileNet backbone | ~95% | Lightweight feature extraction |
| **GANomaly** | GAN-based | Encoder-decoder | ~87% | Lower performance, generation-based |
| **DRAE** | Reconstruction | Autoencoder + discriminator | ~90% | Reconstruction-based approach |

### Model sizes and latency

| Model | Parameters | GPU Latency (256×256) | CPU Latency | Model Size |
|-------|-----------|----------------------|-------------|------------|
| **EfficientAD-S** | ~few hundred K (PDN) | **<1ms** (RTX A6000) | Unknown on Pi | Small |
| **EfficientAD-M** | ~few M (PDN) | ~2ms | Unknown on Pi | Medium |
| **PatchCore** (WideResNet50) | ~68M (backbone) | ~50ms | ~5–10s (memory bank search) | Large (stores embeddings) |
| **PaDiM** (ResNet18) | ~11M (backbone) | ~30ms | ~2–5s | Medium (stores Gaussian params) |
| **SimpleNet** (WideResNet50) | ~68M (backbone) | ~13ms (77 FPS) | ~5–10s | Large |
| **FastFlow** (WideResNet50) | ~68M | ~100ms | ~10s+ | Large |

**Note**: EfficientAD is the standout for speed — it achieves <2ms latency on a modern GPU.
The paper reports processing a 256×256 image in <800μs on an NVIDIA RTX A6000. However,
the PDN (Patch Description Network) uses only 4 convolutional layers, making it lightweight
enough that CPU deployment on edge devices may be feasible.

### Can any run on Raspberry Pi CPU?

**Published benchmarks on Pi** (from Cost-Effective Visual Anomaly Detection project):

| Model | F1 Score (MVTec) | Inference Time (RPi 4B) | FPS |
|-------|------------------|------------------------|-----|
| PaDiM | 0.95 | ~45s | ~0.02 |
| PatchCore | 0.94 | ~52s | ~0.019 |
| CFA | 0.92 | ~38s | ~0.026 |
| FastFlow | 0.90 | ~60s | ~0.017 |

**Verdict**: PaDiM, PatchCore, CFA, and FastFlow all run on Pi 4B but at
**unacceptably slow speeds for real-time monitoring** (~0.02 FPS).

**EfficientAD on Pi**: Not yet benchmarked in published work. Its architecture is much
lighter than PatchCore/PaDiM (4-layer PDN vs 68M-parameter WideResNet50), so it may
achieve 1–5 FPS on Pi 5 CPU. This is the most promising candidate for edge deployment but
requires empirical testing.

**OpenVINO on Intel hardware**: Anomalib exports to OpenVINO IR format, which accelerates
inference on Intel CPUs (not Raspberry Pi ARM). For Intel NUC-class hardware, OpenVINO
could achieve 10+ FPS with EfficientAD.

### Input resolutions

- Standard: 256×256 or 224×224 (center-cropped from larger images)
- EfficientAD: fully convolutional, accepts variable sizes (paper uses 256×256)
- PaDiM: uses 224×224 (ResNet18) or 256×256 (WideResNet50)
- PatchCore: typically 256×256 center-cropped

All models resize/crop to fixed resolution before processing.

### Training data requirements (normal-only)

| Model | Min. Normal Images | Recommended | Training Time |
|-------|-------------------|-------------|---------------|
| PatchCore | ~10–30 | 100–300 | Seconds (feature extraction only) |
| PaDiM | ~10–30 | 100–300 | Seconds (Gaussian fitting) |
| EfficientAD | ~50–100 | 200–500 | Minutes (student-teacher training) |
| SimpleNet | ~50–100 | 200–500 | Minutes (adaptor + discriminator training) |

**Key research finding** (from Gutierrez et al., 2021):
- With only 10 normal images + data augmentation, Mahalanobis/Ledoit achieves AUC >0.9
  on MVTec AD
- With 50 images + augmentation, AUC reaches ~0.927
- PatchCore and PaDiM are robust to small sample sizes due to pretrained backbones

**Industrial deployment data** (from SwitchOn/EthonAI):
- ~200 conforming product images sufficient for production-ready accuracy
- No defect images required
- Deployment in ~45 minutes
- 99.5% real-world accuracy across 50+ manufacturing sites

### MVTec AD benchmark performance

MVTec AD: 15 categories (5 texture, 10 object), 5,354 images total.

| Model | Image AUROC | Pixel AUROC | Notes |
|-------|-------------|-------------|-------|
| EfficientAD-S | 99.8% | 98.8% | Best speed/accuracy tradeoff |
| SimpleNet | 99.6% | 98.1% | Single-stream inference |
| PatchCore | 99.1% | 98.3% | Memory bank approach |
| PaDiM | 97.9% | 97.5% | Gaussian modeling |
| FastFlow | 95.3% | 97.1% | Flow-based |
| DRAEM | 98.0% | 97.9% | Reconstruction + discriminator |

**Important caveat**: MVTec AD images are lab-quality with controlled lighting,
centered objects, and uniform backgrounds. Real-world industrial performance is
typically lower. A 2025 benchmark paper (Baitieva et al.) found that models achieving
>99.9% on MVTec AD showed "significant performance degradation" on real production data.

---

## 2. Anomaly detection for manufacturing

### Is anomaly detection used in practice?

**Yes.** Multiple commercial deployments exist:

| Company/Product | Approach | Scale | Throughput |
|----------------|----------|-------|------------|
| **SwitchOn AI Visual Inspection** | Anomaly detection (200 normal images) | 50+ global sites | 1,100+ units/min |
| **EthonAI Inspector** | Generative anomaly detection | Manufacturing | Production-grade |
| **Landing AI** | Visual inspection with anomaly detection | Enterprise | High throughput |
| **Cognex** | Multiple approaches including anomaly | Industrial | Real-time |
| **Intel Anomalib** | Library for industrial AD | Research + deployment | Variable |

**SwitchOn reports**: 99.5% real-world accuracy using only ~200 conforming images
per product type, with deployment in ~45 minutes. This is the strongest published
evidence that anomaly detection works in production manufacturing.

### Limitations vs supervised classification

| Dimension | Anomaly Detection | Supervised Classification |
|-----------|------------------|--------------------------|
| **Training data** | Normal images only | Normal + defective images |
| **Defect types detected** | Any deviation from normal | Only trained defect classes |
| **False positive rate** | Higher (flags normal variation as anomaly) | Lower (learned decision boundary) |
| **Specificity** | "Something is wrong" | "This specific failure mode" |
| **Actionability** | Lower (requires human interpretation) | Higher (can trigger specific responses) |
| **Retraining** | Per product/setup | Per defect type (can accumulate) |
| **Threshold sensitivity** | High (must tune carefully) | Lower (natural decision boundary) |
| **Novel defect detection** | **Strength** — catches unseen defects | **Weakness** — misses untrained classes |

### When does anomaly detection outperform supervised learning?

1. **Unknown defect types**: When you cannot enumerate all possible failures
2. **Rare defects**: When defective examples are too scarce for supervised training
3. **High product variety**: Many product variants where per-variant supervised training is impractical
4. **Rapid deployment needed**: When you need inspection working in hours, not weeks
5. **Early production**: When no defective examples exist yet

### When does anomaly detection underperform?

1. **High normal variation**: When "normal" itself is diverse (colors, textures, geometries)
2. **Subtle defects**: Defects that are within normal variation bounds
3. **Actionable classification needed**: When the system must say "this is a spaghetti failure, pause the print" not just "something is wrong"
4. **False positive sensitivity**: When operators will ignore frequent false alarms
5. **Domain shift**: When production conditions change frequently

**Practitioner warning** (from Hypernology, 2026):
> "Anomaly detection says: 'this part is different.' Defect qualification says: 'this part
> failed on this dimension, here is the failure mode, here is where it falls on your
> tolerance specification.' Production teams learn to ignore" raw anomaly flags.

---

## 3. Specific to 3D printing

### Has anyone applied anomaly detection to 3D-print monitoring?

**Yes**, several published works:

1. **Visual Defect Detection in FFF Using Embedding Similarity** (Ebrahimian et al., 2026):
   - Used ResNet50 embeddings + L2 distance to compare each layer against reference
   - Achieved 100% detection of catastrophic defects with zero false positives
   - Processing time: <2s per layer on standard CPU
   - Limitation: requires reference images (normal or simulated from G-code)

2. **GuidedFlow** (Paul & Roy, CVPR 2026 Workshop):
   - Attention-guided normalizing flow for anomaly detection in additive manufacturing
   - Trained on normal 3D-printed objects (NylonX Carbon Fiber)
   - Created AM3D-AD dataset: 537 images, 129 videos of printed objects
   - Outperformed SOTA on both AM3D-AD and MVTec AD

3. **Monitoring Anomalies in 3D Bioprinting with Deep Neural Networks** (Jin et al., 2021):
   - Applied to bioprinting (not FDM), but demonstrates the approach
   - Deep neural networks for anomaly detection in layer-by-layer printing

4. **Physics-informed Deep Learning for 3D Printing** (Uhrich et al., 2024):
   - Used PINNs to predict temperature fields and detect anomalies
   - Pre-trained foundation model + transfer learning per layer
   - Real-time anomaly detection by comparing predicted vs. observed thermal fields

5. **Open-source projects using anomaly-inspired approaches**:
   - **CAXTON** (Cambridge): Multi-head neural network for error detection, trained on
     >1M images. Uses parameter deviation labels (not pure anomaly detection, but
     similarity-based).
   - **Various YOLO-based systems**: Supervised, not anomaly-based.

### Could normal-print variation make anomaly detection harder?

**Yes, significantly.** This is the core challenge for 3D-print anomaly detection:

| Variation Factor | Impact on Anomaly Detection |
|-----------------|---------------------------|
| **Filament color** | Red filament looks nothing like blue filament. Anomaly model trained on red will flag blue as anomalous. |
| **Part geometry** | A cube looks nothing like a vase. Model must be retrained per geometry or see sufficient variety. |
| **Lighting conditions** | Day vs night, overhead vs side lighting changes appearance dramatically. |
| **Printer model** | Different printers have different bed surfaces, frame structures, nozzle types. |
| **Print orientation** | Same part printed in different orientations looks different from camera. |
| **Background clutter** | Tools, filament spools, other objects in frame. |
| **Camera angle/position** | Small camera movement changes the entire scene. |

**Contrast with MVTec AD**: MVTec AD categories each have a single object type (bottle,
capsule, etc.) with controlled lighting and background. 3D printing has none of this
consistency.

**Possible mitigations**:
1. **Per-printer/per-geometry models**: Train a separate anomaly model for each
   printer+geometry combination. Achievable but reduces the "one model fits all"
   advantage.
2. **Normal variation training**: Collect diverse normal images (many colors, geometries,
   lighting) and train a single model on all of them. Risk: dilutes normality definition,
   missing real anomalies.
3. **Feature-space normalization**: Use domain adaptation or feature normalization to
   reduce sensitivity to color/lighting while preserving anomaly sensitivity.
4. **Reference-based approach**: Compare current print against expected G-code rendering
   (as in Ebrahimian et al.). Requires accurate simulation but avoids normality modeling.

### Would a "normality" model need retraining for each new print?

**Probably yes, for reliable detection.** Published evidence:

- SwitchOn/EthonAI: 200 images per product type, retrain per SKU
- Manufacturing anomaly detection generally: retrain when production conditions change
- 3D printing: each new geometry is effectively a new "product type"

**Practical implication**: If a user prints 10 different objects per week, they would need
to collect ~200 normal images and retrain for each one. This is a significant friction
point compared to a supervised classifier that generalizes across geometries.

**Exception**: If the anomaly model is trained on a very diverse set of normal prints
(hundreds of geometries, multiple colors), it might generalize. But this dilutes the
normality definition and increases false positives.

---

## 4. Practical considerations

### How many normal images are needed for training?

| Scenario | Images Required | Notes |
|----------|----------------|-------|
| **Minimum viable** | 10–50 | With data augmentation, PaDiM/Mahalanobis achieve AUC >0.9 on MVTec |
| **Production-ready** | 100–200 | SwitchOn uses ~200 per product type in manufacturing |
| **Robust generalization** | 300–500 | Covers normal variation more thoroughly |
| **Diverse multi-geometry** | 1,000+ | If training one model for many print types |

**Key insight**: The number depends heavily on how much normal variation exists. For a
single printer, single geometry, single color: 50–100 images may suffice. For a general
model covering multiple printers, colors, geometries: 500+ images may be needed.

### Does the model need retraining for each new printer/camera setup?

**Yes, typically.** Anomaly detection learns the specific visual distribution of "normal"
from the training images. A different printer, camera angle, or lighting setup changes
that distribution.

**Mitigation strategies**:
1. **Camera calibration/normalization**: Fix camera position, normalize lighting
2. **Domain adaptation**: Fine-tune on a few images from the new setup
3. **Feature-space alignment**: Use techniques like domain-randomization during training

**Published evidence**: The 2025 paper "From Lab to Factory" (Hönel & Nordqvist) found
that "most methods do not handle low data quality well or exude low robustness in
unfavorable, but typical real-world settings" and that "for practitioners it may be very
difficult to identify the actual underlying problem when such methods underperform."

### Inference cost compared to a YOLO classifier

| Model | GPU Latency | CPU Latency (estimated) | Model Size | RAM |
|-------|-------------|------------------------|------------|-----|
| **YOLOv8n-cls** (supervised) | ~1–2ms | ~50–100ms | ~3–6MB | ~50–100MB |
| **EfficientAD-S** (anomaly) | <1ms | ~??? (untested on Pi) | Small | Low |
| **PatchCore** (anomaly) | ~50ms | ~5–10s | Large (embeddings) | High |
| **PaDiM** (anomaly) | ~30ms | ~2–5s | Medium (Gaussian params) | Medium |
| **SimpleNet** (anomaly) | ~13ms | ~5–10s | Large | High |

**Key finding**: YOLO classifiers are dramatically faster at inference than most
anomaly detection methods, particularly those requiring memory bank search (PatchCore)
or Gaussian distance computation (PaDiM). EfficientAD is the exception — it is
comparable to or faster than YOLO on GPU.

**For Raspberry Pi CPU specifically**: YOLOv8n achieves ~2–5 FPS. PatchCore/PaDiM achieve
~0.02 FPS. EfficientAD may achieve 1–5 FPS (unconfirmed).

### Can anomaly detection and supervised classification be combined?

**Yes, and this is an active research area.** Published approaches:

1. **Two-stage pipeline**: Anomaly detector flags "something is wrong," then a supervised
   classifier identifies the specific failure type. Only runs the classifier when anomaly
   is detected — saves compute.

2. **Feature fusion**: Combine anomaly scores with class probabilities as inputs to a
   meta-classifier.

3. **Semi-supervised learning**: Use anomaly detection on normal data + a small number of
   labeled anomalies (Deep SAD, DevNet, DRA).

4. **Anomaly-guided classification**: Use anomaly localization to focus the classifier's
   attention on suspicious regions.

**From the 2025 survey** (Baitieva et al.):
> "Supervised AD might theoretically offer advantages in boundary refinement, but practical
> limitations arise from the impossibility of capturing complete defect variability, the
> scarcity of anomalous training data, and the small size of anomalies in the image."

**Recommended hybrid approach for TinyVision**:
- **Primary**: Supervised YOLO classifier (fast, specific, actionable)
- **Secondary**: Anomaly detector as safety net (catches untrained failure modes)
- **Logic**: If YOLO says "normal" but anomaly detector says "abnormal," flag for review

---

## 5. Alternative: one-class classification

### One-class SVM

- **Approach**: Finds hyperplane separating normal data from origin in kernel space
- **Strengths**: Well-understood, works on small datasets
- **Weaknesses**: Does not scale well to high-dimensional data (images), kernel
  computation is O(n²)
- **Edge feasibility**: **Poor** for images. Works on tabular/sensor data.
- **Pi deployment**: Tested on Raspberry Pi 3 for network intrusion detection —
  "computationally heavier at inference" than alternatives

### Deep SVDD (Deep Support Vector Data Description)

- **Approach**: Neural network maps normal data into tight hypersphere; anomalies are
  far from center
- **Published**: Ruff et al., ICML 2018
- **Training**: Autoencoder pretraining → center computation → SVDD training
- **Performance**: Competitive on MNIST/CIFAR-10, used in industrial anomaly detection
- **Edge feasibility**: **Poor for real-time deployment**
  - "Deep SVDD is particularly effective on structured data but incurs a high computational
    cost, making it less suitable for edge-level runtime deployment" (Paolini et al., 2025)
  - Tested on Raspberry Pi for network data — "lack of optimized support for real-time
    deployment" and "inability to meet latency constraints"
- **Parameters**: ~6,848–8,384 (much smaller than PatchCore), but inference is still
  ~2.6ms on GPU

### Autoencoder-based approaches

- **Approach**: Train encoder-decoder on normal data; high reconstruction error = anomaly
- **Strengths**: Lightweight, well-understood
- **Weaknesses**: Tend to generalize to anomalies (reconstruct them too well), lower
  detection performance than embedding methods
- **Edge feasibility**: **Good** for sensor data, **moderate** for images
  - TFLite quantized autoencoder: 3MB, 45ms on Pi 3B+
  - Simple dense autoencoder (INT8): 800KB, 12ms on Pi 3B+
  - Conv autoencoder (INT8 TFLite): 3MB, 45ms on Pi 3B+

### Comparison table

| Method | Image AUROC (MVTec) | GPU Latency | CPU Latency | Edge Feasibility | Training Data |
|--------|--------------------:|-------------|-------------|------------------|---------------|
| **One-class SVM** | N/A (not for images) | N/A | Slow | Poor | Normal only |
| **Deep SVDD** | ~90–95% | ~2.6ms | ~100ms+ | Poor | Normal only |
| **Autoencoder** | ~85–90% | ~5–10ms | ~45ms (TFLite) | Good | Normal only |
| **PatchCore** | ~99.1% | ~50ms | ~5–10s | Poor | Normal only |
| **PaDiM** | ~97.9% | ~30ms | ~2–5s | Poor | Normal only |
| **EfficientAD** | ~99.8% | <1ms | ??? | **Promising** | Normal only |
| **YOLO (supervised)** | ~95–99% | ~1–2ms | ~50–100ms | **Excellent** | Normal + defective |

**Conclusion**: For image-based anomaly detection, embedding methods (PatchCore, PaDiM)
and student-teacher methods (EfficientAD) vastly outperform classic one-class methods
(OCSVM, Deep SVDD, autoencoders). EfficientAD is the only anomaly detection method that
competes with supervised YOLO on inference speed.

---

## 6. Anomaly detection vs supervised classification

### When anomaly detection is the better choice

1. **No defective examples available**: New product, new process, no failure history
2. **Defect types are open-ended**: Cannot enumerate all possible failures
3. **Very rare failures**: Insufficient defective examples for supervised training
4. **Rapid deployment needed**: Hours to set up, not weeks
5. **High product variety**: Many SKUs, each needing inspection
6. **Unknown unknowns**: Need to catch failure modes you haven't imagined

### When supervised classification is the better choice

1. **Specific failure modes are known**: Spaghetti, layer shift, warping are well-defined
2. **Actionable response needed**: "Pause the print" vs "something might be wrong"
3. **Low false-positive tolerance**: Operators will ignore frequent false alarms
4. **Normal variation is high**: Different colors, geometries, printers
5. **Edge deployment required**: YOLO is 100–1000x faster than most anomaly methods on CPU
6. **Continuous improvement**: New failure types can be added incrementally

### For 3D-print failure detection specifically

| Factor | Favors Anomaly Detection | Favors Supervised |
|--------|------------------------|-------------------|
| Failures are rare | ✓ | |
| Known failure types (spaghetti, warping) | | ✓ |
| Need to catch novel failures | ✓ | |
| Need specific failure identification | | ✓ |
| Edge speed requirement | | ✓ (YOLO) |
| Multiple printer types/colors | | ✓ (generalizes better) |
| Rapid setup for new printer | ✓ (no defect data needed) | |
| Pause/cancel on failure | | ✓ (actionable) |

---

## 7. Can anomaly detection and supervised classification be combined?

**Yes, this is likely the optimal approach.** Published evidence and reasoning:

### Published hybrid approaches

1. **Two-stage pipeline** (most practical):
   - Stage 1: Anomaly detector flags abnormal frames (fast filter)
   - Stage 2: Supervised classifier identifies specific failure type (only when flagged)
   - Benefit: Classifier only runs on suspicious frames, reducing false-positive workload

2. **Score fusion**:
   - Combine anomaly score + class probabilities into a meta-decision
   - Anomaly score provides "confidence calibration" — low anomaly score reinforces
     "normal" classification

3. **Anomaly-aware training**:
   - Train supervised classifier with anomaly scores as additional input feature
   - Allows classifier to leverage anomaly detector's "normality" assessment

### Recommended architecture for TinyVision

```
Camera frame
    │
    ├──→ EfficientAD (anomaly score) ──┐
    │                                   ├──→ Decision Logic ──→ Pause/Alert
    └──→ YOLOv8n (class prediction) ───┘
```

**Decision logic**:
- YOLO says "failure" + high confidence → Act (pause print)
- YOLO says "normal" + anomaly detector says "normal" → Continue
- YOLO says "normal" + anomaly detector says "abnormal" → Warning (possible novel failure)
- YOLO says "failure" + low confidence → Check anomaly score for confirmation

This combines:
- **Speed**: YOLO at 2–5 FPS on Pi, anomaly detector as lightweight supplement
- **Specificity**: YOLO identifies failure type for actionable response
- **Safety net**: Anomaly detector catches failures YOLO hasn't been trained on
- **Low false positives**: Both models must agree for high-confidence actions

---

## 8. Implications for TinyVision

### Does anomaly detection change the recommended MVP?

**Partially.** The current MVP recommends a supervised YOLO classifier. Anomaly detection
suggests enhancements:

1. **Phase 1 (MVP)**: Supervised YOLOv8n classifier for known failures
   - Fastest to deploy, most actionable, proven on Pi
   - Trained on known failure types: spaghetti, layer shift, warping, stringing

2. **Phase 2 (Enhancement)**: Add EfficientAD as anomaly safety net
   - Catches novel/untrained failure modes
   - Requires empirical testing on Pi CPU to validate speed
   - Train on normal prints from target printer

3. **Phase 3 (Advanced)**: Hybrid decision system
   - Combines YOLO specificity with anomaly detection coverage
   - May require more sophisticated threshold tuning

### Data collection implications

Anomaly detection reduces the data collection burden:
- **For supervised only**: Need ~50–200 images per failure type × 5–10 types = 250–2,000
  defective images + 500+ normal images
- **For anomaly + supervised**: Need 200–500 normal images + 50–100 per known failure type
  for the supervised component

The anomaly detector needs only normal images, which are trivial to collect (just print
things normally).

### Edge deployment implications

If EfficientAD can achieve 1–5 FPS on Pi 5 CPU:
- It could run alongside YOLO with minimal overhead
- Combined inference: ~50–100ms per frame (YOLO) + ~200–1000ms (EfficientAD)
- Acceptable for 3D-print monitoring (failures develop over seconds to minutes)

If EfficientAD is too slow on Pi CPU:
- Run anomaly detection on a separate thread/process at lower frequency
- Example: YOLO at 2 FPS (every frame), anomaly check every 5th frame (~0.4 FPS)

### Open questions for implementation

1. **Can EfficientAD run on Pi 5 CPU at acceptable speed?** Must benchmark empirically.
2. **How much normal variation can a single anomaly model tolerate?** Test with 5+
   colors, 3+ geometries in training set.
3. **Does per-printer retraining create too much user friction?** Compare to supervised
   classifier's generalization across printers.
4. **What is the false-positive rate in practice?** Anomaly detectors on real-world data
   often have higher FPR than on MVTec AD.
5. **Can OpenVINO help on non-Intel ARM hardware?** OpenVINO has limited ARM support.

---

## 9. Recommendations

### For TinyVision's MVP

1. **Keep supervised YOLO as primary approach** for the initial MVP. It is faster, more
   actionable, and better understood for edge deployment.

2. **Add anomaly detection research as a parallel track**. Benchmark EfficientAD on
   Raspberry Pi 5 CPU. If it achieves ≥1 FPS, integrate as secondary safety net.

3. **Consider anomaly detection for "zero-shot" deployment scenarios**: When a user
   installs the system on a new printer with no failure history, an anomaly detector
   could provide immediate monitoring while the supervised classifier is being trained
   on collected data.

4. **Document the tradeoffs clearly** so users can choose:
   - "Fast mode" (YOLO only): 2–5 FPS, catches known failures
   - "Safe mode" (YOLO + anomaly): 0.5–2 FPS, catches known + unknown failures
   - "Setup mode" (anomaly only): Immediate monitoring with no failure examples

5. **Investigate hybrid approaches in Phase 2** after the supervised MVP is working.

### For future research

1. **Benchmark EfficientAD on Pi 5** — this is the most impactful missing data point.
2. **Test normal variation tolerance** — train anomaly models on 10+ colors, 5+ geometries.
3. **Evaluate reference-based approaches** — compare current layer against G-code
   rendering (as in Ebrahimian et al.). This sidesteps the normality modeling problem.
4. **Explore semi-supervised approaches** — use small number of labeled failures +
   large normal dataset (Deep SAD, DevNet).
5. **Monitor Anomalib development** — Intel actively maintains it; new lightweight models
   may become available.

---

## Source references

### Anomalib
- Anomalib documentation: https://anomalib.readthedocs.io
- GitHub: https://github.com/open-edge-platform/anomalib (~6k stars, Apache 2.0)
- PatchCore paper: "Towards Total Recall in Industrial Anomaly Detection" (CVPR 2022)
- EfficientAD paper: "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies" (WACV 2024)
- PaDiM paper: "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization" (ICPR 2021)
- SimpleNet paper: "SimpleNet: A Simple Network for Image Anomaly Detection and Localization" (CVPR 2023)

### Manufacturing anomaly detection
- Baitieva et al. (2025): "Beyond Academic Benchmarks: Critical Analysis and Best Practices for Visual Industrial Anomaly Detection" — arXiv:2503.23451
- Shukla et al. (2025): "A systematic survey: role of deep learning-based image anomaly detection in industrial inspection contexts" — Frontiers in Robotics and AI
- Hönel & Nordqvist (2025): "From Lab to Factory: Pitfalls and Guidelines for Self-/Unsupervised Defect Detection on Low-Quality Industrial Images" — arXiv:2506.16890
- ScienceDirect (2023): "Anomaly detection for industrial quality assurance: A comparative evaluation of unsupervised deep learning models"
- Hypernology (2026): "How many training images does AI quality inspection really need?"
- SwitchOn (2026): "Anomaly-Detection-Based AI Visual Inspection" — Vision Systems Design
- EthonAI: "Industrial anomaly detection: Using only defect-free images to train your inspection model"

### 3D-printing anomaly detection
- Ebrahimian et al. (2026): "Visual Defect Detection in Fused Filament 3D Printing, Using Embedding Similarity" — IET Image Processing
- Paul & Roy (2026): "GuidedFlow: An Attention-Guided Framework for Anomaly Detection in Additive Manufacturing" — CVPR 2026 Workshop
- Goh et al. (2022): "Anomaly Detection in Fused Filament Fabrication Using Machine Learning" — 3D Printing and Additive Manufacturing
- Brion & Pattinson (2022): "Generalisable 3D printing error detection and correction via multi-head neural networks" — Nature Communications
- Uhrich et al. (2024): "Physics-informed deep learning to quantify anomalies for real-time fault mitigation in 3D printing" — Applied Intelligence
- Paraskevoudis et al. (2020): "Real-Time 3D Printing Remote Defect Detection (Stringing) with Computer Vision and AI" — Processes

### Edge deployment
- Cost-Effective Visual Anomaly Detection (GitHub: Yunbo-max): PaDiM/PatchCore on Raspberry Pi 4B
- Dayani (2025): "ML-Driven Anomaly Detection for EPICS PV Streams at the Edge: Implementation and Evaluation on Raspberry PI IOCs"
- Paolini et al. (2025): "One-Class Anomaly Detection for Industrial Applications" — MDPI Computers
- Tripathi (2021): "Deploying Anomaly Detection Models on Raspberry Pi"

### One-class classification
- Ruff et al. (2018): "Deep One-Class Classification" — ICML 2018
- Deep SVDD PyTorch: https://github.com/fafalooubb/Deep-SVDD-PyTorch
- pyod DeepSVDD: https://www.aidoczh.com/pyod/_modules/pyod/models/deep_svdd.html

### Low-data anomaly detection
- Gutierrez et al. (2021): "How to train your anomaly detection system" — arXiv:2106.01277
  - Key finding: With 10 images + augmentation, AUC >0.9 on MVTec AD
