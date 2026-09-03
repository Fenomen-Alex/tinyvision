# Task Formulation for 3D-Print Failure Detection

**Status**: Research only.
**Date**: 2026-09-02

This document compares candidate ML formulations for detecting FDM/FFF 3D-print failures
from a fixed webcam, and recommends a primary approach for the TinyVision MVP.

---

## 1. Formulation A: Binary image classification

### Definition

Input: a single camera frame.
Output: `normal` or `failing` (with a confidence score).

### Appropriate architectures

| Architecture | Params | Top-1 (ImageNet) | RPi latency | Size (INT8) | License |
|---|---|---|---|---|---|
| EfficientNet-Lite0 | 4.7M | 75.1% | ~49ms | 5.1MB | Apache 2.0 |
| MobileNetV3-Small | 2.5M | 67.9% | ~61ms | 2.9MB | Apache 2.0 |
| YOLOv11n-cls | 1.6M | 70.0% | ~10ms | 1.6MB | AGPL-3.0 |
| YOLOv8n-cls | 2.7M | 69.0% | ~13ms | 2.7MB | AGPL-3.0 |
| ShuffleNetV2 | 1.37M | 69.4% | ~30ms | 1.47MB | MIT |
| GhostNetV2 | 5.2M | 76.2% | ~25ms | 5.2MB | Apache 2.0 |

### Pros

- Simplest possible formulation. Minimal training data needed.
- YOLO-3DPrint-Monitor (YOLOv8n-cls) reports 99.7% accuracy with 1,884 test images.
- PrintGuard (ShuffleNetV2 + Prototypical Networks) achieves 0.937 F1 on Pi 4B.
- Fastest inference of any formulation — classifier runs at 2–5+ FPS on Pi 5 CPU.
- Naturally produces a confidence score suitable for temporal smoothing.
- Permissive-licensed options exist (EfficientNet-Lite0, GhostNetV2, ShuffleNetV2).

### Cons

- No spatial localization — cannot say *where* the failure is or *what type*.
- Less informative for debugging (user sees "failing" but not why).
- Binary output cannot distinguish spaghetti (urgent) from stringing (minor).
- All normal prints look "normal" regardless of quality.

### Verdict

**Best candidate for the primary MVP path.** Lowest data requirement, fastest inference,
proven on edge hardware, simplest to evaluate. PrintGuard's published benchmark (43x
faster than Obico, F1=0.937 on Pi 4B) demonstrates that a classifier approach is
viable in production.

---

## 2. Formulation B: Multi-class classification

### Definition

Input: a single camera frame.
Output: one of N classes, e.g.:
- `normal`
- `spaghetti`
- `layer_shift`
- `warping`
- `stringing`
- `blobbing`
- `adhesion_failure`
- `other_failure`

### Dataset class feasibility

| Dataset | Available classes | Total images | Balanced? |
|---|---|---|---|
| wengmhu FDM defect | 5 (cracking, layer_shifting, off_platform, stringing, warping) | 1,912 | Off_platform=91 (severe imbalance) |
| Roboflow/seprinder | 5 (blobbing, layer_shift, spaghetti, stringing, warping) | ~3,578 | Unverified |
| Roboflow/syLucauc | 6 (blobs, cracks, over_extrusion, spaghetti, stringing, under_extrusion) | ~5,793 | Unverified |
| klipper-pfd (CookiezRGood) | 4 (spaghetti, blob, cracks, warping) | Roboflow combined | mAP 0.502 |

**Key finding**: The wengmhu dataset has 1,912 images across 5 classes but only 91 images
for `off_platform`. This is insufficient for reliable per-class training of rare failure
modes. The Roboflow datasets are larger but have unverified provenance and per-class
balances.

Multi-class classification requires sufficient examples per class. With <100 examples
for some failure types, the model will have poor recall on rare classes — precisely the
failure modes that matter most.

### Pros

- More informative than binary — user learns *what* failed.
- Can prioritize response (spaghetti → pause immediately; stringing → warn).

### Cons

- Requires more training data per class.
- Severe class imbalance (spaghetti is common; adhesion failure is rare).
- Inference cost similar to binary classification (same architectures).
- May degrade overall recall if the model "spreads" probability across similar classes.

### Verdict

**Secondary path.** Worth exploring in Experiment 2 but not the primary MVP. Binary
classification first; multi-class expansion once a working baseline exists. The class
imbalance problem makes this harder than binary with the available data.

---

## 3. Formulation C: Object detection

### Definition

Input: a single camera frame.
Output: bounding boxes with class labels and confidence scores.

### Relevant models

| Model | mAP50-95 (COCO) | RPi5 latency | Size (INT8) | License |
|---|---|---|---|---|
| YOLOv11n | 39.5 | 56ms (640×640), 45ms (320×320) | 2.6MB | AGPL-3.0 |
| YOLOv8n | 37.3 | 80ms (640×640) | 3.2MB | AGPL-3.0 |

### Existing project results

| Project | Classes | mAP50 | Notes |
|---|---|---|---|
| klipper-pfd | 4 (spaghetti, blob, cracks, warping) | 0.502 | YOLOv8n → TFLite on Pi |
| Obico | spaghetti + general failure | Not published | Darknet YOLO, server-based |

### Does localization provide meaningful value for the MVP?

**Marginal value for the MVP, but useful for later.**

Arguments for:
- Shows the user *where* the failure is (educational, debugging).
- Can mask irrelevant regions (bed edge, background) to reduce false positives.
- Required for multi-failure-type detection in the same frame.

Arguments against:
- mAP 0.502 (klipper-pfd) is far below the binary classifier's 0.937 F1.
- Detection requires more training data (boxes + classes) than classification.
- Slower inference than classification (80ms vs 13ms for similar nano models).
- For the primary use case (pause on failure), knowing *where* is less important
  than knowing *whether*.

### Verdict

**Experiment 2, not Experiment 1.** Detection is useful but the binary classifier
achieves much higher reliability with less data. Detection becomes valuable when
multi-failure-type monitoring is needed or when spatial masking improves precision.

---

## 4. Formulation D: Anomaly detection

### Definition

Input: a single camera frame.
Output: an anomaly score (how different from "normal").

### Key models (Anomalib, Apache 2.0)

| Model | AUROC (MVTec) | GPU latency | CPU latency (RPi 4B) | Pi FPS |
|---|---|---|---|---|
| EfficientAD-S | 99.8% | <1ms | ~20–40ms (estimated, untested) | Unknown |
| PatchCore | 99.1% | ~50ms | ~52s | 0.019 |
| PaDiM | 97.9% | ~30ms | ~45s | 0.022 |

### Could training primarily on normal prints be advantageous?

**Yes, with important caveats.**

Advantages:
- Normal prints are trivial to capture (just print things).
- No failure injection needed during data collection.
- Catches novel/untrained failure modes (unknown unknowns).
- Few-shot friendly: 50–200 normal images can train a model (per SwitchOn/EthonAI).
- Anomalib (Apache 2.0) provides legally clear licensing.

Caveats:
- Normal variation in 3D printing is high: filament color, part geometry, lighting,
  printer model, camera angle all vary. Each change potentially creates false positives.
- Anomaly models may need retraining for each new print geometry/setup.
- PatchCore and PaDiM are far too slow for Pi (0.02 FPS).
- EfficientAD is the only candidate fast enough, but its Pi CPU latency is unverified.
- Anomaly detection provides "something is wrong" but not "what failed" — less actionable.

### Verdict

**Complementary, not primary.** Anomaly detection is valuable as a safety net (catching
failure modes the supervised classifier hasn't seen) but is not reliable enough as the
sole approach due to normal-variation sensitivity. Recommended as Experiment 3 after the
supervised baseline is established.

---

## 5. Formulation E: Temporal inference

### Definition

Aggregate per-frame predictions across time to produce a stable, low-false-positive
decision.

### Candidate approaches (from temporal-decision-logic.md)

| Approach | Cost | Key parameter | Used in |
|---|---|---|---|
| Moving average | O(1) time, O(N) mem | Window N | Construction detection (K≈50) |
| EWMA | O(1) / O(1) | Smoothing α | **Obico** (α=2/13), AWS safety |
| Consecutive-frame | O(1) / O(1) | Count N | Industrial inspection, Bambu Lab |
| Hysteresis | O(1) / O(1) | τ_entry, τ_exit | Work-zone (84.6% FP cut) |
| FSM | O(1) / O(1) | State set, guards | **Obico** (3-tier + warm-up) |

### Are RNNs/transformers justified?

**No.** The deployed leader (Obico) uses EWMA + rolling means + thresholds. Bambu Lab
uses 10-frame accumulation + spatial consistency. Neither uses a learned sequence model.
Published research shows no benefit of learned temporal models over well-tuned
statistical smoothing for the *decision* stage of a per-frame classifier.

### Recommended approach

**EWMA + baseline-relative threshold, wrapped in a minimal FSM with hysteresis and
fire-once semantics.** This is the Obico-proven combination, O(1) cost, trivially
explainable and testable.

Pipeline:
1. Per-frame score p ∈ [0,1]
2. EWMA: E_t = α·p_t + (1-α)·E_{t-1}, α ≈ 0.15
3. Baseline B = long rolling mean
4. FSM: OBSERVING → NORMAL → SUSPECT → FAILING → PAUSED
5. Hysteresis: enter FAILING above τ_high, exit below τ_low
6. Fire-once: pause notification emits once per print
7. Expose sensitivity as user setting

See `temporal-decision-logic.md` for full analysis.

### Verdict

**Essential for production, not an ML task.** Temporal decision logic is deterministic
code, not a trained model. It should be implemented as a simple state machine in the
C++ runtime. Parameters are tuned on held-out data, not learned.

---

## Recommended primary formulation

**Binary classification + temporal smoothing.**

Rationale:
1. Simplest formulation with highest demonstrated accuracy (F1=0.937 on Pi 4B).
2. Fastest inference (13–49ms on Pi 5 CPU).
3. Lowest data requirement (~500+ normal + ~100+ failure images).
4. Naturally produces confidence scores for temporal smoothing.
5. Proven in production by PrintGuard and YOLO-3DPrint-Monitor.

The MVP should:
- Start with binary classification (normal vs failing)
- Use a permissive-licensed architecture (EfficientNet-Lite0 or GhostNetV2) or
  YOLOv11n-cls (if AGPL is acceptable for training; AGPL-3.0 weights are fine for
  open-source distribution)
- Add temporal smoothing (EWMA + FSM) as a deterministic post-processing layer
- Expand to multi-class and/or detection in later experiments
