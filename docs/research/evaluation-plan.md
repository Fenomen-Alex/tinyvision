# Evaluation Plan for 3D-Print Failure Detection

**Status**: Research only.
**Date**: 2026-09-02

This document defines the experiment matrix, evaluation methodology, and temporal decision
logic for TinyVision's first ML experiments.

---

## 1. Experiment matrix

### Experiment 1: Binary classifier + existing/collected dataset

| Field | Value |
|---|---|
| **Task** | Binary classification (normal vs failing) |
| **Dataset** | wengmhu FDM defect (1,912 failure images, MIT) + own-collected normal images (200+). Split by source, not randomly. |
| **Model A** | EfficientNet-Lite0 (Apache 2.0) |
| **Model B** | YOLOv11n-cls (AGPL-3.0) |
| **Model C** | ShuffleNetV2 (MIT) — PrintGuard baseline |
| **Input size** | 224×224 (center crop from 640×480 webcam frame) |
| **Training** | Transfer learning from ImageNet. 50 epochs, Adam optimizer, lr=1e-4, batch=32. FP32 on GPU, export to INT8 ONNX. |
| **Expected training** | ~1–2 hours on single GPU (RTX 3060 class) |
| **Evaluation metrics** | Precision, recall, F1, false-negative rate, false-positive rate, AUROC, confusion matrix |
| **Edge metrics** | Model size, INT8 accuracy degradation, CPU latency on RPi5, RAM consumption, FPS |
| **Purpose** | Establish baseline binary classifier performance |

### Experiment 2: Lightweight detector + existing dataset

| Field | Value |
|---|---|
| **Task** | Object detection (bounding boxes for failure types) |
| **Dataset** | Roboflow/seprinder (~3,578 images, CC BY 4.0) + Roboflow/syLucauc (~5,793 images, CC BY 4.0). Both with provenance caveat. |
| **Model** | YOLOv11n (AGPL-3.0) |
| **Input size** | 320×320 |
| **Training** | Fine-tune from COCO pretrained. 100 epochs, lr=0.001, batch=16, augmentation (mosaic, mixup, HSV, flip). Export INT8 ONNX. |
| **Expected training** | ~2–4 hours on single GPU |
| **Evaluation metrics** | mAP50, mAP50-95, per-class AP, precision-recall curve, false-negative rate per class, detection latency |
| **Edge metrics** | Model size, CPU latency on RPi5 (320 vs 640), RAM, FPS |
| **Purpose** | Determine whether localization adds value; compare detection vs classification |

### Experiment 3: Anomaly detector + normal-print data

| Field | Value |
|---|---|
| **Task** | Anomaly detection (normal vs abnormal, unsupervised) |
| **Dataset** | Own-collected normal prints only (200–500 images from target printer, diverse colors/geometries) |
| **Model A** | EfficientAD-S (Anomalib, Apache 2.0) |
| **Model B** | Autoencoder baseline (TFLite, ~3MB) |
| **Input size** | 256×256 (EfficientAD), 224×224 (autoencoder) |
| **Training** | EfficientAD: ~30 min on GPU. Autoencoder: ~1 hour. Normal images only — no failure examples. |
| **Expected training** | ~1 hour total |
| **Evaluation metrics** | AUROC (using held-out failures as anomaly class), false-positive rate at 95% recall, detection latency |
| **Edge metrics** | CPU latency on RPi5 (critical — must measure), RAM, FPS |
| **Purpose** | Validate anomaly detection as safety net; benchmark EfficientAD on Pi |

### Experiment 4: Temporal smoothing evaluation

| Field | Value |
|---|---|
| **Task** | Evaluate temporal decision logic on per-frame model outputs |
| **Input** | Frame-by-frame predictions from Experiment 1 or 2 model on video sequences |
| **Approaches** | (a) No smoothing, (b) Moving average (N=10, 30, 50), (c) EWMA (α=0.10, 0.15, 0.20, 0.30), (d) Consecutive-frame (K=3, 5, 10), (e) EWMA + FSM + hysteresis |
| **Dataset** | Video sequences of real prints with known failure timestamps |
| **Evaluation metrics** | Detection delay (seconds from first failure frame to trigger), false-positive rate (per hour), false-negative rate, trigger stability (no chattering) |
| **Purpose** | Tune temporal parameters; validate EWMA+FSM approach; compare against simpler baselines |

---

## 2. Evaluation methodology

### Avoiding data leakage

**The critical constraint**: multiple frames from the same print or video must NOT be
randomly distributed across train and test sets. This causes severe data leakage because
consecutive frames are nearly identical.

#### Split strategy

```
Source-level split:
  - Each print/video is assigned entirely to train, val, or test
  - No frame from a print appears in multiple splits
  - Stratify by class to maintain balance
```

#### For video-derived datasets

1. **Identify source prints/videos**: Group frames by their originating print session.
2. **Assign whole sources to splits**: 70% of prints → train, 15% → val, 15% → test.
3. **Never split a print across splits**: Even if it has only a few frames.

#### For the wengmhu dataset

The wengmhu dataset is organized by folder (defect type). Without knowing which images
come from the same print, random splitting is the only option — but this may introduce
mild leakage if multiple images of the same print are in different splits. Document this
as a known limitation.

#### For own-collected data

1. **Capture full video sequences** of each print (not individual frames).
2. **Split at the print level**: Print 1–N for training, Print N+1–M for validation,
   Print M+1–K for test.
3. **Include diverse conditions**: Different colors, geometries, lighting in each split.

### Final test set requirements

The **final test set** must contain:
- Real captures from a webcam pointed at a real 3D printer.
- Not used during any training or hyperparameter tuning.
- Include both normal and failure sequences.
- Include prints with filament colors, geometries, and lighting conditions not seen
  during training (to test generalization).
- At least 30 minutes of normal printing and at least 5 distinct failure events.

### Metrics to report

#### Classification metrics

| Metric | Formula | Why it matters |
|---|---|---|
| **Precision** | TP / (TP + FP) | Low precision = too many false alarms = users disable the system |
| **Recall** | TP / (TP + FN) | Low recall = missed failures = wasted filament |
| **F1** | 2·P·R / (P + R) | Balanced measure |
| **False-negative rate** | FN / (FN + TP) | **Primary concern** — missed failures are costly |
| **False-positive rate** | FP / (FP + TN) | Secondary concern — unnecessary pauses annoy users |
| **AUROC** | Area under ROC curve | Threshold-independent performance |

#### Detection metrics

| Metric | Why it matters |
|---|---|
| mAP50 | Standard detection benchmark |
| mAP50-95 | Stricter localization quality |
| Per-class AP | Identifies which failure types are hardest |

#### Edge deployment metrics

| Metric | Target | How to measure |
|---|---|---|
| Model size | <10MB | File size of INT8 ONNX model |
| INT8 accuracy drop | <2% from FP32 | Compare FP32 vs INT8 on same test set |
| CPU latency | <250ms per frame | Time single-frame inference on RPi5 CPU |
| RAM | <400MB total | RSS during inference |
| FPS | >1 FPS (classification), >0.5 FPS (detection) | Sustained throughput on RPi5 |
| Power | <10W sustained | Measure with USB power meter |

### False-negative vs false-positive tradeoff

For the failure-monitoring use case:

- **False negative** = missed failure → wasted filament + print time. Cost: $1–50+ per
  event depending on print size and material.
- **False positive** = unnecessary pause → user must resume. Cost: annoyance + brief
  interruption.

The operating point should be biased toward **high recall** (low false-negative rate)
at the cost of somewhat higher false-positive rate. Users can adjust sensitivity to
find their preferred operating point.

**Target**: <5% false-negative rate at a false-positive rate of <2 per hour of printing.

The temporal smoothing layer (Experiment 4) is the primary tool for controlling
false-positive rate without sacrificing recall.

---

## 3. Temporal decision logic

### State machine design

```
                    ┌─────────────────────┐
                    │     OBSERVING       │
                    │  (warm-up period)   │
                    │  suppress all       │
                    │  alarms             │
                    └──────────┬──────────┘
                               │ warm-up complete
                               ▼
                    ┌─────────────────────┐
          ┌────────│       NORMAL        │
          │        │  (print proceeding   │
          │        │   without issues)    │
          │        └──────────┬──────────┘
          │                   │ E_t - B > τ_low
          │                   ▼
          │        ┌─────────────────────┐
          │   ┌───▶│      SUSPECT        │
          │   │    │  (possible failure   │
          │   │    │   developing)        │
          │   │    └──────────┬──────────┘
          │   │               │ E_t - B > τ_high
          │   │               │  AND K-of-N
          │   │               │  consecutive
          │   ▼               ▼
          │  ┌─────────────────────┐
          │  │      FAILING        │
          │  │  (failure confirmed │
          │  │   → emit PAUSE)     │
          │  └──────────┬──────────┘
          │             │ print ended or
          │             │ manual reset
          │             ▼
          │  ┌─────────────────────┐
          └──│      PAUSED         │
             │  (waiting for user  │
             │   or print restart) │
             └─────────────────────┘
```

### Pipeline steps

1. **Per-frame model output**: classifier/detector produces confidence score p ∈ [0,1].

2. **EWMA smoothing**:
   ```
   E_t = α * p_t + (1 - α) * E_{t-1}
   ```
   Starting value: α = 0.15 (Obico's 2/13).
   Effect: recent frames weighted more; isolated blips suppressed.

3. **Baseline reference**:
   ```
   B = rolling mean of p over last W frames
   ```
   Starting value: W = 1800 (5 minutes at 6 FPS).
   Purpose: "this moment is worse than the recent normal baseline."

4. **Score difference**:
   ```
   D_t = E_t - B
   ```
   Positive D_t means current score exceeds the baseline (potential failure).

5. **State transitions**:

   | From | To | Condition |
   |---|---|---|
   | OBSERVING | NORMAL | Warm-up period elapsed (first 2–5 min) |
   | NORMAL | SUSPECT | D_t > τ_low (starting: 0.15) |
   | SUSPECT | NORMAL | D_t < τ_low for M consecutive frames (M=5) |
   | SUSPECT | FAILING | D_t > τ_high (starting: 0.35) for K-of-N frames (K=3, N=5) |
   | FAILING | PAUSED | Transition (fire-once pause event) |
   | FAILING | NORMAL | D_t < τ_exit (τ_low, with hysteresis gap) for L frames |
   | PAUSED | NORMAL | Print ended or manual reset |

6. **Hysteresis**: Enter FAILING at τ_high, exit at τ_low (hysteresis band = τ_high - τ_low).
   Prevents chattering near the decision boundary.

7. **Fire-once**: Pause notification emits on exactly one transition into FAILING.
   Repeated high scores do not re-trigger the pause.

8. **Sensitivity control**: Expose a user-facing sensitivity slider that scales
   τ_high and τ_low proportionally. Higher sensitivity = lower thresholds = more
   false positives but fewer false negatives.

### Starting parameter values

| Parameter | Starting value | Rationale |
|---|---|---|
| α (EWMA smoothing) | 0.15 | Obico's documented value |
| B window (baseline) | 1800 frames | ~5 min at 6 FPS (Obico uses ~5 min) |
| τ_low (SUSPECT entry) | 0.15 | Conservative; tune on data |
| τ_high (FAILING entry) | 0.35 | Above normal variation; tune on data |
| K-of-N (confirmation) | 3 of 5 | ~0.8s at 6 FPS; kills isolated blips |
| M (SUSPECT→NORMAL reset) | 5 | ~0.8s of below-threshold to clear suspicion |
| Warm-up duration | 180 frames | ~30s at 6 FPS; tune on data |

**All values are starting points.** They must be tuned on held-out video data and
recorded per the experiment policy. Do not claim these are validated until Experiment 4
is executed.

### What this is NOT

- This is not a learned model. It is deterministic code.
- It does not use RNNs, LSTMs, or transformers.
- It has O(1) time and O(1) memory per frame.
- It is implementable in ~100 lines of C++.

---

## 4. Experiment execution order

```
Experiment 1 (binary classification)
    │
    ├── Establishes baseline model performance
    ├── Produces per-frame confidence scores
    │
    ▼
Experiment 4 (temporal smoothing)
    │
    ├── Uses Experiment 1 model outputs on video data
    ├── Tunes temporal parameters
    ├── Establishes end-to-end detection performance
    │
    ▼
Experiment 2 (detection)
    │
    ├── Compares detection vs classification
    ├── Determines if localization adds value
    │
    ▼
Experiment 3 (anomaly detection)
    │
    ├── Benchmarks EfficientAD on Pi
    ├── Evaluates as safety net
    └── Explores hybrid approaches
```

Experiment 1 and 4 can partially overlap (collect video data while training the
classifier). Experiment 3 can run in parallel with Experiment 2 once the binary
classifier baseline is established.

---

## 5. What we will NOT claim

Per TinyVision's principles:

- We will not fabricate benchmark results.
- We will not claim accuracy without specifying the test set and conditions.
- We will not report metrics on training data.
- We will not compare against Obico without a controlled evaluation.
- We will not claim edge performance without measuring on actual hardware.
- We will not claim temporal parameters are validated until Experiment 4 is executed.
