# Dataset Deep Dive: 3D-Print Failure Detection

**Status**: Research only.
**Date**: 2026-09-02

This document provides detailed analysis of every candidate dataset, with emphasis on
suitability for TinyVision's binary classification and detection tasks.

---

## 1. Caxton (University of Cambridge)

### Basic information

- **Paper**: "Generalisable 3D printing error detection and correction via multi-head
  neural networks" — Brion & Pattinson, Nature Communications, 2022
- **DOI**: 10.1038/s41467-022-31985-y
- **Repository**: https://github.com/cam-cambridge/caxton (MIT license, 64 stars)
- **Data**: https://doi.org/10.17863/CAM.84082
- **HuggingFace mirror**: https://huggingface.co/datasets/cemag/tl-caxton
- **License**: CC BY 4.0 (verified)
- **Image count**: 1,272,273 labeled images (948,396 after filtering)

### What the 1.27M images actually represent

Each image is a **single RGB frame** captured from a nozzle-mounted webcam during active
extrusion. They are independent snapshots captured every ~0.4 seconds during printing.
After each parameter change, 150 images are collected before the next change.

These are **in-situ process images** looking at the nozzle tip and most recently extruded
material — **not** post-print images of completed parts, **not** external webcam views.

### Camera and setup

- **Camera**: Logitech C270 (~$20 consumer webcam)
- **Placement**: Nozzle-mounted, attached next to the extruder on each printer
- **Printers**: 8 Creality CR-20 Pro units
- **Prints**: 192 parts with varied geometries, PLA colors, lighting conditions
- **Firmware**: Marlin 1.1.9

### Labels provided

**Continuous process parameters only — no discrete failure classes:**

| Label | Description |
|---|---|
| `flow_rate` | Relative percentage (optimal = 100%) |
| `feed_rate` | Lateral speed percentage |
| `z_offset` | Absolute mm offset |
| `target_hotend` | Requested temperature °C |
| `hotend` | Actual thermistor reading °C |
| `bed` | Bed temperature °C |
| `nozzle_tip_x/y` | Pixel coordinates of nozzle tip |
| `timestamp`, `print_id` | Identification |

For the paper's neural network task, continuous values are binned into 3 classes
(low/good/high) per parameter, yielding 81 possible combinations (3^4). **There are no
discrete failure-mode labels** (no "spaghetti", "stringing", "warping" classes).

### Filtering pipeline

| Stage | Count | What was removed |
|---|---|---|
| Raw | 1,272,273 | — |
| Complete failures removed | 1,166,552 (91.7%) | Images after print became unrecognizable |
| Gap removal | 1,072,500 | 15 images (~6s) after each parameter change |
| Outlier removal | 991,103 | Unrealistic parameter values from G-code glitches |
| Dark image removal | 946,283 (74.4%) | Mean RGB < 10 |

The most severe/obvious failures are **explicitly excluded**.

### Suitability for TinyVision

**NOT suitable as a primary training dataset** because:
1. Labels are continuous parameters, not failure classes.
2. Complete failures are removed (the easiest cases to detect).
3. No bounding boxes.
4. Nozzle-mounted camera only — no external webcam views.
5. PLA only — generalization to other materials unknown.
6. Single printer model, single camera model.

**Useful for:**
- Understanding the relationship between parameters and visual appearance.
- Pre-training feature extractors on nozzle-view images.
- Informing data collection strategy (camera placement, capture frequency).

### Resolution

Native: 1280×720 (720p). For training: cropped to 320×320 centered on nozzle tip,
then resized to 224×224.

---

## 2. wengmhu FDM 3D Printing Defect Dataset

### Basic information

- **Source**: https://github.com/wengmhu/FDM-3D-Printing-Defect-Dataset
- **Kaggle mirror**: https://www.kaggle.com/datasets/wengmhu/fdm-3d-printing-defect-dataset
- **License**: MIT (verified via GitHub LICENSE)
- **Image count**: 1,912

### Classes and balance

| Class | Count | Percentage |
|---|---|---|
| cracking | 427 | 22.3% |
| layer_shifting | 364 | 19.0% |
| off_platform | 91 | 4.8% |
| stringing | 447 | 23.4% |
| warping | 538 | 28.1% |

### Annotation format

Image classification by folder (defect type). No bounding boxes. Single class per image.

### Resolution

Not stated in the record.

### Camera/viewpoint

Not documented in detail. FFF/FDM printer defect images per the accompanying paper.

### Suitability for TinyVision

**Best commercially-licensed dataset for classification experiments.**

Strengths:
- MIT license is unambiguous for commercial use.
- 5 failure classes with reasonable counts.
- Clean folder-based organization.

Limitations:
- Only 1,912 images total — small by deep-learning standards.
- `off_platform` has only 91 images (severe class imbalance).
- No "normal" class (all images are failures). Need to add normal images.
- Classification only — no bounding boxes for detection training.
- Resolution and camera details not documented.

### Train/test leakage risk

Images are organized by folder. No information about whether images from the same
print are split across train/test. Must split by source/print, not randomly.

---

## 3. Roboflow/seprinder: detect-failure-in-3d-printing

### Basic information

- **URL**: https://universe.roboflow.com/seprinder/detect-failure-in-3d-printing
- **License**: CC BY 4.0 (as stated on page)
- **Image count**: ~3,578 (10 dataset versions)
- **Task**: Object detection (bounding boxes)

### Classes

| Class | Count |
|---|---|
| Blobbing | Unverified |
| Layer shift | Unverified |
| Spaghetti | Unverified |
| Stringing | Unverified |
| Warping | Unverified |

### Resolution

Not stated; Roboflow commonly resizes to 640×640 for training.

### Provenance concerns

**Community-contributed.** Underlying image provenance and original capture conditions
not published. The CC BY 4.0 license applies to the dataset as presented; the original
images may have different licensing.

### Suitability for TinyVision

**Usable with caution for detection experiments.** The class set is relevant (blobbing,
layer_shift, spaghetti, stringing, warping). CC BY 4.0 permits commercial use with
attribution. But provenance is unverified — cannot guarantee the underlying images are
freely licensed.

**Recommendation**: Use for training validation only. Do not redistribute. Record
provenance caveat in dataset records.

---

## 4. Roboflow/syLucauc: 3d-printing-failure-detection

### Basic information

- **URL**: https://universe.roboflow.com/sylucauc/3d-printing-failure-detection
- **License**: CC BY 4.0 (as stated)
- **Image count**: ~5,793 (augmented to ~13,905 in some versions)
- **Task**: Object detection (bounding boxes)

### Classes

| Class |
|---|
| blobs |
| cracks |
| over_extrusion |
| spaghetti |
| stringing |
| under_extrusion |

### Provenance concerns

Same as seprinder — community-contributed, unverified provenance.

### Suitability for TinyVision

**Similar to seprinder.** Larger image count and different class set (includes
over_extrusion and under_extrusion, which are relevant). Same provenance caveats apply.

---

## 5. Kaggle 3D-Printer Defected Dataset

### Basic information

- **URL**: https://www.kaggle.com/datasets/justin900429/3d-printer-defected-dataset
- **License**: **NOT VERIFIED**
- **Image count**: Not confirmed
- **Task**: Binary classification (Defected / Non-defected)

### Suitability for TinyVision

**NOT usable until license is verified.** The binary classification format is directly
relevant, but the unverified license makes this dataset unacceptable per TinyVision
policy.

---

## 6. HuggingFace Javiai: failures-3D-print

### Basic information

- **URL**: https://huggingface.co/datasets/Javiai/failures-3D-print
- **License**: **Unknown**
- **Image count**: 73
- **Task**: Object detection (bounding boxes)

### Classes

| Class | Notes |
|---|---|
| error | General error |
| extrusor | Extruder base |
| part | The printed part |
| spagheti | Spaghetti (note typo) |

### Suitability for TinyVision

**NOT usable — license unknown.** Also too small (73 images) to be useful for training.

---

## 7. Szydlo et al.: 3D-Printing-Data

### Basic information

- **URL**: https://github.com/joanna-/3D-Printing-Data
- **License**: CC BY 4.0 (verified)
- **Content**: Sensor timeseries data (accelerometer + tension), not images

### Suitability for TinyVision

**Not relevant** — this is a sensor dataset, not a vision dataset.

---

## 8. Kaggle 3D-Vis

### Basic information

- **URL**: Kaggle (adiparamartha)
- **License**: Kaggle Terms of Use
- **Image count**: 4,261
- **Task**: Classification (layer-wise fused deposition)

### Suitability for TinyVision

Classification format is relevant but details are limited. Kaggle terms require
individual verification for commercial use.

---

## 9. PrintGuard curated dataset collection

PrintGuard's companion repo (Edge-FDM-Fault-Detection, GPL-2.0) catalogs 7 datasets
with download links:

1. Roboflow: 3d-printer-failure (seprinder) — detection
2. Roboflow: fdm-failure (various) — detection
3. Roboflow: 3d-printing-failure-detection (syLucauc) — detection
4. Roboflow: all_anomalies (various) — detection
5. Kaggle: 3D-Printer Defected — binary classification
6. Kaggle: 3D-Vis — classification
7. Custom training set (not publicly released)

**This is the most comprehensive existing dataset catalog for this domain.**

---

## 10. HuggingFace ApatheticWithoutTheA: 3D-Print-Failure-Detector

### Basic information

- **URL**: https://huggingface.co/ApatheticWithoutTheA/3D-Print-Failure-Detector
- **Model**: YOLOv11s (small, not nano)
- **License**: MIT
- **Training data**: 9,000+ images
- **Classes**: 3 (spaghetti, stringing, zits)

### Suitability for TinyVision

The pretrained weights are MIT-licensed and could serve as a transfer learning starting
point. The model is YOLOv11s (larger than nano), so may need fine-tuning down to nano
for edge deployment.

---

## Dataset summary for TinyVision experiments

### For Experiment 1 (binary classification)

| Dataset | Role | License | Images | Notes |
|---|---|---|---|---|
| Own collection (normal prints) | Normal class | N/A | 200+ needed | Capture from target printer |
| Own collection (failure injection) | Failure class | N/A | 100+ needed | Induce failures deliberately |
| wengmhu FDM defect | Failure class supplement | MIT | 1,912 | 5 failure types, no normal class |
| Kaggle 3D-Printer Defected | If license verified | TBD | Unknown | Binary format matches directly |

### For Experiment 2 (multi-class detection)

| Dataset | Role | License | Images | Notes |
|---|---|---|---|---|
| Roboflow/seprinder | Detection training | CC BY 4.0 (provenance caution) | ~3,578 | 5 failure classes with boxes |
| Roboflow/syLucauc | Detection training | CC BY 4.0 (provenance caution) | ~5,793 | 6 failure classes with boxes |

### For Experiment 3 (anomaly detection)

| Dataset | Role | License | Images | Notes |
|---|---|---|---|---|
| Own collection (normal prints) | Normal-only training | N/A | 200–500 | Diverse colors, geometries, lighting |

---

## Critical dataset issues

1. **No dataset provides both normal and failure classes with clean licensing.** The
   wengmhu dataset has failures only. The Caxton dataset has no failure labels. The
   Roboflow datasets have provenance concerns.

2. **Own data collection is essential.** The MVP requires collecting normal + failure
   images from a specific printer with a specific camera. This is unavoidable.

3. **Video-derived dataset splitting is a major concern.** Multiple frames from the
   same print or video must not be randomly split across train/test. See
   `evaluation-plan.md` for the mitigation strategy.

4. **Caxton is not useful for classification/detection.** Despite its large size
   (1.27M images), the labels are continuous parameters, not failure classes. The
   most severe failures are explicitly removed. It is useful only for feature
   pre-training or parameter estimation research.
