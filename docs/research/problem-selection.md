# TinyVision Problem Selection Research

**Status**: Research only — no training, no dataset downloads, no implementation.
**Date**: 2026-09-02
**Companion data**: See [`datasets/`](./datasets/) for per-dataset provenance records.

This document evaluates five candidate problem areas for TinyVision's first project and
recommends an MVP. All specific numbers are sourced from cited references and were not
fabricated.

---

## Table of contents

1. [Evaluation criteria](#evaluation-criteria)
2. [Candidate 1: 3D-print failure detection](#1-3d-print-failure-detection)
3. [Candidate 2: Machine-state detection](#2-machine-state-detection)
4. [Candidate 3: Agricultural fruit monitoring](#3-agricultural-fruit-monitoring)
5. [Candidate 4: Waste sorting](#4-waste-sorting)
6. [Candidate 5: Small-workshop visual inspection/safety](#5-small-workshop-visual-inspectionsafety)
7. [Dataset comparison](#dataset-comparison)
8. [Licensing assessment](#licensing-assessment)
9. [Problem comparison matrix](#problem-comparison-matrix)
10. [Technical feasibility](#technical-feasibility)
11. [Recommended MVP](#recommended-mvp)
12. [Open questions](#open-questions)

---

## Evaluation criteria

Each candidate is evaluated against TinyVision's core constraints:

- **Local inference** is a first-class requirement (no cloud dependency).
- **Small models**, low RAM, CPU-friendly (Raspberry Pi class hardware).
- **Reproducibility** over clever experimentation.
- **Existing datasets** preferred; data collection must be cheap.
- **Licensing** must be compatible with open-source + commercial use.
- **Edge deployment** via C++/ONNX Runtime.
- **MVP scope** must be achievable in months, not years.

---

## 1. 3D-print failure detection

### Candidate problem

Real-time detection of visual failures in FDM/FFF 3D printing from a fixed camera.
Typical failures: spaghetti, first-layer adhesion failure, warping, stringing, blobbing,
layer shift, nozzle clog. Two related tasks: binary classification (OK vs failing) and
object detection (where is the defect?).

### Target users

Hobbyist and prosumer FDM 3D printer owners using OctoPrint or Klipper. Large, motivated,
technically literate community comfortable with webcams and Raspberry Pi. Secondary: small
print farms, makerspaces.

### Existing solutions

| Solution | Method | Price | Open source |
|---|---|---|---|
| Obico (formerly The Spaghetti Detective) | AI webcam vision (cloud or self-hosted) | Free tier; Pro ~$4/mo | Yes (AGPL) |
| OctoEverywhere Gadget | AI webcam vision | $2.50–5/mo | No |
| PrintWatch | AI webcam vision (OctoPrint plugin) | Free tier + paid | No |
| Bambu Lab built-in | LiDAR + camera + NPU | Included | No (closed) |
| Creality K2 Pro / K1 Max | AI cameras / LiDAR | Included | No (closed) |
| SimplyPrint | AI detection | Free + paid | No |
| QuinlyVision | Multi-failure detection | $29.99–39.99/printer/mo | No |

### Existing open-source projects

| Project | Stars | Architecture | Edge Support |
|---|---|---|---|
| Obico | ~2k | Proprietary model, AGPL platform | Partial (self-hosted) |
| klipper-print-failure-detection | ~29 | TFLite YOLO | Yes (Pi) |
| PrintGuard | ~0 | YOLOv11 ONNX | Yes (Pi 4/5) |
| YOLO-3DPrint-Monitor | ~1 | YOLOv8n-cls | Yes (CPU-only) |
| Various academic repos | 0–30 | YOLO v5/v8/v11 | Partial |

### Public datasets

| Dataset | Size | Task | License | Verified |
|---|---|---|---|---|
| Caxton (Cambridge) | 1,272,273 imgs | Anomaly/regression | CC BY 4.0 | Yes |
| Roboflow/seprinder | ~3.6k imgs | Object detection | CC BY 4.0 | Provenance unverified |
| Roboflow/syLucauc | ~5.8k imgs | Object detection | CC BY 4.0 | Provenance unverified |
| wengmhu FDM defect | 1,912 imgs | Classification (5 classes) | MIT | Yes |
| Kaggle 3D-Vis | 4,261 imgs | Classification | Kaggle TOU | Yes |
| HF/Javiai | 73 imgs | Object detection | **Unknown** | No |

**Dataset verdict**: No single canonical, large, clean, commercially-safe, boxed
3D-printing-failure dataset exists. Sizes range from 73 to ~5.8k boxed images for
detection. The largest usable classification sets are ~2k–4k images.

### Edge inference requirements

- Input: ~640×480 webcam frame
- Latency: sub-250ms CPU on Pi 4/5 is achievable (reported by multiple projects)
- Model size: YOLO-nano class (~4–6 MB quantized INT8)
- Memory: ~300–400 MB RAM
- FPS: 1–15 fps sufficient (failures develop over seconds-to-minutes)

### Strengths

- Very low data-capture cost (printer + webcam already exist; failures can be injected)
- Large, eager user community
- Well-understood model architectures (YOLO fine-tuning)
- Modest, well-established edge compute requirements
- Good potential for synthetic data (G-code toolpath rendering)

### Weaknesses

- Strong incumbent: Obico (AGPL, >100k users, >1M failures detected)
- No canonical licensed detection dataset with full provenance
- Domain shift across printers/cameras/colors/lighting
- Class imbalance (failures are rare)

### Differentiation opportunities

1. Focused edge C++ runtime (most OSS is Python-only)
2. Reproducible, licensed benchmark dataset
3. Local-only default (no cloud)
4. Community data collection pipeline

### Engineering effort

- Baseline demo: ~2–4 person-weeks
- Solid MVP: ~2–4 person-months
- Reproducible dataset + benchmark: +1–3 months

---

## 2. Machine-state detection

### Candidate problem

Using cameras to determine whether a machine (CNC, lathe, 3D printer, industrial
equipment) is running, idle, in error state, in setup, etc. This is distinct from defect
inspection (checking output quality) and predictive maintenance (forecasting failures from
sensor trends).

### Target users

Small workshop operators (CNC, lathe, mill), 3D print farm operators, makerspaces. Broader
than 3D-print failure detection — covers all machine types.

### Existing solutions

The majority of commercial "machine state" monitoring is **sensor-based** (vibration,
current, acoustic), not vision-based:

| Company | Approach | Notes |
|---|---|---|
| IPercept | Vibration/accelerometry on CNC | Claims 30% OEE improvement |
| Augury | Vibration + acoustic sensors | $6.4B market segment |
| Sight Machine | Manufacturing analytics platform | Cloud-dependent, enterprise |
| MachineMetrics | Deep CNC control integration | Pulls data from CNC controllers |
| Guidewheel | Power-draw clip-on sensors | Simple utilization tracking |
| AI-Innovate | AI2Eye (defect detection) | Vision-based, Jetson edge |

**Key finding**: Vision-based systems focus on product defect inspection, not machine state
classification. This is a meaningful gap.

### Existing open-source projects

| Project | Stars | License | Notes |
|---|---|---|---|
| Anomalib (Intel) | ~4.8k | Apache 2.0 | Anomaly detection library, OpenVINO export |
| Ultralytics YOLO | ~40k+ | AGPL-3.0 | General detection, widely used |
| Surface-Defect-Detection | ~4.1k | MIT | Dataset catalog, not runnable |

**No well-maintained open-source project exists for camera-based machine-state detection.**
Existing CNC monitoring projects use vibration/current signals.

### Public datasets

| Dataset | Size | Modality | License | Notes |
|---|---|---|---|---|
| MVTec AD | 5,354 imgs, 15 categories | Images | **CC BY-NC-SA 4.0** | Gold standard; non-commercial |
| MVTec AD 2 | 8,000+ imgs, 8 scenarios | Images | **CC BY-NC-SA 4.0** | Non-commercial |
| VisA (Amazon) | 10,821 imgs, 12 categories | Images | **CC BY 4.0** | Commercially usable |
| Bosch CNC | ~2700 instances | Vibration | CC-BY-4.0 | Not image data |
| NEU Surface Defect | 1,800 imgs, 6 types | Images | Unclear | Steel strip defects |
| KolektorSDD | 399 imgs | Images | CC BY-NC-SA 4.0 | Non-commercial |

**Critical finding**: The primary benchmark (MVTec AD) is **non-commercial only**. No
public image dataset exists for machine-state classification (running/idle/error). The
largest commercially-licensed image anomaly dataset is VisA (CC BY 4.0).

### Edge inference requirements

- YOLOv8n on Jetson Orin Nano: 286 FPS (TensorRT INT8) — far more than needed
- A lightweight classifier or optical flow may suffice for primary state detection
- Target: <100ms latency, <50MB model, <15W power

### Strengths

- Large, growing market ($6.4B → $22.8B by 2034)
- High differentiation potential (vision-based is a gap)
- Strong anomaly detection libraries (Anomalib, Apache 2.0)

### Weaknesses

- No public dataset specifically for machine-state classification
- Domain gap between MVTec/VisA images and real machine interiors
- Temporal aspect (single frame may be ambiguous)
- Camera placement inside machines is non-trivial (coolant, chips, heat)

### Engineering effort

- MVP (3D printer state monitor): ~4–6 months
- Multi-machine support: additional months

---

## 3. Agricultural fruit monitoring

### Candidate problem

Using cameras to monitor fruit growth, detect ripeness, count yields, and identify
disease. Encompasses fruit detection/counting, ripeness classification, and plant disease
detection.

### Target users

Orchard and greenhouse growers, precision agriculture operations, agricultural researchers.
Strong commercial validation ($2.1B market, 19.8% CAGR).

### Existing solutions

| Company | Approach | Notes |
|---|---|---|
| Tevel Aerobotics | Flying autonomous robots for fruit picking | Commercial deployments |
| Agrobot | 24 robotic arms for strawberry harvesting | Greenhouse + open field |
| Harvest CROO | Strawberry harvesting robot | Platform approach |
| Various drone services | Multi-spectral imaging for orchard health | DJI Agras, Sentera |

### Existing open-source projects

| Project | Stars | Architecture | Edge Support |
|---|---|---|---|
| AgML | 332 | PyTorch/TensorFlow | Partial |
| MinneApple | 128 | Mask R-CNN baselines | No |
| CitDet | 11 | YOLOv5/v7, DETR | No |
| Fruit Detection FOMO | 0 | Edge Impulse FOMO | Yes (ESP32-CAM) |

**No mature, actively maintained open-source fruit monitoring project exists.**

### Public datasets

| Dataset | Size | Fruit | License | Notes |
|---|---|---|---|---|
| MinneApple | 1,000 imgs, 41k instances | Apple | **CC BY-NC-SA 3.0 US** | Non-commercial |
| CitDet | 579 imgs, 32k bboxes | Citrus | **CC BY-NC-SA 4.0** | Non-commercial |
| PlantVillage | 54,306 imgs | 14 species | **CC BY 3.0** | Lab images, low res |
| Fruits-360 | ~90,000+ imgs | 131 classes | **MIT** | Studio images |
| Global Wheat Head | 4,948 imgs | Wheat | **CC BY 4.0** | Commercially usable |
| deepNIR | 4,295 imgs (RGB+NIR) | 11 classes | Check source | Hyperspectral |

### Edge inference requirements

- YOLOv5n on RPi 4 (INT8): 12.4 FPS, 240MB RAM
- SSD MobileNet on RPi 4 (INT8): 15.6 FPS, 180MB RAM
- 5–15 FPS acceptable for stationary monitoring

### Strengths

- Abundant public datasets (though many non-commercial)
- Well-understood problem
- Strong commercial validation
- Active research community

### Weaknesses

- Many datasets are **non-commercial** (MinneApple, CitDet, PlantVillage)
- Field conditions (lighting, occlusion, weather) create significant domain gap
- Seasonal nature of fruit makes data collection time-constrained
- Crop-specific — models don't generalize across fruit types
- Most existing work is research-focused, not production-ready edge systems

### Engineering effort

- Single-fruit MVP: ~3–6 months
- Multi-cruit support: significant additional effort

---

## 4. Waste sorting

### Candidate problem

Using cameras to identify and sort different types of waste (plastic, paper, metal,
organic, glass, e-waste, etc.) at the point of disposal. Targets smart bins, small
workshops, and community recycling stations.

### Target users

Municipalities, small workshops, community recycling stations, smart bin operators.
Strong regulatory tailwind: EU PPWR (effective Aug 2026), US state-level EPR (7 states
enacted), UK EPR (2026).

### Existing solutions

| Company | Approach | Notes |
|---|---|---|
| AMP Robotics | AI-powered MRF sorting | $91M+ raised, 160 picks/min |
| ZenRobotics (Terex) | Robotic waste sorting | 500+ categories, 15+ countries |
| TOMRA | Sensor-based sorting | Launched T100 (Feb 2026) |
| Machinex Industries | SamurAI technology | Deployed with AMP |
| Recycleye | AI vision for waste | Sorting robots |
| Bin-e | Smart bins | Consumer/office |

**Gap**: Industrial MRF solutions are dominated by large players. Small/edge/community-scale
is wide open.

### Existing open-source projects

| Project | Stars | Architecture | Edge Support |
|---|---|---|---|
| TrashNet | 711 | CNN (Lua/Torch) | No |
| TrashNet (TensorFlow) | 41 | MobileNet | Possible |
| YOLO-Waste-Detection | 7 | YOLOv8-nano | Yes |
| YOLOv8 Waste Detection | 7 | YOLOv8 | No |

Edge Impulse has a public YOLO11n Recycle Waste project targeting Raspberry Pi 5 with ONNX
and TFLite exports (int8 quantized).

### Public datasets

| Dataset | Size | Task | License | Notes |
|---|---|---|---|---|
| TrashNet | 2,527 imgs | Classification (6 classes) | **MIT** | White background only |
| TrashNet (Roboflow) | ~6,290 imgs | Object detection (19 classes) | **CC BY 4.0** | Community re-annotation |
| TrashNet++ | 2,600 imgs | Object detection (7 classes) | **CC BY 4.0** | Real-world Bangladesh |
| TrashNeXt | 23,625 imgs | Classification (9 classes) | **IEEE subscription** | Largest known open set |

### Edge inference requirements

- YOLOv8n quantized: <7MB, ~0.03s (~33 FPS)
- MobileNetV2 + TFLite: ~3–4MB, ~0.05s
- YOLO11n int8: ~5MB, <0.05s on RPi 5

### Strengths

- Very strong regulatory tailwind (EU PPWR, US EPR)
- Waste is free, abundant, and easily photographed (high data collection feasibility)
- Good edge feasibility (sub-10MB models, 30+ FPS)
- Large, active community (TrashNet 711 stars, multiple Kaggle competitions)
- Clear differentiation gap at small/edge scale

### Weaknesses

- Industrial MRF space is crowded (AMP, ZenRobotics, TOMRA)
- TrashNet is small (2,527 imgs) and lab-conditioned (white background)
- TrashNeXt (largest set) requires IEEE subscription
- Synthetic data for waste is limited
- Community dataset quality varies

### Engineering effort

- Classification MVP: ~2–3 months
- Object detection MVP: ~3–4 months

---

## 5. Small-workshop visual inspection/safety

### Candidate problem

Using cameras in small workshops (woodworking, metalworking, maker spaces) to detect
safety hazards, PPE compliance, tool misuse, hands-in-danger-zones, and quality issues.

### Target users

Small workshop operators, makerspaces, woodworking/metalworking hobbyists, OSHA
compliance officers. Niche but real need — Altendorf HAND GUARD (first BGHM-certified AI
camera for sliding table saws) proves the market exists.

### Existing solutions

| Company | Approach | Notes |
|---|---|---|
| Intenseye | Enterprise construction/logistics CCTV | On-prem GPU servers |
| Visionify | Vision-based safety monitoring | Enterprise focus |
| Tryave | Safety monitoring | Industrial |
| ISEE-CAM | Camera-based safety | Industrial |
| Altendorf HAND GUARD | AI camera for table saws | First BGHM-certified; proprietary |

### Existing open-source projects

Most PPE/industrial safety repos are 0–7 stars, single-contributor. Edge-capable
(YOLO→ONNX on Pi/Jetson) but Python-glued. No dominant open-source project.

### Public datasets

| Dataset | Size | Task | License | Notes |
|---|---|---|---|---|
| Ultralytics Construction-PPE | Varies | PPE detection | AGPL | Construction-focused |
| gsnvb | Varies | Tool detection | CC-BY-4.0 | |
| Mendeley PPE | Varies | PPE detection | CC-BY-4.0 | |
| I3D Tools | ~35k imgs | Tool detection | Check source | |
| Zenodo construction tools | 25k imgs | Tool detection | Check source | |
| RoboFusion | Varies | Synthetic | CC0 | Synthetic data |

**Critical gap**: No public dataset exists for the core hazard scenes — hands-in-danger-zones
around table saws/routers/spindles, guard-misuse.

### Edge inference requirements

- YOLOv8n/11n INT8 on RPi 5: ~23 FPS
- Jetson Orin Nano: tens to hundreds FPS
- Well within TinyVision's target specs

### Strengths

- Real safety need (proven by Altendorf commercial validation)
- Aligned with TinyVision's edge/local direction
- Reusable core infrastructure (person detection, machine-on/off, occupancy)
- Regulatory potential (though AI vision cannot be certified as a safety device per
  ISO 12100/13849)

### Weaknesses

- Core hazard data must be own-collected + synthesized (no public dataset exists)
- AI vision cannot be a certified safety device — only an "assistance system"
- High false-negative risk for rare unsafe events
- Small, fragmented community

### Engineering effort

- Core infrastructure (person + machine state): ~4–6 months
- Hazard-specific detection (hands near cutter): additional months + new data

---

## Dataset comparison

| Candidate | Best datasets | Total usable images | License quality | Data collection cost |
|---|---|---|---|---|
| **3D-print failure** | Caxton (1.27M), Roboflow (~9.4k), wengmhu (1.9k) | ~13k detection, 1.27M anomaly | Mixed (CC BY 4.0, MIT, unknown) | Very low (own printer) |
| **Machine-state** | VisA (10.8k), Anomalib benchmark | ~16k (but no state-specific) | Mixed (MVTec NC, VisA CC BY 4.0) | Moderate (need machines) |
| **Ag fruit** | PlantVillage (54k), Fruits-360 (90k), Global Wheat (5k) | ~150k+ | Mixed (many NC: MinneApple, CitDet) | Seasonal, location-dependent |
| **Waste sorting** | TrashNet (2.5k), TrashNet++ (2.6k), TrashNeXt (23.6k) | ~30k+ | Good (MIT, CC BY 4.0) | Very low (trash is free) |
| **Workshop safety** | Construction-PPE, I3D Tools, RoboFusion | ~60k+ (but no hazard data) | Mixed (AGPL, CC-BY-4.0, CC0) | High (must photograph hazards) |

### Winner: Waste sorting (best dataset landscape)

Waste sorting has the best combination of large, commercially-licensed datasets and
low-cost data collection. 3D-print failure detection has the Caxton set (1.27M images)
but it lacks discrete failure class labels. Agricultural datasets are large but many
are non-commercial.

---

## Licensing assessment

### Commercially compatible datasets (per TinyVision policy)

| Candidate | Datasets with clear commercial license | Risk level |
|---|---|---|
| **3D-print failure** | Caxton (CC BY 4.0), wengmhu (MIT), Roboflow (CC BY 4.0, provenance caution) | Medium |
| **Machine-state** | VisA (CC BY 4.0), Bosch CNC vibration (CC-BY-4.0) | Medium (MVTec is NC) |
| **Ag fruit** | Global Wheat (CC BY 4.0), Fruits-360 (MIT) | Medium (many NC sets) |
| **Waste sorting** | TrashNet (MIT), TrashNet++ (CC BY 4.0), TrashNet Roboflow (CC BY 4.0) | Low |
| **Workshop safety** | gsnvb (CC-BY-4.0), Mendeley PPE (CC-BY-4.0), RoboFusion (CC0), Zenodo tools (check) | Low-Medium |

### License risk summary

- **3D-print failure**: Roboflow datasets have unverified provenance — the stated CC BY 4.0
  may not cover underlying images. HF/Javiai license unknown. Kaggle datasets require
  individual verification.
- **Machine-state**: MVTec AD (the gold standard benchmark) is non-commercial. Only VisA
  is clearly commercially usable.
- **Ag fruit**: MinneApple (CC BY-NC-SA 3.0), CitDet (CC BY-NC-SA 4.0) are non-commercial.
  PlantVillage (CC BY 3.0) is commercial but lab-quality images.
- **Waste sorting**: TrashNet (MIT) is the cleanest licensing story. TrashNet++ (CC BY 4.0)
  is also clean.
- **Workshop safety**: Good licensing for construction PPE datasets. No hazard-specific data.

**License winner**: Waste sorting (TrashNet MIT license is unambiguous).

---

## Problem comparison matrix

| Dimension | 3D-print failure | Machine-state | Ag fruit | Waste sorting | Workshop safety |
|---|---|---|---|---|---|
| **Data availability** | Poor (but cheap to collect) | Poor (no state-specific sets) | Good (many large sets) | Good (TrashNet + variants) | Fair (no hazard data) |
| **License quality** | Medium (provenance risk) | Low (MVTec NC) | Medium (many NC) | **High (MIT, CC BY)** | Medium |
| **Edge feasibility** | **High** (small models proven) | **High** (low FPS needed) | High | **High** (sub-10MB models) | High |
| **Community demand** | **High** (>100k Obico users) | Low-Medium (niche) | Medium (agtech) | **High** (regulatory-driven) | Small (industrial) |
| **Commercial incumbents** | **Strong** (Obico) | Low | Moderate | Moderate (MRF scale) | Low-Medium |
| **Differentiation room** | Medium (edge runtime) | **High** (vision gap) | Medium | **High** (small-scale gap) | Medium |
| **Training difficulty** | Low-Medium | Medium-Hard | Moderate | **Low-Medium** | Medium |
| **Data collection cost** | **Very low** | Moderate | Seasonal/location | **Very low** | High |
| **Regulatory tailwind** | Low | Medium | Medium | **Very strong** (EPR) | Medium |
| **MVP timeline** | 2–4 months | 4–6 months | 3–6 months | **2–3 months** | 4–6 months |
| **Synthetic data potential** | Good | Moderate | Good (Blender) | Moderate | Good (RoboFusion) |

---

## Technical feasibility

All five candidates are technically feasible within TinyVision's target architecture:

| Component | Technology | Status |
|---|---|---|
| Training | PyTorch + Ultralytics YOLO | Well-established for all candidates |
| Export | ONNX, INT8 quantization | Standard workflow |
| Inference runtime | C++ + ONNX Runtime | Production-ready |
| Target hardware | Raspberry Pi 5, Jetson Orin Nano | Verified for all candidates |
| Camera | USB/CSI webcam | Compatible with all |

**Edge inference benchmarks** (from published research):

| Hardware | YOLOv8n FPS | Model Size | RAM | Suitable candidates |
|---|---|---|---|---|
| RPi 5 (TFLite) | 2–5 FPS | <10MB | <500MB | All (1 FPS sufficient for most) |
| Jetson Orin Nano (TRT INT8) | 286 FPS | <10MB | <1GB | All (overkill for most) |

**Recommendation**: Start with YOLOv8n or YOLOv11n quantized to INT8 ONNX. This model
class works for all five candidates with comfortable performance margins.

---

## Recommended MVP

### Primary recommendation: 3D-print failure detection

**Rationale** (in order of importance):

1. **Lowest data-capture cost**: Failures can be intentionally induced on an existing
   printer. No seasonal constraints. No special hardware.
2. **Largest, most accessible user community**: >100k users in the 3D-printing hobby space,
   already comfortable with Raspberry Pi and open-source tooling.
3. **Proven edge feasibility**: Multiple Pi-based implementations exist (PrintGuard,
   klipper-print-failure-detection, YOLO-3DPrint-Monitor).
4. **Clearest differentiation path**: Edge C++ runtime + reproducible benchmark is a
   genuine gap. Obico is a platform; TinyVision can be a focused runtime.
5. **Modest MVP scope**: Binary classifier + optional detector, ONNX on Pi, OctoPrint
   pause hook — achievable in 2–4 months.

**Why not waste sorting** (which scored higher on datasets and licensing)?
Waste sorting is a strong runner-up and should be considered as a second project. However,
3D-print failure detection has a more clearly defined user community, lower data-collection
friction, and a more achievable MVP because the monitoring setup (fixed camera + Pi) is
already standard in the target community. Waste sorting also faces more commercial
competition at the industrial MRF scale, and the small-scale/smart-bin market is less
proven.

### Recommended MVP specification

1. **Target**: Raspberry Pi 4/5 + USB/CSI webcam, local-only inference, optional
   OctoPrint/Klipper integration to pause on failure.
2. **Task**: Primary path is binary classification (OK vs failing, with probability score).
   Secondary optional track: YOLO-nano detector for spatial failure localization.
3. **Model**: Fine-tuned YOLOv8n or YOLOv11n, exported to INT8 ONNX/TFLite. Measure on
   actual Pi hardware: latency (<250ms), RAM (<400MB), model size (<10MB).
4. **Data**:
   - Start with wengmhu FDM defect (MIT, 1,912 classification images)
   - Supplement with Roboflow detection sets (CC BY 4.0, provenance caution)
   - Collect own data: webcam capture of normal prints + deliberate failure injection
   - Record full provenance per TinyVision dataset policy
   - Do not use unknown-license data (HF/Javiai, unverified Kaggle)
5. **Evaluation**:
   - Hold-out real-captured test set (no overlap with training)
   - Report: precision, recall, F1, mAP, **false-negative rate** (called out specifically)
   - Report: latency, RAM, model size on actual edge hardware
   - No fabricated numbers
6. **Runtime**: Small C++/ONNX Runtime inference binary (training in Python, production in C++)
7. **Scope boundary**: Do NOT attempt feature parity with Obico (remote access, mobile app,
   multi-camera, first-layer monitoring). The MVP is a focused local failure monitor.

### Why this MVP works for TinyVision

- Aligns with core principles: local inference, small models, reproducibility
- Demonstrates the full TinyVision stack: Python training → C++ runtime → edge deployment
- Produces a reusable benchmark methodology applicable to future problem areas
- Generates community-trainable dataset infrastructure
- Is achievable within months, not years

---

## Open questions

### Must resolve before implementation

1. **Kaggle "3D-Printer Defected Dataset" license**: Could not be verified during
   research. Do not use until verified.
2. **HF "failures-3D-print" license**: Unknown — not usable until resolved.
3. **Roboflow dataset provenance**: The CC BY 4.0 license stated on Roboflow pages may not
   cover underlying community-uploaded images. Verify before any redistribution.
4. **Obico model weights**: Are Obico's trained model weights available as a standalone
   artifact, or only embedded in the AGPL platform? If available, can they serve as a
   pre-trained starting point?
5. **Failure injection methodology**: What specific failure modes are most valuable to
   detect? (Spaghetti is obvious; stringing, under-extrusion, and subtle layer issues
   are harder.)

### Must resolve during implementation

6. **Threshold tuning for false-negative minimization**: What decision threshold achieves
   <1% false-negative rate without excessive false positives?
7. **Temporal smoothing**: How many consecutive frames must indicate failure before
   triggering a pause? Single-frame triggers risk false alarms.
8. **Camera placement best practices**: What angle, distance, and lighting produce the
   most reliable detection?
9. **OctoPrint/Klipper API integration details**: What hooks are available for pause/cancel?
10. **Quantization accuracy degradation**: What is the actual accuracy loss from FP32 → INT8
    on Pi hardware? Measure, do not assume.

### Future consideration (not blocking MVP)

11. **Waste sorting as second project**: Given strong dataset landscape and regulatory
    tailwind, waste sorting is the recommended follow-up after 3D-print failure detection.
12. **Machine-state detection as expansion**: Once a 3D-print state classifier exists,
    expanding to multi-machine state detection is a natural evolution.
13. **Hardware sensor fusion**: Could vision + vibration/acoustic data improve detection
    reliability? Not for the MVP, but worth exploring.
14. **Community data pipeline**: A mechanism for community-contributed failure data under
    clear licensing would address the dataset gap.

---

## Appendix: Source references

- Obico blog: >100k users, >80M hours monitored, ~1M failures detected
- DWaste paper (arXiv:2510.18513): YOLOv8n-CBAM <7MB, ~0.03s inference
- Anomalib: github.com/open-edge-platform/anomalib (~4.8k stars, Apache 2.0)
- TrashNet: github.com/garythung/trashnet (711 stars, MIT)
- AgML: github.com/Project-AgML/AgML (332 stars, Apache 2.0)
- MVTec AD: CC BY-NC-SA 4.0 (non-commercial)
- VisA: CC BY 4.0 (commercially usable)
- EU PPWR: Regulation (EU) 2025/40, effective August 12, 2026
- US EPR: 7 states enacted (Maine, Oregon, California, Colorado, Minnesota, Maryland, Washington)
- Altendorf HAND GUARD: First BGHM-certified AI camera for sliding table saws
- AI predictive maintenance market: $6.4B (2025) → $22.8B (2034)
- CV in agriculture market: $2.1B (2025), 19.8% CAGR
