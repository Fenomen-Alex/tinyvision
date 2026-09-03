# Problem Selection Research: 3D-Print Failure Detection for TinyVision

**Status**: Research only — no training, no dataset downloads, no implementation.
**Date**: 2026-09-02
**Companion data**: See [`datasets/`](./datasets/) for per-dataset provenance records.

This document evaluates whether 3D-print failure detection is the strongest first
project for TinyVision, an open-source edge computer-vision project. It compares the
candidate against four other problem areas (machine-state detection, agricultural
fruit monitoring, waste sorting, small-workshop visual inspection/safety). All specific
numbers are sourced from the cited references and were not fabricated.

---

## 1. Candidate problem

Real-time detection of visual failures in material-extrusion (FDM/FFF) 3D printing from
a fixed camera pointed at the build plate. Typical failure modes: **spaghetti** (print
detaches from bed, plastic tangles around nozzle), first-layer adhesion failure,
warping/de-lamination, stringing, blobbing, over/under-extrusion, layer shift, nozzle
clog. The system classifies the current print state and can pause/cancel the print to
save time and filament.

Two related but distinct tasks exist:
- **Image classification / anomaly scoring** (is the print OK or failing?): the dominant
  approach among commercial systems (Obico), because most consumer cameras are fixed and
  give a single stable view.
- **Object detection** (where in the frame is the defect?): used by several academic and
  open-source projects; more informative but harder datasets and more edge compute.

This is a *visual-only* monitoring problem — distinct from sensor-based failure detection
(runout sensors, nozzle-contact sensors), which detect a narrower set of failures.

---

## 2. Target users

Primary: **hobbyist and prosumer FDM 3D printer owners** using OctoPrint or Klipper
hosts — a large, motivated, technically literate community that is already comfortable
with webcams, Raspberry Pi, and open-source tooling. Long, unattended/overnight prints are
common, which makes failure detection valuable.

Secondary: small print farms / service bureaus (many printers monitored simultaneously),
makerspaces, educators. These value multi-camera support and automated alerts.

Evidence of community size: Obico alone reports **>100,000 users**, **>80–89.8M hours of
monitored printing**, **~1M detected failures** (their blog stakes: 80M hours / ~1M
failures; another post cites 89.8M hours / 1,067,608 failures as of writing). OctoPrint
has a mature plugin ecosystem (395 plugins as of June 2024, per Wikipedia). The 3D-printing
hobby market is substantial and growing.

---

## 3. Existing commercial solutions

| Solution | Method | Compatible | Price | Open source |
|---|---|---|---|---|
| **Obico** (formerly The Spaghetti Detective, since 2019) | AI webcam vision (cloud or self-hosted) | OctoPrint, Klipper, Bambu | Free tier; Pro ~$4/mo | Yes (AGPL) |
| **Obico Nozzle Ninja** | First-layer focused AI (dedicated nozzle cam) | Obico | — | Part of Obico |
| **OctoEverywhere Gadget** | AI webcam vision | OctoPrint, Klipper | $2.50–5/mo | No |
| **PrintWatch** | AI webcam vision (OctoPrint plugin) | OctoPrint | Free tier + paid | No |
| **Bambu Lab built-in** | LiDAR + camera + NPU | Bambu only | Included | No (closed) |
| **Creality K2 Pro / K1 Max** | AI cameras / LiDAR | Creality | Included | No (closed) |
| **SimplyPrint** | AI detection | OctoPrint/Multi | Free + paid | No |
| **QuinlyVision** | Multi-failure detection | Quinly (printer farm) | $29.99–39.99/printer/mo | No |
| **Hardware sensors (runout, nozzle contact)** | Non-visual | Any (DIY) | $5–15 | N/A |

Key takeaway: the market for *software* AI failure detection is already dominated by one
open-source incumbent (Obico) plus several closed commercial services. Hardware-embedded
AI (Bambu, Creality) is a closed-ecosystem trend.

---

## 4. Existing open-source projects

**Commercial/open-source incumbents**
- **Obico** (github.com/TheSpaghettiDetective/obico-server) — ~2.0k stars, AGPL-3.0.
  Full smart-printing platform; AI failure detection is its core. Cloud backend is
  commercial; self-hostable. This is the strongest existing open-source competitor.
- **OctoPrint-Obico**, **moonraker-obico** (Klipper) — companion plugins (AGPL).

**Community / academic open-source projects**
- **klipper-print-failure-detection** (CookiezRGood, ~29 stars) — TFLite YOLO model on
  Klipper host; detects spaghetti, blobs, cracks, warping; local-only; integrates with
  Moonraker to pause/cancel. *Edge-targeted.*
- **alexgrigoras/failure_detection_for_3d_printing** (MIT, ~22 stars, 2020) — CV module:
  image matching, object tracking, spaghetti detection. Not deep-learning.
- **joanna-/3D-Printing-Data** (CC-BY-4.0, ~26 stars) — sensor dataset, not a runtime.
- **bambu-studio-ai** (heyixuan2, MIT, ~91 stars) — full-stack Bambu agent skill incl.
  monitoring; not a focused failure-detection model.
- **Wartronick/3D-Printer-Failure-Detection** (MIT, ~0 stars) — Raspberry Pi 5, HOG +
  SVM/Random Forest, binary classify; 505-image custom dataset; reports 95% accuracy.
- **EmilyMyaskovski/3d-print-defect-detect** ("PrintGuard", ~0 stars) — Raspberry Pi 4/5,
  YOLOv11 ONNX (ARM-optimized), 640x480@15fps, FastAPI + WebSocket dashboard,
  Telegram alerts. *Edge-targeted.*
- **none-none-n/YOLO-3DPrint-Monitor** (1 star) — YOLOv8n-cls, CPU-only on-device,
  reports <250 ms latency, ~300 MB RAM; closed-loop UART/G-code control.
- **Pratyay1010/embedded-vision-3d-print-monitoring** — lightweight YOLO, dual camera,
  quantized edge deployment.
- **Abhi3886/3D-Printing_Failure_Detection**, **SayanSomya/Defect-Detection-in-3D-Printing**
  (YOLOv5/YOLOv11), **Kasej01/3DAiProject** (1 star, YOLOv8), **LiamLiuXY/** (Raspberry Pi +
  YOLO + Telegram) — mostly academic/demo repos with low activity.

**Pattern**: Many small, single-author, lightly-starred (0–30) academic projects; most use
**YOLO** (v5/v8/v11) or a classification network; several explicitly target edge/Raspberry
Pi deployment and ONNX/TFLite quantization. None approaches Obico in maturity or community.
There is a **gap for a focused, well-engineered, self-contained edge model + C++ runtime**
as distinct from Obico's full platform.

---

## 5. Public datasets

| Dataset | Task | Size | Labels | License | Variability |
|---|---|---|---|---|---|
| **Caxton (Cambridge)** | anomaly / regression | 1,272,273 imgs (948k filtered) | continuous process params, not boxes | CC BY 4.0 | nozzle-cam, 8 printers, 192 parts |
| **detect-failure-in-3d-printing (Roboflow/seprinder)** | object detection | ~3.6k imgs | 5 boxes | CC BY 4.0 | webcam-scale, community |
| **3d-printing-failure-detection (Roboflow/syLucauc)** | object detection | ~5.8k imgs | 6 boxes | CC BY 4.0 | community |
| **FDM-3D-Printing-Defect (wengmhu)** | classification | 1,912 imgs | 5 classes | MIT | printer photos |
| **3D-Vis (Kaggle/adiparamartha)** | classification | 4,261 imgs | multi-class | Kaggle TOU | layer-wise fused deposition |
| **3D-Printer Defected (Kaggle/justin900429)** | classification | unverified | binary | **not verified** | — |
| **failures-3D-print (HF/Javiai)** | object detection | 73 imgs | 4 boxes | **unknown** | nozzle/webcam |
| **3D-Printing-Data (Szydlo)** | sensor anomaly | (sensor, not images) | scenarios | CC BY 4.0 | IMU + JSON |

Plus a number of additional Kaggle sets and Roboflow Universe entries of varying size and
provenance (e.g. "3D printing defects" focusing on ok / bed-not-stick / spaghetti).
The "3D print failure detection" Roboflow model page reports ~4,549 images; the
"YOLO3DP/3D printing failure" page reports ~8,853 spaghetti images — these are aggregated
community counts and provenance is unclear.

---

## 6. Dataset size

There is **no single canonical, large, clean, commercially-safe, boxed 3D-printing-failure
dataset**. Sizes range from **73 images** (HF/Javiai) to ~**5.8k boxed images**
(Roboflow/syLucauc) for detection, to **~1.2M layer images** for the Caxton
parameter-labeled set (no discrete failure classes/boxes). The largest usable classification
sets are ~2k–4k images. For object detection, community datasets are a few thousand images
at most.

By deep-learning standards this is **small**. General fault/anomaly detection often needs
>10k–100k annotated samples, especially for a "normality" baseline covering many print
geometries, colors, and camera setups.

---

## 7. Dataset licensing

- **Caxton**: CC BY 4.0 — commercial OK with attribution. Verified via repository field.
- **wengmhu/FDM defect**: MIT — commercial OK. Verified via GitHub LICENSE.
- **Roboflow seprinder & syLucauc**: CC BY 4.0 as stated on the pages. CC BY permits
  commercial use. **Caveat**: provenance of underlying community-uploaded images is not
  documented; the stated license may not cover the upstream images. Use with caution.
- **Szydlo** sensor set: CC BY 4.0.
- **Kaggle datasets** (3D-Vis, 3D-Printer Defected, etc.): governed by Kaggle terms and the
  per-dataset license. **3D-Printer Defected license was not verified** during research.
- **HF/Javiai**: license **unknown** — not usable until resolved (per TinyVision policy).

**Net**: the strongest and clearly-usable image datasets for TinyVision are the MIT
classification set and (with provenance caution) the two CC BY 4.0 Roboflow detection sets.
No large boxed detection set has a fully verified end-to-end license.

---

## 8. Dataset annotations

- **Caxton**: rich per-image process-parameter labels (flow, speed, Z offset, temps,
  timestamps, nozzle x/y) — ideal for anomaly/regression and closed-loop control, but **not**
  discrete class labels or bounding boxes.
- **Roboflow detection sets**: YOLO-style bounding boxes over 5–6 defect classes
  (spaghetti, blobs, stringing, cracks, over/under-extrusion, warping, layer shift). These
  are quality-but-community annotations.
- **wengmhu**: folder-based classification (5 classes); no spatial labels.
- **HF/Javiai**: bounding boxes with 4 classes, but tiny (73 images) and class labels have
  typos/ambiguity.

Annotation quality is **inconsistent across sources**, which raises data-cleaning effort for
any multi-source fusion.

---

## 9. Dataset quality

The **Caxton** set is the highest-quality and most rigorously collected (automated pipeline,
8 printers, 192 parts, varied colors/lighting), and CC BY licensed — but it is a
nozzle-camera parameter set, best suited to anomaly detection rather than the fixed
webcam whole-part-view used by most consumer setups. The Roboflow sets are real-world
webcam-scale images (relevant distribution) but of unverified provenance and community
labeling quality. Classification sets are clean but single-class and small.

---

## 10. Dataset limitations

- No large, canonical, clean detection dataset with verified end-to-end licensing exists.
- **Domain shift**: failure appearance varies hugely with printer geometry, filament color,
  lighting, and camera angle — a major generalization risk for a small dataset.
- Failure events are **rare and sparse in time**; capturing enough failures to train a
  balanced classifier requires either many hours of prints or deliberate failure injection.
- Segmentation/pixel-level annotation is essentially absent.
- Several usable datasets have unknown/community provenance or unverified licensing.

---

## 11. Existing pretrained models

- Obico's proprietary detection model (AGPL-licensed platform code, but the trained model
  weights are not published as a standalone artifact; the ML is a core proprietary asset).
- Community YOLO weights: YOLOv8/YOLOv11 fine-tunes from the projects above (Kasej01
  `best.pt`, PrintGuard YOLOv11 ONNX, klipper-print-failure TFLite, HF Javiai
  `3dprintfails-yolo5vs`). Quality/latency are not rigorously benchmarked and most are
  single-author.
- Because the task is fine-tuning an off-the-shelf detector/classifier (YOLO n/s,
  EfficientNet, ResNet) on a small custom dataset, **strong pretrained backbones are
  abundant** (Ultralytics, Torchvision). Training a usable model is feasible; the hard part
  is data, not architecture.

---

## 12. Training difficulty

- **Ease of baseline**: Moderate-to-low. Fine-tuning YOLO or a small classifier on a few
  thousand images with a clean metric is straightforward with Ultralytics/Torchvision.
- **Hard parts**: (a) data scarcity and domain shift across printers/cameras/colors;
  (b) class imbalance (failures rare); (c) **false-negative cost** — missing a failure is
  worse than a false positive, requiring a tuned decision threshold on the recall side;
  (d) temporal consistency (a single noisy frame shouldn't trigger a pause).
- Overall: the **learning problem itself is not especially hard**; robustness and data are
  the challenge.

---

## 13. Expected edge inference requirements

Target platform: Raspberry Pi 4/5 (or similar ARM SBC) with a USB/CSI webcam, matching
how most hobbyists already run OctoPrint/Klipper.

- **Input**: webcam frame, ~640x480 or 640x640. Consumer cameras are residential/low-light
  (chamber lights matter; Bambu notes best results with chamber light on).
- **Latency**: a few frames per second to ~1 inference/sec is acceptable — failures develop
  over seconds-to-minutes, so sub-real-time inference is fine (no 30fps requirement).
  Reported references: PrintGuard runs 640x480@15fps restreaming with inference gated by a
  brightness check; YOLO-3DPrint-Monitor reports <250 ms CPU latency, ~300 MB RAM.
- **Model size**: YOLO-nano class (~4–6 MB weights) or an image classifier well under
  20 MB; quantized INT8 preferred. TFLite/ONNX on CPU.
- **Memory**: low hundreds of MB RAM total is realistic and desirable for an SBC.
- **Action**: outputs a failure score → trigger pause/cancel via OctoPrint/Moonraker API.

These requirements are comfortably met by current YOLO-nano/classifier models quantized on a
Pi 4/5; edge feasibility is well established (numerous Pi-based implementations exist).

---

## 14. Potential for collecting our own data

**High and cheap** — this is a strength of the domain.

- The hardware (a 3D printer + a webcam) is already in many hands; we can collect data
  on our own printers with minimal cost.
- Failures can be **intentionally induced** (bed adhesion failure, print without supports,
  clogged nozzle, over/under-extrusion, stringing via wrong retraction, etc.) to generate
  failure images on demand — as done by the wengmhu and Wartronick datasets.
- A webcam pointed at the plate produces an essentially unlimited stream of
  "normal/OK" timelapse frames during all ordinary successful prints — good for the
  normality class and for temporal context.
- Limitation: collecting *spatial* diversity (many printer geometries/colors) is slower;
  see synthetic-data option to broaden coverage.

**Verdict**: collecting our own labeled data is feasible and low-cost, especially if
combined with failure injection and community submissions.

---

## 15. Potential for synthetic data

**Good and under-exploited in practice.**

- Published work (Pearce/FAST group: "Towards Smart Monitored AM", Addit. Manuf. 2022;
  and the synthetic-to-real segmentation paper) shows **Blender physics-based rendering of
  G-code toolpaths** can generate ideal-process reference images, and synthetic-to-real style
  transfer / domain adaptation (cycleGAN) can improve real-image segmentation. Synthetic
  mIoU on a segmentation task reached ~95% (whole part) in that study.
- Toolpath/simulation approaches can generate large labeled datasets cheaply (contours,
  infill, shells, supports) without capturing a real failure.
- Limitation: sim-to-real gap for consumer cameras/lighting; synthetic data is easier for
  **structure** (layer boundaries, geometry) than for **photorealistic failures** (spaghetti
  texture, warping). A hybrid real + synthetic approach is the realistic path.
- Community practice is minimal; this is an open opportunity but not a solved shortcut.

---

## 16. Potential project differentiation

Challenges in standing out:

- **Obico** already offers an open-source, well-known AI failure-detection platform (AGPL)
  with ~2k stars and a huge user base. Directly competing head-on is hard.
- Several closed commercial tools and embedded (Bambu/Creality) options exist.

Differentiation opportunities:

1. **Focused edge model + C++ runtime** — TinyVision's stated direction. Most open-source
   efforts are Python-only and per-hardware; a clean, quantized, ONNX/C++ inference runtime
   that runs on a Pi with a documented latency/RAM budget is a genuine gap.
2. **Truly reproducible dataset + benchmark** — the field lacks a canonical, licensed,
   benchmarked failure-detection dataset with published precision/recall/latency. TinyVision
   could produce the missing "reference" in the space.
3. **Local-only default** — privacy-friendly, no-cloud, which is differentiated vs
   commercial SaaS (Obico cloud, OctoEverywhere).
4. **Multi-class detection** (not just spaghetti) with explicit false-negative-centric
   evaluation — aligns with TinyVision's evaluation principles.
5. **Community data collection pipeline** — a way to aggregate diverse printer/camera
   failure data under a clear license, since no good canonical set exists.

---

## 17. Estimated engineering effort

Assuming a focused MVP (not a full platform), in ascending order:

- **Baseline demo (small)**: collect a few hundred OK/fail images on our printer, fine-tune
  YOLO-nano classifier or YOLOv8n detection, run inference on a Pi, webcam + web viewer.
  Effort: **~2–4 person-weeks**.
- **Solid MVP**: robust data pipeline, failure injection, INT8/ONNX model, C++ inference
  runtime, OctoPrint/Klipper pause integration, a modest benchmark (precision/recall/latency/
  RAM). Effort: **~2–4 person-months**.
- **Reproducible dataset + benchmark + community contribution mechanism**:
  **+1–3 months**.
- **Feature-parity with Obico** (remote access, mobile app, multi-camera, first-layer
  monitoring): **significant** (a multi-quarter platform effort) — NOT recommended for the
  first milestone.

Recommended engineering scope for a first release is the "solid MVP" tier.

---

## 18. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Strong incumbent (Obico, AGPL) already exists | High | Focus on edge C++ runtime + reproducibility, not platform parity |
| No canonical licensed detector dataset | High | Explicit dataset-provenance policy; MIT/CC-BY sets; collect + possibly synthetic |
| Domain shift (printers/cameras/colors/lighting) | High | Robust eval on held-out real prints; document generalization honestly |
| False negatives are costly (missed failure = wasted print) | Medium | Recall-weighted eval; temporal smoothing; tunable threshold |
| Community dataset licensing uncertainty | Medium | Only use verified-license data; record provenance per TinyVision policy |
| Rare-failure class imbalance / data scarcity | Medium | Failure injection + synthetic augmentation + community submissions |
| Sim-to-real gap (if using synthetic) | Medium | Hybrid real+synthetic; validate on real held-out data |
| Scope creep toward full platform | Medium | Enforce the tight MVP scope |

---

## 19. Recommended MVP

A **minimal, edge-runtime-focused 3D-print failure monitor**:

1. **Target**: Raspberry Pi 4/5 + webcam, local-only inference, optional OctoPrint/Klipper
   hook to pause.
2. **Task**: image classifier (OK vs failure, with failure probability score) as the primary
   path — aligned with how commercial systems work and giving good edge efficiency; evaluate
   a YOLO-nano detector as a secondary/optional track for spatial info.
3. **Model**: fine-tuned lightweight model (e.g. YOLOv8n classification or detector),
   exported to INT8 ONNX/TFLite, measured on the Pi for latency (<~250 ms CPU), RAM
   (<~400 MB), and model size.
4. **Data**: combine the MIT-licensed classification set and (with provenance caution) the
   CC BY 4.0 Roboflow detection sets; supplement with our own camera capture + deliberate
   failure injection; record full provenance per TinyVision's dataset policy. Do not use
   unknown-license data.
5. **Eval**: hold out a separate real captured set (no overlap with training) and report
   precision/recall/F1/mAP, with **false-negative rate called out specifically**, plus
   latency/RAM/model-size on the actual edge hardware. No fabricated numbers.
6. **Runtime**: a small C++/ONNX Runtime-based inference binary per TinyVision's technical
   direction (training in Python; production inference in C++).

**Why this is a good first project**: it is visually observable and physically meaningful,
has very low data-capture cost, a large and eager user community, well-understood model
architectures, and modest, well-established edge compute requirements. The main
liabilities — a strong incumbent and no canonical licensed dataset — are mitigated by
differentiating on **edge-runtime quality and reproducibility** rather than platform
feature breadth, and by enforcing strict data provenance.

---

## Candidate comparison (summary)

| Dimension | 3D-print failure | Machine-state | Ag fruit monitor | Waste sorting | Workshop inspect/safety |
|---|---|---|---|---|---|
| Data availability | Poor but collectible cheaply | Poor | Good public sets | Good public sets | Fair/domain-specific |
| Edge feasibility | Good (small models) | Good | Good | Good | Good |
| Community/demand | Large, active | Small/niche | Medium (agtech) | Medium | Small/industrial |
| Commercial incumbents | Strong (Obico) | Low | Moderate (ag sensors) | Moderate | Low-mod |
| Differentiation room | Edge runtime + reproducibility | High | Medium | Medium | Medium |
| Licensing of public data | Mixed | Low | Mixed | Mixed | Low |

**Recommendation**: 3D-print failure detection is a strong first project **provided** the
MVP is scoped tightly around a high-quality edge runtime and a reproducible, licensed
evaluation benchmark (not a full-featured platform). Its unique strengths — very low data
cost, a large user base, and modest edge requirements — outweigh the incumbent-competition
risk when TinyVision differentiates on runtime quality and reproducibility. This selected
direction remains subject to validation during the implementation research phase as required
by the project brief.

### Explicit non-findings / unresolved items

- The exact license and image count of the Kaggle "3D-Printer Defected Dataset" could not be
  confirmed during research (page did not render fully). **Do not use until verified.**
- The HF "failures-3D-print" dataset license is **unknown** — not usable until resolved.
- Precise per-class image counts and the actual image resolution for the Roboflow datasets
  were not fully verified.
- Reported accuracy figures from community repos (e.g. "95% accuracy") are the authors'
  self-reported numbers on their own validation splits and were **not independently
  reproduced**; treat as unverified.
