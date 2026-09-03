# Workshop Safety & Inspection — Dataset Catalog

**Retrieval date:** 2026-09-02 (all sources accessed on this date; no datasets downloaded).
**Companion to:** `docs/research/workshop-safety-detection.md`.
**Policy note (AGENTS.md):** datasets below are RECURRING/DIRECT entries. No dataset may be incorporated until its license status is resolved and this record is completed with all required fields. Unknown or incompatible licensing → dataset excluded.

Legend — Usability:
- ✅ **Use** = license verified compatible with an open, distributable TinyVision.
- ⚠️ **Verify** = license ambiguous, not yet confirmed, or terms restrictive; must be resolved before use.
- ❌ **Exclude** = non-commercial / share-alike / DTA-restricted, or otherwise incompatible.

---

## A. PPE / construction-safety detection (reusable base classes)

### A1. Ultralytics Construction-PPE
- **Source:** Ultralytics (docs.ultralytics.com/datasets/detect/construction-ppe); config in `ultralytics/ultralytics` GitHub.
- **URL:** https://docs.ultralytics.com/datasets/detect/construction-ppe
- **License:** AGPL-3.0 (copyleft — attribute; consider implications if weights/derivatives redistributed).
- **Image count:** 1,416 (train 1,132 / val 143 / test 141); 11,521 reviewed bounding boxes.
- **Annotation format:** YOLO `.txt`.
- **Classes (11):** helmet, gloves, vest, boots, goggles, none, Person, no_helmet, no_goggle, no_gloves, no_boots.
- **Resolution info:** ~178–145 MB download (dataset-level size reported).
- **Collection methodology:** real construction environments, compliant + non-compliant cases.
- **Known limitations:** construction-heavy; worn-vs-missing PPE; small-object imbalance.
- **Intended usage:** object detection training/eval for workplace PPE monitoring.
- **Usability:** ✅ / ⚠️ (AGPL caveat).

### A2. Construction-Safety-GSNVB (RF100)
- **Source:** Roboflow Universe `roboflow-100/construction-safety-gsnvb`; HF mirror `LibreYOLO/construction-safety-gsnvb`.
- **URL:** https://huggingface.co/datasets/LibreYOLO/construction-safety-gsnvb
- **License:** CC-BY-4.0.
- **Image count:** 1,206 split (997 train / 119 val / 90 test, validated locally per IlliaSator repo).
- **Annotation format:** YOLO `.txt`.
- **Classes:** helmet, no-helmet, vest, no-vest, person.
- **Usability:** ✅.

### A3. SH17 — Human Safety & PPE Detection (Manufacturing)
- **Source:** Kaggle `mugheesahmad/sh17-dataset-for-ppe-detection`; GitHub `ahmadmughees/SH17dataset`.
- **URL:** https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection
- **License:** CC BY-NC-SA 4.0 (non-commercial + share-alike). Pexels source imagery (Pexels license per-download).
- **Image count:** 8,099 images; 75,994 instances; 17 classes; avg 9.38 instances/image.
- **Annotation format:** image/XML-level; high-res (1920×1002 to 8192×5462); many small objects (<5% area).
- **Known limitations:** NC/SA restricts commercial + redistribution; web-scraped (Pexels).
- **Usability:** ❌ for commercial TinyVision; ⚠️ research/eval only.

### A4. PPE_Detection (HF, 6-class)
- **Source:** Hugging Face `51ddhesh/PPE_Detection`.
- **URL:** https://huggingface.co/datasets/51ddhesh/PPE_Detection
- **License:** listed as research/education-oriented — **verify exact SPDX before use.**
- **Image count:** object counts (16,014 objects) not image count; classes: vest, safety shoe, mask, helmet, goggles, gloves.
- **Annotation format:** YOLO `.txt` (train/valid/test + data.yaml).
- **Usability:** ⚠️.

### A5. PPE_Detection_V1 (Kaggle, 6-class)
- **Source:** Kaggle `beyzakucuk/ppe-detection-v1` (Roboflow `ppe-6-classes`).
- **URL:** https://www.kaggle.com/datasets/beyzakucuk/ppe-detection-v1
- **License:** page states CC0 (Public Domain) but also shows MIT contributors — **conflicting; verify authoritative license.**
- **Image count:** not published (m/boxes), 6 classes; JPG; YOLO txt; train/valid/test.
- **Classes:** Helmet, No-Helmet, Goggles, No-Goggles, Vest, Person.
- **Usability:** ⚠️.

### A6. PPE Dataset — Mendeley ZKZGHJVPN2
- **Source:** Mendeley Data.
- **URL:** https://data.mendeley.com/datasets/zkzghjvpn2/6 (DOI 10.17632/zkzghjvpn2.6)
- **License:** CC BY 4.0.
- **Image count:** 2,286 images, 640×640.
- **Annotation format:** YOLO (helmet + reflective vest).
- **Collection:** GitHub/Kaggle/Roboflow + real-life photos.
- **Usability:** ✅.

### A7. PPE Dataset — Data in Brief (Huang & Cheng 2025)
- **Source:** Data in Brief / PMC.
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC12398887/ (DOI 10.1016/j.dib.2025.111988)
- **License:** CC BY 4.0 (open access article).
- **Image count:** 2,286.
- **Annotation format:** YOLO (helmet, reflective vest).
- **Usability:** ✅.

### A8. R2PPE (medical PPE surveillance)
- **Source:** Scientific Data (Yang et al. 2025), Zenodo.
- **URL:** https://www.nature.com/articles/s41597-024-04355-0
- **License:** CC BY 4.0.
- **Image count:** 10,034 images (26 videos); 123,751 boxes; 17 PPE adherence/non-adherence classes.
- **Annotation format:** Darwin 2.0 JSON → COCO JSON.
- **Relevance:** healthcare-oriented; illustrates dense annotation, not workshop. ✅.
- **Note:** not workshop-relevant; listed for completeness.

### A9. CPPE-5 (medical PPE)
- **Source:** Hugging Face `merve/cppe-5`; GitHub `cppe-5`.
- **License:** per-source (Flickr CC images; dataset CC-BY-4.0 per paper) — **verify.**
- **Image count:** 1,029 (1,000 train / 29 test); 4,698 boxes.
- **Classes:** Coverall, Face_Shield, Gloves, Goggles, Mask.
- **Annotation format:** COCO-style bbox.
- **Usability:** ⚠️ (medical domain).

---

## B. Tool / hand-tool detection (closest to "small workshop")

### B1. I3D Tools Dataset
- **Source:** Hugging Face `i3dlabiisc/I3D-Tools-Dataset`; GitHub `i3dlabiisc/I3D-Tools-Dataset`.
- **URL:** https://huggingface.co/datasets/i3dlabiisc/I3D-Tools-Dataset
- **License:** paper-affiliated — **verify SPDX before use.**
- **Image count:** ~35,000 @1024×1024.
- **16 classes:** ball_bearing, gear, hammer, measuring_tape, nail, nut, oring, plier, saw, scissors, screw, screwdriver, spring, utility_knife, washer, wrench.
- **Annotation format:** YOLOv8 boxes + pixel segmentation masks + natural-language captions.
- **Usability:** ⚠️ (verify) — strong candidate if license is permissive.

### B2. Small Construction Tools (Zenodo 6530106)
- **Source:** Zenodo.
- **URL:** https://zenodo.org/records/6530106
- **License:** Zenodo record — **verify SPDX.**
- **Image count:** 25,084 images; 25% from real construction site; occlusion/lighting/resolution variation controlled.
- **12 classes:** bucket, cutter, drill, grinder, hammer, knife, saw, shovel, spanner, tacker, trowel, wrench.
- **Annotation format:** YOLO-style `.txt` (class, cx, cy, w, h normalized).
- **Peer-reviewed:** Lee, Jeon, Shin, KSCE J. Civil Eng, DOI 10.1007/s12205-022-1011-7.
- **Usability:** ⚠️ (verify) — real-site subset relevant.
- **Note:** file listed as 109.5 GB (large) — not to be downloaded per current task.

### B3. Tipwei Tool-Detection-Dataset
- **Source:** Hugging Face `tipwei/tool-detection-dataset`.
- **URL:** https://huggingface.co/datasets/tipwei/tool-detection-dataset
- **License:** **verify.**
- **Image count:** small (lab/workshop).
- **7 classes:** screwdriver, breadboard, hex_key, plier, jumper_wire_bundle, wrench, soldering_iron.
- **Annotation format:** YOLO `.txt` + YOLO-World labels.
- **Usability:** ⚠️ (small; verify).

### B4. Electrical Tools Object Detection
- **Source:** Kaggle `venkatesh2410/od-imglables`; GitHub `VK-Venkatesh/electrical-tools-object-detection`.
- **URL:** https://www.kaggle.com/datasets/venkatesh2410/od-imglables
- **License:** **verify** (Google-scraped source raises provenance concern).
- **Image count:** ~500 images; 10 classes.
- **Annotation format:** YOLO `.txt` (70/10/20 split).
- **Usability:** ⚠️ (small, scraped provenance).

### B5. Anthropic-friendly? (n/a) — Ultralytics Platform tool sets
- **Source:** platform.ultralytics.com — `6-tool-datasetyolov11` (25,261 imgs), `toolsv5iyolov8-obb`, `tools`, `mechanical-tools-2`, `industry-tools`.
- **License:** platform terms + per-uploader — **verify each.**
- **Image count:** 488 – 25,261 images; hand + power tools; YOLO / OBB formats.
- **Usability:** ⚠️ (verify per-dataset).

### B6. Automotive Tools Dataset for Computer Vision
- **Source:** Kaggle `katerynabocharova/automotive-tools-dataset-for-computer-vision` (preview; full **$149 commercial purchase** via Gumroad).
- **License:** commercial paid for full set; preview only (11 imgs) public.
- **Image count:** 230 full / 11 preview.
- **Classes:** ratchets, sockets, pliers, screwdrivers, hammer, shears, chisels, PPE goggles/gloves, etc. CVAT-annotated boxes.
- **Usability:** ⚠️ (preview only free; full set paid).

---

## C. Proximity / hazard / human-machine interaction (construction & logistics focus)

### C1. PM-HMCW (proximity monitoring)
- **Source:** GitHub `dyxm/PM-HMCW`; data via TeraBox/Baidu (code `wnp5`).
- **License:** **verify** (paper EAAI 2025).
- **Image count:** 22,500 virtual + real construction images; 3D bbox annotations; proximity categories (Dangerous/Potentially Dangerous/Concerned/Safe).
- **Usability:** ⚠️ (construction domain; verify license/access).

### C2. Forklift-Person-Detector
- **Source:** Roboflow Universe `ai-ulrbv/forklift-person-detector-ljqes`.
- **License:** Roboflow Universe terms — **verify.**
- **Classes:** forklift, person.
- **Usability:** ⚠️ (logistics domain).

### C3. Multispectral Forklift Pedestrian (MDPI Sensors 22-07953)
- **URL:** https://mdpi-res.com/.../sensors-22-07953.pdf
- **License:** data available **on request, non-profit only** → ❌ for commercial.
- **Image count:** 19.4k frames / 20.2k pedestrians; RGB+thermal aligned; occlusion labels.
- **Usability:** ❌ (non-profit only).

### C4. LiHRA (LiDAR HRI risk monitoring)
- **Source:** https://proximityrobotics.github.io/LiHRA/
- **License:** research-only, **not for commercial use** → ❌ for commercial.
- **Image count:** 4,431 labeled point clouds @10 Hz; 6 HRI scenarios (safe/hazard versions).
- **Usability:** ❌ (research-only).

### C5. Human-Robot Collaboration thermal — WLRI-HRC
- **Source:** Zenodo record 14228565.
- **URL:** https://zenodo.org/records/14228565
- **License:** scientific / **non-commercial** (per statement) → ❌ for commercial; research OK.
- **Image count:** 6,485 images / 20,000+ instances; YOLO/COCO/VOC formats.
- **Usability:** ❌ (non-commercial) / ⚠️ research.

---

## D. Gesture recognition for safety (related, niche)
- **HaGRID / HaGRIDv2** — github `hukenovs/hagrid`; CC BY-SA 4.0 variant; ~65,977 subjects; consumer gestures; ⚠️ (SA caveat, not workshop-specific).
- **SHREC'17 hand gestures** — 28 dynamic industrial gestures, depth cameras; public/research; ⚠️.
- **MuViH** — Polytechnique Montréal Borealis; RGB-D cobot hand gesture; **custom DTA (institutional agreement)** → ❌ (not freely redistributable).
- **EgoNRG** — UT Austin TDR DOI 10.18738/t8/dc4j0q; 3,044 videos + 160,639 segmented frames; military/industrial arm gestures; download via TDR API key; ⚠️ verify terms.
- **NUGGET** — whole-body HRI gestures, intralogistics; 11,569 sequences; public/research; ⚠️.
- **RoCoG-v2** — robotic gestures; real (482) + synthetic (106,996); public; ⚠️.

---

## E. Occupancy / person detection (reusable, mature)
- **COCO person** — universal base; CC-BY-4.0-derived; ✅ base.
- **DaRA** (Fraunhofer IML warehousing HAR) — Zenodo 10468175; research; ⚠️.
- **FROG / DROW** — 2D-lidar person detection; research/robotics; ⚠️.
- **Grid-EYE occupancy** — IEEE DataPort 10.21227/0pwe-g126; 8×8 thermal; privacy-preserving; ⚠️ (thermal, small).
- **Sixth Sense indoor spatial awareness** — Zenodo 14936069; RGB-D/LiDAR; research; ⚠️.
- **Person Detection on Construction Sites** (BAuA/BMAS funded) — Zenodo 14884208; German federal occupational-safety institute; verify license; ⚠️.

---

## F. Synthetic / hybrid data (de-risking for missing hazard scenes)
- **RoboFusion** — GitHub `Amr-Khamis-84/RoboFusion-Dataset`; hybrid real+synthetic industrial hazard detection for AMRs; **CC0**; ✅ (strong precedent for our synthetic approach; not workshop-scene itself).
- **PM-HMCW virtual subset** — © construction domain; ⚠️ (see C1).
- **RoCoG-v2 synthetic** — robotic gestures; ✓ synthetic precedent.
- **Altendorf HAND GUARD data** — proprietary; **not public**; ❌ (validates need for our own collection).

---

## Summary of usable (license-clean) anchors
- ✅ **CC-BY-4.0:** A2 gsnvb, A6 Mendeley PPE, A7 Data-in-Brief PPE, A8 R2PPE (medical), F1 RoboFusion (CC0).
- ✅/⚠️ **AGPL-3.0:** A1 Ultralytics Construction-PPE, thalos-forklift checkpoint.
- ⚠️ **Verify before use (license not confirmed):** A4, A5, A9, B1, B2, B3, B4, B5, B6, C1, C2, gesture sets, occupancy sets.
- ❌ **Exclude for commercial TinyVision:** A3 SH17 (NC-SA), C3 (non-profit only), C4 LiHRA (research-only), C5 WLRI-HRC (non-commercial), MuViH (DTA).
- **No public dataset covers** the small-workshop hazard scenes (hands-in-danger-zones, guard misuse around table saws/routers/spindles/drill presses) — **this gap must be filled by our own collection + synthesis.**
