# Machine-State Detection for TinyVision

Research date: 2026-09-02

## 1. Problem Definition

Machine-state detection uses cameras (and optionally other sensors) to determine whether a machine—CNC, lathe, 3D printer, industrial equipment—is in a specific operating state: running, idle, in error, in setup, producing good parts, producing defective parts, etc.

This is a subproblem of the broader industrial monitoring and predictive maintenance space. It is closely related to but distinct from:
- **Anomaly detection**: detecting any deviation from normal, without necessarily classifying the state
- **Defect detection**: inspecting the output product for quality issues
- **Predictive maintenance**: forecasting when a component will fail based on sensor trends

## 2. Existing Commercial Solutions

### Sensor-based (not primarily vision)

| Company | Approach | Notes |
|---------|----------|-------|
| **IPercept** (ipercept.io) | Vibration/accelerometry sensors on CNC machines. Edge analytics. | Claims 30% OEE improvement, 50% reduction in unplanned downtime. Stockholm-based. Works with any CNC brand. Sensor-based, not camera-based. |
| **Augury** (augury.com) | Vibration + acoustic sensors. AI-based machine health. | $6.4B market segment (2025). 1.1B+ hours of machine data. Covers motors, pumps, compressors, CNC. Sensor-based. |
| **Sight Machine** (sightmachine.com) | Manufacturing analytics platform. Semantic model + agentic AI. | 20+ industries. Deploys in days. Primarily data-driven, not vision-focused. |
| **MachineMetrics** | Deep CNC control integration. Real-time monitoring. | Pulls data directly from CNC controllers. Strong for modern CNC machines. |
| **Guidewheel** | Power-draw clip-on sensors. Detects running/idle/off states. | Universal compatibility, fast install. Not vision-based. |
| **Amper** | Clip-on current sensors. Uptime/downtime tracking. | Simple install, basic utilization focus. |

### Vision-based commercial

| Company | Approach | Notes |
|---------|----------|-------|
| **AI-Innovate** (ai-innovate.com) | AI2Eye (defect detection), AIxCore (NVIDIA Jetson edge inference), AICam (synthetic data). | Full stack: vision models + edge hardware. Targets manufacturing QC. |
| **iFactory AI** (ifactoryapp.com) | YOLOv8-based industrial defect detection. TensorRT INT8 on Jetson. | Production-ready edge deployment. Custom model fine-tuning. |
| **3DPrinterOS** | Cloud platform for 3D printer monitoring with AI failure detection (spaghetti detection, layer analysis). | Primarily for 3D print farms. Camera-based. |
| **MVTec** (mvtec.com) | HALCON/MERLIC machine vision software. Industrial anomaly detection. | German company. Industry standard for machine vision. Sells commercial software, provides research datasets. |

### Key observation
The majority of commercial "machine state" monitoring solutions are **sensor-based** (vibration, current, acoustic), not vision-based. Vision-based systems tend to focus on **product quality inspection** (defects in the output) rather than **machine state classification** (is the machine running, idle, errored, etc.). This is a meaningful gap.

## 3. Existing Open-Source Projects

### Anomaly Detection / Industrial Vision Libraries

| Project | URL | Stars | License | Edge Support | Notes |
|---------|-----|-------|---------|--------------|-------|
| **Anomalib** (Intel) | github.com/open-edge-platform/anomalib | ~4.8k | Apache 2.0 | OpenVINO export, ONNX | Largest public collection of anomaly detection algorithms. PatchCore, EfficientAD, SimpleNet, etc. Supports MVTec AD, VisA datasets natively. Anomalib Studio for no-code deployment. |
| **Ultralytics YOLOv8/v11** | github.com/ultralytics/ultralytics | ~40k+ | AGPL-3.0 (commercial license available) | TensorRT, ONNX, TFLite, CoreML, Edge TPU | General object detection, widely used for industrial defect detection. |
| **Surface-Defect-Detection** (Charmve) | github.com/Charmve/Surface-Defect-Detection | ~4.1k | MIT | N/A | Curated list of datasets and papers. Not a runnable system. |
| **Awesome-Industrial-Vision-Datasets** | github.com/SkyShunsuke/Awesome-Industrial-Vision-Datasets | ~10 | MIT | N/A | Dataset catalog (50+ datasets). Work-in-progress. |
| **Awesome-Industrial-Datasets** (jonathanwvd) | github.com/jonathanwvd/awesome-industrial-datasets | N/A | N/A | N/A | Comprehensive catalog of industrial datasets including time-series, audio, and vision. |
| **OpenPnP** | openpnp.org | N/A | BSD-3 | No | Open-source pick-and-place with vision alignment. For electronics manufacturing. |

### CNC-Specific Open-Source

| Project | URL | License | Notes |
|---------|-----|---------|-------|
| **Bosch CNC Machining Dataset** | github.com/boschresearch/CNC_Machining | BSD-3-Clause (code), CC-BY-4.0 (data) | Vibration data from 3 CNC milling machines over 2 years. Time-series, not images. 142 stars. |
| **cnc-operatingstate-lstm** | github.com/stephenseacuello/cnc-operatingstate-lstm | N/A | BiLSTM-Attention for CNC operating state classification using inertial vibration data. University of Rhode Island thesis. |
| **KTH Thesis: CNC Operating State ID** | kth.diva-portal.org/smash/record.jsf?pid=diva2:1900938 | N/A | Decision Tree classifier using triaxial vibration data for CNC state identification (idle, air-cutting, cutting). 5-axis CNC machine. Master's thesis 2024. |

### Key observation
There is **no well-maintained open-source project** specifically for camera-based machine-state detection. The existing CNC monitoring projects use vibration/current signals. The vision-based anomaly detection libraries (Anomalib, YOLO) could be applied to this problem but are not purpose-built for state classification.

## 4. Public Datasets

### Directly Relevant (Machine State / CNC Monitoring)

| Dataset | Source | Size | Modality | License | Notes |
|---------|--------|------|----------|---------|-------|
| **Bosch CNC Machining** | github.com/boschresearch/CNC_Machining | ~2700 instances, 3 machines, 15 processes | Vibration (accelerometer, 2kHz) | CC-BY-4.0 (data), BSD-3 (code) | Real brownfield CNC milling. Normal + anomalous labels. 6 timeframes over 2 years. Available on UCI ML Repository. |
| **CNC Milling Tool Dataset (Piecuch 2025)** | nature.com/articles/s41597-025-04923y | 968 milling cycles, 14 tools | Vibration + current | Open access (Nature Scientific Data) | Full tool lifecycle from initial state to failure. Haas VF-1 + Beckhoff PAC. Available on Kaggle. |
| **Milling Wear** | Kaggle | 16 cases, 167 measurement runs | Vibration, time-series | Information not available | CNC/machining. RUL and condition monitoring. |

### Anomaly Detection (Image-Based, Industrial)

| Dataset | Source | Size | License | Annotations | Notes |
|---------|--------|------|---------|-------------|-------|
| **MVTec AD** | mvtec.com/research-teaching/datasets/mvtec-ad | 5,354 images, 15 categories | **CC BY-NC-SA 4.0** (non-commercial only) | Pixel-precise anomaly masks | Gold standard benchmark. Categories: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper. |
| **MVTec AD 2** | mvtec.com | 8,000+ images, 8 new scenarios | **CC BY-NC-SA 4.0** (non-commercial only) | Pixel-precise (public split); hidden (server) | Expands MVTec AD with varying lighting. Evaluation via public server. |
| **MVTec 3D-AD** | mvtec.com | 4,000+ 3D scans, 10 categories | **CC BY-NC-SA 4.0** (non-commercial only) | 3D anomaly annotations | Industrial 3D sensor data. |
| **MVTec LOCO AD** | mvtec.com | 5 object categories | **CC BY-NC-SA 4.0** (non-commercial only) | Pixel-precise | Logical + structural anomalies. E.g., objects in wrong positions. |
| **VisA (Visual Anomaly)** | Amazon Science / AWS Open Data | 10,821 images (9,621 normal, 1,200 anomalous), 12 categories | **CC BY 4.0** (commercially usable) | Image-level + pixel-level masks | Largest visual anomaly dataset. PCB inspection + multi-instance scenes. Hosted on AWS S3. |
| **NEU Surface Defect (NEU-DET)** | Northeastern University | 1,800 grayscale images, 6 defect types, 200x200px | **License unclear** (academic use assumed) | Bounding box + class labels | Hot-rolled steel strip defects: cracks, inclusions, patches, pitted surface, rolled-in scale, scratches. |
| **KolektorSDD** | vicos.si/resources/kolektorsdd/ | 399 images (52 defective, 347 clean) | **CC BY-NC-SA 4.0** (non-commercial; contact for commercial) | Pixel-level segmentation masks | Electrical commutator surface defects. Small dataset. |
| **KolektorSDD2** | vicos.si | Larger than KolektorSDD | **CC BY-SA 4.0** (commercially usable) | Box annotations | Improved version with bounding box annotations. |
| **GC10-DET** | Kaggle | 3,570 images | N/A | Pixel-level + image-level anomaly | Textures, metal, mechanics defects. |

### License Compatibility Summary

| License | Commercial Use? | TinyVision Compatible? |
|---------|-----------------|----------------------|
| CC BY 4.0 | Yes | Yes |
| CC BY-SA 4.0 | Yes (with share-alike) | Yes (with caution) |
| CC BY-NC-SA 4.0 | **No** | No (unless purely internal R&D) |
| CC-BY-4.0 | Yes | Yes |
| BSD-3-Clause | Yes | Yes |
| MIT | Yes | Yes |

**Critical finding**: MVTec AD (the primary benchmark) and MVTec AD 2 are **non-commercial only**. The largest commercially-licensed image anomaly dataset is **VisA** (CC BY 4.0). The Bosch CNC dataset (vibration, not images) is CC-BY-4.0.

## 5. Existing Pretrained Models

| Model | Source | Dataset | Performance | Edge Deployable? |
|-------|--------|---------|-------------|-----------------|
| **SimpleNet** | arxiv.org/abs/2303.15140 | MVTec AD | 99.60% image AUROC (SOTA) | Via Anomalib |
| **EfficientAD** | arxiv.org/abs/2303.14535 | MVTec AD | 99.10% image AUROC | Yes, fast inference |
| **PatchCore** (Amazon) | arxiv.org/abs/2106.08265 | MVTec AD | 99.10% image AUROC | Yes |
| **FastFlow** | arxiv.org/abs/2111.07677 | MVTec AD | 99.40% image AUROC | Yes |
| **PaDiM** | arxiv.org/abs/2011.08785 | MVTec AD | 97.90% image AUROC | Yes |
| **DRAEM** | arxiv.org/abs/2108.07610 | MVTec AD | 98.00% image AUROC | Yes |
| **YOLOv8** (fine-tuned) | ultralytics.com | Custom datasets | Task-dependent | Yes, TensorRT/ONNX/TFLite |

All anomaly detection models above are available via **Anomalib** (Apache 2.0). They are pretrained/evaluated on MVTec AD which is non-commercial.

For TinyVision's purposes, training from scratch on commercially-licensed data (VisA or custom) is the viable path.

## 6. Training Difficulty

### Assessment: MODERATE to HARD

**Why moderate:**
- Anomaly detection models (unsupervised) only need normal images for training. This is a major advantage for machine-state detection where anomalous states may be rare.
- Pretrained feature extractors (ResNet, EfficientNet, CLIP) can be fine-tuned or used as frozen backbones.
- Anomalib provides turnkey training pipelines.

**Why hard:**
- **Domain gap**: MVTec AD/VisA images don't look like machine interiors. A CNC cutting chamber has coolant, chips, vibration, poor lighting. Transfer learning help is limited.
- **Temporal aspect**: Machine state is inherently temporal. A single frame may be ambiguous (is the machine idle or between cuts?). Video or multi-frame approaches may be needed.
- **State granularity**: Defining the right state taxonomy (running, idle, errored, setup, maintenance) requires domain expertise.
- **Environmental variation**: Coolant spray, chip accumulation, lighting changes, camera fouling are real challenges.
- **Data imbalance**: Most of the time a machine is running normally. Error states are rare but critical to detect.

### Training approach recommendation
1. Start with a simple binary classifier: operating vs. not operating (using motion/optical flow).
2. Expand to multi-state classification once baseline works.
3. Consider unsupervised anomaly detection on the operating state to detect irregularities.
4. Use temporal aggregation (LSTM/Transformer on frame features) for ambiguous states.

## 7. Edge Inference Requirements

### Target Hardware Benchmarks

| Hardware | Power | YOLOv8n FPS (640x640) | YOLOv8s FPS | Model Size Constraint | Price Range |
|----------|-------|----------------------|-------------|----------------------|-------------|
| **Jetson Orin Nano 8GB** | 7-15W | 286 FPS (TensorRT INT8) | ~130 FPS (est.) | ~16GB storage | ~$250 |
| **Jetson Orin NX 16GB** | 10-40W | ~65 FPS (INT8) | ~30 FPS (est.) | ~32GB storage | ~$600 |
| **Jetson Nano** | 5-10W | 5 FPS (PyTorch), 25 FPS (TRT INT8) | Too slow | ~4GB storage | ~$150 |
| **Raspberry Pi 5** | 5-10W | ~2-5 FPS (TFLite) | Too slow | Limited | ~$80 |
| **Intel NPU (OpenVINO)** | Varies | Varies by CPU gen | Varies | N/A | Varies |

### Requirements for Machine-State Detection

| Requirement | Target | Justification |
|-------------|--------|---------------|
| Inference latency | < 100ms per frame | Machine states change on second-scale; 10 FPS is sufficient |
| Model size | < 50MB | Must fit on constrained edge devices |
| Accuracy | > 95% state classification | False state transitions cause false alerts |
| Power consumption | < 15W | Must run continuously on factory floor |
| Camera input | USB or IP camera | Low-cost integration with existing setups |
| Temporal window | 1-5 seconds of frames | For state transition detection |

### Key finding
Running YOLOv8n at 286 FPS on Jetson Orin Nano (INT8) is far more than needed for state detection. A lightweight classifier on extracted features, or even a simple motion detection algorithm (optical flow magnitude thresholding), may be sufficient for the primary state classification. The vision model becomes more important for the secondary task: anomaly detection during operation (detecting chips, coolant issues, vibration-induced blur, etc.).

## 8. Potential for Collecting Own Data

### Feasibility: HIGH

**Advantages:**
- Camera hardware is cheap ($30-100 USB cameras)
- Existing machines can be instrumented with a clip-on camera
- State labels can be derived from machine controller signals (M-code, spindle speed, feed hold) or time-based annotation
- Raspberry Pi / Jetson can serve as recording devices

**Challenges:**
- Camera placement inside CNC/industrial machines is non-trivial (coolant, chips, heat, vibration)
- IP67+ rated cameras or protective enclosures needed
- Lighting inside machines is variable
- Annotation requires domain knowledge

### Recommended approach
1. Mount a camera on a small workshop CNC mill/lathe or 3D printer
2. Record video during various operating states
3. Label using machine G-code timestamps or manual review
4. Start with 3D printing (safer, easier access, well-lit) before CNC

## 9. Potential for Synthetic Data

### Feasibility: MODERATE

**Opportunities:**
- CAD/CAM simulation software can generate visual representations of machining
- 3D rendering engines (Blender, Unity) can simulate machine interiors with realistic lighting
- The MIRAGE paper (arxiv.org/abs/2603.13507) demonstrated using generative AI to create synthetic anomaly images, achieving competitive downstream performance
- Digital twin concepts are actively being explored for CNC machines

**Limitations:**
- Synthesizing realistic chip formation, coolant spray, and tool wear is physically complex
- Domain gap between synthetic and real machine imagery is likely large
- Synthetic data works better for anomaly augmentation than for primary state classification

## 10. Community Size and Demand

### Market size
- AI-driven predictive maintenance market: **$6.4B (2025)**, projected **$22.8B by 2034** (CAGR 15.7%)
- CNC & Machining Centers segment: **18.7% of market** (~$1.2B), CAGR 16.1%
- Computer vision in manufacturing is a rapidly growing subsegment

### Academic activity
- MVTec AD benchmark: hundreds of papers, active leaderboard (SOTA: 99.6% AUROC)
- Many master's/PhD theses on CNC state detection (KTH, URI, and others)
- Active research on edge deployment of industrial vision models
- Papers from 2024-2026 show sustained interest in CNC-specific vision monitoring

### Community demand indicators
- Anomalib: 4.8k GitHub stars, actively maintained by Intel
- Ultralytics YOLOv8: 40k+ stars, massive community
- Kaggle: multiple CNC and industrial defect detection datasets with active competition
- Roboflow Universe: manufacturing category with multiple public datasets

## 11. Related Problems

### Anomaly Detection in Manufacturing
- Well-studied on MVTec AD benchmark
- SOTA models achieve near-perfect results on curated datasets
- Gap remains for real-world deployment (lighting variation, camera angle changes, multiple defect types)
- Key challenge: generalizing beyond the specific product/scene

### Predictive Maintenance with Vision
- Primarily focused on **tool wear detection** (cutting tool condition)
- Uses spindle current, vibration, and increasingly camera data
- KTH thesis and Bosch dataset show vibration-based approaches dominate
- Vision-based approaches are emerging but less mature

### 3D Print Failure Detection
- Active community around OctoPrint/Obico/Spaghetti Detective
- Well-defined problem (spaghetti, layer shift, adhesion failure)
- Camera hardware is simpler (external webcam looking at print bed)
- Less industrial, more maker/prosumer market

## 12. Differentiation Potential

### What TinyVision could offer that doesn't exist today:

1. **Open-source, edge-native machine state classifier** purpose-built for small workshop equipment
2. **Camera-first approach** vs. the dominant sensor-based market
3. **Privacy-preserving local inference** (no cloud dependency)
4. **Multi-machine support** from a single framework
5. **Combined state detection + anomaly detection** pipeline
6. **Accessible to hobbyist/small workshop users** (not enterprise-only pricing)

### Competitive advantages over existing commercial solutions:
- IPercept/Augury: expensive, sensor-based, enterprise-focused
- Sight Machine: cloud-dependent, large manufacturer focus
- MVTec: expensive commercial software
- TinyVision: cheap hardware, local inference, open source

## 13. Estimated Engineering Effort

| Phase | Duration | Effort | Deliverable |
|-------|----------|--------|-------------|
| Data collection (3D printer) | 2-4 weeks | Low | Video dataset of print states |
| Baseline classifier | 2-3 weeks | Medium | Binary running/not-running model |
| Multi-state expansion | 4-6 weeks | Medium | 5-state classifier (running, idle, error, setup, done) |
| Edge deployment (Jetson/RPi) | 2-3 weeks | Medium | ONNX/TensorRT model |
| Anomaly detection integration | 4-6 weeks | High | Anomalib-based anomaly overlay |
| Camera hardware integration | 2-4 weeks | Medium | IP67 camera mount, USB/IP camera driver |
| Total MVP | 4-6 months | Full-time | Working prototype on 1 machine |

## 14. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Camera fouling (coolant, chips) | High | Protective housing, air purge, self-cleaning |
| Insufficient training data | Medium | Start with 3D printing (cleaner, safer), use anomaly detection (normal-only training) |
| Poor lighting conditions | Medium | IR cameras or controlled LED illumination |
| Temporal ambiguity | Medium | Multi-frame / temporal aggregation |
| License restrictions on training data | Medium | Use VisA (CC BY 4.0) or collect own data |
| Edge hardware cost | Low | Orin Nano at ~$250 is affordable; RPi5 for simpler cases |
| Real-world generalization | High | Domain gap between lab and factory floor; iterative testing required |

## 15. Recommended MVP

### Target: 3D Printer State Monitor

**Rationale:**
- 3D printers are safer and more accessible than CNC machines
- Camera placement is straightforward (external webcam)
- Existing community tools (OctoPrint) provide baseline functionality
- Clear state taxonomy: printing, idle, paused, failed, completed
- Maker/prosumer market is large and underserved by commercial solutions

**MVP specification:**
1. USB camera pointed at 3D printer build plate
2. YOLOv8n or EfficientNet-based classifier for 4 states: printing, idle, failed, completed
3. Anomalib-based anomaly detector for print quality issues during operation
4. ONNX Runtime inference on Raspberry Pi 5 or Jetson Orin Nano
5. MQTT output for integration with OctoPrint / Home Assistant
6. Local-only operation, no cloud dependency

**Training data:**
- Self-collected: time-lapse video of 3D prints with state labels
- Augment with VisA (CC BY 4.0) for anomaly detection pretraining
- Avoid MVTec AD for any commercial-facing product due to NC license

**Success criteria:**
- >95% accuracy on state classification
- <500ms latency for state transition detection
- Runs continuously on Raspberry Pi 5 (<10W)
- Zero cloud dependency
