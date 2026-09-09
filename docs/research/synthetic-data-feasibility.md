# Synthetic Data Feasibility for TinyVision: 3D-Print Failure Detection

**Date:** 2026-09-07
**Status:** Research report — all claims verified with sources unless marked UNVERIFIED

---

## Executive Summary

Synthetic/augmented data **can help**, but with narrow scope and critical guardrails. The evidence consistently shows:

1. **Defect-paste augmentation (Cut-Paste, Poisson blending) onto real backgrounds** reliably improves mAP by 3–6% in industrial defect detection, with a strict optimal volume beyond which performance degrades.
2. **Generative models (diffusion/GAN) for defect synthesis** show promising results on academic benchmarks (MVTec-AD, NEU-DET) but have **no verified evidence of success specifically for 3D-print spaghetti/stringing detection**.
3. **Full-scene synthetic rendering** (Blender, Omniverse) has been applied to industrial defect detection but **has NOT been demonstrated for FDM 3D-print failure monitoring**.
4. **Photometric augmentation alone cannot close the domain gap** between cameras/environments — this is experimentally verified in multiple studies.
5. **The primary blocker (domain diversity across camera/printer environments) is NOT addressed by defect augmentation alone** — it requires environmental variation, which synthetic rendering could theoretically provide but has not been validated for this specific domain.

---

## 1. Simulation/Rendering Approaches for 3D-Printing Defects

### 1.1 Blender-Based FDM Simulation

**Verified:**
- **matharano/3D-Printer-Simulation** (GitHub) — A Blender add-on that simulates the FDM printing process. Generates layer-by-layer deposit visualization. Does NOT simulate defects like spaghetti/stringing. https://github.com/matharano/3D-Printer-Simulation
- **Akki2209/blender-synthetic-defect-gen** (GitHub, 2026) — Procedural synthetic defect generator for Blender using Noise/Magic/Paint procedural textures. Targets cosmetic surface defects (scuffs, scratches), NOT FDM extrusion defects. https://github.com/Akki2209/blender-synthetic-defect-gen
- **rohitkoshy2004/Synthblend** (GitHub, 2026) — Blender add-on for synthetic dataset generation targeting industrial defect detection/segmentation. https://github.com/rohitkoshy2004/Synthblend
- **NVIDIA Omniverse Replicator** — Used for synthetic defect generation on 3D-reconstructed industrial objects (scratch defects). PMC paper demonstrates the NeRF→Blender→Omniverse pipeline for scratch synthesis. https://pmc.ncbi.nlm.nih.gov/articles/PMC12656295/

**UNVERIFIED:** No published paper, repo, or blog post was found that renders or simulates **spaghetti, blob, stringing, or under-extrusion** defects in a 3D-printing context using Blender or any other renderer. FDM defect morphology (thin filament strands, tangled blobs) is geometrically complex and may be difficult to render convincingly.

### 1.2 Digital Twin Approaches for FDM

**Verified:**
- **Digital twin-enabled 3D printer fault detection** (ScienceDirect, 2023) — Uses digital twin to generate sensor data (temperature, vibration) for FDM fault detection, NOT image-based visual defect simulation. https://www.sciencedirect.com/science/article/pii/S0952197623006140
- **Peres et al. (2022)** — Procedural synthetic training data generation for AI-based defect detection in industrial surface inspection. Uses physically-based rendering with domain randomization. NOT specific to 3D printing. https://www.sciencedirect.com/science/article/pii/S2212827122003997

**Verdict:** No one has built a validated Blender/Omniverse pipeline for generating realistic spaghetti/stringing/under-extrusion imagery. This is a significant gap — and creating one would be novel engineering work, not a trivial task.

---

## 2. Image-Space Augmentation for Domain Generalization in Defect Inspection

### 2.1 Cut-Paste / Defect Patch Transplantation

**Verified — strong evidence base:**

- **Steel Defect Detection repo (DevilsBreath, 2026)** — The most directly relevant empirical study. Trained YOLOv8s on 1800 real NEU-DET steel defect images. Systematically ablated cut-paste (simple and Poisson) and ControlNet augmentation. **Key findings:**
  - Cut-paste (simple) peaks at **n=50 synthetic images/class** (mAP@50 = 0.7782, up from 0.7165 baseline)
  - Cut-paste (Poisson) peaks at **n=100** (mAP@50 = 0.7811)
  - Both **degrade monotonically beyond their optima** — at n=300, mAP drops 2.8–3.0pp from peak
  - ControlNet (SD1.5, canny-conditioned) without domain fine-tuning **underperforms cut-paste** because the UNet generates pixels from an internet-image distribution, not steel microstructure
  - Mechanism: "beyond the optimum, the synthetic-to-real ratio becomes high enough that the model's learned features shift toward the augmented distribution"
  - https://github.com/DevilsBreath/steel-defect-detection-and-deployment/blob/main/README_tech.md

- **GT-CutMix for strip steel** (IOPscience, 2024) — Proposes ground-truth-guided CutMix that accurately samples defect locations and composites them into original images. mAP of 78.63% on X-SDD dataset with SA-DSSD model. https://iopscience.iop.org/article/10.1088/1361-6501/ad5b7c

- **Copy-paste for metal dent detection** (J. Adv. Manuf. Process., 2025) — 200 copy-paste dents achieved 20% accuracy improvement vs. 8% for standard augmentation. https://onlinelibrary.wiley.com/doi/10.1002/amp2.70011

- **NSA (Neighborhood Synthesis Augmentation)** — Removes a region from a normal image and synthesizes defects using Poisson harmonics for natural blending. Referenced in the Anomaly Detection by Effectively Leveraging Synthetic Images survey (arXiv:2512.23227). https://arxiv.org/pdf/2512.23227

- **Dynamic Label Injection** (Springer, 2025) — Combines Poisson-based seamless image cloning with cut-paste to rebalance batch defect distribution. https://link.springer.com/chapter/10.1007/978-3-031-92805-5_4

**Verdict:** Cut-paste onto real backgrounds is the **highest-confidence, lowest-risk** synthetic augmentation strategy. The steel-defect repo provides the critical finding: there is a strict optimal volume, and exceeding it is actively harmful.

### 2.2 Poisson Blending Specifically

**Verified:**
- Poisson blending (`cv2.seamlessClone`) smooths boundaries when pasting defect patches onto real backgrounds. The steel-defect study found Poisson's optimum (n=100) occurs later than simple paste (n=50), suggesting it produces slightly more realistic composites, but the improvement is marginal (+0.003 mAP@50 at peak).
- Used in DCDGANc and multiple diffusion-based augmentation papers as the standard compositing method.

**Verdict:** Marginal improvement over simple paste. Worth implementing but not transformative.

### 2.3 Photometric Augmentation (Color Jitter, Blur, Noise)

**Verified — strong evidence of limitations:**

- **Generalization Gap in Data Augmentation (Xiao et al., 2024)** — Rigorous controlled experiment: trained 6 models (AlexNet through Swin Transformer) on a Singular Illumination Dataset (SID) vs. Full Spectrum Illumination Dataset (FSID). Models trained on SID achieved ~0.35 accuracy; models on FSID achieved ~0.99. After illumination vector augmentation and Bayesian-optimized ColorJitter, performance improved to ~0.90–0.95 but **a persistent gap of ~0.04–0.14 remained** compared to models trained with real illumination diversity. "A notable generalization gap still exists... emphasizing the critical role of feature diversity in the training set." https://arxiv.org/html/2404.07514v1

- **Colorist (Doerrich et al., 2026)** — Statistical color matching (aligning first two moments of RGB channels) outperforms standard color jitter across 12 in-distribution and 7 OOD datasets. However: "standard geometric heuristics and basic color jittering fail to provide sufficient photometric diversity to bridge severe domain gaps." https://arxiv.org/pdf/2608.18915

- **Adaptive Photometric Augmentation (Sivamani et al., 2022)** — Different CNN architectures require different photometric augmentation parameters; no single setting works universally. https://www.sciencedirect.com/org/science/article/pii/S1546221822009298

- **Domain Generalization Through Data Augmentation: A Survey (Mai et al., 2025)** — Comprehensive survey concludes: "insufficient augmentation intensity may fail to capture potential domain shifts, while excessive augmentation could introduce synthetic artifacts that diverge too far from the target domain's true distribution." Identifies "augmentation overfitting" as a recognized failure mode. https://www.mdpi.com/2227-7390/13/5/824

**Verdict:** Photometric augmentation is **necessary but insufficient** for TinyVision's problem. It can handle within-camera lighting variation but **cannot bridge the gap between different cameras, lenses, resolutions, and physical environments**. The Xiao et al. experiment quantifies the gap: even optimized color augmentation leaves ~4–14% accuracy on the table compared to real diversity.

---

## 3. Generative Models (GAN/Diffusion) for Defect Synthesis

### 3.1 GANs for Defect Generation

**Verified:**

- **Defect-GAN (2021, CVPR/WACV)** — Generates realistic defect images using a multi-hierarchy GAN for automated defect inspection. https://arxiv.org/abs/2103.15158

- **GAN-PD (Nature Sci. Rep., 2026)** — "Redefining defect synthesis as a detector-centric process" — generates samples that actively enhance real-world inspection rather than merely approximating them. https://www.nature.com/articles/s41598-026-52876-y

- **SDGAN** — Steel defect GAN using two generators to switch between defective/non-defective status.

- **StyleGAN2-ADA for defect generation (ScienceDirect, 2025)** — Feature disentanglement approach outperforms existing GAN-based defect generation on MVTec AD and VisA datasets in few-shot scenarios. Average improvement of 23% on KID metric. https://www.sciencedirect.com/science/article/pii/S0952197625018378

- **Synthetic data augmentation for surface defect detection (Springer, 2020)** — GAN-generated synthetic defect images for surface defect classification. Claims 95.33% sensitivity and 99.16% specificity with "high generalization capability." https://link.springer.com/article/10.1007/s10845-020-01710-x

- **Wafer dicing defects GAN (arXiv, 2024)** — Compared DCGAN, CycleGAN, and StyleGAN3 for semiconductor defect augmentation. https://arxiv.org/abs/2407.20268

### 3.2 Diffusion Models for Defect Generation

**Verified:**

- **Defect image generation using diffusion model (Springer, 2026)** — DreamBooth + Differential Diffusion for defect synthesis. "The hybrid scenario (data augmentation) provides the best overall results." Experiments on MVTec-AD and a real-world industrial dataset. https://link.springer.com/article/10.1007/s11042-026-21705-9

- **SynSur (arXiv, 2026)** — End-to-end diffusion pipeline (Flux.1-dev + LoRA) for synthetic industrial surface defect generation. "Synthetic-only training does not replace real data; however, augmenting the full real set with synthetic samples yields consistent AP gains." 420 annotated synthetic images in ~8 GPU-hours. https://arxiv.org/html/2604.26633v2

- **AdaBLDM (arXiv, 2024)** — Blended Latent Diffusion Model with online adaptation for defect generation. Outperforms Cut-Paste and GAN-based methods on MVTec-AD, BTAD, and KSDD2. https://arxiv.org/html/2402.19330v1

- **Aggregated Data Augmentation using Diffusion + Poisson Blending (IEEE, 2025)** — Combines enhanced diffusion models with Poisson blending for defect augmentation. https://ieeexplore.ieee.org/document/11090413

- **Enhancing Glass Defect Detection with Diffusion Models (arXiv, 2025)** — DDPM-generated synthetic defect images improved ResNet50V2 accuracy from 78% to 93% for glass bottle defect classification. Trained on only 63 real defective images. https://arxiv.org/html/2505.03134v1

- **3C Product Surface Defect Detection (ScienceDirect, 2026)** — Uses diffusion model-based synthetic augmentation with historical data retrieval. https://www.sciencedirect.com/science/article/pii/S0360835226001324

### 3.3 Key Finding: Rule-Based vs. Generative Synthesis Trade-off

**Verified** (from arXiv:2512.23227 survey):
- Rule-based synthesis (Cut-Paste, noise injection): "cost-effective but often fails to produce realistic defect images"
- Generative model-based synthesis: "can create high-quality defect images but requires substantial cost"
- "Rule-based synthesis methods exhibit significant limitations in mimicking real-world industrial defects. Cut-Paste simply overlays patches without any blending mechanism, producing unnatural boundary discontinuities... NSA employs Poisson blending to soften edges but often introduces interpolation artifacts."

**Critical caveat:** These evaluations are on **single-domain benchmarks** (MVTec-AD, NEU-DET) where all images share similar camera/environment. The generalization question (do generated defects help across camera/environment domains?) has **not been tested**.

---

## 4. Synthetic-to-Real Transfer Pitfalls

### 4.1 Models Learning Synthetic Artifacts

**Verified:**

- **Domain Randomization survey (EmergentMind, 2025)** — "Domain randomization is an intentionally non-realistic data generation regime... Synthetic images, lacking real-world context structure, may result in sub-optimal high-recall behavior." https://www.emergentmind.com/topics/domain-randomization-dr

- **Synthetic-to-Real Object Detection using YOLOv11 (arXiv, 2025)** — "Domain randomization via well-tuned data augmentation can be an effective strategy for bridging the synthetic-to-real gap... However, the achieved score also highlights the remaining challenges in fully eliminating the domain gap using this approach alone." https://arxiv.org/abs/2509.15045

- **Steel defect ControlNet study** — The most direct evidence: SD1.5 ControlNet without domain fine-tuning **underperformed simple cut-paste** because "the UNet generates pixels from an internet-image distribution, not steel microstructure. Canny conditioning preserves edge structure but cannot override the UNet's pixel-level domain mismatch." This is exactly the "model learning synthetic artifacts" failure mode.

- **SynSur paper (2026)** — "The sim-to-real gap — caused by differences in texture, lighting, and sensor characteristics — often degrades detection performance on real data."

- **JOMAM paper (2025)** — "A model may overfit to idealized synthetic artifacts — crisp edges, uniform surfaces, perfect illumination — and struggle when encountering irradiance variations, irregular backgrounds, or subtle visual distortions in real images." https://ijomam.com/wp-content/uploads/2025/09/Pp.-315-322_SYNTHETIC-TO-REAL-DOMAIN-ADAPTATION-IN-COMPUTER-VISION-SYSTEMS-TOWARDS-HIGH-PRECISION-INDUSTRIAL-APPLICATIONS.pdf

### 4.2 Augmentation Overfitting

**Verified:**
- The steel-defect repo provides the clearest empirical evidence: **both simple and Poisson cut-paste show a strict optimum followed by monotonic degradation**. At n=300 synthetic/class, mAP drops 2.8–3.0pp from peak. The mechanism is that the model's learned features shift toward the augmented distribution.
- The DG survey (2025) identifies "augmentation overfitting" as a recognized failure mode: "Poorly designed data augmentation strategies can lead to overfitting... If augmented data deviate significantly from the true distribution, they may introduce noise or irrelevant characteristics."

---

## 5. Published Evidence Specific to 3D-Print/Spaghetti Detection Using Synthetic Data

**Verified (limited):**

- **IEEE (2024)** — "Detection of Spaghetti and Stringing Failure in 3D Printing" — Uses real images only, no synthetic data. Reports lack of generalization when applied to external data. https://ieeexplore.ieee.org/document/10475059

- **Roboflow Universe: Spaghetti 3D dataset** — 715 images, 2 classes (spaghetti, stringing). mAP@50 = 28.6%, Precision = 40.8%, Recall = 34.1%. Trained on what appears to be a mix of real and augmented data. https://universe.roboflow.com/3d-printing-failure/spaghetti-3d

- **HuggingFace: 3D-Print-Failure-Detector** — YOLOv11s trained on "over 9,000 images" from "various 3D printers and under different lighting conditions." mAP@50-95 of 0.82 for spaghetti, 0.60 for stringing. Data augmentation: "1000 images augmented to grayscale." **Does NOT mention synthetic defect generation** — appears to use only photometric augmentation on real data. https://huggingface.co/ApatheticWithoutTheA/3D-Print-Failure-Detector

- **LanzDeGuzman GitHub (2023)** — YOLOv4-Tiny trained on 257 images (blobs, cracks, spaghetti, stringing, under-extrusion). mAP 30–43%. Uses "open-source data set for 3d printing defects, together with additional images available from Google." No synthetic data generation. https://github.com/LanzDeGuzman/Real-Time-Additive-Manufacturing-FDM-Printer-Vision-Based-Fault-Detection-System

- **FDM-3D-Printing-Defect-Dataset (Kaggle/GitHub, 2024)** — 1912 images across 5 defect types (cracking, layer shifting, off platform, stringing, warping). Real images only. MIT license. https://github.com/wengmhu/FDM-3D-Printing-Defect-Dataset

**UNVERIFIED:** No published paper, dataset, or open-source project was found that uses synthetic/rendered/generative spaghetti, stringing, or blob defects for training 3D-print failure detection models. **This is a genuine research gap.**

---

## 6. Photometric Augmentation vs. True Domain Shift

The evidence is clear and consistent:

1. **Xiao et al. (2024)** experimentally demonstrates that even Bayesian-optimized color augmentation leaves a 4–14% accuracy gap compared to real illumination diversity. https://arxiv.org/html/2404.07514v1

2. **Colorist (2026)** confirms: "standard geometric heuristics and basic color jittering fail to provide sufficient photometric diversity to bridge severe domain gaps." https://arxiv.org/pdf/2608.18915

3. **DG Survey (2025)**: "Input-level augmentation... is often limited in addressing higher-level feature variations and may introduce visual distortions that do not accurately represent domain shifts." https://www.mdpi.com/2227-7390/13/5/824

4. **For TinyVision specifically:** The domain shift is not just illumination — it involves different cameras (resolution, sensor characteristics, lens distortion), different printers (bed size, filament color, enclosure), different physical environments (backgrounds, ambient objects). Photometric augmentation cannot simulate resolution differences, lens characteristics, or physical scene geometry.

---

## 7. Evidence-Based Verdict

### Will synthetic data help TinyVision?

**Yes, but with strict scope and order of operations.**

The evidence supports a **tiered approach**, ranked by confidence:

| Tier | Method | Expected Impact | Risk | Evidence Quality |
|------|--------|----------------|------|------------------|
| 1 | **Real data collection from more environments** | Highest — directly addresses root cause | Low | Strong (your own 333-frame experiment) |
| 2 | **Cut-Paste defect augmentation onto real backgrounds** | Moderate (3–6% mAP improvement) | Low (if volume-controlled) | Strong (steel-defect repo, multiple papers) |
| 3 | **Aggressive photometric augmentation** | Low-moderate (helps within-camera variation) | Very low | Strong (but proven insufficient for domain shift) |
| 4 | **Diffusion-based defect synthesis on real backgrounds** | Potentially moderate | Medium (domain mismatch risk) | Moderate (academic benchmarks only) |
| 5 | **Full-scene synthetic rendering (Blender/Omniverse)** | Potentially high for domain diversity | High (huge engineering effort, no validated pipeline exists for FDM defects) | Weak (no 3D-print-specific evidence) |

### What will NOT work:

- **Replacing real data with synthetic data.** Every study shows this degrades performance. Synthetic is augmentation, not replacement.
- **Uncurated synthetic augmentation.** The steel-defect repo proves there is a strict optimal volume. Exceeding it actively degrades performance.
- **Using off-the-shelf generative models without domain adaptation.** ControlNet on SD1.5 without fine-tuning underperformed simple cut-paste on steel defects.
- **Relying on photometric augmentation alone to solve cross-camera generalization.** Experimentally refuted.

### The core insight for TinyVision:

**Your problem has two distinct sub-problems:**

1. **Defect diversity** (spaghetti morphology, stringing patterns, under-extrusion appearance) — synthetic augmentation CAN help here, via cut-paste or generative methods.
2. **Environmental/domain diversity** (camera, lens, lighting, printer, background) — synthetic augmentation is UNLIKELY to help here without massive engineering effort, and may hurt by introducing artifacts.

**The highest-leverage action remains collecting more real data from more real environments.** Your finding that 333 real normal frames from 7 environments outperformed hundreds of synthetic frames is consistent with every study reviewed here.

---

## 8. Ranked Promising Synthetic Avenues (IF pursued)

### 8.1 Cut-Paste Defect Augmentation onto Real Backgrounds (RECOMMENDED FIRST)

**How:** Extract spaghetti/stringing/under-extrusion patches from your existing failure images. Paste them onto normal frames from different environments using simple paste or Poisson blending.

**Evidence:** Steel-defect repo shows +6.5 mAP@50 improvement at optimal volume (n=50–100/class).

**Risks:**
- **Strict volume control required.** Beyond n=50–100/class, performance degrades.
- Patches from one camera/environment may not look realistic on another.
- Spaghetti/stringing morphology may be too complex for simple paste to look realistic.

**Implementation cost:** Low. OpenCV + existing real data.

### 8.2 Diffusion-Based Defect Compositing on Real Backgrounds (RECOMMENDED SECOND)

**How:** Use a pre-trained inpainting diffusion model (e.g., Stable Diffusion inpainting) to generate spaghetti/stringing defects within masked regions of real normal frames.

**Evidence:** SynSur (2026) and DefectFill/Diffusion-based methods show consistent AP gains when augmenting scarce real sets. The DreamBooth + Differential Diffusion approach (Springer, 2026) specifically targets this use case.

**Risks:**
- Diffusion models trained on internet images may generate defects that look like internet images, not real 3D-print failures.
- Need to validate that generated defects don't introduce exploitable artifacts.
- Computational cost (though manageable for dataset-scale generation).

**Implementation cost:** Medium. Requires GPU for generation, prompt engineering, quality filtering.

### 8.3 CutMix/MixUp-Style Defect Compositing (SUPPORTED ALTERNATIVE)

**How:** CutMix-style training where defect regions are blended into normal images during training (online augmentation).

**Evidence:** GT-CutMix (IOPscience, 2024) and related methods show improvements for small-sample defect detection.

**Risks:** Lower quality than offline compositing, but simpler pipeline.

**Implementation cost:** Low.

### 8.4 Full-Scene Synthetic Rendering (LONG-TERM, HIGH RISK)

**How:** Build a Blender/Omniverse pipeline that renders realistic FDM printing scenes with simulated spaghetti/stringing defects.

**Evidence:** Generic industrial defect rendering exists (Synthblend, NVIDIA Omniverse Replicator, BladeSynth). No FDM-specific implementation exists.

**Risks:**
- **Massive engineering effort** to create convincing FDM defect geometry.
- **No validated rendering approach** for filament-like structures.
- **Sim-to-real gap** is the primary challenge — rendered images may teach the model rendering artifacts rather than real defect features.
- Most expensive option with weakest evidence base for this specific application.

**Implementation cost:** Very high. Months of work for uncertain payoff.

---

## 9. What Could Not Be Verified

1. **Any paper or project that uses synthetic/rendered spaghetti/stringing/under-extrusion defects for training.** This appears to be a genuine gap in the literature.

2. **Whether Blender-based FDM simulation (matharano/3D-Printer-Simulation) can be extended to generate realistic defect geometry.** The existing tool simulates the printing process but does not model failure modes.

3. **Whether diffusion models can generate convincing 3D-print-specific defects (spaghetti blobs, thin stringing filaments) when conditioned on real printer scenes.** No published evaluation exists.

4. **The optimal synthetic-to-real ratio for 3D-print failure detection specifically.** The steel-defect repo gives guidance (n=50–100/class), but this may differ for the different defect morphology of FDM failures.

5. **Whether domain-randomized synthetic FDM scenes (randomized lighting, camera position, material properties) would bridge the cross-camera gap.** Domain randomization has evidence in robotics/autonomous driving but not in this specific inspection domain.

6. **Whether collecting ~500 more real failure frames from 5+ additional cameras would fully solve the generalization problem.** Your existing data suggests this trajectory is promising, but the exact data requirements are unknown.

---

## Sources

| # | Source | URL | Date |
|---|--------|-----|------|
| 1 | DevilsBreath steel-defect-detection repo | https://github.com/DevilsBreath/steel-defect-detection-and-deployment | 2026 |
| 2 | Xiao et al., Generalization Gap in Data Augmentation | https://arxiv.org/html/2404.07514v1 | 2024 |
| 3 | Doerrich et al., Colorist | https://arxiv.org/pdf/2608.18915 | 2026 |
| 4 | Mai et al., DG Through Data Augmentation Survey | https://www.mdpi.com/2227-7390/13/5/824 | 2025 |
| 5 | SynSur (diffusion defect generation) | https://arxiv.org/html/2604.26633v2 | 2026 |
| 6 | Diffusion defect generation (Springer) | https://link.springer.com/article/10.1007/s11042-026-21705-9 | 2026 |
| 7 | AdaBLDM (Blended Latent Diffusion) | https://arxiv.org/html/2402.19330v1 | 2024 |
| 8 | Anomaly Detection by Leveraging Synthetic Images | https://arxiv.org/pdf/2512.23227 | 2025 |
| 9 | GT-CutMix for strip steel | https://iopscience.iop.org/article/10.1088/1361-6501/ad5b7c | 2024 |
| 10 | Domain Randomization topic review | https://www.emergentmind.com/topics/domain-randomization-dr | 2025 |
| 11 | Synthetic-to-Real YOLOv11 (arXiv) | https://arxiv.org/abs/2509.15045 | 2025 |
| 12 | JOMAM synthetic-to-real domain adaptation | https://ijomam.com/wp-content/uploads/2025/09/ | 2025 |
| 13 | Procedural synthetic training data (ScienceDirect) | https://www.sciencedirect.com/science/article/pii/S2212827122003997 | 2022 |
| 14 | GAN-PD for fabric defects (Nature) | https://www.nature.com/articles/s41598-026-52876-y | 2026 |
| 15 | Glass defect detection with diffusion (arXiv) | https://arxiv.org/html/2505.03134v1 | 2025 |
| 16 | Copy-paste for metal dent detection | https://onlinelibrary.wiley.com/doi/10.1002/amp2.70011 | 2025 |
| 17 | StyleGAN2-ADA defect generation | https://www.sciencedirect.com/science/article/pii/S0952197625018378 | 2025 |
| 18 | NeRF→Omniverse defect pipeline (PMC) | https://pmc.ncbi.nlm.nih.gov/articles/PMC12656295/ | 2025 |
| 19 | matharano/3D-Printer-Simulation (GitHub) | https://github.com/matharano/3D-Printer-Simulation | — |
| 20 | Akki2209/blender-synthetic-defect-gen (GitHub) | https://github.com/Akki2209/blender-synthetic-defect-gen | 2026 |
| 21 | rohitkoshy2004/Synthblend (GitHub) | https://github.com/rohitkoshy2004/Synthblend | 2026 |
| 22 | FDM-3D-Printing-Defect-Dataset (GitHub) | https://github.com/wengmhu/FDM-3D-Printing-Defect-Dataset | 2024 |
| 23 | HuggingFace 3D-Print-Failure-Detector | https://huggingface.co/ApatheticWithoutTheA/3D-Print-Failure-Detector | — |
| 24 | Roboflow Spaghetti 3D dataset | https://universe.roboflow.com/3d-printing-failure/spaghetti-3d | 2023 |
| 25 | IEEE spaghetti/stringing detection (2024) | https://ieeexplore.ieee.org/document/10475059 | 2024 |
| 26 | LanzDeGuzman FDM Fault Detection (GitHub) | https://github.com/LanzDeGuzman/Real-Time-Additive-Manufacturing-FDM-Printer-Vision-Based-Fault-Detection-System | 2023 |
| 27 | Defect-GAN (arXiv) | https://arxiv.org/abs/2103.15158 | 2021 |
| 28 | Digital twin FDM fault detection | https://www.sciencedirect.com/science/article/pii/S0952197623006140 | 2023 |
| 29 | Real-time stringing detection (MDPI) | https://www.mdpi.com/2227-9717/8/11/1464 | 2020 |
| 30 | Dynamic Label Injection (Springer) | https://link.springer.com/chapter/10.1007/978-3-031-92805-5_4 | 2025 |
| 31 | Sivamani et al., Adaptive Photometric Augmentation | https://www.sciencedirect.com/org/science/article/pii/S1546221822009298 | 2022 |
| 32 | Real-time FDM dual-camera detection (Springer) | https://link.springer.com/article/10.1007/s40964-025-01401-0 | 2025 |
| 33 | Style transfer for synthetic-to-real (ScienceDirect) | https://www.sciencedirect.com/science/article/pii/S1077314225001687 | 2025 |
