# Stage 2B — Offline Robustness & Generalization Analysis

**Status:** COMPLETE. Perturbation robustness evaluated on the **frozen Experiment 1
ShuffleNetV2 baseline** against the **frozen printing-errors test set**. No retraining,
no threshold retuning, no EWMA/FSM implementation, no real-camera data. All results
below are measured on real held-out test images (inference only) — nothing is
fabricated.

**Question this stage answers:** does the model learn genuine *defect characteristics*
(robust to image-level nuisance variation) or has it latched onto a *recording- or
domain-specific artifact* (fragile to that variation)? Robustness to controlled
perturbations is one signal for the former, and persistence of the same catastrophic
recordings under perturbation is a signal for the latter / for a domain mismatch.

---

## 1. Method

### 1.1 Setup

- Model: **ShuffleNetV2 x1.0** (frozen Experiment 1 checkpoint).
- Test set (frozen, unchanged): **2,859 images, 24 recordings** (failing 1,981 / normal 878).
- Inference: `tv robustness --config <config> --split test --seed 0`, run on Apple Silicon (MPS).
- Perturbations are applied **on-the-fly to in-memory derived copies** of each test
  image; original images and the split are never modified. A seeded RNG makes the
  stochastic perturbations reproducible.

### 1.2 Convention

**Failure-detection view** throughout: *failing* is the positive class, so `recall`
= fraction of true failures caught and `FNR` = missed failures. `FPR` = normal frames
flagged failing. AUROC computed as `roc_auc_score(y_true, 1.0 - fail_scores)`.

### 1.3 Perturbations and severity schedule

Each perturbation has a fixed three-level schedule `[mild, moderate, severe]`:

| perturbation | mild → severe (parameter) | what it simulates |
|---|---|---|
| brightness | gain 1.15 → 1.60 | brighter exposure |
| darken | gain 0.87 → 0.60 | dimmer / shadowed exposure |
| contrast | α 0.85 → 0.55 | flat, washed-out contrast |
| blur | Gaussian σ 1.0 → 3.0 | focus / motion blur, out-of-focus sensor |
| noise | additive σ 10 → 50 | low-light sensor noise, cheap CMOS |
| jpeg | quality 70 → 25 | compression / re-encoding |
| downscale | 0.8 → 0.4× then up | low sensor resolution / digital zoom |
| rotation | 5° → 15° | camera rotation |
| crop | keep 0.9 → 0.7 (center, resize back) | framing / zoom changes |
| perspective | strength 0.5 → 2.0 | camera angle / mounting distortion |
| combined | stacked mild → severe | simultaneous real-world impairments |

Full machine-readable numbers are in
`docs/research/stage2-robustness-report.json`.

---

## 2. Baseline (identity — no perturbation)

Failing-as-positive:

| metric | value |
|---|---|
| accuracy | 0.836 |
| precision | 0.918 |
| recall | **0.8385** |
| F1 | 0.876 |
| FNR | **0.1615** |
| FPR | 0.170 |
| AUROC | 0.900 |

These match the frozen Experiment 1 test numbers exactly, confirming the harness
reproduces the prior evaluation.

### 2.1 By failure type (baseline, failing view)

| failure type | n | recall | FNR | precision |
|---|---|---|---|---|
| under_extrusion | 1184 | 0.853 | 0.147 | 1.000 |
| stringing | 797 | 0.817 | 0.183 | 1.000 |
| normal | 878 | — (no failures) | — | — |

Both real failure types are caught ~82–85% of the time at the default threshold;
precision is 1.0 within-failure-type (the model rarely mixes the two failure types,
it mixes failure↔normal).

### 2.2 By object shape (baseline)

Failure recall is strongly object-dependent. Shapes with failing samples that the
model **misses most**:

| shape code | n | failing recall | FNR |
|---|---|---|---|
| 25 | 127 | 0.118 | 0.882 |
| 8 | 179 | 0.168 | 0.832 |
| 33 | 86 | 0.293 | 0.707 |
| 32 | 164 | 0.793 | 0.207 |
| 16 | 541 | 0.980 | 0.020 |
| 34 | 273 | 0.985 | 0.015 |
| 12 | 166 | 1.000 | 0.000 |
| 13 | 143 | 1.000 | 0.000 |

(Note: shapes 22, 29, 1, 6 contain only normal samples in test, so their failing
recall is trivially 0 — they are uninformative for recall.)

**Interpretation:** false negatives are not random noise — they concentrate in a few
object geometries. This is consistent with the Stage 1.5 finding that ~3 recordings
produce most FNs: failed parts on certain geometries (or those recordings' camera/to-
code conditions) are visually ambiguous, i.e. the defect is either subtle or the
failure photo resembles a normal/low-quality print.

---

## 3. Robustness results (failing-as-positive)

### 3.1 Aggregate summary: AUROC and F1 at each severity

| perturbation | mild | (AUROC) mod | sev | mild | (F1) mod | sev |
|---|---|---|---|---|---|---|
| brightness | 0.912 | 0.922 | 0.895 | 0.881 | 0.884 | 0.852 |
| darken | 0.906 | 0.912 | 0.918 | 0.881 | 0.882 | 0.875 |
| contrast | 0.905 | 0.908 | 0.908 | 0.883 | 0.885 | 0.882 |
| **blur** | 0.880 | **0.739** | **0.682** | 0.855 | **0.695** | **0.633** |
| noise | 0.895 | 0.874 | 0.737 | 0.870 | 0.861 | 0.830 |
| jpeg | 0.891 | 0.892 | 0.901 | 0.857 | 0.864 | 0.861 |
| downscale | 0.891 | 0.883 | 0.841 | 0.871 | 0.870 | 0.812 |
| rotation | 0.884 | 0.887 | 0.894 | 0.860 | 0.850 | 0.859 |
| crop | 0.892 | 0.886 | 0.849 | 0.839 | 0.827 | 0.799 |
| perspective | 0.896 | 0.895 | 0.887 | 0.874 | 0.871 | 0.862 |
| combined | 0.720 | 0.698 | 0.615 | 0.713 | 0.812 | 0.819 |

### 3.2 Full severity table (accuracy, precision, recall, FNR, FPR, AUROC, deltas)

From `stage2-robustness-report.json`:

| perturb | sev | acc | prec | rec | f1 | AUROC | FNR | FPR | ΔF1 | ΔFNR |
|---|---|---|---|---|---|---|---|---|---|---|
| brightness | mild | 0.840 | 0.909 | 0.855 | 0.881 | 0.912 | 0.145 | 0.194 | +0.005 | -0.017 |
| brightness | mod | 0.840 | 0.889 | 0.878 | 0.884 | 0.922 | 0.122 | 0.246 | +0.007 | -0.039 |
| brightness | sev | 0.798 | 0.867 | 0.839 | 0.852 | 0.895 | 0.162 | 0.292 | -0.024 | +0.000 |
| darken | mod | 0.837 | 0.883 | 0.881 | 0.882 | 0.912 | 0.119 | 0.262 | +0.006 | -0.042 |
| darken | sev | 0.820 | 0.845 | 0.907 | 0.875 | 0.918 | 0.093 | 0.377 | -0.002 | -0.069 |
| contrast | sev | 0.841 | 0.908 | 0.857 | 0.882 | 0.908 | 0.143 | 0.195 | +0.006 | -0.019 |
| blur | mod | 0.649 | 0.877 | 0.575 | 0.695 | 0.739 | 0.425 | 0.182 | -0.182 | +0.264 |
| blur | sev | 0.597 | 0.860 | 0.501 | 0.633 | 0.682 | 0.499 | 0.184 | -0.243 | +0.338 |
| noise | sev | 0.718 | 0.713 | 0.992 | 0.830 | 0.737 | 0.008 | 0.900 | -0.047 | -0.153 |
| jpeg | sev | 0.794 | 0.809 | 0.918 | 0.861 | 0.901 | 0.082 | 0.487 | -0.016 | -0.080 |
| downscale | sev | 0.767 | 0.918 | 0.728 | 0.812 | 0.841 | 0.272 | 0.147 | -0.064 | +0.110 |
| rotation | sev | 0.818 | 0.930 | 0.798 | 0.859 | 0.894 | 0.202 | 0.136 | -0.018 | +0.041 |
| crop | sev | 0.748 | 0.893 | 0.723 | 0.799 | 0.849 | 0.277 | 0.195 | -0.077 | +0.116 |
| perspective | sev | 0.819 | 0.914 | 0.816 | 0.862 | 0.887 | 0.184 | 0.173 | -0.014 | +0.023 |
| combined | sev | 0.695 | 0.694 | 1.000 | 0.819 | 0.615 | 0.000 | 0.994 | -0.057 | -0.162 |

### 3.3 Failure-type recall under severe perturbations

| perturb (sev) | under_extrusion recall | stringing recall |
|---|---|---|
| baseline | 0.853 | 0.817 |
| brightness | 0.843 | 0.832 |
| darken | 0.942 | 0.856 |
| contrast | 0.883 | 0.818 |
| **blur** | **0.508** | **0.491** |
| noise | 0.990 | 0.995 |
| jpeg | 0.961 | 0.855 |
| downscale | 0.723 | 0.737 |
| rotation | 0.826 | 0.755 |
| crop | 0.779 | 0.640 |
| perspective | 0.844 | 0.774 |

---

## 4. Interpretation

### 4.1 The model is robust to photometric/intensity nuisance — good sign

Brightness, darken, contrast, and JPEG compression barely move accuracy/AUROC even at
the **severe** level (AUROC stays ~0.90). The model is **not** betting on a trivial
brightness or color hedgedge. Under larger darkening it even *improves* recall
(0.907 at severe darken) — darkening makes failed, tangled filament stand out more
against perceived normal print, and it trades off a higher FPR (0.377) for lower FNR.

**Conclusion:** the model has not learned a recording-specific *brightness* artifact.
This weakens the "pure dataset artifact" hypothesis in the photometric domain.

### 4.2 Fine-detail loss is the model's vulnerability — expected for this task

**Blur is the single most destructive perturbation** (severe: F1 0.633, AUROC 0.682,
FNR 0.499). Stringing and under-extrusion are *textural* defects — thin strands,
irregular surfaces. Blur destroys exactly the fine detail those defects live in, so
the model cannot see them. **Downscale** (FNR +0.110) and **crop** (FNR +0.116) are
the next worst — both remove/upscale spatial detail. Perspective about the geometry
the model saw (angle change) costs less than outright loss of resolution.

**Conclusion:** the model's signal is genuinely the *fine texture* of failures, not a
photometric shortcut. A deployment constraint follows directly: **keep optical focus
and resolution high enough that defect detail survives.** A cheap webcam or ESP32-CAM
that introduces blur / heavy noise / upscaling will hurt.

### 4.3 Heavy noise / combined impairment biases the model toward "failing"

At severe noise, recall jumps to 0.992 but FPR explodes to 0.900 and precision drops
to 0.713 (AUROC 0.737): heavy sensor noise *looks* like print texture to the model.
At severe combined impairment the model predicts nearly everything failing (FPR
0.994, AUROC 0.615), catching all failures but at a massive false-alarm rate.

This is the **domain-gap risk for the target camera**: if the real ESP32-CAM/logitech
frames are noisier or blurrier than the ELP 12MP training data, we can expect a bias
toward false failures (or, if detail is lost, the blur regression of §4.2). Notable:
the moderate combined case (F1 0.812, FNR 0.019) is far better behaved than severe —
a *little* noise is tolerable, extreme is not.

### 4.4 Catastrophic recordings persist under perturbation — a domain/labeling signal

The same recordings that dominate missed failures at baseline recur as the worst
recording for **blur, downscale, rotation, crop, and perspective**:

- `Recording_06012023_12_17_19` (UNDEREX) — severe blur/downscale/crop/rotation all ≈0 recall
- `Recording_15.02.2023_13_19_52` (STRING) — worst for darken/contrast/jpeg/rotation/perspective
- `Recording_01122022_16_06_44` (UNDEREX) — worst for blur (0.0), downscale (0.012), combined
- `Recording_06122022_14_49_53` (STRING) — worst for crop

These recordings fail the model *even when the image is only slightly perturbed*,
across many independent impairments. **A model that has genuinely learned defect
characteristics would be robust on most recordings and only break under extreme
perturbation; instead, these specific recordings are near-zero-recall from baseline
and stay near-zero.** This points to a **recording/domain-specific mismatch** — either
those failures genuinely look normal/low-detail in those photographs (labeling
ambiguity in the source data), or those recordings have camera/lighting conditions
the model never saw – not a general capacity failure. Both architectures (ShuffleNetV2
and MobileNet V3-small) showed the *same* dead recordings in Stage 1.5, reinforcing
that this is data-domain, not model-capacity.

### 4.5 Failure-type asymmetry

Stringing recall drops more than under_extrusion under blur (±0.32 vs ±0.35 absolute
but stringing starts lower), crop (0.640 vs 0.779), rotation and downscale. The thin
strand texture is the first thing fine-detail loss erases. Under-extrusion (wider,
smoother surface differences) survives resolution loss slightly better. Expect the
hardest real-world generalization to be *stringing on a blurry/noisy target camera*.

---

## 5. Evidence classification (required by Stage 2 discipline)

Per the project rules, the deployment-readiness evidence is distinguished explicitly:

| claim | status |
|---|---|
| Model robust to photometric variation on Kaggle test | **established** (this run) |
| Model degrades from fine-detail loss on Kaggle test | **established** (this run) |
| Catastrophic recordings are domain/labeling-inherent | **strongly supported** (baseline + perturbation + cross-architecture) |
| Blur/noise bias on the actual ESP32-CAM / Logitech | **simulated** only — needs real target-camera data (Stage 2A) |
| Real-printer end-to-end FPR/FNR | **not yet measurable** — no printer, collection postponed |

The robustness evidence is *simulated* with respect to the target camera; it tells us
*which* failure modes to expect and what to measure, but cannot substitute for real
captures.

---

## 6. Limitations

- Perturbations approximate, but do not reproduce, real camera optics/noise (no
  chromatic aberration, rolling shutter, sensor pattern noise, physics-based noise).
- Severity levels are heuristic; "mild/moderate/severe" are relative, not tied to a
  calibrated camera.
- `brightness`/`darken` are multiplicative gains; real exposure also affects noise
  (see `combined` for a partial joint model).
- Rotation/crop/perspective warp with `BORDER_REFLECT_101`; real cameras may introduce
  border differences, but the classification model downsizes to 224×224 so borders are
  mostly out of the defect region.
- `by_shape` codes come from the raw Kaggle metadata; shapes with only-normal test
  samples are uninformative for recall (recall is trivially 0) — interpreted with care.
- Single seed for stochastic perturbations (noise/perspective/combined). Report is
  reproducible (`--seed 0`).

---

## 7. Implications for the deployment pipeline

1. **Do not optimize the per-frame threshold using this perturbation sweep** (the
   production threshold stays `0.5` per the frozen Experiment 1 config). The sweep is
   for robustness *understanding*, not tuning.
2. **Resolution/focus budget:** ensure the target camera delivers enough real spatial
   detail (avoid downscaling the ROI below ~0.6 of input, avoid motion blur) — this is
   the dominant controllable lever.
3. **Expect a noise/contrast trade-off on the target camera:** extreme noise pushes
   toward false-failure; extreme blur pushes toward missed failures. Both point to
   needing the temporal layer (EWMA/FSM) more than a better per-frame model, because
   temporal smoothing corrects *flickering* false alarms and recovers brief
   low-confidence misses — and avoids over-Optimizing F1 on a per-frame basis.
4. **Target the known-problem shapes/recordings with real data:** gather clear
   samples of failure onset for the ambiguous geometries (shapes 8/25/33/32) and for
   the catastrophic film conditions, rather than assuming more of the same data fixes it.
5. **Stringing will generalize worst** from the Kaggle domain to a blurry target
   camera. Collect a mix of stringing seen at various focus levels.

---

## 8. Recommendation & next experiment

**No retraining is justified by this stage.** The model already has acceptable
per-frame discriminative power (AUROC 0.90) and is robust to the photometric nuisance
that a naive artifact explanation would predict. Its real weakness is *fine-detail
loss* and a *handful of specific domain/labeling-ambiguous recordings*.

**Recommended next step (Stage 3, in order):**

1. **Implement temporal decision logic (EWMA/FSM) on the frozen per-frame model** —
   FNR-reduction and false-alarm suppression over frames. This is the highest-value,
   lowest-risk next move and directly attacks the two real weaknesses (FNR 0.16 at
   baseline and flickering false alarms) without retraining.
2. In parallel / after: **Stage 2A real target-camera collection** (ESP32-CAM +
   Logitech), because the robustness sweep's *expected* noisier/blurrier target-camera
   degradation must be confirmed or refuted on real frames before investing in
   retraining.
3. Only after both: decide whether retraining on mixed (Kaggle + target) data or a
   reduction in input resolution is warranted. The single most informative future
   experiment is evaluation on real target-camera frames, not more perturbation sweeps.

**Bottom line:** the model behaves like it learned the actual defect texture (robust
to brightness/contrast/JPEG), vulnerabilities are interpretable (fine-detail loss,
specific ambiguous recordings), and the evidence favors proceeding to temporal logic +
real-camera validation — while keeping the per-frame model frozen.

---

## 9. Reproducibility

- Command: `tv robustness --config training/configs/robustness_shufflenet_v2_x1_0.yaml \
  --experiments-dir experiments --split test --seed 0 \
  --output-json docs/research/stage2-robustness-report.json`
- Configuration: `training/configs/robustness_shufflenet_v2_x1_0.yaml` (pins frozen
  `shufflenet_v2_x1_0_printing-errors_cdc9e2cad4`, `data_dir` = printing-errors).
- Frozen checkpoint: `experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/best.pt`.
- Frozen split: `datasets/processed/printing-errors/splits/test.csv` (2,859 rows).
- Raw metadata join: `datasets/Printing_Errors/general_data/all_images_no_filter.csv`
  (joined 100% to test images by filename).
- Hardware: Apple Silicon, MPS. Software: repo `uv.lock` environment.
- Perturbation definitions + severity schedule: `training/tv/robustness.py` (unit
  tested in `training/tests/test_robustness.py`).
