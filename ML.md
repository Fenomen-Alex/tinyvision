# TinyVision ML Status

**Last updated**: 2026-09-03
**Status**: Research complete. Reproducible training/evaluation pipeline built.
Preparing dataset collection. No model declared the winner yet.

---

## Project

First project: **3D-print failure detection** from a fixed webcam.
Target hardware: Raspberry Pi 4/5 + USB/CSI webcam.
Production runtime: C++ / ONNX Runtime.

## Decision record

### Task formulation (decided)

**Binary image classification** is the primary MVP path.

- Simplest formulation with highest demonstrated accuracy.
- PrintGuard achieves F1=0.937 on Pi 4B with ShuffleNetV2.
- Fastest inference (13–49ms on Pi 5 CPU).
- Lowest data requirement.
- Temporal smoothing (EWMA + FSM) is deterministic post-processing, not a learned model.

Multi-class and detection are secondary paths for later experiments.

### Primary model architecture (decided)

**Model is not hard-coded.** The training pipeline (`training/`) is model-agnostic:
every experiment selects its architecture from the registry, and measured results
(not the research table) decide the winner.

#### Experiment 1 measured result (Sept 2026)

All torchvision classification candidates trained on 3d-print-errors binary task
(normal vs failing), evaluated on held-out test set. Full details in
`docs/research/experiment1-results.md`.

| Model | Accuracy | F1 | AUROC | fp32 size | int8 size |
|---|---|---|---|---|---|
| **mobilenet_v3_small** | 0.823 | **0.761** | **0.925** | 6.09 MB | 1.69 MB |
| shufflenet_v2_x1_0 | **0.836** | 0.757 | 0.900 | 5.05 MB | 1.45 MB |
| mobilenet_v3_large | 0.821 | 0.726 | 0.906 | 16.8 MB | 4.42 MB |
| efficientnet_b0 | 0.789 | 0.695 | 0.897 | 16.0 MB | 4.36 MB |
| resnet18 | 0.740 | 0.685 | 0.871 | 44.7 MB | 11.2 MB |

**Provisional winner: mobilenet_v3_small** (best F1 + AUROC, small, fast).
**Runner-up: shufflenet_v2_x1_0** (comparable F1, smallest footprint).

Shadow candidate set (2 models) carried to env experiments. Final selection
confirmed by edge latency/memory benchmarking and temporal-smoothing results.

Original research table (pre-measurement) is preserved below for reference.

### Temporal decision logic (decided)

**EWMA + baseline-relative threshold + FSM with hysteresis + fire-once.**

- α = 0.15 (Obico's documented value, to be tuned)
- Baseline window = ~5 minutes
- FSM states: OBSERVING → NORMAL → SUSPECT → FAILING → PAUSED
- Sensitivity exposed as user setting
- ~100 lines of C++, O(1) per frame

**Stage 3 update (see `docs/research/stage3-temporal-evaluation.md`):** evaluated on the
frozen ShuffleNetV2 baseline over the frozen val/test splits. Findings:
- EWMA smoothing robustly cuts sequence-level false alarms vs. raw (halved on holdout,
  eliminated on dev) with no detection loss — justified.
- The full FSM adds operational semantics but did **not** beat plain EWMA on frame metrics
  on this synthetic dataset.
- The **baseline-relative delta path failed** here (constant-high failure signals absorb
  into their own baseline, missing 5/7 dev recordings); an **absolute-score latch (~0.65)**
  is required alongside any delta guard.
- **Detection-delay is not measurable** on the current synthetic dataset (no verified
  failure-onset ground truth); a real, hand-annotated video experiment is required.
- Dominant bottleneck is **frame-classifier miscalibration** (sustained over-confidence on
  source-labeled-good prints), which temporal logic cannot fix.

**Stage 4 update (see `docs/research/stage4-dataset-forensics.md`):** the prepared dataset
has **no recording-level label** — `normal`/`failing` folders are a *per-frame* mapping of the
source `class` column, so 41/151 recordings span both folders (good→defect transitions). The
Stage 3 "recording label" was a first-frame convention, which mislabels every transition
recording as `normal`. Corrected conclusion: the "miscalibration / over-confidence" concern
applies only to **genuinely all-good** recordings (6 test / 4 val), not to mixed
good→defect transition recordings (whose defect frames are real detections). The strongest
defensible task is **per-frame binary classification**; false alarms should be measured only
on all-good recordings, and temporal "onset" is **not verified ground truth** (class-flip does
not reliably equal physical failure onset).

**Stage 5 update (see `docs/research/stage5-corrected-evaluation.md`):** corrected baseline
and temporal evaluation against per-frame labels. Exp 1 frame-level metrics are confirmed
identical (not superseded). The preserved FSM (use_delta=True, abs 0.65) is catastrophically
broken on corrected labels: recall 0.007 on test. EWMA does not improve frame-level metrics;
its only benefit is smoothing false-alarm runs (80→20 on test). Corrected "catastrophic"
recordings analysis: 3 of 4 were missed-detection problems, not false-alarm problems. Val
threshold sweep shows F1 peaks at 0.7 (0.911 vs 0.908 at frozen 0.5). Key priorities:
(1) own data collection, (2) threshold tuning for deployment, (3) failure-type-specific
analysis, (4) deferred temporal logic redesign.

### Datasets (status)

**Own data collection is essential** for deployment-time validation on real
hardware/camera. Experiment 1 used the NilsHagenBeyer "3d printing errors"
dataset (Apache 2.0) as the binary-classification training set.

| Dataset | Role | License | Status |
|---|---|---|---|
| **NilsHagenBeyer 3d printing errors** | Experiment 1 binary classification | Apache 2.0 | ✅ Used |
| Own collection (normal) | Normal class, deployment validation | N/A | Must collect |
| Own collection (failures) | Failure class | N/A | Must collect + inject |
| wengmhu FDM defect | Failure supplement (multi-class) | MIT | Usable |
| Roboflow/seprinder | Detection experiments | CC BY 4.0 (provenance caution) | Usable with caveat |
| Roboflow/syLucauc | Detection experiments | CC BY 4.0 (provenance caution) | Usable with caveat |
| Caxton (Cambridge) | Not for classification | CC BY 4.0 | Not suitable (continuous params only) |
| Kaggle 3D-Printer Defected | Binary classification | License unverified | **Do not use** |
| HF Javiai | Detection | License unknown | **Do not use** |

### Anomaly detection (status)

**Complementary, not primary.** EfficientAD-S (Apache 2.0) is the only candidate fast
enough for Pi, but its CPU latency is unverified. Benchmark in Experiment 3.

## Current implementation status

### Training pipeline (built, tested)

Located in `training/`, exposed through the `tv` CLI:

- `tv split` — leakage-free train/val/test split at the **source** (print/session) level
- `tv train` — model-agnostic training (config selects architecture from registry)
- `tv evaluate` — held-out test evaluation (accuracy, precision, recall, F1, FNR, FPR, AUROC)
- `tv export` — ONNX + INT8 export for the C++ runtime
- `tv capture` — webcam frame capture for data collection
- deterministic experiment IDs; per-experiment config + status + metrics recorded

End-to-end smoke test (train → evaluate → export → INT8) passed on CPU.

### Next steps

1. ✅ Run Experiment 1 across all classification candidates in the registry (done).
2. Collect own dataset (`tv capture`) for normal + failure classes to supplement
   and validate on real hardware/camera.
3. Run the temporal-smoothing experiment (Experiment 4) on the shadow candidate set
   (mobilenet_v3_small, shufflenet_v2_x1_0).
4. Benchmark edge latency / memory on Raspberry Pi (see `EDGE.md`).
5. Declare the winner from measured edge + accuracy results, then build the C++ runtime.

## Experiment plan

| # | Task | Model | Dataset | Timeline |
|---|---|---|---|---|
| 1 | Binary classification (✅ done) | mobilenet_v3_small, shufflenet_v2_x1_0 (winner + runner-up) | 3d-print-errors + own | Week 1–2 |
| 4 | Temporal smoothing | Experiment 1 model outputs | Video sequences | Week 2–3 |
| 2 | Object detection | YOLOv11n | Roboflow datasets | Week 3–4 |
| 3 | Anomaly detection | EfficientAD-S | Own normal prints | Week 4–5 |

## Key open questions

1. **EfficientAD on Pi 5 CPU**: What is the actual inference latency? Must benchmark.
2. **INT8 quantization accuracy drop**: Measure, do not assume.
3. **Camera placement**: What angle/distance produces best detection?
4. **Failure injection methodology**: Which failure modes are most valuable to detect?
5. **OctoPrint/Klipper API**: What hooks are available for pause/cancel?

## Research documents

| Document | Path |
|---|---|
| Problem selection | `docs/research/problem-selection.md` |
| Task formulation | `docs/research/task-formulation.md` |
| Dataset deep dive | `docs/research/dataset-deep-dive.md` |
| Model comparison | `docs/research/model-comparison.md` |
| Evaluation plan | `docs/research/evaluation-plan.md` |
| Temporal decision logic | `docs/research/temporal-decision-logic.md` |
| Anomaly detection research | `docs/research/anomaly-detection-for-3d-print.md` |
| Stage 3 temporal evaluation | `docs/research/stage3-temporal-evaluation.md` |
| Stage 4 dataset forensics | `docs/research/stage4-dataset-forensics.md` |
| Stage 5 corrected evaluation | `docs/research/stage5-corrected-evaluation.md` |
| Open-source projects | `docs/research/open-source-projects.md` |
| Lightweight models | `docs/research/lightweight-models-comparison.md` |
| Dataset records | `docs/research/datasets/` |
