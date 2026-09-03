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

Research candidates for Experiment 1 (all available in the registry):

| Model | Accuracy | RPi5 latency | Size (INT8) | License |
|---|---|---|---|---|
| EfficientNet-Lite0 | 75.1% (ImageNet) | ~49ms | 5.1MB | Apache 2.0 |
| YOLOv11n-cls | 70.0% (ImageNet) | ~10ms | 1.6MB | AGPL-3.0 |
| ShuffleNetV2 | 69.4% (ImageNet) | ~30ms | 1.47MB | MIT |

Under AGPL-3.0/GPL-2.0 licensing (UL principle 6), a copyleft model as the winner
would constrain how the project may be distributed/relicensed; favor permissive
architectures where results are comparable.

### Temporal decision logic (decided)

**EWMA + baseline-relative threshold + FSM with hysteresis + fire-once.**

- α = 0.15 (Obico's documented value, to be tuned)
- Baseline window = ~5 minutes
- FSM states: OBSERVING → NORMAL → SUSPECT → FAILING → PAUSED
- Sensitivity exposed as user setting
- ~100 lines of C++, O(1) per frame

### Datasets (decided)

**Own data collection is essential.** No existing dataset provides both normal and failure
classes with clean licensing.

| Dataset | Role | License | Status |
|---|---|---|---|
| Own collection (normal) | Normal class | N/A | Must collect |
| Own collection (failures) | Failure class | N/A | Must collect + inject |
| wengmhu FDM defect | Failure supplement | MIT | Usable |
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

1. Collect own dataset (`tv capture`) for normal + failure classes.
2. Add `wengmhu` FDM defect set as failure supplement.
3. Run Experiment 1 across all classification candidates in the registry.
4. Declare winner from measured results, then build the C++ runtime.

## Experiment plan

| # | Task | Model | Dataset | Timeline |
|---|---|---|---|---|
| 1 | Binary classification | EfficientNet-Lite0, YOLOv11n-cls, ShuffleNetV2 | wengmhu + own | Week 1–2 |
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
| Open-source projects | `docs/research/open-source-projects.md` |
| Lightweight models | `docs/research/lightweight-models-comparison.md` |
| Dataset records | `docs/research/datasets/` |
