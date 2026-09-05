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

**Stage 11B update (see `docs/research/stage11b-cluster-generalization-results.md`):**
leave-cluster-out generalization experiment. Held out entire date/session clusters from
training and evaluated on unseen clusters. **Both stringing holdout clusters showed
degradation** vs known-domain stringing: 22122022 FNR 0.062 (known=0.000), 20122022 FNR
0.162 (known=0.001). Under-extrusion controls were mixed: 10012023 performed well (FNR
0.016), but 05012023 degraded significantly (FNR 0.337). Results support the hypothesis
that stringing failures have poor cross-cluster generalization, though the under-extrusion
control variability means this is not a clean comparison. The 20122022 cluster (3 of 5
recordings nearly completely missed) is particularly concerning. Production deployment
must account for per-session performance variation.

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
2. ✅ Cluster domain audit (Stage 10) — confirmed session/date effects.
3. ✅ Leave-cluster-out experiment (Stage 11B) — confirmed cross-domain degradation for
   stringing AND under-extrusion. Clean stringing-only hypothesis NOT established.
4. Acquire external video data: 19–31 independent sessions from ≥10 YouTube sources
   (see `docs/research/stage12-data-acquisition-design.md`). Stage 12A cataloged 13 candidate
   videos; Stage 12B piloted 6 (SP001, SP002, NORMAL001, UE001, UE003, STR001) and determined
    most search-sourced failure videos are heavily edited or tutorial-dominated.
    **STRINGING_SOURCE_FAILED for STR001.** No failure type visually confirmed yet;
    human montage review pending before full acquisition. Stage 12C built a provisional
    external **pilot** dataset (1,546 kept frames, 21 normal + 15 failing labeled,
    exact-md5 dedup, source-atomic split) — see
    `docs/research/stage12c-external-pilot-dataset.md`. **No training on it yet; labels are
    unverified; validation split contains zero labeled frames.** Stage 12D built the
    **human-review workflow** (manifest + contact sheets, 1,006 units; decisions all empty)
    to turn a subset of the six sources into **human-verified** external data —
    see `docs/research/stage12d-human-review-workflow.md`. The pilot showed acquisition
    should favor **continuous footage first, failure type second** (stable printer-cam feeds
    over tutorials/montages). A local **VLM review pipe** (Qwen3-VL-8B via LM Studio,
    proposals only — never ground truth, never merged into labels) now produces opinion
    proposals for the 120 high-value units (`SP001`, `NORMAL001`, `UE003`), smoke-tested on
    one unit; explicit `--batch` required for the full run — see
    `docs/research/stage12d-extra-vlm-review-pipe.md`. Stage 12E audited the full 120/120
    batch run (`docs/research/stage12e-vlm-batch-audit.md`).
5. Label acquired data (binary per-frame: normal / failing / uncertain) — human review
    **stopped at 24 units** (UE003 is largely setup/non-print footage with low information
    value). Stage 12F records the six-unit tranche NORMAL001×3 `failing/stringing`, SP001×3
    `failing/spaghetti` merged into `verified_labels.json` (6 units), plus 18 additional UE003
    decisions kept in the manifest (unmerged); VLM agreed **0/6** on the overlapped audit
    sample (observation, not an accuracy metric) — see
    `docs/research/stage12f-first-human-review.md`.
6. Run Experiment 2: augment training data with external data, evaluate on frozen test set.
7. Benchmark edge latency / memory on Raspberry Pi (see `EDGE.md`).
8. Build the C++ runtime.

## Experiment plan

| # | Task | Model | Dataset | Timeline |
|---|---|---|---|---|
| 1 | Binary classification (✅ done) | ShuffleNetV2 x1.0 (winner) | 3d-print-errors | Week 1–2 |
| 11B | Leave-cluster-out (✅ done) | ShuffleNetV2 x1.0 | 3d-print-errors | Week 4 |
| 2 | External data augmentation | ShuffleNetV2 x1.0 | 3d-print-errors + external video frames | TBD |
| 4 | Temporal smoothing | Experiment 1 model outputs | Video sequences | TBD |
| 3 | Anomaly detection | EfficientAD-S | Own normal prints | TBD |

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
| Stage 10 cluster domain audit | `docs/research/stage10-cluster-domain-audit.md` |
| Stage 11A experiment design | `docs/research/stage11a-cluster-generalization-design.md` |
| Stage 11B execution preflight | `docs/research/stage11b-execution-preflight.md` |
| Stage 11B results | `docs/research/stage11b-cluster-generalization-results.md` |
| Stage 12 data acquisition design | `docs/research/stage12-data-acquisition-design.md` |
| Stage 12A video source candidates | `docs/research/stage12a-video-source-candidates.md` |
| Stage 12B video acquisition pilot | `docs/research/stage12b-video-acquisition-pilot.md` |
| Stage 12C external pilot dataset | `docs/research/stage12c-external-pilot-dataset.md` |
| Stage 12D human review workflow | `docs/research/stage12d-human-review-workflow.md` |
| Stage 12D-extra VLM review pipe | `docs/research/stage12d-extra-vlm-review-pipe.md` |
| Stage 12E VLM batch audit | `docs/research/stage12e-vlm-batch-audit.md` |
| Stage 12F first human review tranche | `docs/research/stage12f-first-human-review.md` |
| Open-source projects | `docs/research/open-source-projects.md` |
| Lightweight models | `docs/research/lightweight-models-comparison.md` |
| Dataset records | `docs/research/datasets/` |
