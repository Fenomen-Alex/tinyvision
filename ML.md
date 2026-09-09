# TinyVision ML Status

**Last updated**: 2026-09-09
**Status**: RESEARCH PROTOTYPE — RC1 trained, exported, and evaluated; **NOT
approved for production/release deployment**. Release finalized as
`v1.0.0-rc1`. Full results in
[`docs/research/rc1-release-candidate.md`](docs/research/rc1-release-candidate.md).

```text
Runtime: validated (dev hardware only)
Model: research prototype (RC1, not production-approved)
Data: training dataset + provenance recorded; target-domain generalization inadequate
Hardware: unvalidated (no physical device / no physical camera)
Real-world generalization: inadequate — dominant issue: normal-source false alarms
Release candidate: RC1 exported; NOT approved for deployment
```

**RC1 outcome (2026-09-09).** Release candidate 1 was successfully trained on
the provenance-typed release dataset and exported to PyTorch / ONNX / INT8
ONNX. Model: **ShuffleNetV2 x1.0**, binary classification (failing vs normal),
image size 224, batch 32, adam 1e-3, 30 epochs, seed 42, decision threshold 0.5
(never tuned). Exact artifacts:

* checkpoint: `experiments/rc1_release_candidate_shufflenet_v2_x1_0/artifacts/best.pt`
* ONNX: `experiments/rc1_release_candidate_shufflenet_v2_x1_0/artifacts/model.onnx` (4.93 MB)
* INT8 ONNX: `experiments/rc1_release_candidate_shufflenet_v2_x1_0/artifacts/model_int8.onnx` (1.42 MB)
* configuration: `training/configs/rc1_release_candidate.yaml`
* dataset ledger: `datasets/processed/release-rc1/derived_manifest.json`

**RC1 is NOT approved for production/release deployment.** The dominant
unresolved issue is **normal-source false alarms and poor cross-environment
generalization**: RC1 flags the independent normal holdout at FPR 1.000 and
immediately (frame ~8) alarms on every real-world normal clip, while the Trier
holdout shows strong environment-correlated predictions (camera-3 environments
missed: layer_shift recall 0.000, normal FPR 0.862). Frozen test performance is
a fair improvement over baseline (FPR 0.170 → 0.091, F1 0.876, AUROC 0.892),
but that does not offset the transfer failures. TinyVision remains a research
prototype.

The **dataset-staging + contributor-intake pipeline** is now implemented
(research/infrastructure only, no training): environment-centric manifests,
immutable environment identity, environment-level leakage-free splitting,
intake validation, human-review verification states, and the next-training
readiness gate — see
[`docs/research/dataset-staging-pipeline.md`](docs/research/dataset-staging-pipeline.md)
and the intake kit under `docs/contributors/`. Staged artefacts live in
`datasets/staging/`. The 5-human-verified-failure-environment requirement is a
**release-class** acquisition gate and was deliberately not required for RC1
training, which used `source_annotation` labels. It still gates any future
production-readiness claim.

The **release-candidate dataset build** (2026-09-08) staged 24 Trier FFF-QC v2
environments (27,154 frames, CC BY 4.0, MD5-verified), kept Bambu's timelapse
contributors as candidate_only, and prepared 2,931 human review units
(contact-sheet sequence units, no pre-filled decisions). `tv env-readiness`:
27/20 training environments, 3/2 normal holdouts, 6/2 failure holdouts, 0/5
human-verified failure environments. Training proceeded on 2026-09-09 with
`source_annotation` labels (see
[`docs/research/release-dataset-build.md`](docs/research/release-dataset-build.md)),
producing **RC1** — evaluated, exported, and *not* approved for deployment
([`docs/research/rc1-release-candidate.md`](docs/research/rc1-release-candidate.md)).

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

**SKZ probe experiments A/B update (see `docs/research/skz-probe-experiments.md`, 2026-09-06):**
the frozen binary baseline (ShuffleNetV2 x1.0, threshold 0.5 on P(failing)) was evaluated
on the 166-image, source-grouped SKZ Nozzle-Cam probe, and an auxiliary-training run was
trained with a **group-split** 135-image SKZ-aux subset (seed 42, fraction 0.8; the 31-image
held-out slice stayed external). Findings:

* **Experiment A (external generalization, no training):** failure detection transfers
  almost perfectly to the SKZ nozzle-cam domain — failing-positive recall 0.966 (FNR
  0.034), Unterextrusion 0.98 / Stringing 1.0 / Spaghetti 0.92. Normal recognition does
  **not** transfer: 46/50 candidate-normal probe frames predicted failing (FPR 0.92,
  AUROC 0.61).
* **Experiment B (auxiliary 135 SKZ images):** on the frozen TinyVision held-out test all
  metrics improved (acc 0.836→0.878, F1 0.876→0.906, FPR 0.170→0.060, AUROC 0.900→0.951);
  no degradation. On the held-out SKZ slice (31 images), it still detected every failure
  (recall 1.0) but did **not** fix the SKZ-normal false-alarm problem (FPR 0.91→1.0,
  AUROC 0.72→0.51; slice too small to be robust).
* Interpretation: the model's weakness is out-of-domain *normal* recognition, not failure
  detection. SKZ normal labels are **candidate** labels (provider annotations); SKZ numbers
  are exploratory.

**Experiment 2 update (`exp2_external_video_aug`, see `docs/research/exp2-external-video-augmentation.md`, 2026-09-06):**
an auxiliary-training run (seed 42, same hyperparameters) was trained on the printing-errors
train plus the **44 human-verified external failing frames** (Stage 12F `verified_labels.json`,
stringing NORMAL001 + spaghetti SP001); the 24 human-review UE003 normal frames were kept as an
**eval-only** external-normal set seen by neither model. Findings:

* **Frozen held-out test regressed**: acc 0.836→0.738, failing-positive recall 0.839→0.628, F1
  0.876→0.769, AUROC 0.900→0.870. FPR fell 0.170→0.013 as a side effect of the same
  normal-leaning shift, not a corrected false-alarm mechanism.
* **External-normal false alarms persist**: UE003 normal frames FPR ~1.0 for the baseline (24/24)
  and 0.958 for 2A (23/24). Adding **failing** frames does not fix out-of-domain *normal*
  recognition.
* **External failing frames**: both models already detect them (recall 0.977). The augmentation
  added nothing, and 44 frames / 2 sources is too small to transfer a domain.
* Conclusion: verified external set is too small for a meaningful augmentation experiment (a
  limitation per policy, not papered over). Next step is collecting/verifying more labelled
  frames — in particular **normal** frames — not tuning this experiment.

**Stage 13 update (`exp3_target_normal_aug`, see `docs/research/stage13-experiment.md`, 2026-09-07):**
trained on the frozen train plus **333 VLM-screened target-domain NORMAL frames from 7 independent
sources** (pending human review). Key results at fixed threshold 0.5:

* **Held-out normal FPR drops 1.000 → 0.258** (120 frames, 2 fully held-out sources) and ~7-8x on
  the 7-video normal clips (0.96→0.11, 0.87→0.14, 0.93→0.11). Confirms the normal-diversity tail risk.
* **Cost: model becomes too normal-biased.** Frozen failure recall falls (FN rises), the confirmed
  external spaghetti `87gsL-Luvis` is missed entirely (raw-failing 0.000), SKZ failure recall
  collapses (1.0 → 0.10). Exp3 is **experimental, not release-usable**.
* Conclusion: normal-FP tail fixed but failure detection sacrificed; motivates a failure-balancing
  round (Stage 14).

**Stage 14 update (`exp4_target_balanced`, see `docs/research/stage14-experiment.md`, 2026-09-07):**
trained identically on frozen train + Stage-13 333 normals + **100 VLM-verified target-domain
FAILING frames from 2 independent fixed-cam sources** (70 spaghetti `ZYfXW0JN2k4`, 30 blob
`3zXQrw0E-Ag`; pending human review). Key results at fixed threshold 0.5:

* **Failure detection restored (best of all checkpoints):** frozen failing recall 0.950, FNR 0.050,
  AUROC 0.925; confirmed external spaghetti `87gsL-Luvis` re-detected (0% → 21%, first event 9);
  SKZ failure recall 0.207 → 0.400.
* **But normal-FP regression is catastrophic:** held-out normal FPR 1.000 (exp3 was 0.258); 7-video
  normal clips ~11-14% → ~39-54%; SKZ normal FPR 0.240 → 0.364.
* **Conclusion: the balancing hypothesis is refuted as stated.** Small-count target-domain
  augmentation cannot satisfy both tails at threshold 0.5; neither Exp3 nor Exp4 is a release
  candidate. **Experimentation STOPPED.** Recommended future direction: scale-diverse real or
  synthetic data collection, not count tweaks on 2-camera YouTube sourcing.
* All 333 normal + 100 failing augmentation frames remain `pending_human_review` (mandatory before
  any release use).

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
| SKZ Nozzle-Cam (SKZ/KunststoffZentrum) | External generalization + auxiliary training (probe) | CC BY-SA 4.0 (HF) / CC BY 4.0 (Zenodo), unresolved | Used (test-split probe; see skz-probe-experiments.md) |

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

> **Historical log (Stages 12–14).** Experimentation is now STOPPED; this list is
> preserved for the record and superseded by
> [`tinyvision-final-synthesis.md`](docs/research/tinyvision-final-synthesis.md).

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
6. ✅ SKZ probe experiments A/B (external generalization + auxiliary-training run) —
    done 2026-09-06, see `docs/research/skz-probe-experiments.md`; run the original
    Exp 2 video-frames augmentation next (still TBD).
7. ✅ Benchmark edge latency / memory on Raspberry Pi (C++ runtime built and
   benchmarked on macOS workstation — see `docs/research/edge-runtime-benchmark.md`;
   Pi cross-platform re-measure remains TBD).
8. ✅ Build the C++ runtime (`runtime/`), achieve Python/POC behavioral parity,
   and establish the deployment path — see `docs/research/cpp-runtime-parity.md` and `EDGE.md`.

## Experiment plan

| # | Task | Model | Dataset | Timeline |
|---|---|---|---|---|
| 1 | Binary classification (✅ done) | ShuffleNetV2 x1.0 (winner) | 3d-print-errors | Week 1–2 |
| 11B | Leave-cluster-out (✅ done) | ShuffleNetV2 x1.0 | 3d-print-errors | Week 4 |
| 2-SKZ | SKZ external generalization + aux training (✅ done) | ShuffleNetV2 x1.0 | 3d-print-errors + SKZ probe aux | Week 8 |
| 2 | External data augmentation (video frames) (✅ done 2026-09-06) | ShuffleNetV2 x1.0 | 3d-print-errors + verified external video frames | Week 8 |
| 4 | Temporal smoothing (✅ done — EWMA + FSM operational in POC and C++ runtime) | Experiment 1 model outputs | Video sequences | Week 3 |
| 3 | Anomaly detection (not run — superseded by synthesis) | EfficientAD-S | Own normal prints | TBD |

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
| SKZ probe experiments A/B | `docs/research/skz-probe-experiments.md` |
| Experiment 2 external video-frames augmentation | `docs/research/exp2-external-video-augmentation.md` |
| Stage 13 target-domain normal augmentation | `docs/research/stage13-experiment.md` |
| Stage 13 target-normal dataset | `docs/research/stage13-target-normal-dataset.md` |
| Stage 14 target-domain failure dataset | `docs/research/stage14-target-failure-dataset.md` |
| Stage 14 failure-balancing experiment | `docs/research/stage14-experiment.md` |
| **Final experimental synthesis** | `docs/research/tinyvision-final-synthesis.md` |
| **Release dataset strategy** | `docs/research/tinyvision-data-strategy.md` |
| **Dataset staging pipeline** | `docs/research/dataset-staging-pipeline.md` |
| **Release-candidate dataset build** | `docs/research/release-dataset-build.md` |
| Open-source projects | `docs/research/open-source-projects.md` |
| Lightweight models | `docs/research/lightweight-models-comparison.md` |
| Dataset records | `docs/research/datasets/` |
