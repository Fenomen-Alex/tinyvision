# Leave-Cluster-Out Generalization Experiment Design

**Status:** design_only
**Objective:** Test whether stringing failures generalize poorly across recording/session domains by withholding entire date clusters from training and evaluating on the unseen cluster.

---

## Recording Inventory

| Split | Recordings | Frames |
|-------|-----------|--------|
| train | 105 | 11254 |
| val | 22 | 2174 |
| test | 24 | 2859 |

**Train stringing recordings:** 23 (2341 frames)
**Train under-extrusion recordings:** 37 (3331 frames)
**Test note:** FROZEN; not used in this experiment

---

## Stringing Recording Table

| Recording ID | Date | Frames | Stringing Frames | Split | Composition |
|-------------|------|--------|-----------------|-------|-------------|
| Recording_14.02.2023_17_53_55 | 14.02.2023 | 181 | 181 | train | PURE_FAILING |
| Recording_14.02.2023_18_15_37 | 14.02.2023 | 85 | 85 | train | PURE_FAILING |
| Recording_14.02.2023_18_39_51 | 14.02.2023 | 94 | 94 | train | PURE_FAILING |
| Recording_15.02.2023_14_15_54 | 15.02.2023 | 52 | 52 | train | PURE_FAILING |
| Recording_20122022_13_39_27 | 20122022 | 138 | 138 | train | PURE_FAILING |
| Recording_20122022_13_53_11 | 20122022 | 3 | 3 | train | PURE_FAILING |
| Recording_20122022_14_15_06 | 20122022 | 2 | 2 | train | PURE_FAILING |
| Recording_20122022_16_09_41 | 20122022 | 3 | 3 | train | PURE_FAILING |
| Recording_20122022_16_55_38 | 20122022 | 21 | 21 | train | PURE_FAILING |
| Recording_21122022_19_21_52 | 21122022 | 113 | 113 | train | PURE_FAILING |
| Recording_21122022_19_53_17 | 21122022 | 174 | 174 | train | PURE_FAILING |
| Recording_22122022_11_00_38 | 22122022 | 146 | 146 | train | PURE_FAILING |
| Recording_22122022_11_49_58 | 22122022 | 165 | 165 | train | PURE_FAILING |
| Recording_22122022_12_03_39 | 22122022 | 174 | 174 | train | PURE_FAILING |
| Recording_22122022_12_19_15 | 22122022 | 93 | 93 | train | PURE_FAILING |
| Recording_22122022_12_37_10 | 22122022 | 140 | 140 | train | PURE_FAILING |
| Recording_22122022_15_18_40 | 22122022 | 158 | 158 | train | PURE_FAILING |
| Recording_22122022_15_50_18 | 22122022 | 102 | 102 | train | PURE_FAILING |
| Recording_22122022_16_35_52 | 22122022 | 119 | 119 | train | PURE_FAILING |
| Recording_22122022_18_04_20 | 22122022 | 163 | 163 | train | PURE_FAILING |
| Recording_23.02.2023_16_44_22 | 23.02.2023 | 96 | 96 | train | PURE_FAILING |
| Recording_23.02.2023_19_58_44 | 23.02.2023 | 24 | 24 | train | PURE_FAILING |
| Recording_23.02.2023_20_15_22 | 23.02.2023 | 95 | 95 | train | PURE_FAILING |

---

## Stringing Clusters

### Cluster `14.02.2023`

- **Definition:** date=14.02.2023
- **Recordings:** 3
- **Total frames:** 360
- **Stringing frames:** 360
- **Rationale:** Natural date/session cluster: 3 recording(s) from 14.02.2023

### Cluster `15.02.2023`

- **Definition:** date=15.02.2023
- **Recordings:** 1
- **Total frames:** 52
- **Stringing frames:** 52
- **Rationale:** Natural date/session cluster: 1 recording(s) from 15.02.2023
- **Exploratory:** YES — Only 1 recording(s) — insufficient for robust evaluation

### Cluster `20122022`

- **Definition:** date=20122022
- **Recordings:** 5
- **Total frames:** 167
- **Stringing frames:** 167
- **Rationale:** Natural date/session cluster: 5 recording(s) from 20122022

### Cluster `21122022`

- **Definition:** date=21122022
- **Recordings:** 2
- **Total frames:** 287
- **Stringing frames:** 287
- **Rationale:** Natural date/session cluster: 2 recording(s) from 21122022
- **Exploratory:** YES — Only 2 recording(s) — insufficient for robust evaluation

### Cluster `22122022`

- **Definition:** date=22122022
- **Recordings:** 9
- **Total frames:** 1260
- **Stringing frames:** 1260
- **Rationale:** Natural date/session cluster: 9 recording(s) from 22122022

### Cluster `23.02.2023`

- **Definition:** date=23.02.2023
- **Recordings:** 3
- **Total frames:** 215
- **Stringing frames:** 215
- **Rationale:** Natural date/session cluster: 3 recording(s) from 23.02.2023

---

## Under-Extrusion Control Clusters

### Cluster `01122022`

- **Definition:** date=01122022
- **Recordings:** 1
- **Total frames:** 82
- **Rationale:** Under-extrusion control cluster: 1 recording(s) from 01122022
- **Exploratory:** YES — Only 1 recording(s) — insufficient for robust evaluation

### Cluster `02012023`

- **Definition:** date=02012023
- **Recordings:** 2
- **Total frames:** 166
- **Rationale:** Under-extrusion control cluster: 2 recording(s) from 02012023
- **Exploratory:** YES — Only 2 recording(s) — insufficient for robust evaluation

### Cluster `02122022`

- **Definition:** date=02122022
- **Recordings:** 1
- **Total frames:** 167
- **Rationale:** Under-extrusion control cluster: 1 recording(s) from 02122022
- **Exploratory:** YES — Only 1 recording(s) — insufficient for robust evaluation

### Cluster `04012023`

- **Definition:** date=04012023
- **Recordings:** 2
- **Total frames:** 236
- **Rationale:** Under-extrusion control cluster: 2 recording(s) from 04012023
- **Exploratory:** YES — Only 2 recording(s) — insufficient for robust evaluation

### Cluster `05012023`

- **Definition:** date=05012023
- **Recordings:** 7
- **Total frames:** 495
- **Rationale:** Under-extrusion control cluster: 7 recording(s) from 05012023

### Cluster `06012023`

- **Definition:** date=06012023
- **Recordings:** 1
- **Total frames:** 124
- **Rationale:** Under-extrusion control cluster: 1 recording(s) from 06012023
- **Exploratory:** YES — Only 1 recording(s) — insufficient for robust evaluation

### Cluster `08012023`

- **Definition:** date=08012023
- **Recordings:** 1
- **Total frames:** 170
- **Rationale:** Under-extrusion control cluster: 1 recording(s) from 08012023
- **Exploratory:** YES — Only 1 recording(s) — insufficient for robust evaluation

### Cluster `10012023`

- **Definition:** date=10012023
- **Recordings:** 8
- **Total frames:** 1079
- **Rationale:** Under-extrusion control cluster: 8 recording(s) from 10012023

### Cluster `11012023`

- **Definition:** date=11012023
- **Recordings:** 2
- **Total frames:** 261
- **Rationale:** Under-extrusion control cluster: 2 recording(s) from 11012023
- **Exploratory:** YES — Only 2 recording(s) — insufficient for robust evaluation

### Cluster `16.02.2023`

- **Definition:** date=16.02.2023
- **Recordings:** 5
- **Total frames:** 520
- **Rationale:** Under-extrusion control cluster: 5 recording(s) from 16.02.2023

### Cluster `17.02.2023`

- **Definition:** date=17.02.2023
- **Recordings:** 2
- **Total frames:** 209
- **Rationale:** Under-extrusion control cluster: 2 recording(s) from 17.02.2023
- **Exploratory:** YES — Only 2 recording(s) — insufficient for robust evaluation

### Cluster `29112022`

- **Definition:** date=29112022
- **Recordings:** 4
- **Total frames:** 317
- **Rationale:** Under-extrusion control cluster: 4 recording(s) from 29112022

### Cluster `29122022`

- **Definition:** date=29122022
- **Recordings:** 1
- **Total frames:** 67
- **Rationale:** Under-extrusion control cluster: 1 recording(s) from 29122022
- **Exploratory:** YES — Only 1 recording(s) — insufficient for robust evaluation

---

## Experimental Runs

### Run `str_holdout_22122022`

- **Held-out cluster:** `22122022`
- **Held-out recordings:** 9
- **Held-out frames:** 1260
- **Held-out stringing frames:** 1260
- **Train recordings (held-in):** 96
- **Train frames (held-in):** 9994
- **Train stringing frames (held-in):** 1081
- **Val recordings:** 21
- **Val frames:** 2080
- **Note:** Excluded 1 val recording(s) from date 22122022 to avoid leakage: ['Recording_22122022_18_48_47']

### Run `str_holdout_20122022`

- **Held-out cluster:** `20122022`
- **Held-out recordings:** 5
- **Held-out frames:** 167
- **Held-out stringing frames:** 167
- **Train recordings (held-in):** 100
- **Train frames (held-in):** 11087
- **Train stringing frames (held-in):** 2174
- **Val recordings:** 21
- **Val frames:** 2082
- **Note:** Excluded 1 val recording(s) from date 20122022 to avoid leakage: ['Recording_20122022_20_06_10']

### Run `ue_control_10012023`

- **Held-out cluster:** `ue_10012023`
- **Held-out recordings:** 8
- **Held-out frames:** 1079
- **Held-out stringing frames:** 0
- **Train recordings (held-in):** 97
- **Train frames (held-in):** 10175
- **Train stringing frames (held-in):** 2341
- **Val recordings:** 21
- **Val frames:** 2055
- **Note:** Excluded 1 val recording(s) from date 10012023: ['Recording_10012023_18_46_21']

### Run `ue_control_05012023`

- **Held-out cluster:** `ue_05012023`
- **Held-out recordings:** 7
- **Held-out frames:** 495
- **Held-out stringing frames:** 0
- **Train recordings (held-in):** 98
- **Train frames (held-in):** 10759
- **Train stringing frames (held-in):** 2341
- **Val recordings:** 19
- **Val frames:** 1861
- **Note:** Excluded 3 val recording(s) from date 05012023: ['Recording_05012023_11_08_00', 'Recording_05012023_11_27_16', 'Recording_05012023_15_08_31']

---

## Leakage Audit Results

### `str_holdout_22122022` — PASS

- [PASS] **L1:** Held-out recordings are correctly excluded from held-in train set — held_out=9, held_in=96, expected=96
- [PASS] **L2:** No frame from held-out cluster appears in held-in training — Clean
- [PASS] **L3:** Session identity remains atomic — All held-out recordings are complete sessions
- [PASS] **L4:** Official validation/test remain untouched — Clean
- [PASS] **L5:** No duplicate recording IDs across held-in train and held-out — Verified
- [PASS] **L6:** No generated fixture accidentally enters training — No synthetic data used
- [PASS] **L7:** No test-derived information influenced cluster selection — Clusters selected from train metadata only
- [PASS] **L8:** No hyperparameters selected from held-out cluster results — Design phase only; no training executed

### `str_holdout_20122022` — PASS

- [PASS] **L1:** Held-out recordings are correctly excluded from held-in train set — held_out=5, held_in=100, expected=100
- [PASS] **L2:** No frame from held-out cluster appears in held-in training — Clean
- [PASS] **L3:** Session identity remains atomic — All held-out recordings are complete sessions
- [PASS] **L4:** Official validation/test remain untouched — Clean
- [PASS] **L5:** No duplicate recording IDs across held-in train and held-out — Verified
- [PASS] **L6:** No generated fixture accidentally enters training — No synthetic data used
- [PASS] **L7:** No test-derived information influenced cluster selection — Clusters selected from train metadata only
- [PASS] **L8:** No hyperparameters selected from held-out cluster results — Design phase only; no training executed

### `ue_control_10012023` — PASS

- [PASS] **L1:** Held-out recordings are correctly excluded from held-in train set — held_out=8, held_in=97, expected=97
- [PASS] **L2:** No frame from held-out cluster appears in held-in training — Clean
- [PASS] **L3:** Session identity remains atomic — All held-out recordings are complete sessions
- [PASS] **L4:** Official validation/test remain untouched — Clean
- [PASS] **L5:** No duplicate recording IDs across held-in train and held-out — Verified
- [PASS] **L6:** No generated fixture accidentally enters training — No synthetic data used
- [PASS] **L7:** No test-derived information influenced cluster selection — Clusters selected from train metadata only
- [PASS] **L8:** No hyperparameters selected from held-out cluster results — Design phase only; no training executed

### `ue_control_05012023` — PASS

- [PASS] **L1:** Held-out recordings are correctly excluded from held-in train set — held_out=7, held_in=98, expected=98
- [PASS] **L2:** No frame from held-out cluster appears in held-in training — Clean
- [PASS] **L3:** Session identity remains atomic — All held-out recordings are complete sessions
- [PASS] **L4:** Official validation/test remain untouched — Clean
- [PASS] **L5:** No duplicate recording IDs across held-in train and held-out — Verified
- [PASS] **L6:** No generated fixture accidentally enters training — No synthetic data used
- [PASS] **L7:** No test-derived information influenced cluster selection — Clusters selected from train metadata only
- [PASS] **L8:** No hyperparameters selected from held-out cluster results — Design phase only; no training executed

---

## Control Condition

**Description:** Comparable leave-cluster-out experiment using under-extrusion. Under-extrusion has 37 train recordings across 13 dates (Stage 10), making it the best-represented failure type. If under-extrusion shows less cluster-holdout degradation than stringing, this supports the hypothesis that stringing has poor cross-cluster generalization specifically.

**Methodology:** Same leave-cluster-out design as stringing, but holding out an under-extrusion date cluster.  Same architecture, same training config, same evaluation metrics.

**Selected clusters:** 10012023, 05012023

**Interpretation:** Compare: (a) stringing held-out FNR vs under-extrusion held-out FNR; (b) stringing validation-to-held-out degradation vs under-extrusion validation-to-held-out degradation.  If stringing degrades more, this supports H1 (stringing has poor cross-cluster generalization).

---

## Baseline

### Experiment 1 Reference

- Checkpoint: `experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/best.pt`
- Val F1@0.65: **0.9135**
- Test F1@0.65: **0.8799**
- Test stringing FNR@0.65: **0.213**
- Test under-extrusion FNR@0.65: **0.155**
- Test normal FPR@0.65: **0.103**
- Note: Historical reference only.  Stage 11B results will be compared against these numbers but the test set is NOT used for model selection or hyperparameter tuning.

**Comparison principle:** For each run, compare: (a) held-out cluster metrics vs ordinary validation metrics from the same run; (b) stringing runs vs under-extrusion control runs; (c) cross-run variance.

---

## Training Configuration

- Model: `shufflenet_v2_x1_0`
- Image size: 224
- Optimizer: adam
- Learning rate: 0.001
- Weight decay: 0.0001
- Epochs: 30 (early stopping patience: 10)
- Batch size: 32
- Precision: fp32
- Seed: 42
- Augmentation: hflip=True, rot=15, cj=0.2, crop=False, normalize=True
- No HP search: True
- No architecture search: True
- No augmentation search: True

---

## Metrics

**Primary:**
- failing_recall: TP / (TP + FN) — fraction of failing frames correctly detected
- fnr: FN / (FN + TP) — fraction of failing frames missed
- f1: 2 * precision * recall / (precision + recall)

**Secondary:**
- precision: TP / (TP + FP) — fraction of predicted-failing that are truly failing
- fpr: FP / (FP + TN) — fraction of normal frames falsely flagged
- accuracy: (TP + TN) / total
- auroc: Area under ROC curve (threshold-independent)

**Threshold:** 0.65
**Frame-level:** True
**Recording-level:** descriptive per-recording summaries only; NOT a recording-classification task

---

## Run Count & Wall Clock

- Stringing holdout runs: 2
- Under-extrusion control runs: 2
- Total: 4
- Per-run estimate: 15-18 minutes (MPS) training + 1-2 minutes eval = 16-20 minutes
- Total wall clock: 64-80 minutes (4 runs sequentially)
- Total disk: ~40 MB (4 checkpoints)

---

## Success & Interpretation Criteria

### Strong evidence of domain generalization failure

Consistent degradation across BOTH independent held-out stringing clusters (22122022 and 20122022), with held-out FNR substantially worse than ordinary validation FNR from the same run.

### Moderate evidence

Degradation in ONE OR MORE held-out stringing clusters, but insufficient independent clusters for strong conclusion.  Under-extrusion control shows less degradation.

### Weak / inconclusive

High between-run variance or insufficient cluster count.  Results do not clearly distinguish domain-generalization failure from inherent difficulty.

### No support

Unseen-cluster performance is broadly comparable to ordinary validation performance.  Stringing and under-extrusion show similar degradation patterns.

### Abort criteria

If F1 on the held-in validation set regresses by >5pp relative to Experiment 1 validation F1 (0.9135), the training itself may have failed and results are not interpretable.

---

## Official Test Set Firewall

- **Frozen:** True
- **May reference:** Historical context only (e.g., 'catastrophic test recording had FNR ~0.898 in Stage 8').

**Must not influence:**
- cluster selection
- split construction
- hyperparameters
- model selection
- threshold selection
- success criteria
- acquisition decisions

---

## Limitations

1. This experiment cannot prove real-world generalization.
1. Date is a proxy for session/provenance, not a causal factor.
1. Causal effects of date, lighting, camera, or printer cannot be established.
1. Production performance is not measured.
1. Limited to 2 independent stringing holdout clusters — statistical power is modest.
1. The under-extrusion control has different sample sizes and may not be a perfect comparison.
1. Only ShuffleNetV2 x1.0 is evaluated; results may differ for other architectures.
1. The frozen Experiment 1 test set provides historical context but is not the primary evaluation target.

---

## Stage 11B Execution Plan

**Clusters held out:** 22122022, 20122022
**Control clusters:** 10012023, 05012023
**Total runs:** 4

**Steps:**
1. Generate temporary CSV splits for each run (preflight)
2. Run leakage audit on each run's splits
3. Train ShuffleNetV2 x1.0 on each run's training split
4. Evaluate on each run's validation split
5. Evaluate on each run's held-out cluster
6. Compare stringing runs vs under-extrusion control runs
7. Report per-cluster and aggregate metrics
8. Interpret against success criteria

**Abort conditions:**
- Validation F1 regresses by >5pp vs Experiment 1
- Any training run fails to converge (val loss does not decrease)
- Leakage audit fails

**Expected artifacts:**
- `experiments/stage11a_<run_id>/artifacts/best.pt`
- `experiments/stage11a_<run_id>/artifacts/metrics.json`
- `experiments/stage11a_<run_id>/config.yaml`
- `experiments/stage11a_<run_id>/status.json`
- `docs/research/stage11a-results.json`
- `docs/research/stage11a-results.md`

---

*Generated 2026-09-04 11:18 UTC — analysis only, no training executed.*
