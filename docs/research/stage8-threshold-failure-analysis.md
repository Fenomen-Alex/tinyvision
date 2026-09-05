# Stage 8 — Threshold and Failure-Type Analysis

**Checkpoint:** `experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/best.pt` (frozen, unmodified)

Analysis only. No retraining, no new data, no hardware. The test set is
evaluated exactly once, using the threshold selected on validation.

## 1. Validation threshold sweep

Positives = failing frames. Metrics at each operating point on the
validation split (never test).

| Threshold | Acc | Prec | Recall | F1 | FNR | FPR |
|---|---|---|---|---|---|---|
| 0.30 | 0.8606 | 0.8820 | 0.9076 | 0.8946 | 0.0924 | 0.2272 |
| 0.40 | 0.8726 | 0.9043 | 0.8998 | 0.9020 | 0.1002 | 0.1783 |
| 0.50 | 0.8813 | 0.9202 | 0.8956 | 0.9077 | 0.1044 | 0.1453 |
| 0.55 | 0.8873 | 0.9328 | 0.8913 | 0.9116 | 0.1087 | 0.1202 |
| 0.60 | 0.8873 | 0.9386 | 0.8850 | 0.9110 | 0.1150 | 0.1083 |
| 0.65 | 0.8910 | 0.9456 | 0.8836 | 0.9135 | 0.1164 | 0.0951 |
| 0.70 | 0.8887 | 0.9488 | 0.8765 | 0.9112 | 0.1235 | 0.0885 |
| 0.80 | 0.8882 | 0.9572 | 0.8673 | 0.9100 | 0.1327 | 0.0727 |

**Validation-optimal threshold (F1):** `0.65`
- F1=0.9135, Prec=0.9456, Rec=0.8836
- FNR=0.1164, FPR=0.0951, Acc=0.8910

### Precision/recall trade-off

Lower thresholds favour recall (fewer missed failures) at the cost of
precision (more false alarms); higher thresholds do the opposite. The
F1 optimum balances these two. The full precision/recall curve is
reported in the `.json` artifact (`thresholds_evaluated`).

## 2. Frozen test evaluation

| Operating point | Acc | Prec | Recall | F1 | FNR | FPR |
|---|---|---|---|---|---|---|
| 0.50 baseline | 0.8360 | 0.9177 | 0.8385 | 0.8763 | 0.1615 | 0.1697 |
| selected 0.65 | 0.8447 | 0.9476 | 0.8213 | 0.8799 | 0.1787 | 0.1025 |

## 3. Failure-type breakdown (test, selected threshold)

Frame-level recall/precision for each fine class. Groups with few
samples are flagged; do not over-interpret.

| Failure type | Frames | Failing true | Recall | Prec | F1 | FNR | FPR |
|---|---|---|---|---|---|---|---|
| normal | 878 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.103 |
| stringing | 797 | 797 | 0.787 | 1.000 | 0.881 | 0.213 | 0.000 |
| under_extrusion | 1184 | 1184 | 0.845 | 1.000 | 0.916 | 0.155 | 0.000 |

## 4. Recording-level error ranking (diagnostic)

**Highest failing FNR:** `Recording_15.02.2023_13_19_52, Recording_06012023_12_17_19, Recording_01122022_16_06_44, Recording_06122022_14_49_53, Recording_03012023_19_32_03, Recording_16.02.2023_17_27_18, Recording_08012023_13_31_14, Recording_17.02.2023_15_18_59`

**Highest FPR on normal frames:** `Recording_03012023_15_32_58, Recording_05012023_13_10_42, Recording_10012023_22_39_52, Recording_03012023_19_32_03, Recording_03012023_15_48_43, Recording_20122022_17_05_36, Recording_06122022_19_29_09, Recording_06122022_14_49_53`

**Most false positives:** `Recording_05012023_13_10_42, Recording_20122022_17_05_36, Recording_03012023_15_32_58, Recording_06122022_19_29_09, Recording_10012023_22_39_52, Recording_07122022_20_17_29, Recording_03012023_15_48_43, Recording_03012023_19_32_03`

**Most false negatives:** `Recording_15.02.2023_13_19_52, Recording_06012023_12_17_19, Recording_01122022_16_06_44, Recording_06122022_14_49_53, Recording_03012023_19_32_03, Recording_08012023_13_31_14, Recording_17.02.2023_15_18_59, Recording_16.02.2023_17_27_18`

**Persistent problematic (both high FNR and high FP count):**
`none`

These rankings are diagnostic only. We do not claim recording-level
classification as the task, and we do not invent explanations without
visual/metadata evidence.

## 5. Hard examples (test, selected threshold)

### 5.1 Highest-confidence false negatives

| Recording | Frame idx | True | Pred | p(fail) | Failure |
|---|---|---|---|---|---|
| Recording_15.02.2023_13_19_52 | 7 | failing | normal | 0.0001 | stringing |
| Recording_15.02.2023_13_19_52 | 22 | failing | normal | 0.0003 | stringing |
| Recording_01122022_16_06_44 | 59 | failing | normal | 0.0003 | under_extrusion |
| Recording_01122022_16_06_44 | 64 | failing | normal | 0.0003 | under_extrusion |
| Recording_01122022_16_06_44 | 58 | failing | normal | 0.0003 | under_extrusion |
| Recording_15.02.2023_13_19_52 | 17 | failing | normal | 0.0003 | stringing |
| Recording_15.02.2023_13_19_52 | 11 | failing | normal | 0.0005 | stringing |
| Recording_15.02.2023_13_19_52 | 15 | failing | normal | 0.0006 | stringing |
| Recording_15.02.2023_13_19_52 | 19 | failing | normal | 0.0006 | stringing |
| Recording_01122022_16_06_44 | 66 | failing | normal | 0.0007 | under_extrusion |
| Recording_01122022_16_06_44 | 62 | failing | normal | 0.0009 | under_extrusion |
| Recording_06012023_12_17_19 | 106 | failing | normal | 0.0010 | under_extrusion |
| Recording_01122022_16_06_44 | 60 | failing | normal | 0.0010 | under_extrusion |
| Recording_06012023_12_17_19 | 98 | failing | normal | 0.0010 | under_extrusion |
| Recording_01122022_16_06_44 | 72 | failing | normal | 0.0010 | under_extrusion |

### 5.2 Highest-confidence false positives

| Recording | Frame idx | True | Pred | p(fail) | Failure |
|---|---|---|---|---|---|
| Recording_20122022_17_05_36 | 1 | normal | failing | 0.9968 | normal |
| Recording_20122022_17_05_36 | 10 | normal | failing | 0.9884 | normal |
| Recording_20122022_17_05_36 | 9 | normal | failing | 0.9853 | normal |
| Recording_20122022_17_05_36 | 8 | normal | failing | 0.9843 | normal |
| Recording_20122022_17_05_36 | 0 | normal | failing | 0.9828 | normal |
| Recording_20122022_17_05_36 | 7 | normal | failing | 0.9734 | normal |
| Recording_03012023_15_32_58 | 23 | normal | failing | 0.9700 | normal |
| Recording_20122022_17_05_36 | 19 | normal | failing | 0.9668 | normal |
| Recording_05012023_13_10_42 | 59 | normal | failing | 0.9588 | normal |
| Recording_03012023_15_32_58 | 18 | normal | failing | 0.9567 | normal |
| Recording_10012023_22_39_52 | 16 | normal | failing | 0.9518 | normal |
| Recording_05012023_13_10_42 | 35 | normal | failing | 0.9504 | normal |
| Recording_03012023_15_32_58 | 24 | normal | failing | 0.9474 | normal |
| Recording_20122022_17_05_36 | 3 | normal | failing | 0.9468 | normal |
| Recording_10012023_22_39_52 | 17 | normal | failing | 0.9421 | normal |

### 5.3 Borderline errors near the selected threshold

| Recording | Frame idx | True | Pred | p(fail) | Failure |
|---|---|---|---|---|---|
| Recording_05012023_13_10_42 | 32 | normal | failing | 0.6511 | normal |
| Recording_05012023_13_10_42 | 55 | normal | failing | 0.6535 | normal |
| Recording_06122022_14_49_53 | 126 | failing | normal | 0.6456 | stringing |
| Recording_06122022_14_49_53 | 148 | failing | normal | 0.6455 | stringing |
| Recording_01122022_16_06_44 | 38 | failing | normal | 0.6452 | under_extrusion |
| Recording_06122022_14_49_53 | 112 | failing | normal | 0.6444 | stringing |
| Recording_05012023_17_24_05 | 20 | normal | failing | 0.6573 | normal |
| Recording_03012023_15_32_58 | 12 | normal | failing | 0.6595 | normal |
| Recording_05012023_13_10_42 | 31 | normal | failing | 0.6604 | normal |
| Recording_05012023_13_10_42 | 41 | normal | failing | 0.6606 | normal |
| Recording_03012023_19_32_03 | 7 | normal | failing | 0.6610 | normal |
| Recording_07122022_20_17_29 | 55 | normal | failing | 0.6615 | normal |
| Recording_06122022_14_49_53 | 146 | failing | normal | 0.6381 | stringing |
| Recording_01122022_16_06_44 | 48 | failing | normal | 0.6365 | under_extrusion |
| Recording_06122022_19_29_09 | 71 | normal | failing | 0.6641 | normal |

## 6. Calibration (validation)

- **Brier score:** `0.0975` (0 = perfect; baseline for a balanced two-class oracle ~0.25)
- **ECE:** `0.0078` (10-bin, weighted by bin occupancy)
- **Systematic:** overconfident-failing=`True`, underconfident-failing=`False`

Bins:

| Bin | n | mean conf | observed freq | gap |
|---|---|---|---|---|
| 0.00-0.10 | 561 | 0.024 | 0.180 | 0.156 |
| 0.10-0.20 | 104 | 0.147 | 0.202 | 0.055 |
| 0.20-0.30 | 51 | 0.250 | 0.176 | 0.074 |
| 0.30-0.40 | 48 | 0.344 | 0.229 | 0.115 |
| 0.40-0.50 | 31 | 0.446 | 0.194 | 0.252 |
| 0.50-0.60 | 43 | 0.542 | 0.349 | 0.193 |
| 0.60-0.70 | 27 | 0.654 | 0.444 | 0.209 |
| 0.70-0.80 | 25 | 0.754 | 0.520 | 0.234 |
| 0.80-0.90 | 59 | 0.857 | 0.610 | 0.247 |
| 0.90-1.00 | 1225 | 0.994 | 0.974 | 0.020 |

## 7. Limitations & recommendation

| Limitation | Note |
|---|---|
| All numbers are frame-level on leaked-free data; recording-level is grouping metadata, not a detection task. | |
| Small failure-type groups (e.g. spaghetti) are not statistically meaningful. | |
| No model retrained, no calibration transform fitted for operational use. | |
| Results are dataset-internal; they do not imply real-world generalization. | |

**Note on calibration vs threshold selection:** if calibration is poor,
hard probability thresholds are less meaningful as calibrated confidences;
they still operate as an effective ranking. No calibration transform was
fitted in this analysis stage; if used later it must be fitted only on
validation and evaluated once on test, to avoid leakage.

---

## Artifacts

- `docs/research/stage8-threshold-failure-analysis.json` — machine-readable full results
- `docs/research/stage8-threshold-failure-analysis.md` — this report
- `training/tv/stage8.py` — analysis module
- `training/tests/test_stage8.py` — tests for the new logic