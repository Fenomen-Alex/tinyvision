# Stage 11B — Leave-Cluster-Out Generalization Results

**Date:** 2026-09-04 14:36 UTC
**Threshold:** 0.65
**Model:** ShuffleNetV2 x1.0 (frozen architecture)

---

## Summary

| Run | Type | Held-out | Frames | Recall | FNR | Notes |
|-----|------|----------|--------|--------|-----|-------|
| `str_holdout_22122022` | Stringing | `22122022` | 1260 | 0.9381 | 0.0619 | single-class |
| `str_holdout_20122022` | Stringing | `20122022` | 167 | 0.8383 | 0.1617 | single-class |
| `ue_control_10012023` | UE control | `ue_10012023` | 1079 | 0.9836 | 0.0164 | mixed |
| `ue_control_05012023` | UE control | `ue_05012023` | 495 | 0.6633 | 0.3367 | mixed |

---

## `str_holdout_22122022`

**Cluster:** `22122022`
**Train frames (held-in):** 9994
**Training epochs:** 16 / 30
**Training time:** 1367s
**Best val loss:** 0.7632

### Held-in Validation

- Samples: 2080
- Failing: 1323, Normal: 757
- **F1: 0.8831**
- Recall: 0.8481, FNR: 0.1519
- Precision: 0.9212, FPR: 0.1268
- Accuracy: 0.8572
- AUROC: 0.9121

### Held-out Cluster

**Single-class evaluation (100% failing).** FPR, AUROC, F1 are undefined or not meaningful.

- Samples: 1260
- Failing: 1260, Normal: 0
- **Detected: 1182, Missed: 78**
- **Recall: 0.9381, FNR: 0.0619**

### Known-domain Stringing (remaining train)

- Samples: 1081
- Recall: 1.0000, FNR: 0.0000
- **Delta recall (held-out minus known): -0.0619**
- **Delta FNR (held-out minus known): +0.0619**

### Per-recording breakdown

| Recording | Frames | Detected | Missed | Accuracy | Mean fail score |
|-----------|--------|----------|--------|----------|-----------------|
| `Recording_22122022_11_00_38` | 146 | 131 | 15 | 0.8973 | 0.8922 |
| `Recording_22122022_11_49_58` | 165 | 164 | 1 | 0.9939 | 0.9937 |
| `Recording_22122022_12_03_39` | 174 | 174 | 0 | 1.0000 | 0.9962 |
| `Recording_22122022_12_19_15` | 93 | 93 | 0 | 1.0000 | 0.9987 |
| `Recording_22122022_12_37_10` | 140 | 140 | 0 | 1.0000 | 0.9884 |
| `Recording_22122022_15_18_40` | 158 | 133 | 25 | 0.8418 | 0.8253 |
| `Recording_22122022_15_50_18` | 102 | 102 | 0 | 1.0000 | 0.9894 |
| `Recording_22122022_16_35_52` | 119 | 82 | 37 | 0.6891 | 0.6584 |
| `Recording_22122022_18_04_20` | 163 | 163 | 0 | 1.0000 | 0.9996 |

---

## `str_holdout_20122022`

**Cluster:** `20122022`
**Train frames (held-in):** 11087
**Training epochs:** 22 / 30
**Training time:** 1948s
**Best val loss:** 0.3518

### Held-in Validation

- Samples: 2082
- Failing: 1417, Normal: 665
- **F1: 0.9241**
- Recall: 0.8807, FNR: 0.1193
- Precision: 0.9720, FPR: 0.0541
- Accuracy: 0.9015
- AUROC: 0.9635

### Held-out Cluster

**Single-class evaluation (100% failing).** FPR, AUROC, F1 are undefined or not meaningful.

- Samples: 167
- Failing: 167, Normal: 0
- **Detected: 140, Missed: 27**
- **Recall: 0.8383, FNR: 0.1617**

### Known-domain Stringing (remaining train)

- Samples: 2174
- Recall: 0.9995, FNR: 0.0005
- **Delta recall (held-out minus known): -0.1612**
- **Delta FNR (held-out minus known): +0.1612**

### Per-recording breakdown

| Recording | Frames | Detected | Missed | Accuracy | Mean fail score |
|-----------|--------|----------|--------|----------|-----------------|
| `Recording_20122022_13_39_27` | 138 | 138 | 0 | 1.0000 | 0.9980 |
| `Recording_20122022_13_53_11` | 3 | 2 | 1 | 0.6667 | 0.6220 |
| `Recording_20122022_14_15_06` | 2 | 0 | 2 | 0.0000 | 0.0109 |
| `Recording_20122022_16_09_41` | 3 | 0 | 3 | 0.0000 | 0.0158 |
| `Recording_20122022_16_55_38` | 21 | 0 | 21 | 0.0000 | 0.0168 |

---

## `ue_control_10012023`

**Cluster:** `ue_10012023`
**Train frames (held-in):** 10175
**Training epochs:** 25 / 30
**Training time:** 2136s
**Best val loss:** 0.6421

### Held-in Validation

- Samples: 2055
- Failing: 1298, Normal: 757
- **F1: 0.8708**
- Recall: 0.8721, FNR: 0.1279
- Precision: 0.8694, FPR: 0.2246
- Accuracy: 0.8365
- AUROC: 0.9282

### Held-out Cluster

- Samples: 1079
- Failing: 975, Normal: 104
- **Detected: 959, Missed: 16**
- **Recall: 0.9836, FNR: 0.0164**
- Precision: 0.9917, FPR: 0.0769
- F1: 0.9876, Accuracy: 0.9778
- AUROC: 0.9951

### Known-domain Stringing (remaining train)

- Samples: 2341
- Recall: 0.9949, FNR: 0.0051
- **Delta recall (held-out minus known): -0.0113**
- **Delta FNR (held-out minus known): +0.0113**

### Per-recording breakdown

| Recording | Frames | Detected | Missed | Accuracy | Mean fail score |
|-----------|--------|----------|--------|----------|-----------------|
| `Recording_10012023_15_02_20` | 128 | 126 | 2 | 0.9844 | 0.9842 |
| `Recording_10012023_15_33_34` | 152 | 152 | 0 | 1.0000 | 0.9958 |
| `Recording_10012023_18_16_22` | 134 | 103 | 9 | 0.9328 | 0.7680 |
| `Recording_10012023_19_39_45` | 139 | 139 | 0 | 1.0000 | 0.9992 |
| `Recording_10012023_20_31_36` | 162 | 111 | 4 | 0.9383 | 0.7359 |
| `Recording_10012023_20_51_12` | 89 | 74 | 0 | 1.0000 | 0.8225 |
| `Recording_10012023_21_52_16` | 132 | 132 | 0 | 1.0000 | 0.9945 |
| `Recording_10012023_22_14_59` | 143 | 122 | 1 | 0.9790 | 0.8440 |

---

## `ue_control_05012023`

**Cluster:** `ue_05012023`
**Train frames (held-in):** 10759
**Training epochs:** 13 / 30
**Training time:** 1116s
**Best val loss:** 0.1901

### Held-in Validation

- Samples: 1861
- Failing: 1160, Normal: 701
- **F1: 0.9380**
- Recall: 0.9647, FNR: 0.0353
- Precision: 0.9127, FPR: 0.1526
- Accuracy: 0.9205
- AUROC: 0.9816

### Held-out Cluster

- Samples: 495
- Failing: 392, Normal: 103
- **Detected: 260, Missed: 132**
- **Recall: 0.6633, FNR: 0.3367**
- Precision: 0.9630, FPR: 0.0971
- F1: 0.7855, Accuracy: 0.7131
- AUROC: 0.8716

### Known-domain Stringing (remaining train)

- Samples: 2341
- Recall: 0.9996, FNR: 0.0004
- **Delta recall (held-out minus known): -0.3363**
- **Delta FNR (held-out minus known): +0.3363**

### Per-recording breakdown

| Recording | Frames | Detected | Missed | Accuracy | Mean fail score |
|-----------|--------|----------|--------|----------|-----------------|
| `Recording_05012023_11_49_39` | 144 | 106 | 23 | 0.8403 | 0.6767 |
| `Recording_05012023_13_33_01` | 88 | 30 | 8 | 0.7955 | 0.4526 |
| `Recording_05012023_13_54_18` | 37 | 37 | 0 | 1.0000 | 0.9940 |
| `Recording_05012023_14_15_36` | 56 | 30 | 6 | 0.8929 | 0.4942 |
| `Recording_05012023_14_32_45` | 37 | 7 | 30 | 0.1892 | 0.2071 |
| `Recording_05012023_15_23_21` | 68 | 49 | 1 | 0.9853 | 0.6942 |
| `Recording_05012023_18_14_03` | 65 | 1 | 64 | 0.0154 | 0.0680 |

---

## Cross-run Comparison

### Stringing holdout runs

| Run | Held-out recall | Known-domain recall | Delta |
|-----|----------------|--------------------|-------|
| `str_holdout_22122022` | 0.9381 | 1.0000 | -0.0619 |
| `str_holdout_20122022` | 0.8383 | 0.9995 | -0.1612 |

### Under-extrusion control runs

| Run | Held-out recall | Held-out F1 | Held-out FNR | Held-out FPR |
|-----|----------------|-------------|--------------|--------------|
| `ue_control_10012023` | 0.9836 | 0.9876 | 0.0164 | 0.0769 |
| `ue_control_05012023` | 0.6633 | 0.7855 | 0.3367 | 0.0971 |

---

## Frozen Context (NOT used for experiment decisions)

Experiment 1 test-set results (historical reference only):

| Metric | Value |
|--------|-------|
| Test F1@0.65 | 0.8799 |
| Test stringing FNR@0.65 | 0.213 |
| Test under-extrusion FNR@0.65 | 0.155 |
| Test normal FPR@0.65 | 0.103 |

---

## Conclusion

Mean held-out stringing FNR: 0.1118
Mean known-domain stringing FNR: 0.0003
Mean delta FNR: +0.1116

- `str_holdout_22122022`: FNR degraded by +0.0619 on unseen cluster (known=0.0000 -> held-out=0.0619). **Evidence of domain generalization failure.**
- `str_holdout_20122022`: FNR degraded by +0.1612 on unseen cluster (known=0.0005 -> held-out=0.1617). **Evidence of domain generalization failure.**

**Under-extrusion control:**
- `ue_control_10012023`: recall=0.9836 FNR=0.0164 F1=0.9876
- `ue_control_05012023`: recall=0.6633 FNR=0.3367 F1=0.7855

### Interpretation

This experiment tests whether stringing failures generalize poorly across 
recording/session domains. Two independent stringing clusters and two 
under-extrusion controls were evaluated. Results should be interpreted as 
evidence for or against domain-specific generalization failure, not as 
definitive statistical proof (n=2 stringing clusters is insufficient for 
strong statistical conclusions).

---

*Generated 2026-09-04 14:36 UTC — Stage 11B complete.*
