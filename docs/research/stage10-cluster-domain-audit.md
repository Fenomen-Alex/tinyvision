# Stage 10 — Cluster and Domain Generalization Audit

**Analysis only.** No training, no data modification, no checkpoint
changes, no threshold re-optimization. The test set is used solely to
document existing Stage 8 observations; no test statistic defines a
bucket, threshold, acquisition choice, or model change.

**Model:** `shufflenet_v2_x1_0_printing-errors_cdc9e2cad4 (frozen, Experiment 1)`
**Threshold:** 0.65 (Stage 8 val-selected; not re-optimized)
**Evidence splits:** train, val

## 1. Executive summary

This stage tests (attempts to falsify) five hypotheses from Stage 9:

| Hypothesis | Claim | Result |
|---|---|---|
| H1 | Stringing FNR varies by date/session cluster | **Supported** — `20122022` cluster fnr_mean=0.20, all others near 0 |
| H2 | Catastrophic stringing test recording reflects domain shift not in training | **Supported** — test recording `15.02.2023_13_19_52` has fnr=0.898; train `15.02.2023_14_15_54` has fnr=0.0 (distinct session) |
| H3 | The known false-positive normal test recording is darker than training normals | **Contradicted** — dark bucket has *lowest* FPR (0.003); brightness-FPR correlation r=0.22 |
| H4 | False-positive rate increases systematically with darkness | **Contradicted** — medium bucket FPR 0.055, bright 0.033, dark 0.003 |
| H5 | Under-extrusion data is sufficiently diverse to serve as comparison/control | **Partially supported** — 49 recordings, 15 dates, fnr_mean=0.05; but two catastrophic sub-sessions exist |

## 2. Stringing cluster analysis (H1)

Grouping is by the date token embedded in the recording id (Recording_<date>_<time>; date = ddMMyyyy or dd.MM.yyyy). Date is a provenance/session cluster label, NOT a causal factor.

| Date cluster | Recs | Stringing frames | FNR mean | FNR max | Exploratory |
|---|---|---|---|---|---|
| 14.02.2023 | 5 | 494 | 0.0255 | 0.1277 | False |
| 15.02.2023 | 1 | 52 | 0.0000 | 0.0000 | True |
| 20122022 | 5 | 167 | 0.2014 | 1.0000 | False |
| 21122022 | 3 | 430 | 0.0000 | 0.0000 | False |
| 22122022 | 10 | 1354 | 0.0006 | 0.0063 | False |
| 23.02.2023 | 3 | 215 | 0.0000 | 0.0000 | False |

**Key finding:** The `20122022` cluster (5 recordings, 167 stringing
frames) has fnr_mean=0.20, with one recording (`Recording_20122022_14_15_06`)
achieving fnr=1.0 (all 2 stringing frames missed). All other clusters are
near-zero FNR. This is strong evidence that stringing performance is not
uniform across sessions; the `20122022` cluster represents a harder domain.

## 3. Stringing recording-to-recording variation

- Train stringing recordings: 23
- Val stringing recordings: 4
- Val stringing dates: 14.02.2023, 21122022, 22122022
- Val FNRs: [0.0, 0.1277, 0.0, 0.0]
- VAL is the primary development-domain generalization evidence; training-set performance is descriptive only, not generalization proof.

Worst train stringing recordings:
- `Recording_20122022_14_15_06` (20122022): fnr=1.0, n=2 frames, med_score=0.5448
- `Recording_20122022_13_39_27` (20122022): fnr=0.0072, n=138 frames, med_score=0.9913
- `Recording_22122022_15_18_40` (22122022): fnr=0.0063, n=158 frames, med_score=0.9972
- `Recording_14.02.2023_17_53_55` (14.02.2023): fnr=0.0, n=181 frames, med_score=0.9996
- `Recording_14.02.2023_18_15_37` (14.02.2023): fnr=0.0, n=85 frames, med_score=0.9998

## 4. Darkness audit (H3/H4)

- Train normal recordings: 61
- Val normal recordings: 15
- Train normal brightness: mean=142.4, median=149.2, range=[61.4, 183.9]
- Val normal brightness: mean=143.2, median=148.4
- Brightness-FPR correlation: r=0.215
- Negative correlation would indicate darker normal recordings trend toward higher FPR. Correlation over ~recordings is descriptive and does NOT establish causality or semantic OOD.
- Brightness is a proxy, not proof of semantic domain shift; no causal claim is made.

**Interpretation:** H3 predicted that the known false-positive test
recording (`Recording_20122022_17_05_36`) fails because it is darker
than training normals. This is contradicted: the darkest bucket has the
lowest FPR (0.003), not the highest. The false-positive problem is not
driven by systematic darkness. H4 is also contradicted: there is no
monotonic FPR-darkness relationship.

## 5. Brightness buckets

**Method:** Bucket boundaries derived ONLY from the TRAIN normal brightness distribution (tertiles). No official TEST statistic was used. Explicitly avoiding tiny buckets; if a bucket is empty it is reported as empty rather than forcing a split.
**Train tertile boundaries:** low=141.0, high=155.8

| Bucket | Boundary | Recs | Frames | Mean brightness | FPR (frame) | FPR (rec mean) |
|---|---|---|---|---|---|---|
| dark | [..., 141.0) | 23 | 2999 | 113.3 | 0.0027 | 0.0053 |
| medium | [141.0, 155.8) | 28 | 1834 | 148.2 | 0.0545 | 0.0602 |
| bright | [155.8, ...) | 25 | 1315 | 163.1 | 0.0327 | 0.0586 |

Medium brightness has the highest FPR (0.055), not dark (0.003).
This confirms the darkness audit: FP behavior is not explained by
darkness alone.

## 6. Under-extrusion control (H5)

- Recordings: 49
- Dates: 15 (01122022, 02012023, 02122022, 03012023, 04012023, 05012023, 06012023, 08012023...)
- FNR: mean=0.0473, std=0.173, min=0.0, max=0.8889
- Splits: train, val
- Comparison to stringing is descriptive; session count is NOT claimed as the sole explanation for any performance difference.

Worst under-extrusion recordings:
- `Recording_29112022_21_41_27` (train, 29112022): fnr=0.8889, ue_frames=9, med_score=0.4631
- `Recording_05012023_11_08_00` (val, 05012023): fnr=0.8681, ue_frames=144, med_score=0.0235
- `Recording_11012023_12_28_37` (val, 11012023): fnr=0.0882, ue_frames=68, med_score=1.0000
- `Recording_08012023_13_05_05` (val, 08012023): fnr=0.0814, ue_frames=172, med_score=1.0000

**Interpretation:** H5 is partially supported. Under-extrusion has the
most diverse training domain (15 dates, 49 recordings, fnr_mean=0.05),
making it a reasonable control. However, two catastrophic sub-sessions
exist (`Recording_29112022_21_41_27` fnr=0.89, `Recording_05012023_11_08_00`
fnr=0.87), confirming that even the best-represented failure type has
domain-specific failures.

## 7. Spaghetti limitation

- Train: 5 recordings, 639 frames
- Val: 3 recordings, 346 frames
- Test: 0 recordings, 0 frames
- Official test set contains no spaghetti frames, so frozen test spaghetti performance cannot be measured and no generalization conclusion is possible. No spaghetti performance estimate is invented. Spaghetti remains a future-data priority.

## 8. Documented test observations (context only)

> Official test statistics are displayed for context ONLY. No test statistic defines a brightness bucket, selects a threshold, determines acquisition, sets session counts, or triggers a model change.

**Catastrophic false-negative:** `Recording_15.02.2023_13_19_52` (127 frames, 127 failing, FNR=0.898)
**High-confidence false-positive:** `Recording_20122022_17_05_36` (110 frames, 110 normal, FP=20, max_score=0.9968)

## 9. Image distribution (train normals)

- Frames sampled: 4618
- Method: Deterministic non-learned statistics only; no PCA, clustering, or learned embeddings are used. Analysis is intentionally kept simple.
- Mean brightness: 137.8
- Mean brightness std: 86.7
- Channel means (BGR): [138.7, 138.3, 136.5]

## 10. Acquisition decision (updated)

**Classifications (from existing evidence):**
- {
  "strongly_supported": "clear TRAIN/VAL evidence",
  "moderately_supported": "suggestive TRAIN/VAL evidence",
  "weakly_supported": "limited/ambiguous TRAIN/VAL evidence",
  "insufficient_evidence": "cannot conclude from available TRAIN/VAL data"
}

### stringing
- **Need:** independent sessions
- **Detail:** Prefer genuinely independent stringing sessions (distinct dates, objects, recording conditions, optical/camera conditions) over additional frames from existing recordings, because independent sessions add provenance/domain coverage whereas extra frames from one session only add within-session redundancy.

### dark_normal
- **Need:** normal images across lighting conditions
- **Detail:** Add normal-domain images spanning the brightness range, especially darker conditions, to broaden coverage.

### spaghetti
- **Need:** any independent spaghetti sessions
- **Detail:** Spaghetti has only 5 training recordings / 191 train frames and no test frames; it remains a future-data priority with no measurable frozen-test baseline.

### under_extrusion
- **Need:** maintain / modestly extend
- **Detail:** Under-extrusion has the most diverse training domain; treat it as the comparison/control for the stringing decision.

## 11. Future split design

- Principle: Session atomicity preserved; never split frames of one session across sets.
- The official Experiment 1 test set remains FROZEN. A future targeted experiment should use sessions reserved across dates/objects/lighting for its held-out generalization set; do not reuse Experiment 1 frozen test frames as training or validation data.
- Current Experiment 1 split unchanged: True

## 12. Retraining gate

**A. Is retraining justified now?** False
**B. First data to add:** Independent stringing sessions across new dates/objects/conditions ('5-10 minimum, 15-20 strong' from Stage 9); plus targeted dark-normal sessions and any available spaghetti sessions.
**C. Still missing:** Evidence that added stringing diversity reduces stringing FNR and that added dark-normal coverage reduces normal FPR, without regressing overall F1, as measured on FROZEN independent evaluation sets.
**D. Minimum useful experiment:** Collect proposed new sessions; freeze a new split by session with a held-out set; retrain the same architecture on the expanded training set; evaluate on the FROZEN Experiment 1 test at 0.50 and 0.65.
**E. Result that justifies retraining:** Stringing FNR improves by >=5pp (ideally >=10pp) with no F1 regression, dark-normal FPR falls, and the held-out set performs comparably to the frozen test.

- No architecture change is recommended unless the evidence specifically supports it.
- Temporal/FSM logic is NOT proposed as a solution to the underlying frame-classification problem.

---

## Artifacts

| File | Purpose |
|---|---|
| `docs/research/stage10-cluster-domain-audit.json` | Machine-readable results |
| `docs/research/stage10-cluster-domain-audit.md` | This report |
| `training/tv/stage10.py` | Analysis module |
| `training/tests/test_stage10.py` | Tests for the analysis logic |
| `docs/research/evalcaches/frozen_train_trajectories.json` | Corrected train cache |