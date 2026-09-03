# Stage 5 — Corrected Evaluation

**Date:** 2026-09-03
**Scope:** Re-establish the correct empirical baseline using frame-level ground truth,
evaluate temporal logic against per-frame labels, and compare historical conclusions.
**Constraint honored:** No retraining, no split modification, no threshold tuning on test.
Preserved Stage 3 parameters evaluated on corrected labels before any changes.

**Authoritative ground truth:** per-frame source `class` (binary normal/failing). Recording
IDs are grouping/sequence metadata only. No recording-level label is used or constructed.

---

## Part A — Correct frame-level baseline

### Raw frame-level (threshold 0.5)

**Test split** (2,859 frames: 1,981 positive, 878 negative):

| Metric | Value |
|--------|-------|
| Accuracy | 0.8360 |
| Precision | 0.9177 |
| Recall | 0.8385 |
| F1 | 0.8763 |
| FPR | 0.1697 |
| FNR | 0.1615 |
| AUROC | 0.9004 |

Confusion matrix (test): `[[TN=729, FP=149], [FN=320, TP=1661]]`

**Validation split** (2,174 frames: 1,417 positive, 757 negative):

| Metric | Value |
|--------|-------|
| Accuracy | 0.8813 |
| Precision | 0.9202 |
| Recall | 0.8956 |
| F1 | 0.9077 |
| FPR | 0.1453 |
| FNR | 0.1044 |
| AUROC | 0.9330 |

Confusion matrix (val): `[[TN=647, FP=110], [FN=148, TP=1269]]`

### Stratified by recording composition

| Subset | n_frames | accuracy | recall | FNR | FPR |
|--------|----------|----------|--------|-----|-----|
| **PURE_NORMAL** (test) | 693 | 0.8600 | — | — | 0.1400 |
| **MIXED** (test) | 1,194 | 0.7923 | 0.8057 | 0.1943 | 0.2811 |
| **PURE_FAILING** (test) | 972 | 0.8724 | 0.8724 | 0.1276 | — |
| **PURE_NORMAL** (val) | 427 | 0.9930 | — | — | 0.0070 |
| **MIXED** (val) | 1,148 | 0.7857 | 0.8301 | 0.1699 | 0.3242 |
| **PURE_FAILING** (val) | 599 | 0.9850 | 0.9850 | 0.0150 | — |

Key observations:
* PURE_FAILING performance is strong (val recall 0.985, test recall 0.872) — the
  model detects defect frames well when the whole recording is defective.
* MIXED recordings have the worst FPR (0.28 test, 0.32 val) — good frames within
  transition recordings are the primary false-alarm source.
* PURE_NORMAL FPR is moderate: 0.14 on test (6 all-good recordings), 0.007 on val
  (4 all-good recordings). The model's genuine false-alarm rate on all-good prints
  is lower than the overall FPR suggests.
* Val PURE_FAILING and PURE_NORMAL performance is notably stronger than test,
  confirming the val/test split has meaningful distributional differences.

### Stratified by failure type (test)

| Failure type | n_frames | recall | FNR |
|-------------|----------|--------|-----|
| normal (all) | 878 | — | — |
| stringing | 797 | 0.8168 | 0.1832 |
| under_extrusion | 1,184 | 0.8530 | 0.1470 |

The model performs comparably on stringing and under-extrusion. The val results
also show 100% recall on spaghetti (49 frames) but with only a small sample.

---

## Part B — Corrected catastrophic-recording analysis

Stage 1.5 identified four "catastrophic" recordings. The corrected per-frame
analysis reveals that three of the four were mischaracterized due to the
recording-label artifact:

| Recording | True labels | Previous interpretation | Corrected interpretation |
|-----------|-------------|------------------------|-------------------------|
| `15.02.2023_13_19_52` | 127 stringing (all failing) | FAILING, mean_score=0.15, FNR ~0.89 | **Still catastrophic:** FNR 0.882, TP=15, FN=112. Genuine detection failure on stringing. |
| `06012023_12_17_19` | 60 normal + 119 underex (MIXED) | "normal-labeled, 60 good→119 underex" | **Not a false-alarm case:** FPR=0, FNR=0.832. This is a missed-detection problem on 119 real defect frames. |
| `01122022_16_06_44` | 4 normal + 82 underex (MIXED) | "normal-labeled, mislabeled" | **Not a false-alarm case:** FPR=0, FNR=0.707. Another missed-detection problem on 82 real defect frames. |
| `06122022_14_49_53` | 14 normal + 150 stringing (MIXED) | "normal-labeled, catastrophic" | **Mostly fine:** FNR=0.207 (detects 119/150 stringing), FPR=0.57 on 14 normal frames only. Model does well on defect frames. |

**Corrected conclusion:** the "catastrophic" problem is a **false-negative concentration**,
not a false-alarm problem. Three of four recordings had real defect frames that the model
missed — the model fails on under-extrusion and some stringing patterns, not on
normal prints. The `06122022_14_49_53` recording, which was treated as a worst-case, is
actually one of the best-performing transition recordings (FNR 0.207 on defect frames).

---

## Part C — Corrected temporal evaluation

All three variants evaluated against **per-frame labels** on the test split (no
synthetic recording label). FSM uses preserved Stage 3 parameters (alpha=0.15,
fail_threshold_abs=0.65, fail_threshold=0.30, confirm=3, warmup=30, use_delta=True).

### Test split

| Variant | acc | prec | rec | F1 | FPR | FNR | FA_frames | FA_runs | longest_FA | miss_frames | miss_runs | transitions |
|---------|-----|------|-----|----|-----|-----|-----------|---------|------------|-------------|-----------|-------------|
| raw (thr 0.5) | 0.836 | 0.918 | 0.839 | 0.876 | 0.170 | 0.162 | 149 | 80 | 13 | 320 | 61 | 275 |
| ewma (α=0.15, thr 0.5) | 0.837 | 0.918 | 0.840 | 0.877 | 0.169 | 0.160 | 148 | 20 | 32 | 317 | 10 | 41 |
| fsm (preserved) | 0.312 | 1.000 | 0.007 | 0.013 | 0.000 | 0.993 | 0 | 0 | 0 | 1,968 | 19 | 3 |

### EWMA vs raw: the run tradeoff

EWMA does **not** meaningfully change frame-level metrics (acc/F1 shift <0.001).
Its real effect is on the **temporal structure** of errors:

* **False-alarm runs:** 80 → 20 (75% reduction in number of alarm events)
* **Longest false-alarm run:** 13 → 32 (2.5× longer worst case)
* **Missed runs:** 61 → 10 (84% reduction)
* **Longest missed run:** 109 → 118 (similar)
* **State transitions:** 275 → 41 (85% reduction in flicker)

EWMA consolidates short flicker events into longer sustained events. This is a
smoothness improvement, not an accuracy improvement. Whether this is desirable
depends on the operational use case: for a pause/cancel mechanism, fewer shorter
alarms may be preferable to fewer longer ones.

### FSM (preserved Stage 3 parameters): failure

The preserved FSM achieves FNR 0.993 on corrected frame labels — it detects only
0.7% of failing frames. This is because:

1. `use_delta=True` with a rising baseline causes delta ≈ 0 on sustained failure
   signals, preventing the FSM from entering FAILING state.
2. `fail_threshold_abs=0.65` only catches the highest-confidence frames (score ≥
   0.65), missing the majority of failing frames that score 0.3–0.6.
3. The 30-frame warmup suppresses all detections at recording start.

Stage 3 reported FSM as having `sequence_detection_rate=1.0` because it used a
recording-level detection criterion (enter FAILING once = detected). On corrected
frame-level labels, the FSM is catastrophically ineffective.

---

## Part D — Threshold methodology

All threshold exploration performed on the **validation split only**. The frozen
threshold (0.5) from Stage 3 / Experiment 1 is preserved as the primary comparison.

### Raw threshold sweep (val)

| threshold | acc | F1 | FPR | FNR | FA_frames | FA_runs | longest_FA |
|-----------|-----|----|-----|-----|-----------|---------|------------|
| 0.3 | 0.861 | 0.895 | 0.227 | 0.092 | 172 | 57 | 51 |
| 0.4 | 0.873 | 0.902 | 0.178 | 0.100 | 135 | 49 | 51 |
| **0.5** (frozen) | 0.881 | 0.908 | 0.145 | 0.104 | 110 | 37 | 33 |
| 0.6 | 0.887 | 0.911 | 0.108 | 0.115 | 82 | 28 | 22 |
| 0.7 | 0.889 | 0.911 | 0.089 | 0.124 | 67 | 19 | 19 |
| 0.8 | 0.888 | 0.910 | 0.073 | 0.133 | 55 | 22 | 11 |

Val F1 peaks at 0.6–0.7 (F1=0.911) vs 0.5 (F1=0.908). The frozen threshold 0.5
is conservative (lower FNR at the cost of higher FPR). Raising to 0.7 would
substantially reduce FPR (0.145→0.089, -39%) with modest FNR increase
(0.104→0.124, +19%).

### EWMA threshold sweep (val, α=0.15)

| threshold | acc | F1 | FPR | FNR | FA_frames | FA_runs | longest_FA |
|-----------|-----|----|-----|-----|-----------|---------|------------|
| 0.3 | 0.834 | 0.876 | 0.297 | 0.096 | 225 | 14 | 91 |
| 0.4 | 0.847 | 0.883 | 0.235 | 0.109 | 178 | 15 | 88 |
| **0.5** (frozen) | 0.852 | 0.885 | 0.189 | 0.126 | 143 | 19 | 80 |
| 0.6 | 0.854 | 0.885 | 0.151 | 0.143 | 114 | 12 | 76 |
| 0.7 | 0.850 | 0.878 | 0.118 | 0.168 | 89 | 13 | 44 |
| 0.8 | 0.840 | 0.866 | 0.074 | 0.206 | 56 | 13 | 21 |

EWMA consistently underperforms raw on val F1 (0.885 vs 0.911 at best). This
confirms: **EWMA does not improve the classifier; it only smooths the output.**

### Threshold freezing

For the corrected test evaluation, the threshold 0.5 is preserved (Stage 3 / Exp 1
default). The val sweep confirms that 0.5 is reasonable but not optimal for F1;
a deployment tuning pass should explore 0.6–0.7 if FPR reduction is prioritized.

---

## Part E — Comparison of historical conclusions

| Historical conclusion | Source | Status | Reason |
|----------------------|--------|--------|--------|
| Frame metrics: acc 0.836, F1 0.876, FNR 0.162, FPR 0.170 | Exp 1 | **NOT superseded** | Frame-level labels are authoritative; Exp 1 metrics were already frame-level and correct. |
| Spaghetti has low recall (FNR 0.54) | Exp 1 error analysis | **NOT superseded** | Failure-type stratification is consistent with corrected analysis. |
| "Catastrophic" recordings are mislabeled/out-of-distribution | Stage 1.5 | **PARTIALLY superseded** | 3 of 4 "catastrophic" recordings were misclassified by the recorder-label artifact. Only `15.02.2023_13_19_52` remains genuinely catastrophic (FNR 0.88 on real defect frames). The others are missed-detection problems, not labeling errors. |
| Model learns fine texture, not brightness shortcuts | Stage 2 | **NOT superseded** | Robustness conclusions are independent of the recording-label issue. |
| EWMA halves sequence-level false alarms | Stage 3 | **Superseded** | "Sequence-level" used the recording-label construction. On corrected per-frame labels, EWMA does not reduce FPR (0.170 vs 0.169) and only reduces the number of false-alarm runs (80→20). |
| FSM with absolute latch (0.65) detects all events | Stage 3 | **Superseded** | Stage 3's "sequence_detection_rate=1.0" used the fire-once recording-level criterion. On corrected frame labels, the preserved FSM has recall 0.007. |
| FSM has FNR 0.397 on holdout | Stage 3 | **Superseded** | That FNR was computed relative to the recording label. On corrected frame labels, FSM FNR is 0.993. |
| "Miscalibration" is the dominant bottleneck | Stage 3 | **Narrowed** | The "sustained over-confidence on source-labeled-good prints" concern applies only to the 6 genuinely all-good test recordings (FPR 0.14 on that subset), not to the 9 mixed transition recordings (which have real defect frames). The false-alarm problem is narrower than Stage 3 implied. |
| Detection delay is not measurable | Stage 3 | **Confirmed** | Consistent with corrected analysis. No verified temporal onset ground truth exists. |

---

## Part F — Is temporal logic justified?

### Does EWMA reduce false-alarm runs?

**Yes.** EWMA reduces the number of false-alarm runs from 80 to 20 (75%) on test.
It also reduces missed-failure runs from 61 to 10 (84%).

### Does EWMA increase or reduce FNR?

**Neither meaningfully.** FNR changes from 0.1615 (raw) to 0.1600 (EWMA) on test.
Frame-level FNR is essentially unchanged.

### Does EWMA sacrifice recall?

**No meaningful sacrifice.** Recall changes from 0.8385 to 0.8400 on test.

### Does FSM add value beyond EWMA?

**No.** The preserved FSM (use_delta=True, abs 0.65) has recall 0.007 — it is
essentially useless. Even the best FSM variants explored on val achieve FNR >0.85.

### Does EWMA justify its operational complexity?

**Marginal.** EWMA provides:
* Fewer, longer false-alarm runs (smoothness)
* Fewer, longer missed-failure runs
* No frame-level metric improvement

The smoothness benefit is real but comes at the cost of longer worst-case alarm
duration (longest FA run: 13→32 frames). Whether this tradeoff is beneficial
depends on the operational use case:
* **For pause/cancel automation:** longer sustained alarms may actually be worse
  (a 32-frame alarm is worse than 5 short 1-2 frame alarms for a human operator
  who can ignore brief flicker).
* **For operator notification:** fewer alarm events reduce alert fatigue.

**Conclusion:** EWMA is a defensible post-processing choice for operational
smoothness, but it is not a classification improvement. The FSM in its current
form should not be used.

---

## Part G — Evidence-based recommendation

### Corrected baseline summary

The frozen ShuffleNetV2 (Experiment 1 checkpoint) achieves on the corrected test
labels: acc 0.836, F1 0.876, FNR 0.162, FPR 0.170, AUROC 0.900. These numbers
are identical to the original Experiment 1 results because Experiment 1 was
already frame-level — no historical metric needs correction.

### Primary task

**Per-frame binary classification (normal vs failing)** remains the correct
primary task. The frame-level labels are authoritative and the model performs
reasonably on them.

### Next experiment priorities

1. **Own data collection.** The val/test FPR difference (0.145 vs 0.170) and
   PURE_FAILING recall gap (0.985 vs 0.872) indicate distributional shift between
   val and test that the synthetic dataset cannot resolve. Collecting real print
   data with known ground truth is the single highest-impact next step.

2. **Threshold tuning for deployment.** The val sweep shows F1 peaks at 0.7
   (F1 0.911, FPR 0.089) vs the default 0.5 (F1 0.908, FPR 0.145). A deployment
   pass should tune the threshold on collected real data, not on this synthetic
   dataset.

3. **Failure-type-specific analysis.** Stringing (FNR 0.183) and under-extrusion
   (FNR 0.147) have comparable but distinct failure modes. Investigating whether a
   multi-class or ensemble approach improves per-type recall is warranted.

4. **Mixed-recording FPR reduction.** The MIXED subset's FPR (0.28) is the
   dominant false-alarm source. Understanding why the model misclassifies good
   frames within transition recordings (early frames of a print that later fails)
   may reveal a domain-specific bias worth addressing.

5. **Temporal logic: defer.** EWMA provides marginal smoothness benefit only. The
   FSM should be redesigned from scratch if temporal logic is pursued (the current
   delta-guard approach does not work). Temporal evaluation should use
   human-annotated onset timestamps, not the class-flip proxy.

### What is NOT needed

* Retraining on the current synthetic dataset (the bottleneck is data quality,
  not architecture).
* FSM with the current parameters (use_delta=True, abs 0.65).
* Any metric re-evaluation on the current test set (the frame-level baseline is
  stable).

---

**Artifact:** `docs/research/stage5-corrected-results.json`
**Module:** `training/tv/correctedeval.py` (+ 13 tests, all pass)
**Full suite:** 100 tests pass (87 pre-existing + 13 correctedeval)
