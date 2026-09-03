# Experiment 1 — Error Analysis & Stage 2 Baseline

**Status:** Stage 1.5 — baseline validation of the two carried-forward Experiment 1
candidates (`mobilenet_v3_small`, `shufflenet_v2_x1_0`).

**Scope:** confusion matrices, per-class and binary (normal-vs-failing) metrics,
val-vs-test, threshold sweeps, FP/FN examples, confidence distributions,
per-recording metrics, model size + host latency, failure-type breakdown.

**Convention (important):** this document uses the **failure-detection view** —
*failing* is the positive class. This is the view that matters for 3D-print
failure detection (we must catch failures). The Stage-1 headline F1 figures were
reported in the *normal-as-positive* view; both are given below where relevant.

---

## 1. Models evaluated

| model | test acc | failing F1 | failing recall | failing FNR | failing precision | AUROC | per-class F1 (report) |
|---|---|---|---|---|---|---|---|
| mobilenet_v3_small | 0.823 | 0.860 | 0.7814 | 0.219 | 0.955 | 0.925 | 0.761 (normal-pos) |
| shufflenet_v2_x1_0 | 0.836 | 0.876 | 0.8385 | 0.162 | 0.918 | 0.900 | 0.757 (normal-pos) |

Test set (frozen): **2,859 samples, 24 recordings** — failing 1,981 / normal 878.

### Confusion matrices (rows = true, cols = pred)

**mobilenet_v3_small**

| | pred failing | pred normal |
|---|---|---|
| true failing | 1548 (TP) | **433 (FN)** |
| true normal | 73 (FP) | 805 (TN) |

**shufflenet_v2_x1_0**

| | pred failing | pred normal |
|---|---|---|
| true failing | 1661 (TP) | **320 (FN)** |
| true normal | 149 (FP) | 729 (TN) |

---

## 2. Per-class metrics (test)

| class | model | precision | recall | F1 | support |
|---|---|---|---|---|---|
| normal | mobilenet_v3_small | 0.650 | 0.917 | 0.761 | 878 |
| failing | mobilenet_v3_small | 0.955 | 0.781 | 0.860 | 1981 |
| normal | shufflenet_v2_x1_0 | 0.695 | 0.830 | 0.757 | 878 |
| failing | shufflenet_v2_x1_0 | 0.918 | 0.839 | 0.876 | 1981 |

Both models detect the **normal** class with high/higher recall than failing
(in the m3s case failing recall 0.781 < normal recall 0.917). Failing precision is
very high in both (0.955 / 0.918), meaning **false alarms are rare** — the dominant
error mode is **missed failures** (FN), not spurious alarms.

The two imperfections trade off against each other:
- **m3s**: highest precision (0.955), fewer false alarms (FP=73), but more misses
  (FN=433, FNR 0.219).
- **shufflenet**: better recall (0.839), fewer misses (FN=320, FNR 0.162), more
  false alarms (FP=149).

For a safety-critical failure detector, **FNR is the metric that matters most**.
On that basis **shufflenet_v2_x1_0 is the better baseline**: it misses 320 of 1981
failures (16.2%) vs m3s's 433 (21.9%), at a modest cost of ~2x the false-alarm rate.

---

## 3. Val vs test

| model | split | acc | failing F1 | failing FNR | failing precision |
|---|---|---|---|---|---|
| mobilenet_v3_small | val | 0.857 | 0.880 | 0.195 | 0.969 |
| mobilenet_v3_small | test | 0.823 | 0.860 | 0.219 | 0.955 |
| shufflenet_v2_x1_0 | val | 0.881 | 0.908 | 0.104 | 0.920 |
| shufflenet_v2_x1_0 | test | 0.836 | 0.876 | 0.162 | 0.918 |

Val exceeds test for both (expected), and the drop is modest (F1 −0.02 to −0.03).
Ranking is stable across splits: **shufflenet > m3s** on accuracy and failing-F1 in
both val and test. There is no evidence of a systematic generalization gap unique
to one architecture; the test is slightly harder for both.

---

## 4. Failure-type breakdown (is one failure type harder?)

Failing origin classes in the frozen test split (from the raw CSV class column):
**UNDEREX 1,184 (60%), STRING 797 (40%)**; no SPAGHETTI samples reach the test
split.

Pooled failing-recall by original failure type (weighted by per-recording recall):

| model | UNDEREX recall | STRING recall |
|---|---|---|
| mobilenet_v3_small | 0.806 | 0.745 |
| shufflenet_v2_x1_0 | 0.853 | 0.817 |

STRING is consistently a few points harder than UNDEREX for both models, but the
gap is small and no single failure type dominates the error budget. The catastrophic
recordings below include **both** an all-STRING recording and an all-UNDEREX one,
so the difficulty is *not* simply "one failure type is hard." It is **recording /
domain specific**.

---

## 5. Per-recording variance — the decisive finding

Both architectures fail on the **same small set of recordings**. Estimated FN
contribution (failing_count × (1 − per-recording failing recall)):

**mobilenet_v3_small** — worst recordings (of 433 total FN):

| recording | failing | recall_fail | est FN |
|---|---|---|---|
| Recording_06012023_12_17_19 (UNDEREX) | 119 | 0.017 | ~117 |
| Recording_15.02.2023_13_19_52 (STRING) | 127 | 0.150 | ~108 |
| Recording_06122022_14_49_53 (STRING) | 150 | 0.567 | ~65 |

Top 3 recordings ≈ 290 FN ≈ **67% of all m3s misses**.

**shufflenet_v2_x1_0** — worst recordings (of 320 total FN):

| recording | failing | recall_fail | est FN |
|---|---|---|---|
| Recording_15.02.2023_13_19_52 (STRING) | 127 | 0.118 | ~112 |
| Recording_06012023_12_17_19 (UNDEREX) | 119 | 0.168 | ~99 |
| Recording_01122022_16_06_44 (UNDEREX) | 82 | 0.293 | ~58 |

Top 3 recordings ≈ 269 FN ≈ **84% of all shufflenet misses**.

Over the rest of the test set both models perform very well (many recordings have
failing recall ≥ 0.93). **The overall FNR is almost entirely due to a handful of
recordings.**

### Why this matters

Because two architecturally very different CNNs fail on the *same* recordings with
*same classes present at normal rates elsewhere*, the misses are **not explained by
model capacity or architecture**. They are explained by either:

1. **Out-of-distribution recordings** — a particular print geometry, lighting,
   camera distance, or failure appearance that the training recordings did not
   contain; or
2. **Labeling noise** in those specific recordings (mislabeled frames).

Both hypotheses point to the same Stage 2 action: **evaluate the frozen models on
independently captured target-camera/printer data** before deciding whether to
retrain, and inspect the offending recordings visually.

---

## 6. Threshold behavior

P(failing) is strongly bimodal: the histogram is dominated by the two extremes
(very-low and very-high confidence), with sparse mid-range scores.

- m3s: 0.0–0.1 → 915 samples; 0.9–1.0 → 1299; middle bins ≤ 136 each.
- shufflenet: 0.0–0.1 → 657; 0.9–1.0 → 1489; middle bins ≤ 148 each.

Consequence: the decision threshold has a **wide flat optimum**. Sweeping the
P(failing) threshold from ~0.1 to ~0.5 leaves failing-F1 essentially flat
(~0.86–0.89 for m3s). The model is highly confident on most samples, so the choice
of a single threshold is not the critical lever for this baseline.

Implication for temporal logic: a clean, well-separated per-frame confidence signal
is a good substrate for EWMA / finite-state smoothing in Stage 2. False alarms are
rare and confidence is decisive, so temporal filtering mainly needs to suppress
brief flicker rather than rescue weak detections.

---

## 7. False-positive / false-negative examples

Full excerpts (paths, true/pred, fail_score, confidence) are recorded in:
`experiments/mobilenet_v3_small_printing-errors_b7ac0cdc2e/deepeval/test.json` and
`experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/deepeval/test.json`
(`example_excerpts` block). Only the 6 most confident FP and FN per model are saved
there to keep the record small. These should be visually inspected as part of
Stage 2 validation.

---

## 8. Model size & host latency

Both are small enough for edge deployment on Raspberry Pi 4/5. Latency below is
**host (MPS/Metal) latency**, not edge latency; real ONNX Runtime / ARM figures
belong to a later benchmark.

| model | fp32 size | int8 size | host fp32 latency | host int8 latency |
|---|---|---|---|---|
| mobilenet_v3_small | 6.09 MB | 1.69 MB | 1.84 ms | 6.72 ms |
| shufflenet_v2_x1_0 | 5.05 MB | 1.45 MB | 1.99 ms | 4.95 ms |

(int8 size from the Stage-1 ONNX int8 exports; host latency from the Stage-1
benchmark. Both are well within target.)

---

## 9. Confidence distributions

Stored per model in `deepeval/<split>.json` (`confidence_histogram.failing_score`).
Both are strongly bimodal with a clear low-confidence base and a high-confidence
failure mass; see Section 6. This supports a simple, robust decision rule.

---

## 10. Recommendation

**Adopt `shufflenet_v2_x1_0` as the frozen Stage-1 baseline** for all further
Stage 2 work:

- Best accuracy (0.836) and failing-F1 (0.876) on the frozen test set.
- **Lowest false-negative rate (FNR 0.162)** — the priority metric for failure
  detection.
- Smallest model footprint and best int8 host latency.
- Stable ranking on both val and test.

Its one weakness is a higher false-alarm rate (FP=149 vs 73) and lower normal recall
(0.830 vs 0.917); for a failure detector this is the less costly error mode, and
false alarms can be further suppressed by temporal smoothing.

**Do not optimize the frozen test set.** The per-recording analysis shows the few
problem recordings are unchanged across architectures; tuning on them would overfit
to recording-specific artifacts. Address them via Stage 2 real-camera data instead.

---

## 11. Open question / risk

The three catastrophic recordings are likely out-of-distribution and/or mislabeled.
Until they are visually inspected against independently captured target data, the
baseline's reported FNR (0.162) should be treated as **optimistic for a real
deployment** — real-camera conditions may differ more than the test split reflects.
This is exactly what the Stage 2 spec (below) is designed to measure.

---

# Stage 2 Specification (baseline + temporal logic)

## 11a. Stage 2 — Phase A: real-camera baseline validation (BEFORE retraining)

**Do NOT retrain.** Evaluate the frozen, Kaggle-trained `shufflenet_v2_x1_0`
checkpoint (and m3s for comparison) on independently collected target
camera/printer data.

### Dataset structure (Layout A — same as training)

```
datasets/printing-errors-2/<split>/
  failing/        Recording_<YYYYMMDD_HHMMSS>/  <frame>.jpg
  normal/         Recording_<YYYYMMDD_HHMMSS>/  <frame>.jpg
```

One directory per recording session; frames sequential from a fixed camera.
A CSV snapshot per split with `path,label` (path = `label/Recording_.../frame.jpg`).

### Capture procedure

- Camera: the **target webcam** (USB or CSI) at a fixed distance/angle to the
  printer bed, matching the inference deployment setup.
- Recording granularity: one recording per print (or per continuous period),
  so timestamps map cleanly to real failure events.
- Failures: record full failing prints (from a visible defect onset onward) and
  full normal prints. **Do not** curate only "ideal" frames — include challenging
  conditions (low light, motion blur, moving extruder, partial occlusion) to expose
  the per-recording generalization gap seen in Stage 1.
- Labeling: assign each recording a class (failing / normal) at capture time; do
  not hand-pick frame labels. Capture metadata (printer, material, camera, lighting)
  per recording.
- Reproducibility: record the exact capture command, camera device, resolution,
  FPS, and any `tv capture` flags used.

### Commands

```
tv split --data-dir datasets/processed/printing-errors-2/config <split_ratios>   # 0.7/0.15/0.15, class-balanced, recording-atomic
tv eval-deep --config training/configs/experiment1_shufflenet_v2_x1_0.yaml \
             --experiments-dir experiments --split test --output-json <out>
tv eval-deep --config training/configs/experiment1_mobilenet_v3_small.yaml \
             --experiments-dir experiments --split test --output-json <out>
```

Compare the frozen models' metrics on this new split against Section 1. If a few
recordings again dominate the misses **and match the categories seen in Stage 1**
(same visual failure style), that is evidence of a systematic domain mismatch and
justifies a retraining window after Stage 2 Phase B.

### Known tooling gap

`tv capture` exists but does **not** support per-recording class labeling or
metadata. Capturing target data currently requires a small manual/shell workflow
that writes frames into Layout A directories and records a CSV. This should be added
to `cli/` before data collection if feasible; otherwise use a documented script in
`scripts/`.

## 11b. Stage 2 — Phase B: temporal decision logic

After Phase A validates the per-frame baseline on real data, implement temporal
smoothing on top of the **frozen** per-frame model:

- **EWMA** of P(failing) across frames with a suspicion/confirmed hysteresis band,
  or an equivalent **finite-state machine** (normal → suspect → failing), keyed on
  sustained low-confidence or sustained high-confidence runs.
- Tune the smoothing parameters on the real-camera split (not the frozen test).
- Re-measure: FNR, FPR, and time-to-alert (frames between defect onset and alarm).

Threshold choice is forgiving (Section 6), so the temporal layer should focus on
**FNR reduction and alarm stability**, not on rescuing weak per-frame confidence.

---

## 12. Supporting data

- `experiments/mobilenet_v3_small_printing-errors_b7ac0cdc2e/deepeval/{test,val}.json`
- `experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/deepeval/{test,val}.json`

These are the machine-readable, reproducible outputs produced by `tv eval-deep`
that this analysis summarizes.

## 13. Reproducibility

- Evaluation: `tv eval-deep --config <config.yaml> --split <test|val> --output-json <out>`
  run from repo root (config `data_dir` is relative).
- Seeds/configs/MPS hardware recorded in each `experiments/<id>/config.yaml`.
- AUROC computed via `roc_auc_score(y_true, 1.0 - fail_scores)` because this
  environment's `roc_auc_score` takes label `1` as the implicit positive class
  (equivalent AUROC for P(failing)).
