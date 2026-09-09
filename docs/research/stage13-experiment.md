# Stage 13 Experiment Report: Target-Domain Normal Augmentation (Experiment 3)

Status: evaluation-only, experiment report
Created: 2026-09-07
Related: [stage13-target-normal-dataset.json](stage13-target-normal-dataset.json),
[exp3-target-normal-design.md](exp3-target-normal-design.md),
[real-world-video-evaluation.md](real-world-video-evaluation.md)

## 1. Question

The frozen TinyVision baseline emits persistent false "failing" alarms on
independent normal-print footage (the 7-video external evaluation reported
every live normal print as ~99% failing). The central hypothesis under test in
Stage 13:

> The catastrophic false-positive rate on independent normal footage is
> caused primarily by insufficient diversity of normal target-domain training
> data.

Experiment 3 tests this by adding a small, source-grouped, VLM-screened set of
verified-normal target-domain frames to the frozen training set (everything else
unchanged) and re-measuring all evaluation surfaces with the existing validated
inference configuration. No threshold tuning, no architecture change, no
modification of historical datasets or the 7-video holdout.

## 2. Method

### 2.1 Training configuration

* Baseline: `shufflenet_v2_x1_0_printing-errors_cdc9e2cad4` (frozen)
  * ShuffleNetV2-x1.0 binary classifier, `datasets/processed/printing-errors`,
    epochs 30, adam lr 1e-3 wd 1e-4, batch 32, augmentation
    (hf flip, rot 15, color jitter 0.2, normalize), seed 42, image 224.
* Experimental (Exp3): `exp3_target_normal_aug_shufflenet_v2_x1_0`
  * Identical architecture / optimizer / augmentation / epochs / seed.
  * Training set = frozen printing-errors train (11,254 rows:
    5,863 failing + 5,391 normal) **+ 333 target-domain NORMAL frames**
    from 7 independent sources (derived split:
    `datasets/processed/external-aug-exp3-normal`).
  * val / test = byte-identical copies of the frozen printing-errors splits.
  * Augmentation frames are VLM-screened (`decision==normal AND
    part_visible==True`), status pending human review. All sampled 30-50 s
    apart; source-grouped so no source is split across roles.
  * Trained 30 epochs, early-stopped at epoch 21, best val at epoch 11
    (val acc 0.8997, val loss 0.2691).

### 2.2 Held-out evaluation surfaces

| surface | what | source-grouping |
|---------|------|-----------------|
| Frozen original test | `datasets/processed/printing-errors/splits/test.csv`, 2,859 rows (1,981 failing / 878 normal), evaluated with `evaluate_classification` | frozen, unchanged |
| Held-out normal sources | 120 VLM-screened normal frames from 2 fully held-out sources (`LY3jlcO34f4`, `zkJgLanCSUw`) | source-level holdout; never in any training split |
| SKZ held-out slice | `tv skz-eval --subset skz_eval` on 31 probe images across 4 classes | deterministic token split, seed 42 |
| 7-video external holdout | C++ runtime (`./runtime/build/tinyvision`) on the 7 frozen external YouTube clips, `--sample-fps 1 --warmup 8 --confirm 2`, threshold 0.5 | unchanged clips; frames never trained on |

Decision threshold 0.5 everywhere (no tuning). Failing-as-positive is the
primary view.

### 2.3 Files

* Frozen test + held-out normal compare:
  `docs/research/stage13-exp3-comparison.json`
* SKZ reports: `docs/research/skz-baseline.json`, `skz-exp3.json`,
  `skz-baseline-evaltok.json`, `skz-exp3-evaltok.json`
* 7-video per-model runtime reports:
  `datasets/external/real-world-eval/runs/*.json` (baseline),
  `datasets/external/real-world-eval/exp3_runs/*.json` (Exp3)
* Derived split: `datasets/processed/external-aug-exp3-normal/{splits,derived_manifest.json}`

## 3. Results

### 3.1 Frozen original test (unchanged split)

Failing-as-positive (threshold 0.5):

| metric | baseline | Exp3 | Δ |
|--------|----------|------|---|
| accuracy | 0.8360 | 0.7859 | -0.050 |
| precision (failing) | 0.9177 | 0.9265 | +0.009 |
| recall (failing) | 0.8385 | 0.7506 | -0.088 |
| F1 (failing) | 0.8763 | 0.8293 | -0.047 |
| FPR (normal misread as failing) | 0.1697 | 0.1344 | -0.035 |
| FNR (missed failures) | 0.1615 | 0.2494 | +0.088 |
| AUROC (failing) | 0.9004 | 0.8774 | -0.023 |

Confusion matrices (rows=true, cols=pred, [failing=0, normal=1]):
baseline `[[1661, 320], [149, 729]]`; Exp3 `[[1487, 494], [118, 760]]`.

Interpretation: Exp3 reduces false alarms on the original test modestly while
giving back failure recall (-0.088) and AUROC (-0.023). Not catastrophic, but a
real regression on the frozen benchmark.

### 3.2 Held-out normal sources (120 frames, 2 completely held-out sources)

The decisive test for the hypothesis: 100% of held-out normal frames that the
baseline calls failing.

| metric | baseline | Exp3 |
|--------|----------|------|
| normal frames evaluated | 120 | 120 |
| predicted failing | 120 | 31 |
| **normal FPR** | **1.000** | **0.258** |

| source | baseline predicted-failing | Exp3 predicted-failing | Exp3 FPR |
|--------|---------------------------|------------------------|----------|
| `LY3jlcO34f4` (60) | 60 | 6 | 0.100 |
| `zkJgLanCSUw` (60) | 60 | 25 | 0.417 |

Interpretation: the hypothesis is **confirmed** — adding a small, diverse,
verified-normal target-domain set collapses the catastrophic normal false-alarm
rate from 100% to ~26% on fully held-out independent sources. The two held-out
sources behave differently (10% vs 42%), meaning source-level variability
remains large and 120 frames is informative but not conclusive.

### 3.3 SKZ held-out slice (`tv skz-eval --subset skz_eval`, 31 images)

| metric (failing-positive) | baseline | Exp3 |
|---------------------------|----------|------|
| AUROC (failing) | 0.7227 | 0.3318 |
| precision / recall / F1 | 0.667 / 1.000 / 0.800 | 0.400 / 0.100 / 0.160 |
| normal FPR | 0.909 (10 of 11) | 0.273 (3 of 11) |
| spaghetti recall (4 img) | 1.000 | 0.000 |
| stringing recall (5 img) | 1.000 | 0.200 |
| under-extrusion recall (11 img) | 1.000 | 0.091 |

Interpretation: 31 images is a thin slice. Baseline masks almost everything as
failing (including 91% of normals). Exp3 normalizes the normal frames but its
failure recall collapses on this nozzle-cam domain — the sharpest trade-off in
this report. This held-out slice is thus **strong evidence the model became more
conservative overall** (predicts "normal" more freely), which fixes false alarms
but hurts failure detection, particularly in the SKZ nozzle-cam domain that
differs substantially from the ELP-style target domain.

### 3.4 7-video external holdout (C++ runtime, validated config)

Per-frame raw failing-decision fraction and FSM event timing (threshold 0.5,
sample-fps 1, warmup 8, confirm 2):

| video | intended | visual (VLM) | baseline raw-failing | exp3 raw-failing | baseline first event | exp3 first event |
|-------|----------|--------------|----------------------|------------------|----------------------|------------------|
| `nbkf372s_cA` | normal | continuous normal print | 0.961 | 0.110 | 8 | 8 |
| `UzULqvGoi1Y` | normal | continuous normal print | 0.870 | 0.141 | 11 | 67 |
| `0TY6LVvMpUo` | normal-ish review | normal | 0.926 | 0.114 | 8 | 15 |
| `38gnsbGqcx4` | failure (blob) | **VLM: normal print** | 0.787 | 0.036 | 9 | 159 |
| `G56LQfQCUnA` | failure (under/over ext) | **VLM: normal print** | 0.847 | 0.411 | 8 | 9 |
| `87gsL-Luvis` | failure (spaghetti) | **VLM: spaghetti failure** | 0.818 | 0.000 | 8 | never |
| `mGH3p3AF1YY` | failure (spaghetti) | **VLM: spaghetti/stringing** | 1.000 | 0.562 | 8 | 9 |

Note: `38gnsbGqcx4` and `G56LQfQCUnA` were classified *normal* by the VLM
reviewer (see `real-world-video-evaluation.md`); their concurrent "intended
failure" label came only from YouTube titles, never treated as ground truth. So
the two visually-confirmed external failures in this set are the spaghetti
clips `87gsL-Luvis` and `mGH3p3AF1YY`.

Interpretation:
* On all 3 normal/intended-normal clips, Exp3 drops raw-frame false alarms
  ~7-8x (0.96→0.11, 0.87→0.14, 0.93→0.11).
* FSM event timing improves on `UzULqvGoi1Y` (11→67) and `0TY6LVvMpUo` (8→15)
  but `nbkf372s_cA` still emits an early event at sample 8 (8 samples into the
  clip the raw score is above threshold), even though after ~sample 17 it drops
  to normal and stays mostly normal (43 of 516 FSM-failing frames vs 508
  baseline).
* **However: the confirmed spaghetti failure `87gsL-Luvis` is now completely
  missed (raw-failing 0.000, no event), and the other confirmed spaghetti
  `mGH3p3AF1YY` weakens (1.000→0.562).** This is the most important external
  failure-detection regression.
* The blob/under-extrusion clips (`38gnsbGqcx4`, `G56LQfQCUnA`) becoming
  mostly-normal is consistent with the VLM verdict that those clips visually show
  normal prints — so that change is arguably correct, not a regression.

## 4. Conclusion

The Stage 13 central hypothesis is **confirmed**: adding a small (333-frame),
diverse, source-grouped, verified-normal target-domain set drops the independent
normal-print false-alarm rate from ~100% to ~26% on held-out normal sources and
~7-8x on the 7-video normal clips, using the unchanged validated configuration.

However, the fix is not free at fixed threshold 0.5:

1. Failure recall on the frozen original test falls 0.839 → 0.751 (FNR nearly
   doubles), AUROC 0.900 → 0.877.
2. The two visually-confirmed external spaghetti failures degrade:
   `87gsL-Luvis` is now missed entirely; `mGH3p3AF1YY` weakens.
3. On the SKZ held-out slice, failure recall collapses (1.0 → 0.10) while FPR
   improves; the model became globally more "normal"-biased.

Net: Exp3 fixes the dominant false-positive failure mode at the cost of failure
detection robustness. It is a strong, evidence-backed confirmation of the tail
risk the baseline had (normal diversity), but **not** a ready-to-ship model.
Recommended next steps:

* Rebalance the second training augmentation round with target-domain
  *failing* examples (e.g., a handful of real spaghetti/stringing clips) so the
  decision boundary is not pushed purely toward "normal".
* Consider a *small* curated addition of target-domain failing/near-failing
  frames and re-measure the same four surfaces (the dataset handling and
  evaluation harness are now in place).
* Retain both checkpoints; do not silently replace the baseline.
* Complete the pending human review of the 333 stage-13 normal frames before any
  release use.

## 5. Caveats

* All stage-13 augmentation labels are VLM-screened, status
  `pending_human_review`, not human ground truth.
* 120 held-out normal frames / 2 sources is informative but limited; per-source
  spread is large (0.10 vs 0.42).
* SKZ held-out slice is only 31 frames and a different (nozzle-cam) domain.
* No threshold tuning; threshold 0.5 fixed everywhere as required.
* 7-video clips are evaluation-only; the two non-failing clips are not
  retrained on by either model.