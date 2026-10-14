# Reliability audit: the independent-normal-holdout FPR = 1.0 finding

Date: 2026-10-13
Scope: trace and confirm the reported false-positive-rate (FPR) failure on the
independent target-domain normal holdout, verify the metric path, and record
what is and is not reproducible. Companion to
[`../reliability-evaluation.md`](../reliability-evaluation.md).

## Finding

The rc1 release candidate predicts **every frame of the independent normal
holdout as failing** at the operational threshold (0.5), i.e. FPR = 1.0.

Source artifact: `docs/research/rc1-core-eval.json` ->
`external_normal_holdout`:

| field | value |
|---|---|
| evaluated_frames | 120 |
| sources | `LY3jlcO34f4`, `zkJgLanCSUw` (60 frames each) |
| predicted_failing | 120 |
| false_positive_rate | 1.0 |
| per-source fpr | 1.0 (both sources) |

## The metric path is correct — this is a genuine failure, not a reporting bug

Verified this session:

1. **Label indexing.** `class_to_idx` is sorted alphabetically, so
   `failing=0`, `normal=1`; `FAIL_IDX = 0`
   (`training/tv/exp3_external.py:42`).
2. **Score direction.** The held-out surface is scored with
   `exp2_external.infer`, whose failure score is `P(failing)`; a frame is
   predicted failing when that score is at or above the threshold.
3. **Definition.** `_binary_metrics(..., FAIL_IDX)` computes
   `FPR = FP / (FP + TN)`. On an all-normal surface with all frames predicted
   failing, `FP = 120`, `TN = 0`, so FPR = 1.0. That is the arithmetic the
   artifact reports.
4. **Reproduction of the frozen original test.** Re-scoring the committed
   rc1 checkpoint on `splits/test.csv` at threshold 0.5 reproduces the
   committed `frozen_original_test` confusion exactly:
   `TN=798, FP=80, FN=372, TP=1609`
   (compare `docs/research/rc1-core-eval.json:6-18`). This confirms the
   scoring path, the checkpoint, and `FAIL_IDX=0` are all consistent — the
   FPR=1.0 result is therefore a real out-of-domain generalisation failure of
   the model, not a mislabelled class, inverted score, or metrics bug.

## What is not yet independently reproducible

The stage-13 normal holdout manifest and its frames live under
`datasets/external/stage13-normal/` and are **intentionally not committed**
(they were removed from the working tree and are gitignored). They cannot be
re-downloaded within this task's constraints. Consequences:

* The exact FPR=1.0 number cannot be regenerated on demand right now; it is
  **confirmed from the committed artifact and corroborated** by the exact
  frozen-test reproduction above, but not re-executed.
* The new reliability framework reports this surface as a missing-data error
  and returns **INCONCLUSIVE**, never a pass — see below.

## Reproducibility note on the existing harness (two-step pipeline)

The committed `rc1-core-eval.json` was produced by **two** scripts, and this
ordering matters:

1. `scripts/eval_rc1_core.py` writes the frozen-test block and the
   held-out-normal block. Its
   `ft.get("metrics_failing_positive")` / `ft.get("metrics_normal_positive")`
   calls (`scripts/eval_rc1_core.py:32-33`) return `None` against the current
   `evaluate_frozen_test`, because `evaluate_classification` returns
   `ClassificationMetrics.to_dict()`
   (`training/tv/metrics.py:30-42`, keys
   `precision, recall, f1, accuracy, fnr, fpr, auroc, confusion_matrix,
   sample_count, classes`, class-1-positive) plus `class_to_idx`/`checkpoint`.
2. `scripts/eval_rc1_frozen_test.py` then re-scores the frozen test and
   **overwrites** those three keys (`metrics_failing_positive`,
   `metrics_normal_positive`, `auroc_failing`) with the explicit
   failing-as-positive values.

So the committed values are correct, but the frozen-test
failing-positive metrics come from step 2, not step 1; step 1's `.get(...)`
placeholders are filled later. The held-out-normal block comes from
`evaluate_heldout_normal`, which does return those keys.

This ordering dependency is exactly the kind of silent drift the new framework
removes: metrics have a single definition, a fixed positive class, undefined
values are `null` with a reason instead of `0.0` or a missing key, and each
surface is derived from the same scoring path.

## Release gate added

`tv reliability` (see `../reliability-evaluation.md`) turns this into a
permanent, versioned acceptance decision:

* Suite: `training/configs/reliability/suite.yaml` (`rc1-reliability`) with
  `dev_val`, `frozen_test` and `target_normal_holdout` (role
  `normal_holdout`, the FPR=1.0 surface).
* Policy: `training/configs/reliability/policy.yaml` requires
  `normal_holdout` (`min_samples: 100`, `min_sources: 2`, `max_fpr: 0.1`).
  A restored holdout that again produces FPR=1.0 fails the gate (exit 2).
  A missing holdout is INCONCLUSIVE and also blocks release (exit 3).

### Live run (this machine, external holdout absent)

    reliability validate-suite -> dev_val [ok] n=2174, frozen_test [ok] n=2859,
                                  target_normal_holdout [MISSING]
    reliability evaluate       -> exit 2
      dev_validation   FAIL        (recall 0.8603 < min 0.9 at selected threshold 0.3)
      locked_test      FAIL        (recall 0.8430 < min 0.9)
      normal_holdout   INCONCLUSIVE (required evidence missing)
      failure_holdout  WARN        (optional; no verified independent failing holdout yet)
      cross_environment WARN       (optional; none staged yet)
      parity           PASS        (ONNX disagreement 0.0, max |dscore| 3e-6)
      deployment       PASS

The gate blocks the release for two independent reasons: the model does not
meet the recall floor, and the normal-holdout evidence that exposed the
false-alarm failure is not present. Neither can be reported as a pass.

## Tests

`training/tests/test_reliability.py` (26 tests) locks in the behaviour,
including a direct regression test of this finding: an all-normal surface
predicted entirely failing yields `fpr == 1.0`, `fnr == None`, and a gate
FAIL against `max_fpr`. It also covers leakage blocking release and a
missing required surface ending INCONCLUSIVE (exit 3) end-to-end.

Full suite: 407 passed, 17 skipped, 0 failed. The six previously
erroring/failing `tests/test_exp2_external.py` cases now **skip explicitly**
when the external experiment-2 data is absent, matching the
`test_skz_experiment.py` pattern: a missing-data skip, never a pass. That data
is absent because the external `video-frames` label JSONs are **tracked in git
but deleted in this working tree**, and the frames they reference are
gitignored and were never committed — so the tests cannot run here; a
`FileNotFoundError` is caught and reported as an explicit skip.

## Reproduced evidence (not committed)

`tv reliability evaluate` was re-run against the current rc1 checkpoint and
the frozen test reproduced `docs/research/rc1-core-eval.json` exactly
(`TN=798, FP=80, FN=372, TP=1609`; P=0.9526 R=0.8122 FPR=0.0911 at 0.5). The
120-frame independent normal holdout could not be reproduced: its frames are
not committed and the source is non-redistributable, so the FPR=1.0 number
remains confirmed-by-artifact and corroborated, not independently re-executed.

Gate behaviour was verified through the real CLI with exit codes: PASS=0,
optional-missing WARN=0, FAIL=2 (real suite), required-missing
INCONCLUSIVE=3, leakage=2.

