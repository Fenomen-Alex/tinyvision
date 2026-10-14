# Model improvement plan (prepared, not run)

Constraint: this plan is **not executed** — no training was performed. It states
the evidence-backed diagnosis, the experiments to run, and the acceptance
criteria, so a future run is reproducible. Nothing here changes the RC1 verdict.

## Diagnosis (evidence)

| Evidence | Source |
|---|---|
| RC1 within-venue: frozen-test FPR 0.091, F1 0.876, AUROC 0.892 | [`rc1-core-eval.json`](rc1-core-eval.json), [`rc1-release-candidate.md`](rc1-release-candidate.md) |
| RC1 transfer: normal holdout **FPR 1.000**; alarms on every real-world normal clip (~frame 8) | [`../../ML.md`](../../ML.md), [`reliability-audit.md`](reliability-audit.md) |
| Trier holdout: predictions environment-correlated (camera-3 layer_shift recall 0.000) | [`../../ML.md`](../../ML.md) |
| val/test are same-venue recording splits; suite not environment-independent | [`evaluation-independence.md`](evaluation-independence.md) |
| Mild augmentation shared across rc1/exp3/exp4 | [`training-pipeline-audit.md`](training-pipeline-audit.md) |
| INT8 raises FPR by ~4 points | [`deployment-parity.md`](deployment-parity.md) |

**Diagnosis: domain shift / cross-environment overfitting, not a threshold or
calibration problem.** Within-venue performance is fine; the model keys on the
single ELP venue's appearance. Threshold sweeps on `val` cannot fix it
(monotone recall/FPR trade-off, no point satisfies recall 0.9 and FPR 0.2 — see
`docs/reliability-evaluation.md`).

## Hypotheses

* **H1 (primary):** appearance/domain-shift augmentation (stronger color,
  brightness/contrast, blur, camera-like noise, JPEG, scale/crop, mild
  geometric) improves normal-holdout FPR without hurting within-venue recall.
* **H2:** reducing the printing-errors (single-venue) share of training data,
  or up-weighting multi-environment data, reduces venue overfitting.
* **H3:** class weighting / balanced sampling raises the operating point
  margin (currently no knob).
* **H4:** INT8 must be validated independently under the normal-holdout FPR
  floor (measured +4 pt FPR on dev_val).

## Experiments to run (each = one new config; never mutate a run's config)

| ID | Change vs rc1 | Tests |
|---|---|---|
| `rc2_domain_robust` | H1 augmentation stack | H1 |
| `rc2_env_balanced` | H2 sampling/weighting | H2 |
| `rc2_class_weight` | H3 class weight | H3 |
| `rc2_int8_gate` | H4 (no training; export + holdout gate) | H4 |

Prepared configs live in [`../../training/configs/planned/`](../../training/configs/planned/)
and are **not** wired into any run.

## Acceptance criteria (unchanged policy; do not loosen)

A candidate is only an improvement if, on an **environment-independent** surface
(Trier holdout or collected footage), evaluated by
`tv reliability evaluate --policy training/configs/reliability/policy.yaml`:

* `normal_holdout` FPR ≤ 0.1 **and** the gate is PASS (not INCONCLUSIVE), and
* rc1's within-venue recall is not materially degraded (≥ 0.9 on `dev_validation`
  / `locked_test`).

The threshold stays selected on `dev_validation` only; it must never be tuned on
the locked test or the holdout. If no environment-independent surface is
available, the candidate is INCONCLUSIVE and is not a release.

## Data prerequisites (blocking)

Improvements cannot be *proven* without an environment-independent validation
surface. Required (see [`dataset-inventory.md`](dataset-inventory.md)):
extract the Trier holdout (≈48 GB) **or** collect own multi-environment footage,
and clear `tv env-readiness` (≥5 human-verified failure environments, full
failure-type coverage).

## What not to do

* Do not tune thresholds on the locked test or the normal holdout.
* Do not loosen `max_fpr` / `min_recall` to make a candidate pass.
* Do not report within-venue numbers as generalization.
* Do not treat INT8 as equivalent to fp32.
