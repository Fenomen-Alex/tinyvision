# Reliability evaluation and release acceptance

The reliability framework makes evaluation of a frozen checkpoint repeatable,
auditable and leakage-resistant, and turns the release decision into an explicit
versioned policy. It replaces ad-hoc scoring scripts: same preprocessing, same
positive-class convention, same metrics definition every time.

Entry point (run from the repo root):

    training/.venv/bin/python -m tv.cli reliability <subcommand>

Subcommands:

    validate-suite --suite S [--dataset-root D] [--strict]
        Dry-run of a suite: checks each surface's data is present, labels map,
        required roles exist, and no source-level leakage. `--strict` exits 1
        if required surface data is missing.

    evaluate --suite S --policy P --model NAME=PATH [--model ...]
             [--config C] [--thresholds T,T,T] [--threshold T]
             [--dataset-root D] [--output-dir O] [--image-size 224]
        Scores every model over every surface of the suite (PyTorch on CPU,
        deterministic), auto-discovers sibling `model.onnx` / `model_int8.onnx`
        for parity, applies the acceptance policy, and writes
        `summary.json` + `summary.txt`. `--policy` is **required**: a release
        decision needs acceptance criteria, so a policy-less run is a usage
        error (exit 2) rather than a silent pass. Exit codes below.

    parity --suite S --framework CKPT --other ARTIFACT [--surface ID]
           [--threshold T] [--output-json F]
        One-off framework-vs-ONNX/INT8 comparison on a single surface (default:
        the first suite surface; use --surface to pick another).

Parity scope differs between the two commands and is printed in the summary:
`parity` compares the **whole** chosen surface, while `evaluate` compares the
**largest** surface bounded to the first `PARITY_MAX_SAMPLES` (512) samples —
so an `evaluate` parity line reads `parity[onnx] on <surface> n=512 ...` and
its numbers are not directly comparable to a full-surface `parity` artifact.

## Score convention

Failing is the positive class (`class_to_idx` sorted alphabetically:
`failing=0`, `normal=1`). Therefore:

    score       = P(class = failing) = softmax(logits)[0]
    prediction  = failing  iff  score >= threshold

All metrics in this framework use `positive_class=0`. The framework matches the
production ONNX preprocessor (`_build_transforms`, same `image_size`),
so parity numbers reflect genuine format drift, not preprocessing differences.

## Undefined metrics

A metric that is mathematically undefined on a surface (e.g. FPR with zero
normal samples, FNR/recall with zero failing samples) is reported as `null`
with a reason — never silently `0.0`. A gate whose required metric is `null`
is INCONCLUSIVE, not PASS.

## Missing evidence is a blocker, not a pass

If a surface's data file is absent (external/holdout datasets are intentionally
not committed), the surface is recorded with an `error`, and any gate that
requires it is INCONCLUSIVE. `evaluate` exits 3. A release can progress only
when the evidence for every `required` gate exists and passes.

## Leakage checks

`check_source_leakage` flags:

* duplicate image across any two surfaces;
* a source group shared between a training-facing surface (`dev_validation`)
  and an independent-holdout role (`normal_holdout`, `failure_holdout`,
  `cross_environment`, `locked_test`).

Leakage warnings set the overall evaluation to FAIL (exit 2), because they
invalidate the holdout claim.

## Acceptance policy

`training/configs/reliability/policy.yaml` defines gates keyed by surface role:

* `required: true`  — gate must PASS; missing/insufficient evidence is
  INCONCLUSIVE.
* `required: false` — absence yields WARN and does not block release.
* limit keys: `max_fpr`, `max_fnr`, `min_recall`, `min_precision`,
  `min_samples`, `min_sources`, `min_environments`,
  `max_model_size_mb`, `max_latency_ms`,
  parity limits `max_onnx_prediction_disagreement`/
  `max_onnx_score_abs_drift` and `require_onnx`/`require_int8`.

**Provisional floors.** The policy is `status: draft`. Its metric floors
(`min_recall`, `max_fpr`, `max_fnr`, size/latency budgets) are **provisional
engineering targets**, not empirically derived requirements. They must not be
weakened to make a particular model pass; changing a floor requires an
explicit, documented decision with a stated justification (and new evidence
where relevant). The current rc1 model does not meet them — that is the
intended signal, not a reason to lower them.

Status precedence in `overall_status`: any **required** gate FAIL ⇒ FAIL;
else any **required** gate INCONCLUSIVE ⇒ INCONCLUSIVE; else any **optional**
gate FAIL ⇒ WARN; else PASS. A missing optional surface WARNs only its own
gate and leaves `overall_status` at PASS (it never blocks). Exit code:
FAIL ⇒ 2, INCONCLUSIVE ⇒ 3, leakage ⇒ 2, otherwise 0 (WARN does not block).
If no gate decisions exist at all (no policy applied), the outcome is
INCONCLUSIVE (exit 3), never a pass.

Exit codes: `0` pass/warn, `2` FAIL, leakage, or CLI usage error, `3`
INCONCLUSIVE.

## Threshold policy

Threshold selection uses ONLY the `dev_validation` surface
(`objective: max_recall_subject_to_max_fpr`). If the validation surface cannot
support that choice (no normal+failing samples, or nothing satisfies the FPR
budget), selection is INCONCLUSIVE and the run falls back to the fixed
threshold **without claiming it was tuned**. Every surface reports metrics at
both the selected and the fixed threshold, plus a sweep.

## Current suite: `rc1-reliability`

Surfaces (see `training/configs/reliability/suite.yaml`):

| Surface | Role | Data | Notes |
|---|---|---|---|
| `dev_val` | dev_validation | `splits/val.csv` | threshold selection only |
| `frozen_test` | locked_test | `splits/test.csv` (2859) | frozen original test |
| `target_normal_holdout` | normal_holdout | `stage13-normal` manifest (120 frames, 2 sources) | **not committed**; the FPR=1.0 surface |

With the external manifest restored, evaluation of the rc1 model reproduces the
documented result: `target_normal_holdout` FPR = 1.0 at threshold 0.5, which
fails `normal_holdout.max_fpr = 0.1` and blocks release (exit 2).
With the manifest absent, the run is INCONCLUSIVE (exit 3) — it never reports a
passing release on missing evidence.

## Reproducibility record

Each report records the suite YAML, policy YAML and experiment config
(whichever are supplied) by SHA-256, the exact software versions, every score
per image, and the threshold selected. Identical inputs must produce identical
`summary.json` (CPU inference only).