# RC2 first controlled experiment — Kaggle under-extrusion (printer-disjoint)

**Status:** diagnostic. Labels are **print-level, not human-verified** (agent has no
vision). Nothing here satisfies the release policy. Numbers are real outputs of the
audited evaluator and gate; they are not benchmark claims.

**Date:** 2026-10-14 · **Branch:** `feat/realdomain-automated-pipeline` (HEAD `b52cf74`)
· **Seed:** 42 · **Device:** MPS (Apple)

> **Update (same day): domain-shift ablation (photometric normalisation vs appearance
> augmentation).** See [Section 9](#9-domain-shift-ablation-photometric-normalisation-vs-appearance-augmentation).
> Headline: both interventions raise threshold-independent AUROC on the **held-out**
> printer 103 (baseline 0.708 → photometric 0.809–0.887 → appearance 0.851–0.933,
> stable across seeds 42/43), but **neither yields a transferable operating point**
> (the validation-selected threshold gives 0 held-out recall) and printer-022
> discrimination stays below chance. Recommendation (Section 12): prioritise
> collecting matched normal/failing streams on the intended printer/camera.

## 1. Why Kaggle

The RC2 diagnostic was blocked by too few **independent normal streams** and failing
prints (see [`rc2-realdomain-results.md`](rc2-realdomain-results.md)). The acquired
Kaggle competition dataset provides print-level labels: 27 good (normal) and 25
under-extrusion (failing) prints, each print one independent stream, each printer one
environment. See
[`datasets/kaggle-early-detection-3d-printing-issues.md`](datasets/kaggle-early-detection-3d-printing-issues.md).

## 2. Pre-training checks (performed before training)

| Check | Result |
|---|---|
| Source licence / permitted use | Kaggle **Competition Rules** — research-only, non-redistributable. Verified from the rules page; rules accepted by the user. |
| CC BY-NC exclusion | EmptyDeck (CC BY-NC 4.0) was **not** used for training; only the research-only Kaggle data was. |
| Label mapping | print-level `has_under_extrusion`: `1` = failing = positive (`FAIL_IDX=0`), `0` = normal. `class_to_idx` printed by training: `{failing: 0, normal: 1}`. |
| Print-level consistency | No `print_id` has mixed labels (checked); grouping by `print_id` is leak-safe. |
| Environment separation | Splits are **printer-disjoint**: train 101/102/104, val 022, held-out 103. `scripts/kaggle_rc2_prepare.py` asserts no printer spans splits (`leak_check: ok`). |
| Manifest integrity | `validate_manifest` → 0 warnings; `build_streams` → 52 streams / 81,060 frames. |
| Suite leakage | `tv reliability validate-suite` → OK (no duplicate images / shared source groups across surfaces). |

**Distributability:** all data used is research-only/non-redistributable, so the trained
checkpoint is **not distributable**. It and all run artifacts live outside Git under
`/Users/alex/tinyvision-data/kaggle-rc2-experiment/run1/` (checkpoint
`…/experiments/kaggle_rc2_run1_shufflenet_v2_x1_0/artifacts/best.pt`).

## 3. Experiment matrix (small, reproducible)

One model, one seed — deliberately not a hyperparameter search.

| Item | Value |
|---|---|
| Model | `shufflenet_v2_x1_0` (torchvision, ImageNet pretrained) |
| Input | 224×224, ImageNet normalisation |
| Augment | hflip, rot 10°, colour-jitter 0.2, RandomResizedCrop, normalise |
| Optimiser | Adam, lr 5e-4, wd 1e-4 |
| Epochs / batch | 8 / 32, early-stop patience 4 |
| Train cap | 100 frames/stream (27 streams → 2,700 frames; CPU/MPS-feasible) |
| Eval surfaces | **full** frames (val 7,492; locked fails 6,952; normal holdout 2,975) |
| Seed / device | 42 / mps |

Training: best val loss 1.471 (epoch 6), 465 s. Train acc ≈ 0.98, val (frame) acc
0.25–0.43 → clear **cross-printer overfitting**.

Provenance/hashes: `run1/provenance.json` (checkpoint, config, split CSVs, dataset
manifest sha256; all recorded). Reproduce with:

```
scripts/kaggle_rc2_prepare.py --src <data>/splits --out <run> --data-dir <run> --train-per-stream 100
tv train --config <run>/config.yaml --experiments-dir <run>/experiments
tv reliability evaluate --suite <run>/suite.yaml --policy <policy> --model rc2=<ckpt> --config <run>/config.yaml
scripts/kaggle_rc2_report.py --run <run> --cache <run>/score-cache --out <run>/reports/per_stream.json --model rc2=<ckpt> --model rc1=<rc1-ckpt>
```

## 4. Results (identical samples for RC1 and RC2)

Threshold 0.5. Stream decision = median P(failing) over the stream's frames ≥ 0.5.

### Per-frame

| surface | model | n | FPR | FNR | P | R | F1 | AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dev_val (printer 022) | rc2 | 7,492 | 0.642 | 0.522 | 0.339 | 0.478 | 0.397 | **0.382** |
| dev_val | rc1 | 7,492 | 0.991 | 0.133 | 0.376 | 0.867 | 0.525 | **0.289** |
| locked_test (printer 103, failing only) | rc2 | 6,952 | n/a | 0.949 | 1.000 | 0.051 | 0.097 | n/a |
| locked_test | rc1 | 6,952 | n/a | 0.291 | 1.000 | 0.709 | 0.829 | n/a |
| normal_holdout (printer 103, normal only) | rc2 | 2,975 | 0.000 | n/a | n/a | n/a | n/a | n/a |
| normal_holdout | rc1 | 2,975 | 0.424 | n/a | n/a | n/a | n/a | n/a |

### Per-stream (print = independent unit)

| surface | model | normal streams | false alarms | stream FAR (95% CI) | failing streams | detected | event recall (95% CI) |
|---|---|---:|---:|---|---:|---:|---|
| dev_val | rc2 | 10 | 6 | 0.60 (0.31–0.83) | 7 | 4 | 0.57 (0.25–0.84) |
| dev_val | rc1 | 10 | 10 | 1.00 (0.72–1.00) | 7 | 6 | 0.86 (0.49–0.97) |
| locked_test | rc2 | 0 | – | – | 4 | 0 | **0.00 (0.00–0.49)** |
| locked_test | rc1 | 0 | – | – | 4 | 4 | 1.00 (0.51–1.00) |
| normal_holdout | rc2 | 4 | 0 | **0.00 (0.00–0.49)** | 0 | – | – |
| normal_holdout | rc1 | 4 | 2 | 0.50 (0.15–0.85) | 0 | – | – |

### Reading the numbers

* **Neither model generalises across printers.** On the unseen validation printer
  (022) both rank normal frames **above** failing frames (AUROC rc2 0.382, rc1 0.289 —
  confirmed directly, not an orientation bug: mean P(failing) is higher for normals).
* **RC1** collapses to "always failing" off-domain (dev FAR 1.0, normal-holdout FAR
  0.5) but "detects" fails (recall 1.0) — a degenerate alarm-everything point.
* **RC2** is less trigger-happy: zero false alarms on the held-out printer's normals
  (FAR 0.0) — but it also predicts almost everything normal there, so it **misses every
  held-out failing print** (recall 0.0, CI upper 0.49). The 0.0 FAR is therefore a
  degenerate operating point, not evidence of good normal discrimination.
* Against the RC1 false-alarm failure (FPR=1.0), RC2 reduces per-stream FAR on a new
  printer (1.0 → 0.0–0.6) but pays for it with lost failure recall. The primary RC2
  hypothesis (real in-domain data closes the normal false-alarm failure) is **not
  supported** by this run.

## 5. Release gate

`tv reliability evaluate --policy training/configs/reliability/policy.yaml`:
**RELEASE GATE: FAIL (blocks_release=True)**.

* dev_validation: **FAIL** — FPR 0.642 > 0.3; recall 0.478 < 0.9.
* locked_test: INCONCLUSIVE — FPR undefined (no normal samples in the locked surface).
* normal_holdout: PASS (FPR 0.0) — but see the degenerate-point caveat above; labels are
  print-level, not human-verified.
* parity: INCONCLUSIVE — no ONNX/INT8 artifact.
* threshold selection: INCONCLUSIVE — no threshold in the release grid `[0.3…0.7]`
  satisfies FPR ≤ 0.2 on dev_val.

**RC1 gate comparison (same suite):** also FAIL on dev_validation (FPR 0.991, recall
0.867 still below the FPR floor). RC2 improves FPR but not enough to pass.

## 6. Limitations

* **Labels are print-level and not human-verified** — the agent cannot see images; every
  metric is against dataset-supplied labels. No release claim is possible.
* **No failing normal-holdout independence:** the locked_test surface holds only failing
  streams (printer 103), so its per-frame FPR is undefined; the normal-holdout and
  locked_test are the same printer (class-disjoint groups, so no leakage, but a single
  environment).
* **Single failure mode** (under-extrusion) and **nozzle-camera domain** only.
* **Small training sample** (100 frames/stream, 3 printers): this run is a first
  controlled signal, not a tuned model. The cross-printer collapse may partly reflect
  the cap and lack of domain-randomised augmentation.
* Only 5 labeled printers exist total, so environment-held-out train/val/test is tight
  by construction.

## 7. Single highest-value next step

The result is a **domain-shift problem, not a calibration or capacity problem**. The
highest-value next step is to attack cross-**printer** generalisation directly:
hold out whole printers, and train with strong appearance/photometric
domain-randomisation plus per-printer standardisation (or a small domain-adaptation /
style-normalisation step), then re-run this exact suite. If per-printer normalisation
does not lift held-out AUROC above 0.5, the dataset's printer confound — not the
model — is the blocker, and effort should shift to collecting normal/failing streams
from a *matched* environment (same printer/camera) for the target deployment.

## 8. Artifacts

* `run1/provenance.json` — seed, hashes (checkpoint/config/splits/manifest), env.
* `run1/config.yaml`, `run1/suite.yaml`, `run1/splits/`.
* `run1/reports/{diagnostic,release}/summary.{json,txt}` — gate evidence.
* `run1/reports/per_stream.json` — per-frame + per-stream metrics, Wilson CIs.
* `run1/score-cache/*.npy` — cached P(failing) per checkpoint/surface.
* `scripts/kaggle_rc2_prepare.py`, `scripts/kaggle_rc2_report.py` — reproducible drivers.

All under `/Users/alex/tinyvision-data/kaggle-rc2-experiment/run1/` (outside Git;
research-only, non-redistributable).

---

# 9. Domain-shift ablation: photometric normalisation vs appearance augmentation

**Question:** does photometric variation or training-time appearance augmentation
improve discrimination across *unseen* printers?

This is a controlled ablation on the **same** printer-disjoint splits as Section 3/4
(train printers 101/102/104, validation printer 022, held-out printer 103). Printer 103
was never used for training, preprocessing-statistic fitting, augmentation tuning,
calibration, or threshold selection. Two interventions were compared against an exact
baseline reproduction, each repeated at a second seed. No hyperparameter search.

New artifacts live outside Git under
`/Users/alex/tinyvision-data/kaggle-rc2-experiment/ablation/` (research-only,
non-redistributable). Provenance/hashes: `ablation/provenance.json`; consolidated
metrics: `ablation/reports/per_stream.json`; in-repo summary:
[`rc2-domain-shift-ablation.json`](rc2-domain-shift-ablation.json).

## 9.1 Minimal pipeline change + regression tests

The shared loader only supported ImageNet normalisation and a fixed augmentation set.
Two backward-compatible extensions were added (defaults preserve old behaviour):

* `AugmentationConfig.normalize_mean` / `normalize_std` — custom per-channel statistics,
  used for **both** training and evaluation/inference (`tv/data/loaders.py`
  `normalize_stats`/`_build_transforms`).
* `AugmentationConfig.appear_brightness/contrast/saturation/hue`, `blur_prob`,
  `blur_sigma`, `noise_std` — train-only appearance augmentations.
* `tv/train.py` now stores `normalize_mean`/`normalize_std`/`image_size` in the
  checkpoint; `tv/reliability.py` `score_framework` reads them, so inference reuses the
  exact training normalisation (falls back to ImageNet when unset — baseline unchanged).

Regression tests (`training/tests/test_normalization.py`, 5 tests) pin the invariant that
the inference transform equals the deterministic training/validation transform for custom
statistics, and that `score_framework` honours the checkpoint's statistics.

## 9.2 Baseline reproduction (exact)

Config copied verbatim from `run1/config.yaml` (`ablation/configs/baseline.yaml`).
Training with seed 42 reproduced run1 **exactly**:

* `best_val_loss` 1.471287 — identical to run1.
* All 338 model parameter tensors bit-identical to `run1/.../best.pt`
  (the checkpoint file sha differs only because extra metadata keys are now stored).
* `score_framework` on `dev_val`: max abs diff **0.0** vs the saved
  `run1/score-cache/7afe59dcd0aca464_dev_val.npy`.
* Release gate: **FAIL (exit 2)**, identical surface metrics to the run1 gate.

Reproduction is exact, so the interventions below are compared against a faithful
baseline.

## 9.3 Interventions

**E2 photometric normalisation** (`ablation/configs/photometric.yaml`): only the input
normalisation changed, to per-channel mean/std fit on the **2,700 training frames only**
(`scripts/kaggle_rc2_photometric_stats.py`, using the shared deterministic
`Resize((224,224))+ToTensor` path): `mean=[0.44555, 0.445957, 0.434777]`,
`std=[0.308143, 0.313864, 0.335357]`. Identical deterministic preprocessing at
train/val/inference.

**E3 appearance augmentation** (`ablation/configs/appearance.yaml`): baseline
preprocessing (ImageNet normalisation) retained; added **train-only** moderate
appearance augmentations — `ColorJitter(brightness/contrast/saturation=0.3, hue=0.05)`,
`GaussianBlur(p=0.3, sigma∈[0.1,1.2])`, additive Gaussian noise `std=0.03`. Validation
and inference preprocessing unchanged and deterministic.

## 9.4 Exact commands

```bash
REPO=/Users/alex/Projects/tinyvision
RUN=/Users/alex/tinyvision-data/kaggle-rc2-experiment/run1
ABL=/Users/alex/tinyvision-data/kaggle-rc2-experiment/ablation

# fit training-only photometric statistics
$REPO/training/.venv/bin/python $REPO/scripts/kaggle_rc2_photometric_stats.py \
  --data-dir $RUN --out $ABL/train_photometric_stats.json

# train each variant (seed 42 in config; seed-43 configs derive from these)
for cfg in baseline photometric appearance photometric_seed43 appearance_seed43; do
  $REPO/training/.venv/bin/tv train --config $ABL/configs/$cfg.yaml \
    --experiments-dir $ABL/experiments
done

# per-frame + per-stream metrics, incl. combined held-out printer-103 AUROC,
# and a validation-only selected threshold
$REPO/training/.venv/bin/python $REPO/scripts/kaggle_rc2_report.py \
  --run $RUN --cache $ABL/score-cache --out $ABL/reports/per_stream.json \
  --model baseline=$ABL/experiments/kaggle_rc2_run1_shufflenet_v2_x1_0/artifacts/best.pt \
  --model photometric=$ABL/experiments/rc2_ablation_photometric_shufflenet_v2_x1_0/artifacts/best.pt \
  --model appearance=$ABL/experiments/rc2_ablation_appearance_shufflenet_v2_x1_0/artifacts/best.pt \
  --model photometric_s43=$ABL/experiments/rc2_ablation_photometric_seed43_shufflenet_v2_x1_0/artifacts/best.pt \
  --model appearance_s43=$ABL/experiments/rc2_ablation_appearance_seed43_shufflenet_v2_x1_0/artifacts/best.pt

# release gate (fails by design; run for the record)
$REPO/training/.venv/bin/tv reliability evaluate --suite $RUN/suite.yaml \
  --policy $REPO/training/configs/reliability/policy.yaml \
  --model baseline=$ABL/experiments/kaggle_rc2_run1_shufflenet_v2_x1_0/artifacts/best.pt \
  --config $ABL/configs/baseline.yaml --output-dir $ABL/reports/release_baseline
```

`--image-size 224`, `--batch-size 32`, seed 42 (or 43), model `shufflenet_v2_x1_0`,
8 epochs / patience 4, Adam lr 5e-4 wd 1e-4, MPS. Checkpoint sha256 (first 16): baseline
`4552aa1342caa5fd`, photometric `15460109626f0bb5`, appearance `36b8653901df7169`,
photometric_s43 `e29d29f6ed7ef814`, appearance_s43 `4e4390361c5107e6`.

## 9.5 Results (identical samples and metric definitions for all models)

`holdout103_combined` concatenates the class-disjoint **held-out printer 103** surfaces
(locked_test failing streams + normal_holdout normal streams) so that a
threshold-independent AUROC is defined on the held-out environment. `dev_val` is
printer 022 (both classes). AUROC is threshold-independent; the operating-point columns
use threshold 0.5, and the last three columns use the threshold selected on `dev_val`
only (max recall s.t. FPR ≤ 0.10).

| model | seed | best_val_loss | dev AUROC | **heldout103 AUROC** | dev sFAR@.5 | dev sRecall@.5 | p103 sFAR@.5 | p103 sRecall@.5 | val-t | dev FPR@val-t | p103 recall@val-t | p103 FAR@val-t |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 42 | 1.4713 | 0.382 | **0.708** | 0.60 | 0.571 | 0.00 | 0.000 | 0.98 | 0.065 | 0.00 | 0.00 |
| photometric | 42 | 1.3177 | 0.225 | **0.887** | 1.00 | 0.714 | 0.00 | 0.250 | 0.90 | 0.099 | 0.00 | 0.00 |
| appearance | 42 | 1.7864 | 0.345 | **0.933** | 1.00 | 1.000 | 0.25 | 1.000 | 0.98 | 0.071 | 0.00 | 0.00 |
| photometric | 43 | 1.4123 | 0.327 | **0.809** | 0.90 | 0.571 | 0.00 | 0.000 | 0.95 | 0.073 | 0.00 | 0.00 |
| appearance | 43 | 1.2297 | 0.300 | **0.851** | 1.00 | 1.000 | 0.00 | 0.250 | 0.92 | 0.075 | 0.00 | 0.00 |

Per-stream 95% Wilson CIs (threshold 0.5, `holdout103_combined`):

| model | p103 failing detected | p103 event recall (95% CI) | p103 false alarms | p103 FAR (95% CI) |
|---|---|---|---|---|
| baseline | 0/4 | 0.00 (0.00–0.49) | 0/4 | 0.00 (0.00–0.49) |
| photometric (42) | 1/4 | 0.25 (0.05–0.70) | 0/4 | 0.00 (0.00–0.49) |
| appearance (42) | 4/4 | 1.00 (0.51–1.00) | 1/4 | 0.25 (0.05–0.70) |
| photometric (43) | 0/4 | 0.00 (0.00–0.49) | 0/4 | 0.00 (0.00–0.49) |
| appearance (43) | 1/4 | 0.25 (0.05–0.70) | 0/4 | 0.00 (0.00–0.49) |

`dev_val` per-stream (printer 022) at 0.5: baseline FAR 0.60 / recall 0.571; photometric
FAR 1.00 / recall 0.714; appearance FAR 1.00 / recall 1.000 (all models alarm on nearly
every printer-022 normal print).

## 9.6 Threshold-independent vs operating point

* **Threshold-independent (AUROC).** On the held-out printer 103, both interventions
  improve ranking over baseline, consistently across seeds: baseline 0.708; photometric
  0.887 / 0.809; appearance 0.933 / 0.851. All exceed the 0.5 directional floor, and the
  interventions beat the baseline. On `dev_val` (printer 022), however, **every** model is
  below chance (0.225–0.382) and the interventions are *worse* than baseline.
* **Operating point.** At a fixed 0.5 the improvements are mixed: photometric (42)
  detects 1/4 held-out failing prints with **0** held-out false alarms; appearance (42)
  detects 4/4 but flags 1/4 held-out normal prints (FAR 0.25) and alarms on nearly all
  printer-022 normals. At the threshold selected on `dev_val` (0.90–0.98, driven high
  because printer-022 scoring is anti-correlated) **held-out recall collapses to 0 for
  every model** — no learned global threshold transfers.

## 9.7 Release gate

`tv reliability evaluate --policy training/configs/reliability/policy.yaml`:

* **baseline:** FAIL (exit 2) — identical to the run1 gate (dev_validation FPR 0.642 >
  0.3, recall 0.478 < 0.9; locked_test INCONCLUSIVE; normal_holdout PASS).
* **appearance (best held-out AUROC):** FAIL (exit 2) — dev_validation FPR 0.979 > 0.3;
  **normal_holdout FAIL** (FPR 0.163 > 0.10); locked_test INCONCLUSIVE.

Neither the baseline nor the best intervention passes; the appearance model additionally
fails the normal-holdout gate that the baseline passed.

## 9.8 Tests

`cd training && .venv/bin/python -m pytest tests -q` → **452 passed, 17 skipped**
(447 prior + 5 new normalisation-parity tests). No existing test changed.

## 10. Did either intervention improve held-out discrimination without collapsing recall?

**Yes on threshold-independent discrimination; no on a usable operating point.**

* Both interventions raise held-out (printer-103) AUROC over baseline, stable across two
  seeds (photometric 0.81–0.89; appearance 0.85–0.93 vs baseline 0.71), so the minimum
  directional signal (> 0.5) is met and improved. At threshold 0.5 these gains partially
  translate without a recall collapse for the seed-42 runs (photometric 25% held-out
  recall at 0 FAR; appearance 100% held-out recall at 25% held-out FAR).
* But the seed-43 repeats are weaker (0–25% held-out recall), and **no model produces a
  threshold that transfers**: the validation-selected operating point gives 0 held-out
  recall for all five models, because the only single-printer surface with both classes
  (`dev_val`, printer 022) is anti-correlated, pushing the FPR-constrained threshold to
  ~0.9–0.98.
* Printer 022 behaves anomalously: every model scores its normal prints *above* its
  failing prints (AUROC < 0.5), and the interventions make this worse. This is a real
  ranking signature, not an orientation bug, but with weak print-level labels and no
  vision it cannot be attributed to data vs model.

So the interventions are a genuine directional improvement on one held-out printer but do
not deliver an operating point — they move the ranking, not the decision boundary.

## 11. Limitations (ablation-specific)

* **Weak, unverified labels.** Print-level weak labels only; no human verification. A
  high held-out AUROC cannot be confirmed to reflect true failure discrimination, and
  printer-022's inverted ranking cannot be explained.
* **Held-out AUROC is one environment, 8 prints.** `holdout103_combined` has 4 failing +
  4 normal streams; Wilson CIs are wide (e.g. appearance event-recall 0.51–1.00) and the
  0.0/1.0 cells are small-sample.
* **Single failure mode** (under-extrusion), nozzle-camera domain, only 5 labelled
  printers (tight environment-held-out design).
* **Small training set** (100 frames/stream) and one frozen architecture; results are a
  controlled signal, not a tuned comparison.

## 12. Recommendation

**Prioritise collecting matched normal/failing streams on the intended printer and
camera.** The ablation shows the interventions only improve a threshold-independent
ranking metric on one held-out printer while making the in-domain validation printer
worse and never yielding a transferable threshold — i.e. the binding constraints are the
**printer/environment confound in the data** and the lack of **verified in-domain
labels**, not model-side appearance handling. New matched-environment streams (with
frame-verified labels) would let threshold selection and evaluation happen in-domain and
would directly test whether the printer-103 ranking signal is real.

If matched data collection is delayed, the single justified model-side follow-up is a
narrow **per-printer / test-time normalisation or calibration** step (not more
augmentation): it is the only thing that addresses the observed cross-printer score-scale
shift, and it can be evaluated on the existing printer-disjoint split. Do not expand into
a broad hyperparameter search: nothing in this dataset currently supports a release
claim.
