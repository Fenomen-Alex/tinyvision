# SKZ Nozzle-Cam Probe Experiments (A: External Generalization, B: Auxiliary Training)

**Date**: 2026-09-06
**Pipeline**: `training/` (`tv` CLI) · **Baseline model**: `shufflenet_v2_x1_0`
**SKZ probe**: `datasets/external/skz-probe/` (manifest.json, schema `skz-probe` v1, 166 images)
**License**: recorded, not resolved (HF card CC BY-SA 4.0; Zenodo record 18628777 CC BY 4.0).

## Purpose

Answer two separate, reproducible questions about the current TinyVision binary
classifier (normal vs failing):

1. **Experiment A (external generalization):** does the model generalize to the
   SKZ Nozzle-Cam domain?
2. **Experiment B (auxiliary-data benefit):** does adding a group-split SKZ subset
   to training improve the frozen TinyVision held-out test without degrading it,
   and what happens to a *held-out* SKZ slice?

## Task compatibility

The TinyVision model is **binary**; the probe records four image-level classes.
Mapping used (faithful, never a redefinition of the original task):

| Probe derived class | Binary label | Note |
|---|---|---|
| stringing, spaghetti, under_extrusion | `failing` | failure-type detail preserved in the report |
| normal | `normal` | **candidate** label: "no defect-class box in this annotation", provider-annotated, NOT human-verified ground truth |

The model therefore cannot distinguish the four SKZ classes from one another; it
is evaluated only on the binary failing/normal decision, with per-failure-type
recall reported within the failing group.

## Grouping / independent-units rule

Group key = the `Profile_<params>__<variant>` token (inferred, `grouping_verified=false`).
**36/166 probe tokens span more than one derived class** (transition prints whose
later frames carry defect boxes), so the unit of independence is the token across
ALL classes. The Experiment-B train/eval split (`split_by_group`, seed 42,
train fraction 0.8) never splits a token:

- **SKZ-aux** (added to training): 135 images, 60 groups — normal 39 / stringing 11 / spaghetti 46 / under_extrusion 39
- **SKZ-eval** (held out, never trained): 31 images, 16 groups — normal 11 / stringing 5 / spaghetti 4 / under_extrusion 11

The 16 stringing images were never balanced or duplicated; 11 went to aux, 5 to eval.

## Experiment A — SKZ external generalization (no training)

Command: `tv skz-eval --checkpoint experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/best.pt --subset all|skz_eval`

### Baseline checkpoints & immutable results

- Checkpoint used: `experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/best.pt` (the frozen baseline used throughout Stages 3–11B; binary, `failing=0/normal=1`, decision threshold on P(failing) = 0.5).
- Recomputing the baseline on the frozen test set reproduces the stored/immutable results exactly (acc 0.836, normal-positive F1 0.757, AUROC 0.900; failing-positive F1 0.876, recall 0.839, FPR 0.170).

### Result A: full probe (166 images, 76 independent groups, threshold 0.5)

| Metric (failing-positive) | Value |
|---|---|
| accuracy | 0.699 |
| precision | 0.709 |
| recall | 0.966 |
| F1 | 0.818 |
| FNR | 0.034 |
| FPR | 0.920 |
| AUROC (failing) | 0.611 |

Per-failure-type recall: Unterextrusion 0.98 (60 images[^1]), Stringing 1.0 (17[^1]),
Spaghetti 0.92 (50). Multi-defect images (11): 10/11 predicted failing.
Confusion matrix: `[[TP=112 FN=4],[FP=46 TN=4]]`.

**Candidate-normal behaviour (critical):** 46/50 candidate-normal probe images are
predicted `failing` (FPR 0.92). The model detects SKZ *failures* very well but
essentially never recognises SKZ nozzle-cam *normal* frames as normal. Note the
provenance caveats: normal is a provider-derived candidate (not verified), and
35/50 of those normal frames come from tokens whose later frames carry defect
boxes (transition prints) — however even the 11 single-frame eval normals had
FPR 0.91, so this is a genuine domain-shift/false-alarm problem, not only a
labelling artefact. AUROC 0.61 reflects the same score-distribution shift.

### Result A: held-out slice (31 images, 16 groups) — the B-comparable subset

| Metric (failing-positive) | Value |
|---|---|
| accuracy | 0.677 |
| precision | 0.667 |
| recall | 1.0 |
| F1 | 0.800 |
| FNR | 0.0 / FPR 0.909 |
| AUROC | 0.723 |

## Experiment B — Auxiliary SKZ training (separate run)

Config: `training/configs/skz_expB_aux.yaml` (id `expB_skzaux_shufflenet_v2_x1_0`).
Identical training setup to the Experiment-1 baseline (shufflenet_v2_x1_0, 30
epochs, Adam lr 1e-3, wd 1e-4, seed 42, ES patience 10, fp32, 224×224, same
augmentation). Only the training set differs.

Derived split (built with `tv skz-prep --seed 42 --train-fraction 0.8`,
manifest in `datasets/processed/skz-aux-expB/derived_manifest.json`):

- train = printing-errors train (11,254) + SKZ-aux (135) = **11,389** (failing 5,959 / normal 5,430; SKZ candidate-normals are explicitly candidates)
- val = printing-errors val (2,174) — byte-identical copy
- test = printing-errors test (2,859) — byte-identical copy

The existing printing-errors splits, the probe, and the baseline are all untouched.
Training early-stopped at epoch 15 (best val_loss 0.4419).

### Result B: TinyVision held-out test (frozen, failing-positive)

| Metric | Baseline | Exp B | Δ |
|---|---|---|---|
| accuracy | 0.836 | 0.878 | **+0.042** |
| precision | 0.918 | 0.970 | +0.052 |
| recall | 0.839 | 0.850 | +0.012 |
| F1 | 0.876 | 0.906 | **+0.030** |
| FNR | 0.162 | 0.150 | −0.012 |
| FPR | 0.170 | 0.060 | **−0.109** |
| AUROC | 0.900 | 0.951 | +0.050 |

All metrics improved; no degradation. False-alarm rate roughly cut by a factor of
three. The two historically hardest test recordings (`06012023_12_17_19`,
`01122022_16_06_44`) remain largely misclassified by B (failing recall 0.008 / 0.134),
so the known miscalibration problem on source-labelled-good recordings persists.

### Result B: held-out SKZ slice (31 images, 16 groups) — never used in training

| Metric (failing-positive) | A (baseline) | B | Δ |
|---|---|---|---|
| accuracy | 0.677 | 0.645 | −0.032 |
| precision | 0.667 | 0.645 | −0.022 |
| recall | 1.0 | 1.0 | 0 |
| F1 | 0.800 | 0.784 | −0.016 |
| FPR | 0.909 | 1.0 | +0.091 |
| AUROC | 0.723 | 0.509 | −0.214 |

B still detects every held-out SKZ failure (recall 1.0, all failure types), but on
this small slice the SKZ *normal* false-alarm behaviour did not improve — it went
slightly worse (FPR 1.0; AUROC at chance 0.51). The 39 candidate-normal SKZ-aux
training images did not transfer a usable "nozzle-cam normal" concept to eval
normals.

## Comparison table (the requested form)

| Model | TinyVision held-out | SKZ external |
|---|---|---|
| Existing baseline | acc 0.836 / F1 0.876 / FPR 0.170 | **A**: acc 0.699 / recall 0.966 / FPR 0.92 |
| TinyVision + SKZ | **B**: acc 0.878 / F1 0.906 / FPR 0.060 | **B**: acc 0.645 / recall 1.0 / FPR 1.0 |

(The SKZ external columns come from the **held-out 31-image slice** for an
apples-to-apples B comparison; the A row additionally has the full-166 result above.)

## Interpretation

1. **External generalization (A):** the current binary model transfers *failure
   detection* to the SKZ Nozzle-Cam domain almost perfectly (recall 0.966, FNR 0.034,
   every failure type ≥ 0.92). It does **not** transfer *normal recognition*:
   candidate-normal frames are called failing almost all the time (FPR 0.92,
   AUROC 0.61). This matches Stage 3–5 findings about over-confidence on
   out-of-distribution "normal" content: the failure class is well represented,
   the normal class is not separable across domains by this model.
2. **Auxiliary benefit (B):** adding 135 SKZ images to training **improved** the
   royalty-free TinyVision held-out test on every metric (no degradation) — the
   single biggest change was FPR 0.170 → 0.060. B did **not** fix the SKZ normal
   false-alarm problem on the held-out slice (FPR 1.0, AUROC ~chance), though the
   slice is tiny (31 images / 16 groups) and not statistically robust.
3. **Caveats:** SKZ normal labels are candidates; SKZ annotations are provider
   annotations and correctness is not claimed; probe group keys are inferred
   (`grouping_verified=false`); full train-side leakage is unverifiable (HF tree
   pagination) but irrelevant to these numbers because eval used only held-out
   probe tokens. All SKZ numbers are exploratory.
4. The engineering takeaway for the *external* SKZ domain is not "keep training on
   it" (B's overall effect on SKZ-normal was flat-to-worse) but that this model
   needs a real, verified normal-signal concept (ideally own data / verified
   frames) before any nozzle-cam deployment. The strongest measurable, positive
   result of B is the held-out TinyVision improvement.

## Reproducibility

Recorded per experiment (config.yaml at each experiments/<id>/):
experiment IDs, model+checkpoint, dataset+manifest versions, token split
(seed 42, fraction 0.8 → groups listed in `derived_manifest.json` `skz_split`),
image/class counts, hyperparameters, software/hardware stamps, metrics, and
artifact paths. All commands below are deterministic.

## Commands

```bash
# baseline verification (reproduces stored results)
tv evaluate --config training/configs/experiment1_shufflenet_v2_x1_0.yaml

# Experiment A (external generalization, no training)
tv skz-eval --checkpoint experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/best.pt \
  --subset all     --output-json experiments/skz_expA_external_eval/artifacts/probe_all.json
tv skz-eval --checkpoint experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/best.pt \
  --subset skz_eval --output-json experiments/skz_expA_external_eval/artifacts/probe_heldout.json

# Experiment B (auxiliary training; separate run)
tv skz-prep --printing-errors-dir datasets/processed/printing-errors \
  --out-dir datasets/processed/skz-aux-expB --train-fraction 0.8 --seed 42
tv train --config training/configs/skz_expB_aux.yaml
tv evaluate --config training/configs/skz_expB_aux.yaml
tv eval-deep --config training/configs/skz_expB_aux.yaml --split test \
  --output-json experiments/expB_skzaux_shufflenet_v2_x1_0/deepeval/test.json
tv skz-eval --checkpoint experiments/expB_skzaux_shufflenet_v2_x1_0/artifacts/best.pt \
  --subset skz_eval --output-json experiments/expB_skzaux_shufflenet_v2_x1_0/artifacts/skz_heldout.json
```

Exact software: Python 3.11.15, torch 2.14.0, torchvision 0.29.0,
ultralytics 8.4.138, opencv 5.0.0 (as stamped in the baseline config).

## Artifacts

- `experiments/skz_expA_external_eval/` (config.yaml, status.json, `artifacts/probe_all.json`, `artifacts/probe_heldout.json`)
- `experiments/expB_skzaux_shufflenet_v2_x1_0/` (config.yaml, status.json, `artifacts/best.pt`, `artifacts/skz_heldout.json`, `deepeval/test.json`)
- `datasets/processed/skz-aux-expB/` (`derived_manifest.json`, `splits/{train,val,test}.csv`, `splits/splits.json`)

[^1]: failure-type image counts exceed derived-class counts when a type also
appears inside a different derived class (e.g. 11 multi-defect images), so Stringing
appears in 17 probe images and Unterextrusion in 60. A multi-defect image is counted
under every type it carries.