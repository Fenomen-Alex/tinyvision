# Experiment 2 — External Video-Frames Augmentation (Verified Human-Reviewed Frames Only)

**Date**: 2026-09-06
**Pipeline**: `training/` (`tv` CLI) · **Baseline model**: `shufflenet_v2_x1_0`
**Experiment ID**: `exp2_external_video_aug_shufflenet_v2_x1_0`
**Training config**: `training/configs/exp2_external_video_aug.yaml`
**Derived split**: `datasets/processed/external-aug-exp2/` (`prepare_exp2_splits`)
**Module**: `training/tv/exp2_external.py` · **CLI**: `tv exp2-prep`, `tv exp2-eval`

## Purpose

Determine whether augmenting TinyVision training with **verified, human-reviewed external
video-frame footage** (the project's own Stage 12B camera footage) improves the frozen
TinyVision held-out test without degrading it — the same controlled baseline-vs-2A design
used for the SKZ auxiliary-data experiment. Because the project's failure-detection priority
is avoiding **false negatives**, and because a persistent weakness of the model is
out-of-domain *normal* false alarms, Experiment 2 additionally reports false-positive rates
on external normal frames **for both the baseline and the 2A model**.

## Question restriction

This experiment uses a strict label gate: **only human-verified decisions** from
`datasets/external/video-frames/labels/verified_labels.json` can enter training, and only
human-review manifest decisions can feed evaluation. VLM proposals, provisional Stage 12B
labels, provider-derived labels, `uncertain` decisions, and undecided units are all excluded.
If the resulting verified set is too small for a meaningful augmentation experiment, that is
itself a result — reported as a limitation rather than papered over.

## The verified external data (as of 2026-09-06)

| Item | Value |
|---|---|
| Verified failing units (merged) | **6** — `NORMAL001__seq000/001/002` (stringing, 21 frames) + `SP001__seq000/001/002` (spaghetti, 23 frames) |
| Verified failing frames | **44** (2 sources: NORMAL001, SP001) |
| Verified under_extrusion | **0 frames** (not verified) |
| Verification source | `verified_labels.json` (Stage 12F), sha256 `27038df1e7dbb81ff2da57b8d04e3d1d77ddf9ec6242d50409322b4204f50642` |
| Normal (manifest decision only) | **24 frames, 8 units** — `UE003` (`UE003__frm0000..0004` + `UE003__seq000/001/002`), human-review decisions in `review_manifest.json`; **not merged** into `verified_labels.json` |
| Excluded | 15 uncertain units; 977 undecided units |

The 24 UE003 normal frames are manifest decisions that Stage 12F left unmerged. They are used
**evaluation-only** — never as training labels — and are never silently merged.

## Group / role rule

Sources are whole-group assigned. A source is never split across train and eval.

| Source | Role in Exp 2A | Binary label |
|---|---|---|
| `NORMAL001`, `SP001` | Training augmentation (the 44 verified failing frames) | `failing` |
| `UE003` | External-normal evaluation only (both baseline and 2A) | `normal` |

Binary mapping (faithful, matching the model's binary task): `stringing`/`spaghetti`/
`under_extrusion` → `failing`; `normal` → `normal`; `uncertain` → excluded. Failure-type
detail is preserved in provenance.

## Split construction

`tv exp2-prep` copies the frozen printing-errors `train.csv` verbatim (11,254 rows) and
appends the 44 verified failing frames as absolute-path rows, producing:

| Split | Frames | Change vs frozen |
|---|---|---|
| train | **11,298** | +44 external failing (failing 5,863→5,907) |
| val | 2,174 | byte-identical to frozen |
| test | 2,859 | byte-identical to frozen |

The derived `splits.json` points at the printing-errors root as `data_dir`; relative
printing-errors rows resolve against it, absolute external paths pass through. This keeps the
**frozen** val/test byte-identical, so the primary comparison (baseline vs 2A on the frozen
test) is leakage-free and directly comparable. The 24 UE003 normal frames are **never** in
the derived split. A `derived_manifest.json` records coverage, role-by-source, sha256
integrity of the two label sources, and caveats.

## Run summary (deterministic, seed 42)

- **Baseline (immutable, stored)**: `experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4`,
  trained on printing-errors train (11,254). Stored frozen-test metrics used for comparison
  (recompute reproduces them exactly).
- **2A**: `experiments/exp2_external_video_aug_shufflenet_v2_x1_0`, same config/hyperparameters
  (seed 42, ShuffleNetV2 x1.0, adam lr 1e-3, batch 32, epochs 30/patience 10, aug =
  hflip/rot15/colorjitter0.2/no-crop/ImageNet-normalize), trained on the 11,298-frame derived
  train. Early stopping at epoch 12, best **val** loss 0.660. `artifacts/best.pt` saved.

## Frozen test result (primary comparison, threshold 0.5 on P(failing))

| Metric (failing-positive) | Baseline | 2A | Δ |
|---|---|---|---|
| accuracy | 0.836 | 0.738 | −0.098 |
| precision | 0.918 | 0.991 | +0.073 |
| recall | 0.839 | 0.628 | **−0.211** |
| F1 | 0.876 | 0.769 | −0.107 |
| false-positive rate | 0.170 | 0.013 | −0.157 |
| false-negative rate | 0.161 | 0.372 | **+0.211** |
| AUROC | 0.900 | 0.870 | −0.031 |

**Result: the 44-frame verified augmentation did not improve the frozen test — it degraded
it**, driven by a large recall loss (838→628 failing frames recalled on the 2,859 test
frames; 737 false negatives vs 320). The FPR improvement is a side effect of the same
normal-leaning shift, not a corrected false-alarm mechanism. The augmentation is far too
small (44 frames, 2 sources, ~0.4% of training) relative to the printing-errors diversity to
move a learned failure manifold usefully.

## External evaluation (descriptive, not a benchmark; threshold 0.5)

External numbers are **descriptive only**: UE003 is a single source, NORMAL001/SP001 are two
sources; balances are 24/44 frames. No external tuning happened.

### UE003 normal frames (24, seen by neither model)

| Model | Predicted failing | FPR |
|---|---|---|
| Baseline | 24 / 24 | 1.000 |
| 2A | 23 / 24 | **0.958** |

`UE003__frm0000` is the only normal frame any model has accepted. Both models keep flagging
this camera's genuinely-normal footage as failing. **The model's out-of-domain *normal*
false-alarm weakness is not fixed by 44 failing frames** — consistent with the SKZ
Experiment A result (FPR 0.92 on candidate-normal frames).

### Failing frames (44, NORMAL001 + SP001)

| Model | Kind | Recall | FNR |
|---|---|---|---|
| Baseline (never trained on them) | generalization | 0.977 (43/44) | 0.023 |
| 2A (trained on them) | training-fit (memorization check) | 0.977 (43/44) | 0.023 |

Both models already recognize these verified failures almost perfectly; the augmentation
added nothing on the failing side.

## Conclusions

1. **The verified external set is too small for a meaningful augmentation experiment.** 44
   failing frames / 2 sources cannot plausibly transfer a domain; the observed frozen-test
   regression is the informative result.
2. **Adding failing frames does not fix normal false alarms.** UE003 normal FPR stays ~1.0
   for both models. The open problem remains **out-of-domain normal recognition**, which
   needs *normal* data — the 24 UE003 normal frames were deliberately left eval-only.
3. The failing frames both models know best; failure detection is not the binding constraint
   for this footage.

## Limitations

- 44 failing frames (2 sources) and 24 normal frames (1 source): tiny, single-domain-shot
  signals; external numbers are descriptive, never a benchmark.
- `UE003__frm0000` accuracy is a single frame; the per-unit FPR table must be read as
  per-source behavior, not a statistic.
- No verified under_extrusion external frames exist; failure-type coverage is stringing +
  spaghetti only.
- 2A evaluated on NORMAL001/SP001 is a **training-fit** (memorization) result, labeled as
  such, not a generalization claim.
- 977 undecided units and 15 uncertain units remain unreviewed; adding human-verified normal
  frames and more failing sources is the natural next step, not more tuning of this
  experiment.

## Reproducibility

- Config, module, tests: `training/configs/exp2_external_video_aug.yaml`,
  `training/tv/exp2_external.py`, `training/tests/test_exp2_external.py`.
- Derived split builder: `tv exp2-prep --printing-errors-dir … --out-dir …`.
- Evaluation: `tv eval-deep --config … --split test` (frozen test, stored JSON in
  `deepeval/test.json`) and `tv exp2-eval --checkpoint … --subset normal|failing --kind
  generalization|training_fit` (external JSON kept in the experiment `artifacts/` dir,
  gitignored).
- Source files unchanged: `verified_labels.json`, `vlm_review_proposals.json`, frozen splits;
  `review_manifest.json` carries only the pre-existing Stage 12F UE003 decisions.