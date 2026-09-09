# Release Candidate 1 — Dataset, Training, and Evaluation

Status: evaluated; first release candidate on the provenance-typed dataset
Created: 2026-09-09
Related: [release-dataset-build.md](release-dataset-build.md),
[tinyvision-data-strategy.md](tinyvision-data-strategy.md),
[dataset-staging-pipeline.md](dataset-staging-pipeline.md),
[stage14-experiment.md](stage14-experiment.md)

## 1. Decision context

Per the release-candidate directive, training proceeded **on the existing
staged data** with **Trier source annotations**, not VLM proposals and not
manufactured human labels. The `new_failure_envs_human_verified_min=5`
readiness gate (data-strategy §9.7, G-4) is a release-class acquisition
heuristic; it does not block training, but no release-readiness claim is made
here and human verification of Trier failure environments remains pending.

No threshold was tuned. No model, dataset-design, or data-distribution
redesign. Frozen val/test splits are byte-identical copies of the frozen
printing-errors splits.

## 2. Release training dataset (`datasets/processed/release-rc1`)

Built by `scripts/build_release_training_split.py`; ledger:
`datasets/processed/release-rc1/derived_manifest.json` (sha256 of the Trier
env manifest + per-env sample files + frozen splits).

Composition of `train.csv` (28,383 frames, 28,383 unique, 0 missing):

| provenance | kind | label set | rows |
|------------|------|-----------|------|
| `trier(source_annotation)` | failing | source labels (CC BY 4.0) | 8,792 |
| `trier(source_annotation)` | normal | source labels (CC BY 4.0) | 8,293 |
| `printing-errors(frozen)` | failing | frozen source labels | 5,863 |
| `printing-errors(frozen)` | normal | frozen source labels | 5,391 |
| `video-frames NORMAL001/SP001` | failing | human_reviewed (verified_labels.json) | 44 |

Environment counts:

* Train environments: **27** total (11 holdings: 8 normal / 3 failing; 16
  staged: Trier 15 — 9 failing [3 extrusion_error, 3 layer_shift, 3 stringing]
  + 6 normal — and Bambu 1 normal).
* **0** staged/holdout environments are human-verified; all Trier labels are
  `source_annotation` (folder-level, FFF-QC v2 authors), provenance preserved.
* Holdouts: new normal 3, new failure 6 (Trier `test_data` root) + frozen
  Kaggle val/test untouched.

Excluded from training (label policy): all 2,931 VLM proposals, Stage-13
VLM-screened normals (333), Stage-14 VLM-screened failures (142), UE003
eval-only frames. No VLM-only label is in this training set.

Splits: `train.csv` 28,383 / `val.csv` 2,174 / `test.csv` 2,859 —
`val.csv` and `test.csv` byte-identical to frozen `printing-errors`.

Leakage checks (stateless, reproducible):
* Trier holdout leak scan: 0 train rows under `test_data`; env manifest
  `external_holdout` environments absent from train.
* Stage13/14/screening path scan: 0 rows.
* Byte-identity of val/test: verified.
* Trier env manifest + Bambu staging validation: `env-validate --dir
  datasets/staging/trier` → 0 errors, 0 warnings.

## 3. Training configuration

Config: `training/configs/rc1_release_candidate.yaml`
(binary-identical hyperparameters to the frozen baseline / Exp3 / Exp4).

| field | value |
|-------|-------|
| model | shufflenet_v2_x1_0, classification, image_size 224 |
| data | batch 32, num_workers 4, pin_memory (MPS ignores) |
| optimizer | adam 1e-3, weight_decay 1e-4 |
| augment | hflip, rot ±15°, color jitter 0.2, normalize |
| epochs | 30 (early stop, patience 10) |
| precision / seed | fp32 / 42 |
| decision threshold | **0.5 (fixed; no tuning)** |
| class idx | failing=0, normal=1 |

Training run: completed `status=completed`, elapsed 2113.8 s (~35 min, MPS),
best val loss 0.39816 at epoch 2, early stop at epoch 12. Runs on this machine
historically took ~2135 s, so rc1 cost is normal.

Checkpoint: `experiments/rc1_release_candidate_shufflenet_v2_x1_0/artifacts/best.pt`

Export (`tv export`):

| artifact | size | avg CPU latency |
|----------|------|-----------------|
| `model.onnx` (fp32) | 4.93 MB | 1.95 ms |
| `model_int8.onnx` | 1.42 MB | 4.94 ms |

## 4. Evaluation results

All surfaces use threshold 0.5, image_size 224, failing-as-positive primacy;
no tuning anywhere. Command-identically reproduced exports under
`docs/research/` : `rc1-comparison-frozen.json`, `rc1-core-eval.json`,
`rc1-trier-holdout.json`, `rc1-skz-eval.json`, `rc1-skz-all.json`;
real-world reports under `datasets/external/real-world-eval/rc1_runs/*.json`.

### 4.1 Frozen original test (2,859 frames; failing-positive)

| candidate | P | R | F1 | ACC | FPR | FNR | AUROC |
|-----------|------|------|------|------|------|------|-------|
| baseline | 0.918 | 0.839 | 0.876 | 0.836 | 0.170 | 0.161 | 0.9004 |
| Stage 13 (exp3) | 0.927 | 0.751 | 0.829 | — | 0.134 | 0.249 | 0.8774 |
| Stage 14 (exp4) | 0.974 | 0.820 | 0.890 | 0.860 | 0.050 | 0.180 | 0.9246 |
| **rc1** | **0.953** | **0.812** | **0.877** | **0.842** | **0.091** | **0.188** | **0.8922** |

rc1 sits between Stage 14 (highest precision/AUROC) and baseline (highest
recall): strictly better FPR than baseline (−0.079) and Stage 13 (−0.043) at
recall within 0.027 of baseline.

### 4.2 Held-out target-domain normal sources (120 frames, LY3jlcO34f4 + zkJgLanCSUw)

| candidate | FPR |
|-----------|------|
| baseline | 1.000 |
| Stage 13 | 0.258 |
| Stage 14 | 1.000 |
| **rc1** | **1.000** |

rc1, like baseline and Stage 14, calls every genuinely-normal external frame
failing. Stage 13 is the only model with any normal restraint, at heavy recall
cost. This is the dominant unsolved failure mode.

### 4.3 Trier failure / normal holdouts (NEW surface; 9 environments, 10,069 frames)

Source annotations; environment-level. Aggregate failing-positive:
precision 0.731, recall 0.687, F1 0.708, FPR 0.538, FNR 0.313.

By environment (recall on all-failing envs; FPR on all-normal envs):

| environment (test_data) | kind | frames | recall | FPR | note |
|--------------------------|------|--------|------|------|------|
| cam1 extrusion_error | failure | 1,186 | 0.975 | — | near-perfect |
| cam1 layer_shift | failure | 1,117 | 0.472 | — | weak |
| cam1 no_error | normal | 1,069 | — | 0.000 | excellent |
| cam2 extrusion_error | failure | 1,156 | 0.988 | — | near-perfect |
| cam2 layer_shift | failure | 1,116 | 0.263 | — | weak |
| cam2 no_error | normal | 1,075 | — | 1.000 | fails |
| cam3 extrusion_error | failure | 1,160 | 1.000 | 0.758 | onset not learned |
| cam3 layer_shift | failure | 1,117 | 0.000 | — | missed |
| cam3 no_error | normal | 1,073 | — | 0.862 | fails |

Implication: predictions are strongly environment-correlated (whole envs flip
together). The model keys substantially on camera/lighting/domain features, not
solely on print morphology; extrusion_error generalizes but layer_shift does
not, and no camera-3 environment is classified sensibly. Frames before defect
onset are labeled failing in the source (folder-level annotation), which
depresses precision on pre-onset frames (visible in cam3 extrusion FPR).

### 4.4 SKZ Nozzle-Cam probe (secondary transfer; failing-positive)

| candidate | subset | FPR | recall | AUROC |
|-----------|--------|------|--------|-------|
| baseline | skz_eval (31) | 0.909 | 1.000 | 0.7227 |
| Stage 13 | skz_eval (31) | 0.273 | 0.100 | 0.3318 |
| Stage 14 | skz_eval (31) | 0.364 | 0.400 | 0.5182 |
| **rc1** | **skz_eval (31)** | **0.455** | **0.500** | **0.6409** |
| rc1 | all (166) | 0.540 | 0.638 | 0.5821 |

rc1 is the best transfer candidate after baseline on separation, and per-type
recall is: stringing 1.000, spaghetti 0.500, unterextrusion 0.273.

### 4.5 Real-world YouTube footage (independent; C++ `tinyvision`, temporal FSM)

Per-video: `first_failing_event_frame/failing_frames` and `last_state`
(baseline | exp3 | exp4 | rc1).

| video | intended | baseline | exp3 | exp4 | rc1 |
|-------|----------|----------|------|------|-----|
| 0TY6LVvMpUo | normal-ish (edited) | 8/476 FAIL | 15/146 PAUS | 9/475 FAIL | 10/474 FAIL |
| UzULqvGoi1Y | normal (real-time) | 11/1042 FAIL | 67/34 PAUS | 30/1023 FAIL | 8/1045 FAIL |
| nbkf372s_cA | normal (real-time) | 8/508 FAIL | 8/43 PAUS | 43/473 FAIL | 8/508 FAIL |
| 38gnsbGqcx4 | failure (blob) | 9/216 FAIL | 159/66 FAIL | 28/197 FAIL | 8/217 FAIL |
| G56LQfQCUnA | failure (extrusion) | 8/698 FAIL | 9/70 PAUS | 8/77 PAUS | 8/698 FAIL |
| 87gsL-Luvis | failure (spaghetti) | 8/25 FAIL | –/0 NORM | 9/24 FAIL | 8/25 FAIL |
| mGH3p3AF1YY | failure (spaghetti) | 8/8 FAIL | 9/7 FAIL | 9/7 FAIL | 9/7 FAIL |

rc1 fails fast on everything — near-instant alarms on all 3 normal videos and
full coverage on all 4 failure videos. It inherits baseline’s false-alarm
behavior; Stage 13 remains the only candidate that stays mostly PAUSED/NORMAL
on normal footage (but then misses 87gsL-Luvis and only flags G56LQfQCUnA
briefly).

## 5. Comparison against frozen baseline / Stage 13 / Stage 14 — verdict

Evidence: rc1 is the **best balanced candidate on the frozen test** (ties
baseline F1 0.877 → 0.877, cuts FPR from 0.170→0.091, FNR 0.188 vs Stage14
0.180; only Stage 14 has higher AUROC 0.925). On genuinely external normals,
however, rc1 behaves like baseline/Stage 14 (FPR 1.0) and Stage 13 remains the
only candidate with any restraint (FPR 0.258) — and it does so because it was
trained WITH the very human-reviewed frames that rc1’s no-VLM-only-label policy
banned.

**Release candidate NOT approved.** Decision basis (failing-positive primacy,
false negatives first):

1. **Normal-source false alarms are unsolved.** FPR 1.000 on the independent
   normal holdout and instant FAIL alarms (`ev=8`) on every real-world normal
   clip == baseline behavior. For a failure-detection tool, this means the
   device cries wolf constantly; user value ≈ 0 until fixed.
2. **Target-domain generalization is environment-correlated, not
   morphology-driven** (Trier holdout): camera-3 environments are
   catastrophically wrong (layer_shift recall 0.000, normal FPR 0.862), and
   whole-environment flipping indicates the model exploits lighting/camera
   identity more than defects.
3. The only candidate that fixed #1/#2 to any degree (Stage 13) did so only by
   violating the no-VLM-only-labels constraint we chose for rc1. rc1's honest
   label policy therefore did not buy transfer robustness; the data was the
   wrong lever.

## 6. What this means for the project

- Trier source annotations are confirmed usable for training, but the
  **annotated onset** (frame-level) still matters: folder-level labels
  penalize pre-onset frames.
- The blocker to a usable release is **normal-source false-alarm behavior**,
  not the number of failure environments. The honest next experiments are
  (a) a normal-dominated calibration/threshold operating-behavior study —
  still no intra-eval tuning; and (b) verifying whether the unresolved
  width/delta temporal FSM (`use_delta`, `fail_threshold_abs`) can gate
  instant 8-frame alarms. Both are runtime/FSM questions, not dataset
  questions; a decision on them is a data-leakage-free path forward.
- These were the only candidates trained under the current policy; there were
  no further stages, and no further candidates were invented.

## 7. Trier on-disk footprint and exact safe cleanup commands

Current footprint (nothing deleted):

```
56G  datasets/raw/trier-fffqc-v2            (total)
45G    ├── base_data.zip   (18G)  ── used: extracted/base (2.5G)
        ├── additional_defect_data.zip (15G) ── used: extracted/... 
        ├── test_data.zip  (12G)  ── used: extracted/test (4.9G)
10G    └── extracted/                      (index + frames)
673M datasets/staging/trier                (manifests, samples, review units,
                                            evidence sheets, human_candidates)
3.6M datasets/processed/release-rc1        (derived train/val/test + ledger)
```

The training requirement is only the extracted frames + index under
`datasets/raw/trier-fffqc-v2/extracted/`; the ZIP archives are redundant once
extraction integrity is verified. `extracted/test_data` serves the Trier
holdout evaluation and must be retained. Safe commands (idempotent, no data
loss of labels — sample files reference frame paths only):

```bash
# 1. Verify extracted-frame integrity before deleting archives:
git status --short datasets/raw/trier-fffqc-v2   # ensure no pinned pointers to zips

# 2. Delete the three duplicate archives ONLY (45 GB):
rm -f \
  datasets/raw/trier-fffqc-v2/base_data.zip \
  datasets/raw/trier-fffqc-v2/additional_defect_data.zip \
  datasets/raw/trier-fffqc-v2/test_data.zip

# 3. (Optional) drop review-unit render intermediates you no longer need:
#    673 MB staging includes render artifacts; keep candidate_definitions.json,
#    human_candidates.json, and manifests. Safe to remove only the rendered
#    contact/evidence sheets:
#   rm -rf datasets/staging/trier/review/units datasets/staging/trier/review/evidence
```

If you want to entirely remove the Trier raw source (after the Trier holdout
surface is retired) and reclaim the remaining ~10 GB:

```bash
# rm -rf datasets/raw/trier-fffqc-v2
#   → REQUIRES re-downloading from Zenodo 18628777 to rebuild the raw source.
```

Following step 2 the footprint drops from ~65 GB to ~20 GB with no capability
loss; every artifact produced for rc1 (splits, ledger, config, checkpoint,
ONNX, all eval JSON) resides outside `datasets/raw`.

## 8. Artifact index

| artifact | path |
|----------|------|
| dataset ledger | `datasets/processed/release-rc1/derived_manifest.json` |
| splits | `datasets/processed/release-rc1/splits/{train,val,test}.csv` |
| config | `training/configs/rc1_release_candidate.yaml` |
| checkpoint | `experiments/rc1_release_candidate_shufflenet_v2_x1_0/artifacts/best.pt` |
| ONNX | `.../artifacts/model.onnx`, `.../model_int8.onnx` |
| experiment status | `.../status.json` (completed, 2113.8 s) |
| frozen test comparison | `docs/research/rc1-comparison-frozen.json` |
| core eval (test + heldout normals) | `docs/research/rc1-core-eval.json` |
| Trier holdout | `docs/research/rc1-trier-holdout.json` |
| SKZ | `docs/research/rc1-skz-eval.json` (skz_eval), `rc1-skz-all.json` (all) |
| real-world | `datasets/external/real-world-eval/rc1_runs/*.json` (7) |
| split builder | `scripts/build_release_training_split.py` |
| eval scripts | `scripts/eval_rc1_core.py`, `eval_rc1_frozen_test.py`, `eval_frozen_candidates.py`, `eval_trier_holdout.py` |
| training log | `datasets/staging/rc1_train.log` |