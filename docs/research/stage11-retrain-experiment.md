# Stage 11 — Retraining Experiment Design

**Design only.** No training is executed in this stage.
**Objective:** Test whether adding targeted independent sessions improves generalization, measured on the frozen Experiment 1 test set and a new held-out evaluation set.

## 1. Baseline (Experiment 1)

- **Model:** `shufflenet_v2_x1_0`
- **Checkpoint:** `experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/best.pt`

| Metric | Value |
|---|---|
| F1_at_0_50 | 0.8763 |
| F1_at_0_65 | 0.8799 |
| FNR_at_0_65 | 0.1787 |
| FPR_at_0_65 | 0.1025 |
| stringing_FNR_at_0_65 | 0.213 |
| under_extrusion_FNR_at_0_65 | 0.155 |
| normal_FPR_at_0_65 | 0.103 |

**Training pool:** 105 recordings, 11254 frames
- Stringing: 23 recordings
- Under-extrusion: 37 recordings
- Spaghetti: 5 recordings
- Normal: 40 recordings

## 2. Stage 10 evidence driving this design

- **H1_stringing_clustered:** 20122022 date-cluster has fnr_mean=0.20 vs near-zero elsewhere. Stringing performance is not uniform across sessions.
- **H2_catastrophic_test_ood:** Test recording 15.02.2023_13_19_52 (fnr=0.898) is from a distinct session than any training stringing recording on that date.
- **H3_dark_not_fp_driver:** Contradicted. Dark bucket has lowest FPR (0.003). Brightness-FPR correlation r=0.22 (weak positive).
- **H4_no_darkness_fnr_gradient:** Contradicted. Medium bucket FPR 0.055, bright 0.033, dark 0.003.
- **H5_under_extrusion_control_ok:** Partially supported. 49 recordings/15 dates/fnr_mean=0.05, but two catastrophic sub-sessions exist.

## 3. Data requirements

**Principle:** INDEPENDENT SESSIONS, not raw frame count. Each session = a distinct print job, ideally a different date. Session atomicity preserved: never split frames of one session across sets.

### Priority 1: stringing
**Rationale:** Primary failure mode. Clustered by date/session. Stage 8 stringing FNR=0.213 on frozen test.

| Target | Sessions | Frames/session | Total frames |
|---|---|---|---|
| Minimum | 5-10 | 50-200 | 250-2000 |
| Strong | 15-20 | 80-200 | 1200-4000 |

**Diversity requirements:**
- Distinct print objects (if possible)
- Different lighting conditions
- Different print stages where stringing occurs
- Mix of severe and mild stringing
- Dates not already in training set

**Held-out:** Hold out 1-2 sessions from a distinct date range or object as the new held-out evaluation set.

### Priority 2: normal
**Rationale:** Dark-normal FPR is currently low (0.003) but coverage is narrow. The known FP test recording (brightness ~76) may represent an uncovered condition.

| Target | Sessions | Frames/session | Total frames |
|---|---|---|---|
| Minimum | 5-10 | 50-150 | 250-1500 |
| Strong | 10-15 | 80-150 | 800-2250 |

**Diversity requirements:**
- Varied ambient lighting (bright, dim, mixed)
- Different objects in frame
- Different camera orientations

### Priority 3: spaghetti
**Rationale:** Only 5 training recordings (191 frames), 0 test frames. Cannot measure frozen-test spaghetti performance.

| Target | Sessions | Frames/session | Total frames |
|---|---|---|---|
| Minimum | 5-8 | 50-150 | 250-1200 |
| Strong | 10-15 | 80-150 | 800-2250 |

### Priority 4: under_extrusion
**Rationale:** Already well-represented (37 train recordings). Maintain or modestly extend.

| Target | Sessions | Frames/session | Total frames |
|---|---|---|---|
| Minimum | 3-5 | 50-150 | 150-750 |

**Labeling:** Same binary normal/failing + fine failure type as existing. Metadata: date, object, lighting condition, camera settings.

## 4. Split strategy

- **Unit:** session (recording), never frame
- **New data ratio:** 60% train / 20% val / 20% held-out
- **Held-out set:** Sessions from new data held out at acquisition time. This becomes the second evaluation set to measure improvement.
- **Frozen test set:** The existing Experiment 1 test set remains FROZEN. Used only for final comparison.
- **Existing data:** All existing train/val recordings remain in their current splits. New data is ADDED to the existing train/val pools.
- **Directory:** New sessions follow the existing layout:
  datasets/processed/printing-errors/{failing,normal}/Recording_<date>_<time>/frame.png
- **CSV update:** New rows appended to train.csv and val.csv. splits.json updated with new counts. metadata.csv updated with new fine-class mappings.

## 5. Training protocol

- Architecture: `shufflenet_v2_x1_0 (same as Experiment 1)`
- Image size: 224
- Optimizer: adam (lr=0.001, wd=0.0001)
- Epochs: 30 (early stopping patience=10)
- Batch size: 32
- Precision: fp32
- Seed: 42
- Augmentation: {"horizontal_flip": true, "random_rotation": 15, "color_jitter": 0.2, "random_crop": false, "normalize": true}
- Hardware: MPS (Apple M1 Pro) or CPU

**Expected training runtimes (measured/estimated):**

| Train frames | Estimate | Source |
|---|---|---|
| 11,254 | ~16 min | Measured (Experiment 1) |
| 15,000 | ~22 min | Estimated (linear) |
| 20,000 | ~29 min | Estimated |
| 25,000 | ~36 min | Estimated |

## 6. Evaluation protocol

### Frozen Test Eval
- Run Stage 8 evaluation pipeline on the frozen Experiment 1 test set with the new checkpoint.
- Metrics: F1 at 0.50, F1 at 0.65, FNR at 0.65 (overall), FPR at 0.65 (overall), Stringing FNR at 0.65, Under-extrusion FNR at 0.65, Normal FPR at 0.65, Calibration (Brier, ECE)
- Runtime: ~30s on MPS

### Held Out Eval
- Evaluate on the new held-out evaluation set (sessions held out from new data acquisitions).
- Metrics: Same as frozen_test_eval
- Runtime: ~15s on MPS (smaller set)

### Train Cache Rebuild
- Rebuild frozen_train_trajectories.json with the new checkpoint for Stage 10-style analysis.
- Runtime: ~40s on MPS

### Stage10 Rerun
- Re-run Stage 10 cluster/domain audit with new train cache to verify stringing cluster FNR improved.
- Runtime: ~47s on MPS

### Threshold Sweep
- Run Stage 8 threshold sweep on val split to confirm 0.65 remains optimal or select new threshold.
- Runtime: ~10s on MPS

## 7. Runtime estimates

> All estimates based on measured runtimes on Apple M1 Pro (MPS). CPU will be ~2-3x slower. Scaling is roughly linear with frame count.

**Measured baselines:**
- experiment_1_training: 968s (11254 train frames, 30 epochs, early stop at ~20)
- inference_speed: ~282 fps on MPS
- train_cache_rebuild: ~40s (11254 frames)
- stage10_analysis: ~47s (image I/O on ~50 normal recordings)
- stage8_evaluation: ~30s (val+test, 5033 frames)

**Projected totals by acquisition scenario:**

| Scenario | New frames | Total train | Training | Eval | Pipeline |
|---|---|---|---|---|---|
| Minimum Acquisition | 500-2000 | 11754-13254 | 17-20 minutes | 2-3 minutes | 20-25 minutes |
| Strong Acquisition | 3000-8000 | 14254-19254 | 20-28 minutes | 2-3 minutes | 23-33 minutes |
| Ideal Acquisition | 6000-15000 | 17254-26254 | 25-38 minutes | 3-4 minutes | 29-45 minutes |

**Data preparation:** 1-5 hours (depending on data source)
**Pipeline after prep:** 20-45 minutes

## 8. Success criteria

### Minimum
- Stringing FNR on frozen test improves by >=5pp (from 0.213 to <=0.163), without F1 regression.
- stringing_FNR_target: <=0.163
- F1_regression_allowed: False

### Meaningful
- Stringing FNR improves by >=10pp AND overall F1 improves by >=1pp on frozen test.
- stringing_FNR_target: <=0.113
- F1_improvement_target: >=0.8899

### Strong
- Stringing FNR on frozen test approaches under-extrusion FNR (currently 0.155), and held-out set shows comparable performance.
- stringing_FNR_target: <=0.170
- held_out_comparable: True

### Abort criteria
- If F1 on frozen test regresses by >2pp, or if any failure type FNR increases by >5pp, the experiment is a regression.
- F1 regression threshold: <0.8599
- FNR regression threshold: >0.2287 (any type)

## 9. Decision tree

### After minimum acquisition
- **minimum_met:** Proceed to strong acquisition
- **minimum_not_met:** Analyze which failure type regressed. Consider: (a) more data of the failing type, (b) different augmentation, (c) learning rate adjustment

### After strong acquisition
- **meaningful_met:** Proceed to ideal acquisition or deployment planning
- **meaningful_not_met:** Evaluate whether the bottleneck is data diversity or model capacity. Consider: (a) architecture search, (b) different pretrained backbone, (c) more targeted data

### After ideal acquisition
- **strong_met:** Ready for deployment evaluation (Stage 12)
- **strong_not_met:** Model capacity may be the limiting factor. Consider: (a) larger ShuffleNet, (b) MobileNetV3, (c) EfficientNet-B0

## 10. Gating checklist

> All items must be checked before executing the experiment.

| ID | Check | Required |
|---|---|---|
| G1 | New data collected and labeled | 5-10 independent stringing sessions minimum |
| G2 | Data organized in existing directory layout | datasets/processed/printing-errors/{failing,normal}/Recording_<date>_<time>/ |
| G3 | metadata.csv updated with new fine-class mappings | All new frames have fine class (normal/under_extrusion/stringing/spaghetti) |
| G4 | New splits created (train.csv, val.csv updated) | Session-level split, no frame leakage |
| G5 | Held-out sessions identified and reserved | 1-2 sessions from distinct date range |
| G6 | Frozen test set untouched | test.csv and test frames unchanged |
| G7 | Config YAML created from template | experiment_id assigned, all hyperparameters set |
| G8 | Training pipeline tested on small subset | Verify no crashes, correct class mapping |

## 11. Limitations

- Runtime estimates assume MPS; CPU will be ~2-3x slower.
- Data preparation time is not included in pipeline estimates.
- Success criteria assume the bottleneck is data diversity, not model capacity.
- The frozen test set is a single snapshot; held-out set provides a second signal.
- No claims of real-world generalization are made.
- Session atomicity means some useful within-session frames may be lost at split boundaries.

---

## Artifacts

| File | Purpose |
|---|---|
| `docs/research/stage11-retrain-experiment.json` | Machine-readable design |
| `docs/research/stage11-retrain-experiment.md` | This report |
| `scripts/analyze_stage11.py` | Design generator |