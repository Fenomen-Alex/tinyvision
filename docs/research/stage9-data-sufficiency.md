# Stage 9 — Data Sufficiency & Acquisition Design

**Analysis only.** No training, no data modification, no checkpoint changes.
This report answers: *What independent data is actually missing, why do we
believe it is missing, and how should we acquire it without contaminating
the evaluation?*

## 1. Dataset inventory summary

| Split | Recordings | Total frames | Normal frames | Failing frames |
|---|---|---|---|---|
| train | 105 | 11254 | 5391 | 5863 |
| val | 22 | 2174 | 757 | 1417 |
| test | 24 | 2859 | 878 | 1981 |

## 2. Frame-vs-recording concentration

### train
- recordings: 105, frames: 11254
- frames/recording: min=2, max=247, median=102, mean=107.2
- top 5 recordings: 9.3% of all frames
  - Recording_06122022_14_16_00: 247 frames (MIXED_NORMAL_LABEL)
  - Recording_13012023_14_38_10: 228 frames (PURE_NORMAL)
  - Recording_06122022_19_52_40: 211 frames (PURE_NORMAL)
  - Recording_14.02.2023_17_53_55: 181 frames (PURE_FAILING)
  - Recording_06122022_18_53_57: 178 frames (PURE_NORMAL)

### val
- recordings: 22, frames: 2174
- frames/recording: min=39, max=181, median=93.0, mean=98.8
- top 5 recordings: 37.2% of all frames
  - Recording_08012023_13_05_05: 181 frames (MIXED_FAILING_LABEL)
  - Recording_13122022_14_51_28: 167 frames (MIXED_NORMAL_LABEL)
  - Recording_05012023_11_08_00: 165 frames (MIXED_FAILING_LABEL)
  - Recording_01122022_17_44_58: 152 frames (PURE_NORMAL)
  - Recording_21122022_19_03_19: 143 frames (PURE_FAILING)

### test
- recordings: 24, frames: 2859
- frames/recording: min=46, max=190, median=118.5, mean=119.1
- top 5 recordings: 31.0% of all frames
  - Recording_17.02.2023_15_18_59: 190 frames (PURE_FAILING)
  - Recording_07122022_20_17_29: 187 frames (PURE_NORMAL)
  - Recording_06012023_12_17_19: 179 frames (MIXED_FAILING_LABEL)
  - Recording_10012023_20_09_14: 166 frames (MIXED_FAILING_LABEL)
  - Recording_06122022_14_49_53: 164 frames (MIXED_FAILING_LABEL)

## 3. Failure-type diversity

| Failure type | Train rec | Train frames | Val rec | Val frames | Test rec | Test frames |
|---|---|---|---|---|---|---|
| spaghetti | 5 | 191 | 3 | 49 | 0 | 0 |
| stringing | 23 | 2341 | 4 | 371 | 7 | 797 |
| under_extrusion | 37 | 3331 | 12 | 997 | 11 | 1184 |

## 4. Stringing investigation

- **Training recordings:** 23
- **Training frames:** 2341
- Frames/recording: min=2, max=181, median=102, mean=101.8
- Tiny sessions (<10 frames): 3 — `Recording_20122022_13_53_11, Recording_20122022_16_09_41, Recording_20122022_14_15_06`
- Small sessions (10-50 frames): 2
- Large sessions (>=50 frames): 18
- Date distribution: {'14.02.2023': 3, '21122022': 2, '22122022': 9, '20122022': 5, '23.02.2023': 3, '15.02.2023': 1}
- Composition breakdown: {'PURE_FAILING': 23}

**Stage 8 observation (evaluation context, not a training-selection signal):**
- Recording `Recording_15.02.2023_13_19_52` is a PURE_FAILING stringing recording with 127 frames.
- FNR at threshold 0.65: **0.898** (114/127 frames missed).
- The frozen model produces near-zero fail scores for the majority of its frames.

## 5. Normal-domain diversity

### Training normal distribution
- Pure normal recordings: 40
- Total normal frames: 5391
- Frames/recording: min=8, max=228, median=100.5
- Image brightness (sampled): mean=154.4, stdev=6.5
- Image contrast (sampled): mean=104.7

### Problematic test recording (Stage 8 observation)
- Recording: `Recording_20122022_17_05_36` (PURE_NORMAL, 110 frames)
- Image brightness: mean=76.4, stdev=3.9
- Image contrast: mean=79.8
- **Note:** brightness ~76 vs training mean ~156 suggests substantially different lighting. This is a proxy observation, not proof of semantic OOD.

## 6. Train/val/test distribution comparison

| Split | Recordings | Total | Normal% | under_ext rec | stringing rec | spaghetti rec |
|---|---|---|---|---|---|---|
| train | 105 | 11254 | 47.9% | 37 | 23 | 5 |
| val | 22 | 2174 | 34.8% | 12 | 4 | 3 |
| test | 24 | 2859 | 30.7% | 11 | 7 | 0 |

**Observations:**
- Under_extrusion is best represented across all splits (37/12/11 recordings).
- Stringing has reasonable recording count (23) but date concentration is high.
- Spaghetti is severely underrepresented in all splits (5/3/? training sessions).

## 7. Data quality / label structure

- Total concerns flagged: 8

  - [scarce_spaghetti] train/Recording_13122022_16_24_29: 
  - [tiny_recording] train/Recording_20122022_13_53_11: Very few frames; limited diversity from this session.
  - [tiny_recording] train/Recording_20122022_14_15_06: Very few frames; limited diversity from this session.
  - [tiny_recording] train/Recording_20122022_16_09_41: Very few frames; limited diversity from this session.
  - [scarce_spaghetti] train/Recording_29112022_18_50_45: 
  - [scarce_spaghetti] val/Recording_03012023_15_07_03: 
  - [scarce_spaghetti] val/Recording_13122022_14_51_28: 
  - [few_normal_frames_in_mixed] test/Recording_01122022_16_06_44: 

## 8. Recommended targeted data acquisition

### Stringing (Priority 1)
**Rationale:** Stage 8 found FNR=0.898 on one PURE_FAILING stringing test recording. The training set has 23 stringing recordings (2341 frames), but 4 have <10 frames and date diversity is concentrated (22122022, 14.02.2023).
- Minimum exploratory: 5-10 independent sessions / 50-200 frames
  - 5-10 new independent sessions would roughly double the session count. Each session should come from a distinct print job (different dates if possible) to test generalization beyond the existing date clusters.
- Stronger target: 15-20 independent sessions / 80-200 frames
- Ideal target: 25-30 independent sessions
- Diversity requirements:
  - Multiple distinct print objects (if possible)
  - Different lighting conditions (if possible)
  - Different print stages where stringing occurs
  - Mix of severe and mild stringing
- Held-out: Hold out 1-2 sessions from a distinct date range or object as a final held-out set, separate from the existing frozen test set.

### Normal (Priority 2)
**Rationale:** Recording_20122022_17_05_36 is a PURE_NORMAL test recording with brightness ~76 vs training normal ~156. This represents a potentially missing normal condition (darker lighting or different camera setting).
- Minimum exploratory: 5-10 independent sessions / 50-150 frames
- Stronger target: 10-15 independent sessions / 
- Diversity requirements:
  - Varied ambient lighting (bright, dim, mixed)
  - Different objects in frame (if applicable)
  - Different camera orientations/angles if eventually relevant

### Spaghetti (Priority 3)
**Rationale:** Spaghetti has only 5 training recordings (191 frames) and 4 validation recordings (49 frames). This is the scarcest failure type.
- Minimum exploratory: 5-8 independent sessions / 50-150 frames
- Stronger target: 10-15 independent sessions / 

### Under Extrusion (Priority 4)
**Rationale:** Under_extrusion has 37 training recordings (3331 frames) — the best-represented failure type. Additional data is lower priority but could improve generalization at the margins.
- Minimum exploratory: 3-5 independent sessions / 50-150 frames

### General principles
- Prioritize INDEPENDENT SESSIONS over raw frame count.
- Each new session should be from a distinct print job, ideally on a different date.
- Record metadata: date, object, lighting condition, camera settings if available.
- Label requirements: same binary normal/failing + fine failure type as existing.
- Split strategy: new sessions split 60/20/20 by session (not frame) into train/val/held-out.
- The existing frozen test set remains untouched for evaluation.
- Held-out sessions from new data provide a second evaluation set to measure improvement.

## 9. Proposed future retraining experiment

**Objective:** Test whether adding targeted independent data improves generalization to previously unseen recordings.

**Baseline:** ShuffleNetV2 x1.0, checkpoint `experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/best.pt`
- Test F1 at 0.50: 0.8763
- Test F1 at 0.65: 0.8799
- Test FNR at 0.65: 0.1787

**Old test set:** FROZEN. Do not touch. Used only for final comparison.
- 24 recordings, 2859 frames

**New data split:** session/recording (never frame)
- 60% train / 20% val / 20% held-out evaluation
- Sessions from new data held out at acquisition time; these become the second evaluation set to measure improvement.

**Primary metrics:**
- FNR on stringing failure type (test set, at threshold 0.65)
- FPR on normal frames (test set, at threshold 0.65)
- F1 overall (test set, at threshold 0.50 and 0.65)

**Success criteria:**
- Minimum: Stringing FNR improves by >=5pp on the frozen test set, without F1 regression on other failure types.
- Meaningful: Stringing FNR improves by >=10pp AND overall F1 improves by >=1pp on the frozen test set.
- Strong: Stringing FNR on the frozen test set approaches under_extrusion FNR (currently 0.155), and the new held-out set shows comparable performance.

## 10. Retraining gate

**A. Is the evidence sufficient to retrain immediately?** No

**B. What evidence/data is missing?**
1. The current training set has adequate under_extrusion coverage (37 sessions) but stringing is concentrated in date clusters (22122022, 14.02.2023) with some tiny sessions (2-24 frames). Adding 5-10 new independent stringing sessions before retraining would test whether the model can learn stringing from diverse sessions, not just from the current date clusters.
2. The normal-domain false-alarm problem (Recording_20122022_17_05_36, brightness ~76 vs training ~156) suggests the training normal set may not cover darker lighting conditions. 5-10 normal sessions under varied lighting would test this.
3. Spaghetti is severely underrepresented (5 training sessions, 191 frames). Adding 5-8 sessions would bring it closer to stringing's diversity.

**C. What data acquisition unblocks retraining?**
Minimum: 5-10 new independent stringing sessions (50-200 frames each) from distinct print jobs and dates. Optionally: 5-10 normal sessions under varied lighting, and 5-8 spaghetti sessions.

**D. What constitutes a successful retraining result?**
After retraining with the expanded dataset:
- Stringing FNR on the frozen test set improves by >=5pp (ideally >=10pp).
- Overall test F1 does not regress.
- The new held-out evaluation set shows comparable performance to the frozen test.
- Calibration does not materially degrade.
- No single recording dominates the error profile.

**Priority order:**
- 1. Independent session diversity (not raw frame count)
- 2. Failure-type diversity (stringing > spaghetti > under_extrusion)
- 3. Normal-domain diversity (varied lighting conditions)
- 4. Object/condition diversity (if metadata supports it)
- 5. Raw frame count (lowest priority — more of the same sessions adds little)

## 11. Limitations & methodological caveats

- All analyses are metadata/statistical only; no model inference was performed.
- Image statistics (brightness, contrast) are non-learned proxies, not proof of semantic OOD.
- Frame counts are not independent samples; recording identity is the diversity unit.
- Test-set observations are documented for context only; they do not drive data-selection decisions.
- No claims of real-world generalization are made.

## Artifacts

| File | Purpose |
|---|---|
| `docs/research/stage9-data-sufficiency.json` | Machine-readable results |
| `docs/research/stage9-data-sufficiency.md` | This report |
| `training/tv/stage9.py` | Analysis module |
| `training/tests/test_stage9.py` | Tests for the analysis logic |