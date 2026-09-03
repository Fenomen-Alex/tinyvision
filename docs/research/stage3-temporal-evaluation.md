# Stage 3 — Offline Temporal Decision-System Evaluation

**Status:** COMPLETE. The frozen ShuffleNetV2 frame classifier was run once over the
frozen **val** and **test** splits; frame-level failure probabilities were processed by
raw, EWMA-only, and EWMA+FSM decision logic and compared. No retraining, no modification
of the frozen test set, no real-camera data. All numbers below are measured, not
fabricated.

**Headline finding:** temporal smoothing **reduces sequence-level false alarms** (roughly
halves them vs. raw) with no loss of failure *detection*, but it does **not** reduce the
dominant frame-level false-negative problem, cannot recover the systematic catastrophic
miss, and — most importantly — the dominant bottleneck on this dataset is **frame-classifier
miscalibration** (the frozen model outputs high failure probability on source-labeled-`good`
prints), which temporal logic cannot fix. **Valid detection-delay cannot be measured** on
this synthetic dataset; a real, hand-annotated video experiment is required.

---

## A. Methodology

### A.1 What was evaluated

The production decision pipeline concept (see `temporal-decision-logic.md`):

```
frame -> frozen ShuffleNetV2 -> P(failing) -> [EWMA] -> [FSM] -> NORMAL/SUSPECT/FAILING/PAUSED
```

Three decision variants were compared on **the same frozen per-frame scores**:

| variant | logic |
|---|---|
| `raw` | per-frame threshold on P(failing) at 0.5 (no temporal logic) |
| `ewma` | EWMA of P(failing), α=0.15, threshold at 0.5 |
| `fsm` | EWMA (α=0.15) → absolute-score latch (0.65) + delta guard (0.30), confirm=3 consecutive, warmup=30, hysteresis (enter 0.30 / recover 0.01), fire-once |

The `fsm` parameters were **selected on the dev (val) split only**; they were then frozen
and applied unchanged to the held-out test split.

### A.2 Data / splits

- Frozen checkpoint: `experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/best.pt` (ShuffleNetV2 x1.0, unchanged).
- Frozen splits share **no recordings** (leakage-free by construction):
  - **dev = `val`**: 22 recordings, 2,174 frames — used only for parameter selection.
  - **holdout = `test`**: 24 recordings, 2,859 frames — used only for final evaluation.
- Frames within each recording were **re-ordered by capture timestamp** (the UTC-ms suffix
  in each filename) so per-recording trajectories are true temporal sequences.

### A.3 What ground truth exists (and what does not) — the pivotal determination

Per the Stage 3 constraint ("do not assume image-level labels define temporal
transitions"), ground truth was classified **per recording** as follows:

| reliability class | definition | test n |
|---|---|---|
| `failing` | folder label = failing (all concordant with a defect fine-class) | 9 |
| `normal_confirmed` | folder = normal **and** every frame fine-class = `good` | 6 |
| `normal_ambiguous` | folder = normal but the source dataset's own per-frame labels mark frames as a defect | 9 |

Decisions:

1. **Sequence truth** (failing = positive) is only trustworthy on `failing` recordings.
   All 9 test `failing` recordings are concordant (dominant fine-class = a defect), so
   **failure-detection rate is a legitimate metric**.
2. **Negative truth** (no-failure) is only trustworthy on `normal_confirmed` recordings.
   **Sequence false-alarm rate is reported only over these 6**, because 9 of the 15
   `normal`-labeled test recordings actually contain defect frames per the source dataset's
   own labels — a documented data-quality finding, not a temporal artifact.
3. **Event-onset ground truth does not exist in verified form.** The only frame-level
   "onset" signal is the source dataset's own per-frame `class`, which conflicts with the
   folder labels and is derived from synthetic `modified_layers` design metadata (not an
   independently verified failure-onset timestamp). It is therefore used **only as a proxy
   for relative-delay comparison**, never as trusted absolute delay.

---

## B. Available ground truth — determination result

- **Usable positive truth:** 9 failing recordings (folder) — reliable.
- **Usable negative truth:** 6 confirmed-normal recordings (folder + all-good fine class) — reliable.
- **Excluded from naive false-alarm claims:** 9 `normal_ambiguous` recordings. On these,
  the folder label and the source dataset's own frame labels disagree (they contain
  under-extrusion/stringing frames yet sit in `normal/`). This is almost certainly the root
  cause of the Stage 1.5 "catastrophic recordings" — the model flags these as failing not
  because it is wrong per the fine-class labels, but because the *folder* labels are wrong
  for them.
- **NOT available:** verified failure-onset timestamps. **Valid detection-delay (frames or
  seconds to trigger) cannot be measured** on this dataset. This is stated explicitly per
  the Stage 3 constraint.

---

## C. Results

### C.1 Consolidated comparison (reliable subsets)

| split | variant | det rate | missed seq | seq false-alarm rate | n seq-FA | mean FNR | mean FPR |
|---|---|---|---|---|---|---|---|
| dev (val) | raw | 1.000 | 0 | 0.500 | 2/4 | 0.013 | 0.007 |
| dev (val) | ewma | 1.000 | 0 | **0.000** | 0/4 | 0.065 | 0.000 |
| dev (val) | fsm | 1.000 | 0 | **0.000** | 0/4 | 0.435 | 0.000 |
| **holdout (test)** | raw | 1.000 | 0 | 0.833 | 5/6 | 0.108 | 0.173 |
| **holdout (test)** | ewma | 1.000 | 0 | **0.500** | 3/6 | 0.155 | 0.128 |
| **holdout (test)** | fsm | 1.000 | 0 | 0.500 | 3/6 | 0.397 | 0.249 |

- **Failure detection (sequence):** every variant detects all 9 failing recordings
  (det rate 1.0, zero missed events) on holdout. Detection is not the differentiator here.
- **Sequence false alarms:** EWMA smooths raw 0.833 → 0.5 on holdout (and 0.5 → 0.0 on
  dev). Confirmation/hysteresis in the FSM suppresses *isolated-frame* spikes (e.g.
  `20122022_17_05_36`, max P(failing)=0.997 but 0 FAILING frames, versus raw which would
  fire) — the one robust, cross-split benefit of temporal logic.
- **Frame-level FNR:** EWMA (0.155) and raw (0.108) are low *on average*, but the
  catastrophic recording `15.02.2023_13_19_52` has FNR 0.89 even when detected. **Temporal
  logic does not materially reduce frame-level FNR**, and the strict FSM *increases* it
  (0.397) because its state machine demands sustained confirmation and latches.

### C.2 Frame-classifier miscalibration is the dominant problem

The frozen model outputs **sustained high P(failing) on source-labeled-good recordings**:

| recording (holdout) | folder | fine | max P(failing) | FSM FPR |
|---|---|---|---|---|
| `05012023_13_10_42` | normal | all good | 0.959 | 0.529 |
| `07122022_20_17_29` | normal | all good | 0.797 | 0.706 |
| `06122022_19_29_09` | normal | all good | 0.942 | 0.258 |
| `01122022_16_06_44` | normal | most under-ext | 0.993 | fired |

These are **not temporal-noise false positives** — they are *sustained* over-confident
output on prints the source labels as entirely good. EWMA/hysteresis suppress *isolated*
spikes but cannot correct *sustained* over-confidence; that requires frame-model
recalibration/retraining or genuinely-normal real-camera data. This dominates the residual
false-alarm rate and cannot be fixed by the decision layer.

### C.3 Detection delay — not measurable

`n_delay_measurable = 0` on holdout: no failing recording supports a valid proxy-onset
delay (the proxy onset either conflicts with the folder label, or the recording is
constant-defect with no real normal→defect boundary in the decision-relevant window).
**No absolute detection-delay number is reported**, because to do so would be fabrication
on this dataset.

### C.4 Design finding: baseline-relative delta fails on this data

The decided `temporal-decision-logic.md` design used a **baseline-relative delta**
(`signal - rolling baseline`) to avoid slow-drift absorption. Empirically this **fails on
this dataset**: on dev, the delta-only FSM (α=0.15, confirm=3, warm=30) detected only
**2/7** failing recordings (`det=0.286`, mean FNR 0.853) because many defected prints are
constant-high from the start — the EWMA absorbs into its own baseline and delta → 0,
reproducing the exact "slow drift absorbed into baseline" pitfall the design doc warned
about, but harder: the failure is present *throughout*, not a drift. Adding an
**absolute-score latch** restored dev detection to 7/7. **Recommendation: keep a high
absolute-confidence latch (≈0.65) alongside any delta guard.**

---

## D. Chosen / default parameters

Selected on **dev only** (saturated operating region; not tuned on test):

| param | value | notes |
|---|---|---|
| α (EWMA) | 0.15 | Obico-aligned |
| warmup | 30 frames | suppress early alarms |
| confirm | 3 consecutive | suppress isolated spikes |
| fail_threshold_abs | 0.65 | high-confidence hard latch (essential) |
| fail_threshold (delta guard) | 0.30 | secondary |
| recover_threshold | 0.01 | hysteresis |
| fire-once | yes | single action per recording |

**Caveat:** dev selection was saturated (det=1.0, seq-FA=0.0 across a wide parameter range),
so these values are indicative, not knife-edge-tuned. They transferred consistently
(identical seq-FA on holdout = 0.5 for EWMA and FSM), but the frame-level FNR of the FSM
is worse than EWMA on holdout. **On this dataset a plain EWMA threshold (α=0.15, t=0.5)
gives the best accuracy/simplicity balance**; the FSM adds operational semantics
(warm-up, latch, single action) without a metric gain.

---

## E. Limitations (explicit)

1. **No verified temporal ground truth.** All delay numbers would be proxy-only; none are
   reported because none are measurable here.
2. **Folder-label noise.** 60% of `normal`-labeled test recordings contain defect frames by
   the source dataset's own labels. Any "false-alarm rate" over the whole `normal` set
   would be misleading; results are restricted to `normal_confirmed`.
3. **Synthetic nature of the defect onset.** The failure "onset" is a derived design-metadata
   boundary, not a real physical transition observed in video.
4. **Small reliable subsets.** Only 6 confirmed-normal and 9 failing recordings on holdout;
   dev was used for selection, leaving a small held-out sample. Sequence-level rates have
   wide uncertainty.
5. **Frame-model miscalibration dominates.** Residual frame-level FPR is a model-calibration
   issue that temporal logic cannot address.
6. **Frozen checkpoint / short recordings.** Recordings (46–190 frames) are short; warm-up
   and baseline effects can span much of a recording.

---

## F. Benefit justification

Temporal logic is **conditionally justified**:

- **Justified:** a plain EWMA threshold (or EWMA + confirmation/hysteresis) robustly cuts
  sequence-level false alarms vs. raw (raw 0.833 → 0.5 on holdout; 0.5 → 0.0 on dev) with
  no detection loss, and gives fire-once/warm-up/latching operational semantics at O(1) cost.
  This matches the deployed state of the art (Obico, Bambu).
- **Not justified by this data alone:** the full multi-state FSM does not beat EWMA on
  frame metrics here, so its extra complexity is **not** warranted by measured numbers on
  this synthetic dataset — only by the operational semantics it would provide on real
  prints (which this offline test cannot measure).
- **Cannot fix the real bottleneck:** the dominant miss (`15.02.2023_13_19_52`, FNR 0.89)
  and the sustained over-confidence on good prints are **frame-model** problems. The largest
  expected real-world gain is a better-calibrated frame model and genuinely-normal data,
  not the decision layer.

---

## G. Exact requirements for a real-video experiment (needed to validate delay + lever)

To make temporal claims (especially detection-delay and real false-alarm rate) valid, we
need real captures we do not currently own:

1. **Several full prints** captured at fixed FPS (1–10) with the target camera (ESP32-CAM
   and/or USB webcam), covering: normal prints, and failures starting partway through a
   print (clean normal → defect transition), not just constant-defect prints.
2. **Hand-annotated failure-onset frame** (or wall-clock time) per failing print —
   an *independent* ground-truth onset to measure delay against. The user must provide this;
   we cannot derive it from the synthetic dataset.
3. **Genuinely-normal prints** (verified by the user as defect-free end-to-end) to measure a
   true false-alarm rate under realistic lighting/motion.
4. **Multiple prints per condition** (≥5 failing, ≥5 normal) so sequence-level rates are
   statistically meaningful.
5. Record capture timestamp ordering + FPS so frames map to time; log any started/paused
   printer state to validate fire-once and recovery semantics.

Until such data exists: **do not report detection-delay in seconds as a project metric**
(it would be unmeasured), and treat sequence false-alarm rates as preliminary.

---

## Artifacts

- Structured results: `docs/research/stage3-comparison.json`, `docs/research/stage3-recording-detail.json`
- Frozen inference caches: `docs/research/evalcaches/frozen_val_trajectories.json`, `.../frozen_test_trajectories.json`
- Code: `training/tv/temporal.py` (EWMA + FSM, pure-DSL), `training/tv/temporaleval.py` (harness)
- Tests: `training/tests/test_temporal.py`, `training/tests/test_temporaleval.py`
