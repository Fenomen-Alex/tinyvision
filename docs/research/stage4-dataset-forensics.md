# Stage 4 — Dataset Forensic Audit

**Date:** 2026-09-03
**Scope:** Determine whether the 3D-print-errors dataset's labels actually support
the task being evaluated, and explain the Stage 3 label disagreement between
`normal`/`failing` recording labels and per-frame defect classes.
**Constraint honored:** No retraining, no split modification, no new large
inference, no fabrication.

---

## A. Labeling semantics (how the data is actually labeled)

**The source dataset labels every frame independently.** The raw metadata file
`all_images_no_filter.csv` has one row per image with a per-frame `class`
column: `0`=GOOD, `1`=UNDEREXTRUSION, `2`=STRINGING, `4`=SPAGHETTI.

The prepared TinyVision dataset (`datasets/processed/printing-errors/`) maps each
frame's `class` to a binary folder label **per frame** via
`scripts/prepare_dataset.py`:

```python
CLASS_MAP = {0: "normal", 1: "failing", 2: "failing", 4: "failing"}
```

So the `normal/`/`failing/` folders do **not** label whole recordings. A single
recording can appear under **both** folders when its frames transition
good → defect during the print.

**There is no recording-level label in the source dataset.** The dataset record
confirms the `recording` column is a session ID captured via an OctoPrint macro,
with per-frame classification labels only. The "recording label" (good vs
defected print) used in earlier stages was a **TinyVision construction**, not a
source label. It was derived two different, fragile ways:

* `temporaleval.py` line 120: `rec_labels = {rec: frames[0]["label"] ...}` — the
  recording's label is just the **first frame's** binary label by timestamp.
* Recording-folder membership (a recording counted in `failing` if any frame is
  failing, and in `normal` if any frame is normal — double counting mixed
  recordings).

**Verification:** 41 of 151 recordings contain frames under both `normal/` and
`failing/`. The per-frame fine classes come directly from the raw CSV `class`
(see `robustness.load_resolved_metadata`), so the model's per-frame training
targets are genuine source labels.

### Implication for Stage 3's contradiction

Stage 3 reported "9 of 15 `normal`-labeled holdout recordings contain
source-labeled defect frames." This is **not noise or contradiction** — it is the
expected consequence of a good→defect transition inside those recordings. A
printed part that starts fine and later fails produces good frames *and* defect
frames; the prepared folders record each frame's own class. The apparent
"disagreement" exists only because TinyVision had (incorrectly) erected a
single-valued "recording label" on top of per-frame labels.

---

## B. Disagreement statistics (per split, not one number)

Recording composition taxonomy (by frame composition, not an invented label):

* `PURE_NORMAL` — every frame `good` (reliable negative set).
* `PURE_FAILING` — every frame a defect class (reliable positive set).
* `MIXED_NORMAL_LABEL` / `MIXED_FAILING_LABEL` — contains both good and defect
  frames (a genuine good→defect transition); majority label decides the suffix.
* `UNKNOWN_AMBIGUOUS` — unparseable / empty.

| Split | Records | Frames | Mixed records | Frames in mixed | PURE_NORMAL | PURE_FAILING | Mixed breakdown |
|-------|---------|--------|---------------|-----------------|-------------|--------------|-----------------|
| train | 105 | 11,254 | 21 (20%) | 2,537 (22.5%) | 40 | 44 | 13 MIXED_FAILING / 8 MIXED_NORMAL |
| val   | 22  | 2,174  | 11 (50%) | 1,148 (52.8%)  | 4  | 7  | 10 MIXED_FAILING / 1 MIXED_NORMAL |
| test  | 24  | 2,859  | 9  (37.5%) | 1,194 (41.8%) | 6  | 9  | 9 MIXED_FAILING |

Whole-dataset raw class counts: GOOD 7,026 (43.1%), UNDEREXTRUSION 5,512,
STRINGING 3,509, SPAGHETTI 240.

The "label disagreement" is therefore concentrated in mixed recordings, which are
a large minority in every split (20–50%). Interpreting a *mixed* recording as a
single "normal" or "failing" unit is not supported by the source.

---

## C. Catastrophic analytics (using cached predictions)

Cache: `docs/research/evalcaches/frozen_test_trajectories.json` (gitignored
artifact, not rerun). The model is the frozen Experiment-1 ShuffleNetV2
checkpoint.

The cache's recording `label` = first-frame label, which **mislabels every
good→defect transition recording as `normal`**. Re-deriving reliability from the
per-frame fine classes gives the correct picture for the 24 test recordings:

| Cache label | Forensic composition | Count | Meaning |
|-------------|----------------------|-------|---------|
| failing     | PURE_FAILING         | 9     | Start already-defect; labels concordant |
| normal      | PURE_NORMAL          | 6     | Genuinely all-good (real negative set) |
| normal      | MIXED (9 MIXED_FAILING) | 9  | Start good then go defect; **not** false alarms on the defect frames |

**Corrected interpretation of Stage 3's "over-confidence on source-labeled-good
prints":** this concern applies to only the **6 genuinely all-good** test
recordings, not the 9 mixed ones. The mixed recordings' late frames *are* real
defects that the model correctly flags. Per-recording alarm counts on the
all-good set:

* `Recording_05012023_13_10_42` — 34/68 frames alarmed (worst; avg score 0.52)
* `Recording_05012023_17_24_05` — 8/84
* `Recording_06122022_19_29_09` — 12/97
* `Recording_07122022_20_17_29` — 19/187
* `Recording_20122022_17_05_36` — 24/110
* `Recording_29112022_22_13_20` — 0/147 (perfect)

The genuine false-alarm problem is thus concentrated in a handful of all-good
recordings (notably `05012023_13_10_42`), far narrower than Stage 3's framing
implied. The Stage 3 temporal-false-alarm reductions (EWMA halving sequence false
alarms) still hold, but the *denominator* (all-good recordings) is smaller than
reported, so the magnitude of the false-alarm "problem" was overstated.

---

## D. Trustworthy metrics (re-interpretation of prior experiments)

* **Frame-level classification metrics (Exp 1 / 1.5 / 2B / 3):** **reliable.**
  Every frame has a genuine source `class` label. Per-frame accuracy/precision/
  recall/F1, FNR, FPR, AUROC are well-defined and meaningful. The frozen
  ShuffleNetV2 test numbers (acc 0.836, prec 0.918, rec 0.8385, F1 0.876,
  FNR 0.162, FPR 0.170, AUROC 0.900) stand.
* **"Recording classification" / recording-label metrics (Stage 3 reliability
  classes `normal_confirmed`, `normal_ambiguous`, `failing`):** **conditional /
  partly misleading.** The recording label was `frames[0]["label"]`, an arbitrary
  first-frame convention. `normal_ambiguous` = mixed transition recording, whose
  defect frames are *not* false alarms. These particular numeric comparisons
  (e.g. FPR on normal-labeled recordings) should be reinterpreted with the
  recovered composition.
* **Temporal event onset / detection-delay:** **not interpretable** — see E. The
  Stage 3 "detection delay not measurable (n=0)" result is consistent: there is
  no reliable onset ground truth.

No historical metric is rewritten; the interpretation of which metrics are
trustworthy vs. constructed is corrected.

---

## E. Does temporal event ground truth exist?

**No trustworthy recording-level event ground truth exists.**

1. The source provides **per-frame class only**. A recording is never labeled as
   "failing at time T."
2. The class-flip layer does **not** reliably map to a verified physical onset:
   * Some transition recordings have empty `modified_layers` (e.g.
     `Recording_01122022_16_06_44`, `ex_mul=1.05`) yet flip to a defect class,
     so the flip cannot be attributed to injection metadata.
   * Some recordings with populated `modified_layers`
     (e.g. `Recording_06012023_12_17_19`) flip at layers inconsistent with the
     listed injection points, and the model's own score is ~0 on that flip —
     suggesting the source label at that point is questionable or the defect is
     visually subtle.
   * The dataset record says under-extrusion was "**partially** injected
     synthetically," explicitly acknowledging mixed labeling provenance.
3. The class-flip could be either a physically-observed defect onset or a
   label-provenance artifact; the metadata cannot disambiguate it.

**Conclusion:** the good→defect frame transition is a plausible *proxy* for
onset, but it is **not verified temporal ground truth**. Reporting detection
delay or FPR-at-time-to-onset as measured quantities would be over-claiming.

---

## F. Strongest defensible ML task

**Per-frame binary classification (normal vs failing),** with recording-level
stratification *only* as an evaluation grouping — never as a ground-truth label.

* Each frame's label is a genuine source class; the task is well-defined.
* False-negative rate and false-positive rate are computable per frame and per
  recording, with the all-good (`PURE_NORMAL`) recordings as the reliable
  negative set and `PURE_FAILING` / defect-frames-in-mixed as the positive set.
* Temporal smoothing (EWMA) remains a legitimate *post-processing* layer on a
  per-frame classifier; it just must not be evaluated against a non-existent
  recording-level/event label.

Recommended reframings for **Stage 5**:
* Keep the per-frame binary classifier as the core model.
* Evaluate false alarms **only** on `PURE_NORMAL` recordings (6 test / 4 val),
  not on mixed recordings.
* Evaluate detection on `PURE_FAILING` + the defect frames of mixed recordings.
* If temporal onset is pursued, treat onset as a **research question** requiring
  human visual verification of the transition boundary, not as dataset ground
  truth.
* Reconsider the highly-correlated mixed transitions: frame-level FNR/FPR on
  mixed recordings is dominated by a good→defect order effect, so metrics should
  be reported separately for pure vs. mixed recordings.

---

## G. Representative set for human review (Task 9 — no visual hallucination)

The raw CSV `class` metadata establishes these labels; we do **not** assert a
frame visually shows a defect unless a human confirms. These are the frames that
should be spot-checked by eye to validate the labeling scheme:

**Transition-boundary candidates (first defect-labeled frame of each mixed test
recording):** under `datasets/processed/printing-errors/failing/<rec>/…`

| Recording | First defect frame | fine (source) | model score |
|-----------|--------------------|---------------|-------------|
| 01122022_16_06_44 | ELP_12MP_01.12.2022_166991166820.png | under_extrusion | 0.443 |
| 03012023_15_32_58 | ELP_12MP_03.01.2023_167276044824.png | under_extrusion | 0.973 |
| 03012023_15_48_43 | ELP_12MP_03.01.2023_167276685303.png | under_extrusion | 0.942 |
| 03012023_19_32_03 | ELP_12MP_03.01.2023_167277465302.png | under_extrusion | 0.743 |
| 06012023_12_17_19 | ELP_12MP_06.01.2023_167300854113.png | under_extrusion | **0.002** ⚠ |
| 06012023_12_43_32 | ELP_12MP_06.01.2023_167301046029.png | under_extrusion | 0.968 |
| 06122022_14_49_53 | ELP_12MP_06.12.2022_167033853829.png | stringing | 0.366 |
| 10012023_20_09_14 | ELP_12MP_10.01.2023_167338198812.png | under_extrusion | 0.999 |
| 10012023_22_39_52 | ELP_12MP_10.01.2023_167339110557.png | under_extrusion | 1.000 |

**All-good high-alarm frames (potential false alarms):** under
`datasets/processed/printing-errors/normal/<rec>/…`

| Recording | alarmed/total | top-score frame | model score |
|-----------|---------------|-----------------|-------------|
| 05012023_13_10_42 | 34/68 | ELP_12MP_05.01.2023_167292533577.png | 0.959 |
| 05012023_17_24_05 | 8/84 | ELP_12MP_05.01.2023_167294036199.png | 0.712 |
| 06122022_19_29_09 | 12/97 | ELP_12MP_06.12.2022_167035631347.png | 0.942 |
| 07122022_20_17_29 | 19/187 | ELP_12MP_07.12.2022_167044576707.png | 0.797 |
| 20122022_17_05_36 | 24/110 | ELP_12MP_20.12.2022_167155844883.png | 0.997 |

`Recording_06012023_12_17_19`'s first defect-label frame (score 0.002) is the
highest-priority review item: the model does not see a defect, so the source
label there is suspect (injection-metadata mismatch) or the defect is visually
subtle. **Human review of these 15 frames is required before trusting the
transition labels or the false-alarm characterization.**

---

## Recommendation for Stage 5

Adopt **per-frame binary classification (normal vs failing)** as the primary
task, and change *evaluation* to use the composition-recovered reliable sets:
false alarms measured only on `PURE_NORMAL` recordings; detection measured on
`PURE_FAILING` plus defect frames of mixed recordings. Relegate "recording/event
onset" to a human-reviewed research question rather than ground truth. Before
training anything new, the 15 frames in section G should be visually verified.

Evidence base: `docs/research/stage4-recording-audit.json` (151-recording audit),
`training/tv/forensics.py`, `training/tests/test_forensics.py` (4 tests, all
pass; full suite 87 pass).
