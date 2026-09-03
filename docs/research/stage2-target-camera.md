# Stage 2A — Target-Camera Domain Validation

**Status:** READY FOR MANUAL DATA COLLECTION. Implementation and tooling are
complete; no real target-camera dataset has been collected yet and **no
evaluation results are reported or fabricated in this document.**

The goal of Stage 2A is to assess how well the **frozen Experiment 1
ShuffleNetV2 baseline** generalizes to data captured with the actual target
camera/printer environment, before any retraining or temporal logic (EWMA/FSM)
is introduced.

**The Experiment 1 ShuffleNetV2 model and its test set are frozen and must not
be modified.** Stage 2A only *collects new data* and *evaluates* the frozen model
against it.

---

## 1. Data collection protocol

Collect real prints from the **target printer + target camera** at the exact
placement (distance, angle, resolution) intended for the deployed sensor. Keep
each capture session focused and procedurally recorded.

### Required: normal prints

A print with no defects, run to completion (or a representative sustained period)
from the fixed camera. This provides the "normal" negative class.

### Failing prints

A print that develops a visible defect. Record from before the defect appears
through the failure (or until stopped) so frames span the normal→failing
transition. Important: mix failure *types* and *severities* so we can measure
per-failure-type generalization honestly.

### Recommended session matrix

To make the evaluation meaningful rather than a single-condition snapshot,
collect sessions spanning at least:

| axis | recommendation |
|---|---|
| failure state | normal and failing (both required) |
| failure types | stringing, under-extrusion, spaghetti/warped, others as they occur |
| failure severity | mild, moderate, severe (at least a few of each) |
| print objects | at least 3 distinct object geometries/materials |
| lighting | at least 2 lighting conditions (e.g. room light, dimmer/shadowed) |
| camera position | at least 2 camera positions/angles if the mount permits |
| sessions per cell | ≥ 5 sessions per class as a minimum target; more is better |

A pragmatic floor for a first result: **~8–10 normal sessions and ~8–10 failing
sessions**, distributed across objects/lighting/failure types, each ~10–20
seconds. But because the model is frozen and we evaluate the whole collection as
a held-out test, even fewer sessions yield a valid (if noisy) first read — see
§6.

### Procedure per session

1. Position the printer/camera and confirm framing.
2. Run `tv capture --output-dir datasets/target-camera --class normal --object <id> --printer <id> [--notes ...]`
   (or `--class failing` with `--failure-type <type>`).
3. Confirm the on-screen class label before recording begins (the CLI prompts if
   you omit `--class`, and requires a valid choice).
4. Record. The session is written to
   `datasets/target-camera/<class>/session_XXX/frames/` with `metadata.json`.
5. Repeat, varying one axis at a time so attribution is clean.

---

## 2. Dataset structure

Session-oriented Layout A:

```
datasets/target-camera/
  normal/
    session_001/
      metadata.json
      frames/
        session_001_000000.jpg
        ...
    session_002/...
  failing/
    session_003/
      metadata.json
      frames/
  ...
```

After building the evaluation manifest, the pipeline also writes:

```
datasets/target-camera/splits/
  splits.json        # mode: holdout, data_dir, stats
  test.csv           # path,label for every session's frames
```

`datasets/target-camera/` is gitignored (it is raw collected data). Provenance
lives in `metadata.json` and reproducible evaluation outputs are stored under the
experiment directory (see §6).

---

## 3. Metadata schema

`metadata.json` is written at capture time. **Only supplied values are included —
no values are ever invented.** Optional fields (`printer`, `object`,
`failure_type`, `notes`) appear only when provided.

| key | type | always? | notes |
|---|---|---|---|
| `session_id` | str | yes | e.g. `session_001`, auto-sequenced per class |
| `class` | str | yes | `normal` or `failing` — enforced before recording |
| `captured_at` | str (ISO 8601 UTC) | yes | capture start timestamp |
| `camera` | dict | yes | `device_index`, `width`, `height`, `fps`, `backend` (from OpenCV) |
| `resolution` | str | yes | e.g. `640x480` |
| `frames` | int | yes | number of frames saved |
| `actual_fps` | float | yes | measured frames/second |
| `duration_seconds` | float | yes | measured duration |
| `printer` | str | only if provided | user-supplied printer id |
| `object` | str | only if provided | user-supplied print/object id |
| `failure_type` | str | only if provided | e.g. `stringing`, `under_extrusion` |
| `notes` | str | only if provided | free text |

---

## 4. Leakage prevention

- **Session-level isolation is structural.** `tv capture` writes every session
  under exactly one class directory with a single `session_XXX` id. Frames from a
  session can never be split across splits because they belong to one session
  directory.
- **`build_holdout` places every session in exactly one held-out `test` split.**
  Since there is exactly one split, no session can appear in two splits.
- **The target-camera collection is disjoint from Experiment 1 training data by
  construction** (different printer/camera/objects). The frozen model never saw
  any target-camera frame.
- `metadata.json` is never treated as an image (discovery only matches
  image suffixes).
- The splitter (`_discover_sources`) also ignores frame files only when they live
  under a session's `frames/` subdir or directly in the session dir; it does not
  merge sessions across classes.

---

## 5. Evaluation methodology

The model is **frozen** — nothing on the target-camera collection changes the
weights. Therefore a conventional `train/val/test` split is not appropriate: there
is no training set to leak into. The statistically sound first result is to treat
the **entire collected collection as one held-out test set**, which is exactly
what `tv holdout` does.

This is valid because every target-camera session is out-of-distribution relative
to the Experiment 1 training data, so each session is genuinely held-out data for
the question "does this frozen model generalize to my deployment conditions?"

Caveats, stated explicitly:

- With small `n`, per-session variance is high; a single session with unusual
  geometry/lighting can dominate aggregate metrics (Stage 1.5 showed exactly this
  for catastrophic recordings). Report per-session metrics alongside aggregates
  and inspect the worst sessions.
- If we later *tune* the decision threshold or temporal logic on this data, that
  data can no longer serve as an unbiased test. For tuning, reserve a
  leave-one-session-out or independent-collection protocol instead of reusing the
  same sessions.

### Metrics reported

`tv eval-deep` reports: accuracy, precision, recall, F1, AUROC (failing view),
confusion matrix, false-positive rate, false-negative rate, per-session metrics,
metrics by failure type (from `metadata.json` when present), confidence
distributions, and a threshold sweep. Compare these against the Experiment 1
test performance in `docs/research/experiment1-error-analysis.md`.

---

## 6. Exact commands

### 6.1 Collect a normal session

```bash
tv capture --output-dir datasets/target-camera \
    --class normal \
    --object cube-01 --printer prusa-mk3 \
    --duration 20 --fps 2 --camera 0
```

### 6.2 Collect a failing session

```bash
tv capture --output-dir datasets/target-camera \
    --class failing \
    --object cube-01 --printer prusa-mk3 \
    --failure-type stringing \
    --duration 20 --fps 2 --camera 0
```

If `--class` is omitted the CLI prompts and strictly accepts only
`normal`/`failing` (or `n`/`f`) before recording, making it hard to label a
session wrongly. Use `--duration 0` for "record until Ctrl+C".

### 6.3 Build the held-out evaluation manifest

```bash
tv holdout --data-dir datasets/target-camera
```

### 6.4 Evaluate the frozen ShuffleNetV2 baseline

```bash
tv eval-deep \
    --config training/configs/eval_target_camera_shufflenet_v2_x1_0.yaml \
    --experiments-dir experiments \
    --split test \
    --metadata-dir datasets/target-camera \
    --output-json experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/deepeval/target-camera.json
```

The config pins `experiment_id: shufflenet_v2_x1_0_printing-errors_cdc9e2cad4`
(= the frozen checkpoint) and sets `dataset.data_dir: datasets/target-camera`, so
`eval-deep` **loads the frozen `best.pt` and does not retrain**.

### 6.5 Optional: conventional stratified split (later tuning)

When there is enough data to hold out a tuning subset, use session-level splits:

```bash
tv split --data-dir datasets/target-camera --ratios 0.7,0.15,0.15 --seed 42
```

splits at the session level (no session crosses splits). Prefer `holdout` for the
first result.

---

## 7. Reproducibility

- Every config and seed is recorded; evaluation is deterministic given the frozen
  checkpoint and session metadata.
- The full machine-readable report is saved to
  `experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/deepeval/target-camera.json`.
- Captured session provenance is in `metadata.json`; it is raw data and gitignored,
  but the aggregated evaluation output is committed.
- AUROC uses `roc_auc_score(y_true, 1.0 - fail_scores)` for the failing view (see
  `experiment1-error-analysis.md`).

---

## 8. Unresolved limitations

- **No real data collected yet.** Nothing in this document is a measured result;
  the implementation has been smoke-tested on synthetic (blank-image) data only
  to validate the pipeline machinery.
- **`tv capture` cannot label class transitions mid-session.** A print that starts
  normal and later fails is labelled by its dominant/declared class. Capturing the
  full normal→failing transition therefore requires either (a) recording such a
  session and manually re-labeling frames, or (b) two sessions with a sync
  timestamp. This is a known gap and should be revisited for temporal (Stage 2B)
  work.
- **Small-`n` variance and single-condition bias** are expected on first
  collection; follow the session matrix in §1 and inspect per-session metrics.
- **Failure-severity and failure-type labels are only as reliable as the person
  capturing.** Recording them in `metadata.json` at capture time (not after) keeps
  them honest.
- **Threshold reuse**: after collecting and evaluating, do not tune the threshold
  on this same data and then re-report it as out-of-sample.
