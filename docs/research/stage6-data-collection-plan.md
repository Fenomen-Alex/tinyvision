# Stage 6 — Real-World Data Acquisition and Experiment Design

**Date:** 2026-09-03
**Status:** Design only. No hardware integration, no model changes, no retraining, no
long-running jobs. **No real-world data has been collected and none is fabricated here.**
**Constraint honored:** the frozen Experiment 1 ShuffleNetV2 checkpoint and its splits are
untouched; threshold 0.5 and validation-selected threshold 0.7 are treated as fixed decision
points for future evaluation, not tuned on any real-world data in this document.

This Stage builds directly on the Stage 2A target-camera protocol
(`docs/research/stage2-target-camera.md`) and the existing tooling
(`tv capture`, `tv holdout`, `tv eval-deep`, `tv deepeval`) against the frozen
`shufflenet_v2_x1_0_printing-errors_cdc9e2cad4` checkpoint. Stage 6 is the **experiment-design
and sample-size justification** layer on top of that machinery, and it explicitly accommodates
the fact that **no 3D printer is available today** — only an ESP32-CAM and a USB Logitech webcam.

The next bottleneck being addressed is **domain validity**, not another architecture. Stage 5
established that the Kaggle/source evaluation is well understood; the open question is whether
the frozen classifier behavior transfers to our own deployment environment, and how large a
gap we must bridge before retraining is justified.

---

## Part A — Define the target domain

### A.1 Intended deployment scenario

The MVP is a **fixed, always-on visual failure monitor for a single home/user FDM (FFF) 3D
printer**. The camera is mounted once, pointed at the print volume, and runs unattended for the
duration of a print. Its only real-time output is a binary decision — `normal` or `failing` —
generally consumed by a pause/notify automation. Everything downstream of the per-frame
decision (EWMA smoothing, state machine, pause hook) is deterministic post-processing and is
out of scope of what this Stage tries to *validate* (though it will be *measured*).

### A.2 Target-domain dimensions

| Dimension | Known now | Unknown until a setup exists | Cannot be determined without a printer + fixed setup |
|---|---|---|---|
| Deployment scenario | Fixed near-print monitor; binary normal/failing decision; unattended during print | — | — |
| Camera model | ESP32-CAM (OV2640) and Logitech USB webcam (model TBD; the Kaggle source was an ELP-USB13MAFKV76, and the Caxton reference used a Logitech C270 — ours is a different, actual unit) | Exact sensor/AOV/autofocus behavior of our units | — |
| Camera placement | Nozzle-adjacent view is the proven domain (256×256 crop centered on nozzle in Kaggle); external bed-wide view is the Obico norm. Ours is undecided and must be fixed before data collection | Distance, angle, focal length, AOV for our mount | Meaningful elevation/angle/zoom until we physically mount a camera |
| Resolution | Un-derived inputs from 256×256 PNG (Kaggle); training crop 320×320 → 224×224 (Stage 2 / dataset deep-dive) | Actual capture res of our sensors (640×480 on ESP32-CAM and webcam, configurable) | — |
| Lighting | Significant lighting/shadow sensitivity is expected; Kaggle had controlled room light; Caxton explicitly dropped dark images (mean RGB < 10) | Our ambient lighting, shadows from the moving gantry, window daylight drift | Repeatable lighting fixtures/curtains until the rig exists |
| Object/print variation | PLA on Ender-3 class printer, gray filament, black bed dominates the source domain; 34 geometries in Kaggle | Our filament colors/materials, bed color, printer brand, object geometries | — |
| Frame rate | Inference is fast (~13–150ms range across candidates on Pi 5 CPU); decision cadence is the bottleneck choice | Actual sustained end-to-end FPS on eventual hardware+camera | — |
| Inference rate | Model is fast enough for real-time on Pi 5 CPU (see `task-formulation.md`, `evaluation-plan.md`) | — | — |
| Decision frequency | Frames ≈ decisions per frame in the klatest design; temporal smoothing only changes *alarm semantics* | Whether we sample at capture FPS or decimate | Real CPU budget, camera FPS, end-to-end latency |

**Known-now summary:** We know the claim we are testing (does ShuffleNetV2 transfer?), the
frozen checkpoint and decision points (0.5 and 0.7), and the evaluation machinery. We do **not**
know our physical rig geometry, our sensor behavior, our lighting, or the real end-to-end cost.

**Explicit non-assumption:** We assume **nothing** about a printer being present. Camera
benchmarking (Part E, Part H) is a printer-independent activity we can start immediately.

### A.3 What is genuinely undetermined (do not guess)

- **Camera mounting geometry.** No elevation, distance, angle, zoom, or field-of-view number
  is asserted. It must be fixed empirically once a mount exists.
- **Expected frame rate.** No target FPS is assumed; it is an input we will measure
  (camera FPS, capture FPS, inference FPS are separate and each measured).
- **Expected object motion / gantry occlusions.** No claim about how the print head moves
  across the frame; this drives whether the source's nozzle-crop view is representative.
- **Failure prevalence in normal use.** We will intentionally control class mix in our
  dataset (Part B); we will not predict real failure rates in the field.

---

## Part B — Design the minimum dataset

### B.1 Design principles

1. **Independence is the unit of validity.** Every frame within a print is temporally
   correlated (Stage 4 showed consecutive frames are near-duplicates; the splitter already
   splits at the `recording`/session level). The statistically meaningful unit is therefore
   the **independent print session**, not the frame or the recording-length label. Number of
   sessions is the real sample size; frames mostly add precision about within-session behavior.
2. **Session-level isolation is already enforced structurally** by `capture_session` (each
   session in one class dir with one session id; `build_holdout` keeps every session in one
   split). Stage 6 adds the *counts and matrix* that make the isolation meaningful.
3. **Vary one axis at a time** so that a per-session or per-axis attribution is clean. Each
   failure type, lighting condition, object, and camera is a distinct axis we deliberately
   cross.
4. **Prefer few long, well-documented sessions over many duplicate-frames sessions.** The
   goal is density of distinct conditions, not raw frame volume.

### B.2 Collection protocol (for when a printer exists)

Each captured session runs `tv capture` (or `capture_session`) into `datasets/target-camera/`
(gitignored), producing Layout A with `metadata.json` provenance. The session matrix below
defines the *minimum* set of sessions. A session is one print (or one contiguous staged segment
of a print) captured from the fixed camera at the fixed placement.

#### Required session matrix (minimum, printer-pending)

| Axis | Value / range | Why it matters |
|---|---|---|
| Total independent sessions | ≥ 20 (see Part G) | Independence unit; drives split and CI width |
| Normal sessions | ≥ 12 | Dominant real-world condition; measure PURE_NORMAL FPR honestly |
| Failing sessions | ≥ 8 | Scarcest and most valuable; measure recall per type |
| Failure types | ≥ 4: stringing, under-extrusion, spaghetti, warping/adhesion | Stage 5 confirmed per-type recall differs (stringing FNR 0.18, underex 0.15); a single-type sample would over/under-claim |
| Failure severity | mild, moderate, severe within the failing set | Threshold 0.7 was tuned on val; real severity distribution may shift it |
| Objects/geometries | ≥ 3 distinct objects | Generalization across silhouettes/texture is untested |
| Filament/material | ≥ 2 colors, ≥ 1 color/material not in Kaggle | Domain shift to real printers is partly photometric |
| Camera configs | 1 fixed production placement + recorded alternates | We must *commit* to one MVP placement, but evaluate sensitivity to it |
| Lighting conditions | ≥ 2 (e.g. room light, dimmer/deeper shadow) | Lighting is a known dominant failure mode for this family |
| Frame sampling rate | 1–2 FPS during reasoning runs; 10–30 FPS short span for capture-FPS tests | Decision cadence (~1–2 FPS) is separate from camera capability |

**Why each dimension each matters** (aggregated from the matrix):

- **Normal vs failing mix** directly maps to the FPR vs FNR tradeoff that Stage 5 quantified
  (FPR 0.14 pure-normal, 0.28 mixed; FNR varies). You must see both under the same
  lighting/object to isolate the misclassification from the condition shift.
- **Failure-type coverage** is required because Stage 5 found failure-type-specific recall; a
  deployment tuned only on stringing would inherit underex under-detection.
- **Independence of sessions** prevents the near-duplicate-frame leakage that the source
  dataset exhibited and that any naive random split would introduce.
- **Lighting and object** are the two most likely drivers of domain shift (Stage 2 hypothesis
  about texture vs brightness; Caxton dropped dark images). They must be crossed, not kept
  constant.

### B.3 Per-session metadata (mandatory)

The existing `metadata.json` schema captures: `session_id`, `class`, `captured_at` (ISO UTC),
`camera` (device id, width, height, fps, backend), `resolution`, `frames`, `actual_fps`,
`duration_seconds`, and optional `printer`, `object`, `failure_type`, `notes`. Stage 6 adds no
new mandatory field that is not already in that schema — we just mandate that the optional
fields are populated (object, failure_type, printer) and that `notes` records lighting
condition/disturbances. Do **not** invent values; leave any genuinely unknown field absent.

---

## Part C — Ground truth

### C.1 Primary ground truth is frame-level, session-associated

Consistent with Stage 4's finding (there is no recording-level label in the source data; the
only defensible ground truth is per-frame), the **primary ground truth in our collection is the
per-frame binary class**, carried alongside stable per-frame provenance:

- **frame-level class** (`normal` / `failing`) — the label a downstream classifier predicts.
  `capture_session` records the session class at capture, and frames within a session inherit
  it; for a print that transitions mid-session, see C.4.
- **session_id** — the independence unit for splitting (Part D).
- **failure_type** (optional, only on failing frames) — e.g. `stringing`, `under_extrusion`,
  `spaghetti`, `warping`. Populated on a per-session basis from `failure_type`.
- **object** — which geometry/material the print is.
- **event/onset annotation** (optional, separate layer) — a human-recorded timestamp marking
  when a visible defect can first be reliably identified; see C.3.

### C.2 Do not construct a recording-level label as primary

No recording is reduced to a single label for the purpose of the primary evaluation. We will
not define e.g. "a session is `failing` if any frame is failing." Session is a **grouping
key for splitting and variance reporting**, not the classification target. This avoids
re-introducing the Stage 3 artifact (a first-frame/any-frame convention mislabeling good→defect
recordings).

### C.3 Event / onset annotation

- An **onset/event** is a human-annotated timestamp (ISO UTC, rounded to the nearest frame)
  at which a trained annotator can first confidently see the defect. This is a **separate,
  secondary** layer; it is **not** used to fabricate a hard class boundary.
- If physical failure onset cannot be established reliably (e.g. a gradual under-extrusion, or
  ambiguity against lighting), record the onset field as **unknown / absent** rather than
  guessing a timestamp. Per TinyVision practice: record only what is confirmable.
- Stage 5 confirmed the source dataset has **no verified onset ground truth** (the class-flip
  does not reliably equal physical onset). Our own onset annotations are the first place real
  onset ground truth could exist — but only if we actually annotate them, and we will not
  claim onset precision where we only have a class-flip.

### C.4 Handling ambiguous frames and mid-session transitions

- `capture_session` labels a whole session as one class. A print that starts good and later
  fails is a **transition session**. For these, we must reconcile the structural (one-class)
  schema with per-frame truth:
  - **Preferred:** record the transition session as the class at capture but carry frame-level
    `normal`/`failing` labels separately (an annotator relabels frames around the transition),
    so the frame-level truth is authoritative. This is the known gap of `capture_session`
    flagged in Stage 2A §8, to be handled by a relabel pass that writes a per-frame label
    manifest, never by folding the session to a single label.
  - **Ambiguous frames** (blur, head occlusion, glare, mid-motion) get an explicit
    `ambiguous` / unlabeled state and are **withheld** from both training and evaluation of the
    primary classifier (or reported separately), rather than force-labeled. Do not guess.
  - **Do not treat a transition session as a binary example of either endpoint.** This is the
    exact error Stage 4 corrected.

---

## Part D — Prevent leakage

### D.1 The split unit is the independent session

A future train/validation/test split assigns **entire sessions** (the independence unit, Part B)
to splits; no frame of a session appears in more than one split. This is already the behavior of
the source-level splitter and `build_holdout`. Random frame assignment is explicitly forbidden.

### D.2 Splits that must never cross

| Leakage vector | What we prevent | Mechanism |
|---|---|---|
| Adjacent-frame leakage | Near-duplicate frames land in train and test | Session-level assignment; consecutive frames co-located in one session, hence one split |
| Same-print leakage | Two samples of the same physical print land in different splits | Session (one print) is atomic |
| Same-object leakage | The same object geometry printed twice appears in both train and test, leaking silhouette/appearance priors | Assign all sessions of a given `object` id to one split where possible; at minimum keep a held-out object for test generalization at E. Failing that, document it |
| Same-camera-condition leakage | The same camera/lighting condition appears in both train and test | Fix an MVP camera+lighting for the production-like eval; hold out sessions by camera+condition so the final test includes conditions unseen in training |
| Same-session temporal correlation | Frames from one continuous capture across train/test | session_id atomic assignment (already structural) |

### D.3 Which dimensions to hold out for the final test

For the **final generalization test** (the claim "the frozen model generalizes to a new real
setup"), hold out:

1. **Whole sessions** (mandatory) — the atomicity [D.1].
2. **At least one dedicated object** not used in any tuning subset, to probe object
   generalization.
3. **At least one camera/lighting condition** not present in any tuning subset, to probe
   photometric generalization.
4. **The entire test set is disjoint from Experiment 1 training by construction** — it is
   different printer/camera/objects, and `eval-deep` loads the frozen checkpoint without
   retraining (Stage 2A §5). This is already leakage-free.

**Important caveat (stated, not solved by design):** because the model is frozen and our real
data is all out-of-distribution, a *holdout* of the whole real collection is the correct first
test. If we later **tune** a threshold or temporal parameter on some of this real data, that
portion can no longer be an unbiased test; reserve a leave-some-sessions-out or a second
independent collection for any tuning (Stage 2A §5 caveat). Stage 6 plans for threshold 0.7 as a
*fixed, pre-registered* decision point so we are not tempted to tune on test.

---

## Part E — Camera experiment

Two tracks on the available cameras (ESP32-CAM and USB Logitech webcam). These are
**printer-independent** and can begin immediately (Part H). Do **not** assume one camera wins;
measure, do not assert.

### E.1 Dimensional matrix to measure per camera

| Dimension | What to measure | Why | Expected relationship to model |
|---|---|---|---|
| Actual FPS | Sustained `read`-bound FPS at each resolution | Sensor/interface ceiling vs target 1–2 FPS decision | If actual FPS < decision cadence, cadence upper-bounded |
| Resolution | Native + supported modes actually delivered | AOV vs noise vs downstream 224×224 crop | 640×480 vs 320×240 delivered res may differ from spec |
| Capture stability | Frame-to-frame exposure/white-balance/auto-focus drift | Autofocus/AWB churn can shift input distribution | Misclassifies if the valid signal is fine texture (Stage 2) |
| Dropped frames | Count of consecutive empty reads (mirrors `capture.py` logic) | Drop pattern affects temporal continuity and EWMA | EWMA correctness depends on honest frame cadence |
| Inference latency | Per-frame model time on eventual CPU (and on dev machine as a soft proxy) | Not the only cost, but the largest per-frame term | — |
| End-to-end latency | capture→preprocess→infer→decision wall time | Real cadence, informs decision frequency | — |
| CPU/RAM | RSS and CPU% during continuous capture+infer | Edge budget (eval-plan targets <400MB, <10W) | — |
| Image-quality differences | Sharpness, exposure, depth-of-field, noise (webcam vs OV2640) | The classifier input distribution changes per sensor | Likely the largest domain-shift term |
| Lighting sensitivity | Same scene under fixed lighting changes | Camera auto-gain/AWB response to light | Lighting robustness (Stage 2/5) |
| Focus/mount sensitivity | Focus distance, fixed vs autofocus | The source domain was a focused near-nozzle crop | Wrong focus kills fine-texture signal |

### E.2 Scientific comparison protocol

- **Same scene, same lighting, same target**: mount both cameras on the same rig (no printer
  needed; a static textured target suffices) and capture simultaneously or back-to-back to make
  image-quality/auto-corrections comparable.
- **Record all sessions with `capture_session` metadata** so camera, resolution, fps, and
  backend are stored per session — including for the benchmark (mirrors existing schema).
- **Report per-camera means and per-condition splits**, not cherry-picked single frames.
- **Do not tune the model per camera.** Compare the frozen model's output distribution and
  metrics across cameras on identical held-out content using the pre-registered decision
  points (0.5, 0.7). The winner is whichever gives closer validation data to the Kaggle-baseline
  numbers *and* acceptable real cadence — not whichever "looks better" in one photo.

---

## Part F — Evaluation protocol

### F.1 Experiment sequence (fixed order, pre-registered decisions)

1. **Frozen ShuffleNetV2** — load `shufflenet_v2_x1_0_printing-errors_cdc9e2cad4`, no retrain,
   no weight change. (`eval-deep` with the target-camera config.)
2. **Threshold 0.5** — the Stage 3 / Exp 1 default operating point; report all metrics at it.
3. **Threshold 0.7** — the validation-selected operating point from Stage 5. **Pre-registered**
   as a fixed decision point, NOT tuned on the real-world test set. This lets us compare
   operating points honestly on out-of-distribution data.
4. **EWMA** — evaluate the smoothing on real *video sequences* (onset-annotated), not just on
   frame i.i.d. metrics, because EWMA's purported value is run/flicker behavior, which is
   temporal by nature.
5. **raw vs EWMA comparison** — run/event metrics (false-alarm runs, missed-failure runs,
   state transitions) as in Stage 5, now on real onset-ground-truth video.
6. **Only then consider retraining** — if the frozen model's real-world FNR/FPR gap is
   unacceptable AND we have enough independent sessions, retraining is on the table. No
   retraining happens in this Stage.

Threshold 0.7 is deliberately treated as a **decision boundary fixed before seeing real data**,
to preserve test-set unbiasedness (Part D).

### F.2 Primary metrics

- **FNR (recall)** — the primary concern (missed failures are costly; eval-plan target <5%).
- **FPR** — secondary; on PURE_NORMAL sessions specifically (Stage 5 showed this is the clean
  false-alarm view).
- **F1** — balanced summary.
- **AUROC** — threshold-independent (using `roc_auc_score(y_true, 1.0 - fail_scores)` per
  established convention).
- **False-alarm runs** — number and longest duration (the EWMA-relevant behavior).
- **Missed-failure runs** — number and longest duration.

### F.3 Secondary metrics

- **Inference latency** and **end-to-end latency** (per camera, per resolution).
- **FPS** (capture, inference, sustained end-to-end).
- **Memory (RSS) / CPU%** on the eventual edge hardware; on dev machine as a soft proxy only.
- **Robustness by failure type** — recall per `failure_type`, and separately per object and per
  lighting condition, to locate where the domain shift hurts.

### F.4 Test-set discipline

The final real-world test set is not used for any training or threshold/temporal tuning
(Part D). Any tuning uses a separately reserved subset (the non-test sessions), reported as
such. No metric is fabricated or reported without specifying the exact test subset and
conditions.

---

## Part G — Determine the minimum useful experiment

### G.1 Rationale underlying counts

The dominant error, with small real-world sample sizes, is **per-session variance** (Stage 1.5
showed a single catastrophic session can dominate aggregate recall). The number that matters is
the number of **independent sessions**, because adjacent frames are not independent. Reports
like "5 normal + 5 failing sessions" (the Stage 2A pragmatic floor) give a valid but very noisy
first read; for a defensible FPR/FNR comparison across failure types and lighting, more sessions
are needed.

### G.2 Practical minimum vs ideal target (distinguished)

| Quantity | Practical minimum (exploratory) | Ideal target (for a defensible claim) | Basis |
|---|---|---|---|
| Total independent sessions | 20 | 50+ | Per-session variance; CI width |
| Normal sessions | 12 | 30 | PURE_NORMAL FPR needs enough normal sessions to measure a <2/hour rate |
| Failing sessions | 8 | 20 | Recall per type; ≥4 types → need ≥4 failing per type for any per-type claim |
| Failure types | 4 | 5+ (incl. rare ones) | Stage 5 per-type recall differs |
| Objects | 3 | 5+ | Object-generalization probe |
| Camera configurations | 1 production + 1 alternate | 2 production-like + alternates | Part E comparison |
| Lighting conditions | 2 | 3–4 | Lighting robustness is a top domain risk |
| Frames per session | enough to span the relevant period | full print spans | Frames add within-session precision, not independence |

**Bottom line:** **~20 independent print sessions (12 normal, 8 failing, ≥4 failure types,
≥3 objects, ≥2 lighting conditions, 1 fixed camera placement) is an exploratory dataset,
NOT a statistically sufficient production benchmark.** It is a minimum first read useful for
detecting large domain shifts and guiding retraining priorities — not for establishing
generalization. It is deliberately separated from the **50+ sessions** needed to make a
robust retraining decision with reasonable per-type recall estimates and CI widths. We treat
the minimum as the exploratory run and the ideal as the target before any retraining decision
is trusted.

This cannot be fully justified to tight error bars from the synthetic evidence alone — the
evidence supports the *directions* (independence matters, per-type recall differs, cross-session
variance dominates) but the exact session number is a principled exploratory floor, honestly
separated from any production-adequate claim.

---

## Part H — What we can do now without a printer

These are **real experiments on the real cameras and the real frozen model**, usable without
any printer, and they are **not** presented as printer experiments.

### H.1 Worth doing now (recommended)

1. **Offline video/pipeline benchmark (Stage 7, COMPLETED)**: verified the end-to-end
   pipeline (MP4 → VideoCapture → preprocessing → frozen ShuffleNetV2 → decision) using
   labelled video fixtures assembled from Kaggle PNG recordings. Achieved 81-100%
   prediction-level match vs PNG path, E2E processing at ~26-27 FPS (36-39 ms/frame) on
   Mac M1 Pro CPU. Preprocessing negligible (<1 ms), inference dominates (~97% of E2E).
   Full report: `docs/research/stage7-video-benchmark.md`.
2. **Camera pipeline benchmarking (Part E)**: measure actual FPS, delivered resolution,
   dropped-frame rate, exposure/AWB/focus drift, and image-quality of the ESP32-CAM vs the
   Logitech webcam on an identical static textured scene, using `capture_session` metadata.
   This is the highest-value printer-independent experiment because it directly bounds the
   domain-shift term and the cadence budget. It requires no printer and no new production code
   (reuses `capture_session`). **Not yet done** — requires physical camera access.
3. **Preprocessing / resolution benchmarking**: confirm input pipeline (capture → 224×224 crop
   → normalize) time on both cameras; the frozen model's real per-frame cost on the eventual
   CPU. Bounds decision frequency. **Not yet done** — requires physical camera access.
4. **Image-quality and autofocus/AWB sensitivity**: quantify whether a consumer webcam's
   auto-corrections destabilize the input distribution the way the focused Kaggle crop did
   not. **Not yet done** — requires physical camera access.

### H.2 Worth doing but lower priority / conditional

- **ESP32-CAM MJPEG stream decode feasibility:** OV2640 typically streams JPEG; whether OpenCV
  `VideoCapture` on the ESP32-CAM block/stream (and at what res/fps) gates whether the ESP32-CAM
  is even usable as the MVP sensor. Worth a quick check as part of H.1 item 1, not a separate
  project.
- **Ad-hoc lighting fixtures (curtains/lamp)** to create the 2+ lighting conditions now, before
  a printer arrives, so the lighting axis is pre-built. Optional but useful.

### H.3 Explicitly deferred (do not start now)

- **Any retraining / long-running inference / training job** (forbidden this Stage).
- **Real printer captures** (no printer, and we will not fake them).
- **Production hardware integration** (ESP32-CAM firmware tuning, C++ runtime bring-up,
  OctoPrint/Klipper pause hooks) — out of scope until the domain validity question is answered.
- **Inventing real-world data** of any kind.

I will **not implement** these automatically now. H.1 items are identified as the worth-doing
next immediate actions; I'll proceed with whichever the user wants once the plan is reviewed
and the exact scope (and any new files/schema) is agreed. This document introduces **no new
production code and no new tests** (test-capture and the deepeval/holdout machinery already
exist), so no tests are added in this Stage.

---

## Component deliverables / artifacts

- **Plan:** `docs/research/stage6-data-collection-plan.md` (this document).
- **Machine-readable experiment schema:** not created in this Stage. The information needed to
  reproduce the plan lives in the existing `metadata.json` and config YAMLs. A machine-readable
  collection-manifest schema becomes worth adding only once real collection begins (when a
  printer exists), and would be authored together with any supporting CLI, at which time
  tests would be added. Not fabricated ahead of need.
- **ML.md:** updated in Stage 5 (committed `8c260a1`). Stage 6 adds a design document, not a
  status change; no further ML.md update needed.
- **Stage 7 (offline video benchmark, COMPLETED):** report at `docs/research/stage7-video-benchmark.md`,
  results at `docs/research/stage7-video-benchmark.json`, fixtures at `docs/research/stage7-fixtures/`.
  Code: `training/tv/data/video.py`, `scripts/generate_video_fixtures.py`, `scripts/benchmark_video.py`,
  `training/tests/test_video.py`. Not yet committed (awaiting review).

---

## Summary — the six questions Stage 6 answers

1. **Domain shift quantity** → measured by the real-collection "test vs Kaggle-baseline" gap in
   Part F §1–3, decomposed by object/lighting/failure-type (F.3).
2. **Does frozen ShuffleNetV2 generalize?** → F.1 step 1–3; no retrain until step 6.
3. **Does threshold 0.7 stay sensible?** → pre-registered double operating point 0.5 vs 0.7 on
   out-of-distribution data (F.1).
4. **Which failure types are missed?** → per-type recall (F.3) requiring ≥4 types (Part G).
5. **Is temporal smoothing useful in real video?** → raw vs EWMA on onset-annotated video (F.1
   4–5) with run/event metrics.
6. **What data is needed before retraining?** → the Part G exploratory minimum (~20 sessions) for a
   first look; 50+ sessions for a defensible retraining decision.