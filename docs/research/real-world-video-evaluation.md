# Real-World Footage Evaluation of the Frozen Edge Pipeline

Status: evaluation-only, external validation
Created: 2026-09-06
Related docs: [cpp-runtime-parity.md](cpp-runtime-parity.md), [stage12a-video-source-candidates.md](stage12a-video-source-candidates.md), [edge-runtime-benchmark.md](edge-runtime-benchmark.md)

## 1. Purpose

Test the frozen C++ pipeline against 3D-printing footage it has never seen
before: 7 independent YouTube videos (3 intended normal, 4 intended failure).
The questions:

1. Does the pipeline behave plausibly on independent footage?
2. Is normal-print false alarming still the dominant failure mode?
3. Is failure detection qualitatively useful when the failure is real?
4. Are there any software / runtime problems in non-camera pipeline use?
5. What should the project do next?

Constraints honored: no runtime architecture changes, no Raspberry Pi work, no
threshold tuning, no retraining, no dataset redesign, no new model, no
modification of historical datasets / verified labels / review manifests / VLM
proposals / SKZ artifacts. All material below is evaluation-only and is kept
separate from training and verified-label data.

## 2. Method

### 2.1 Configuration under test

The frozen model and configuration are exactly the validated edge/POC parity
configuration (no parameters were changed for this evaluation):

* Model: `experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/model.onnx`
  (fp32 ShuffleNetV2-x1.0, input `[1,3,224,224]`, output `[1,2]`, fail_idx = 0).
* Runtime: `./runtime/build/tinyvision`, JSON report schema `poc-inference`.
* Sampler: `--sample-fps 1 --warmup 8 --confirm 2`, threshold 0.5 (defaults).

The C++ runtime runs every 30th frame (1 fps at 30 fps sources). The decision
FSM starts OBSERVING for 8 samples, then emits `failing_event` once 2
consecutive samples exceed the 0.5 threshold.

### 2.2 Evaluation videos

All videos were freshly downloaded with `yt-dlp` on 2026-09-06 into
`datasets/external/real-world-eval/<youtube-id>/{video.mp4,video.webm}`. The
intended category (normal vs failure) is the acquisition *hypothesis* and is
checked visually via the VLM review section — titles and channel claims were
never treated as ground truth.

| id | title | channel | upload | intended | res / fps | frames | dur (s) |
|----|-------|---------|--------|----------|-----------|--------|---------|
| `nbkf372s_cA` | Bambu Lab P1P real time print | VivaLaChance | 2023-07-08 | normal (real-time) | 1280x720 / 30 | 16010 | 533.7 |
| `UzULqvGoi1Y` | 300x Real Time Print Demo | MakerMade | 2020-10-14 | normal (real-time) | 1280x720 / 30 | 32655 | 1088.5 |
| `0TY6LVvMpUo` | AnkerMake M5 review | TECH SHAIDER | 2022-04-08 | normal-ish (review, edited) | 1280x720 / 25 | 12595 | 503.8 |
| `38gnsbGqcx4` | Elegoo OrangeStorm Giga blob of death | 3DRundown | 2025-01-13 | failure (blob of death) | 1280x720 / 30 | 6980 | 232.9 |
| `G56LQfQCUnA` | Fixing Under/Over Extrusion ESteps | Maker-Mindset | 2022-06-28 | failure (under/over extrusion) | 1280x720 / 30 | 21898 | 729.9 |
| `87gsL-Luvis` | Printing Spaghetti | bLiTzJoN | 2016-04-11 | failure (spaghetti) | 1280x720 / 30 | 1042 | 34.7 |
| `mGH3p3AF1YY` | 3d Print Spaghetti (Fail) | Brian Johnson | 2018-12-02 | failure (spaghetti) | 640x480 / 25 | 433 | 17.3 |

Provenance notes (dataset policy): videos are not redistributed and remain
evaluation-only; they are not part of any label set or training data, and are
**not** licensed for reproduction. Title/channel/upload date were captured from
`yt-dlp` metadata at retrieval time (2026-09-06).

### 2.3 Visual review protocol (VLM, non-authoritative)

This agent has no image-input capability, so the repo's established visual
observer (local LM Studio `qwen/qwen3-vl-8b`, 127.0.0.1:1234) reviewed the
footage using the Stage-12D-extra conventions: proposals are opinions, never
ground truth, never merged into labels.

For each video the extractor (`scripts/real_world_extract.py`) produced:

* a temporally ordered 12-frame contact sheet (t01..t12),
* a 5-frame strip around the runtime's first emitted failure event
  (`*_eventstrip.jpg`),
* single frames at the model's lowest-P and highest-P sampled moments
  (`*_minP.jpg`, `*_maxP.jpg`).

The prompts were neutral: no titles, channels, upload claims, or prior
predictions were disclosed to the model (the exact neutral prompt lives in
`scripts/real_world_vlm.py`). Proposals are stored in
`datasets/external/real-world-eval/vlm_proposals.json`.

Because grid cells are downscaled to 384x216, the VLM can miss subtle cues
(e.g. thin under-extrusion lines, early-warp lift). A "normal" verdict is
therefore only evidence that **no gross failure** (spaghetti, big blob, obvious
disintegration) was visible at the sampled moments.

### 2.4 Structural cross-checks

* `datasets/external/real-world-eval/continuity_analysis.json` — mean absolute
  difference between adjacent video frames at 24 evenly spaced windows per
  video. Low, stable values with no large jumps indicate continuous real-time
  capture; high values / jumps would indicate timelapse or editing. See section 5.
* `datasets/external/real-world-eval/review/structural_signal.json` — crude
  object-coverage time series at the 12 grid timestamps. Only meaningful for a
  few scenes; kept for the record.

### 2.5 Python/C++ sanity check

`tv poc` (Python pre-trained pipeline) was run with the identical configuration
on one intended-normal (nbkf372s_cA) and one intended-failure
(87gsL-Luvis) video and compared frame-by-frame against the C++ JSON reports
(the 0.15 raw-P tolerance from `cpp-runtime-parity.md` = the empirical FFmpeg
decoder-build floor, not a correctness tolerance).

## 3. Results

All numbers come from the C++ runtime JSON reports in
`datasets/external/real-world-eval/runs/*.json`.

| id | first alarm (sample / raw frame) | FAILING frames | % post-warmup FAILING | raw mean | raw min | raw max | raw≥0.5 fraction | VLM (grid/event) |
|----|-----------------------------------|----------------|------------------------|----------|---------|---------|------------------|------------------|
| `nbkf372s_cA` | 8 / 279 | 508 / 516 | 98.4% | 0.924 | 0.116 | 1.000 | 0.961 | normal (t01..t12 clean build) |
| `UzULqvGoi1Y` | 11 / 372 | 1042 / 1053 | 99.0% | 0.750 | 0.106 | 1.000 | 0.870 | normal (clean build) |
| `0TY6LVvMpUo` | 8 / 234 | 476 / 484 | 98.3% | 0.890 | 0.070 | 1.000 | 0.926 | normal print scene; 1 frame = UI screen |
| `38gnsbGqcx4` | 9 / 310 | 216 / 225 | 96.0% | 0.705 | 0.015 | 1.000 | 0.787 | normal (finished objects / closeups) |
| `G56LQfQCUnA` | 8 / 279 | 698 / 706 | 98.9% | 0.793 | 0.004 | 1.000 | 0.847 | normal (solid blocks) |
| `87gsL-Luvis` | 8 / 279 | 25/33 (25/26 post-warmup) | 75.8% overall | 0.823 | 0.066 | 0.998 | 0.818 | **failing / spaghetti** |
| `mGH3p3AF1YY` | 8 / 234 | 8/16 (8/8 post-warmup) | 100% post-warmup | 0.934 | 0.605 | 0.998 | 1.000 | **failing / stringing+spaghetti** |

Every video emits the failure alarm between its 8th and 11th sampled second
and holds FAILING for the remainder of the clip.

## 4. Python/C++ parity on independent footage

| video | samples | max \|ΔP\| | mean \|ΔP\| | state diffs | events |
|-------|---------|------------|-------------|-------------|--------|
| nbkf372s_cA | 516 | 0.073 | 0.008 | 0 / 516 | identical (failing_event) |
| 87gsL-Luvis | 33 | 0.049 | 0.007 | 0 / 33 | identical (failing_event) |

Both runtimes produce identical state sequences and events on independent
footage; raw-P agrees well within the 0.15 decoder-build floor. Runs:
`datasets/external/real-world-eval/py_runs/*.json`. **The C++ runtime is not
the source of the behavior described below — the frozen model's outputs are.**

## 5. Visual observations (VLM proposal + structural signals, not ground truth)

Note: the VLM attaches 0.95 confidence to nearly every call, so confidence
values are not informative; the `visible_evidence` sentences are what matter.

* **nbkf372s_cA** (intended normal): grid, event strip and extreme frames all
  described as a printer steadily building a clean layered object; the
  sequence reads as continuous real-time printing (mean adjacent-frame diff
  2.6, stable). The runtime nonetheless reports FAILING for 98.4% of samples,
  beginning at t≈9 s while the print is on its first layers.
  → **clearly wrong (false alarm on a genuine normal print).**

* **UzULqvGoi1Y** (intended normal): VLM describes a red object building
  layer-by-layer on a blue build bed with no visible defects; continuous
  footage (mean adjacent-frame diff 0.9, the most stable clip). Runtime FAILING
  99.0%, alarm at t≈12 s.
  → **clearly wrong (false alarm on a genuine normal print).**

* **0TY6LVvMpUo** (review content): mostly a printer building a white
  cloud-like object; one max-P frame is a dark UI/product screen the VLM called
  "not a print" (uncertain). Runtime FAILING 98.3% including that UI frame.
  → **false alarm; content is partly non-print footage, so not a clean case,
  but consistent with the same over-alarming pattern.**

* **38gnsbGqcx4** (blob-of-death content): at the sampled moments the VLM sees
  finished, clean objects and closeups held in hand — no gross blob appears at
  the 12 grid timestamps or event strip. Runtime FAILING from t≈10 s (this is a
  comment/reaction video, not a continuous failure build).
  → **false alarm; no visible gross failure at alarmed moments. The runtime
  alarmed "correctly" here only in the sense that it alarms on everything.**

* **G56LQfQCUnA** (ESteps calibration): VLM sees solid clean blocks; the video
  is a calibration walkthrough (closeups of calibration pieces + printer states)
  rather than a gross failure. Under/over extrusion at the fine ESteps scale is
  below what the small grid cells can resolve. Runtime FAILING 98.9% from t≈9 s
  — before any diagnostic claim in the video has occurred.
  → **false alarm / unverifiable; no evidence the alarm had any basis.**

* **87gsL-Luvis** (spaghetti): VLM: yellow tangled spaghetti mass, failure
  visible across grid/strip/extremes; continuous footage. Runtime FAILING for
  25 of 26 post-warmup samples.
  → **plausible correct detection of a gross failure.**

* **mGH3p3AF1YY** (spaghetti/stringing): VLM: pink chaotic/stringy strands
  growing from t07 of the grid, gross failure; raw P ≥ 0.605 for every single
  sample (raw≥0.5 fraction = 1.000). Because the clip is only 17 s, the FSM's
  8-sample warmup leaves only 8 post-warmup samples — all FAILING.
  → **plausible correct detection at the raw-classifier level.**

Continuity analysis: all seven videos show low absolute adjacent-frame
differences (means 0.9–9.5, p90 ≤ 14.5, zero jumps above 60 in 24 sampled
windows each). None is a timelapse; `nbkf372s_cA`, `UzULqvGoi1Y` and
`87gsL-Luvis` are the cleanest continuous real-time captures. This supports the
claim that the two "normal" verdicts were genuinely normal *prints*, not
montages or timelapses the model could legitimately be confused by.

## 6. Classification of observations per video

| id | visual status (VLM+structure) | runtime behavior | assessment |
|----|------------------------------|------------------|------------|
| nbkf372s_cA | normal, real-time | FAILING 98.4%, alarm at 9 s | clearly wrong |
| UzULqvGoi1Y | normal, real-time | FAILING 99.0%, alarm at 12 s | clearly wrong |
| 0TY6LVvMpUo | normal print + UI screen | FAILING 98.3%, alarm at 8 s | false alarm |
| 38gnsbGqcx4 | normal scenes at sampled moments | FAILING 96.0%, alarm at 10 s | false alarm |
| G56LQfQCUnA | normal/clean at sampled moments | FAILING 98.9%, alarm at 9 s | false alarm |
| 87gsL-Luvis | visible spaghetti | FAILING 25/26 post-warmup | plausible correct detection |
| mGH3p3AF1YY | visible stringing/spaghetti | raw≥0.5 for all 16 samples | plausible correct detection |

No quantitative precision/recall/accuracy claims are made anywhere in this
document: the video set is small, category labels for 4 of the 7 clips are not
reliable screen truth, and the set is evaluation-only.

## 7. Findings

1. **No, the frozen pipeline does not behave plausibly on independent footage.**
   It emits `failing_event` within its first ~8–11 sampled seconds and then
   holds FAILING for 96–99% of every one of the seven videos — including two
   independently confirmed continuous real-time normal prints.
2. **Normal-print false alarming is confirmed as the dominant failure mode on
   external footage.** The two clean real-time normal prints
   (`nbkf372s_cA` raw mean 0.924, `UzULqvGoi1Y` raw mean 0.750) are classified
   as failing essentially 100% of the time. This extends the earlier internal
   UE003 weakness into a systematic, reproducibility-clean external result: the
   frozen classifier treats "a 3D printer currently mid-print, filmed by an
   arbitrary camera" as a failing state. The single most urgent data gap is
   diverse *normal* printing footage (different printers, angles, lighting).
3. **No software problem found.** Python and C++ produce identical state/event
   sequences on independent footage (0 state diffs, events identical), with raw
   P within 0.073 max absolute deviation — inside the 0.15 decoder-build floor.
   The fault is in the frozen model's outputs, not the runtime.
4. **Gross failure detection is qualitatively real when the failure is obvious.**
   The two spaghetti clips received high raw P throughout (min 0.605, post-
   warmup FAILING 100% / 25-of-26) and the VLM independently described the same
   gross failure. But the evidence stops there: the blob / ESteps videos were
   also alarmed from second ~9, far earlier than any plausible failure, so those
   clips provide no support for detection ability — only more false alarms.
5. **Recommended next step**: do **not** tune thresholds against this unseen
   set (that would silently convert evaluation data into a tuning set and is
   excluded by the task constraints). The path forward is a model/ dataset
   change: collect or license diverse normal-printing footage (preferably
   fixed-camera, real-time, from common printer bays), expand the training
   distribution, then re-validate against this same frozen evaluation set. In
   parallel, standardize the capture standards for future evaluation videos
   (fixed camera angle, real-time only, printer-mounted camera for consistency
   with the intended deployment).

## 8. Limitations

* Small set (7 videos), and 2 of them (`0TY6LVvMpUo`, `38gnsbGqcx4`) are
  produced/edited review content rather than clean print footage.
* `G56LQfQCUnA` and `38gnsbGqcx4` contain fine-grained or transient conditions that both the 384px-grid VLM and the 224px model are too coarse to
  adjudicate; "false alarm" for them should be read as "no visible basis for
  the alarm", not "proven no failure present anywhere in the clip".
* VLM proposals are a non-authoritative second observer; they are recorded and
  kept separate, not ground truth.
* Evaluation metrics are directional; no accuracy numbers claimed.

## 9. Artifacts

* Videos: `datasets/external/real-world-eval/<id>/video.*`
* Runtime reports: `datasets/external/real-world-eval/runs/*.json`
* Python POC reports: `datasets/external/real-world-eval/py_runs/*.json`
* Review images: `datasets/external/real-world-eval/review/`
* VLM proposals: `datasets/external/real-world-eval/vlm_proposals.json`
* Structural signals: `datasets/external/real-world-eval/review/structural_signal.json`
* Continuity analysis: `datasets/external/real-world-eval/continuity_analysis.json`
* Tooling: `scripts/real_world_extract.py`, `scripts/real_world_vlm.py`
* Structured data mirror: `docs/research/real-world-video-evaluation.json`