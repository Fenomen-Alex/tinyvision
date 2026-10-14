# RC2 real-domain diagnostic: automated pipeline results

**Status:** diagnostic, NOT a release candidate. Labels are weak/VLM, not
human-verified. **Nothing here satisfies the release policy** and the numbers
must not be quoted as benchmark results.

This run exercises the automated real-domain pipeline end to end:

```
documented YouTube ids ──▶ tv acquire ──▶ tv autolabel ──▶ tv streams --split-dir
                                              │                    │
                                              ▼                    ▼
                                    (VLM frame proposals)   source-isolated splits
                                                                   │
                                                        tv train ──┴──▶ rc2_realdomain
                                                                   │
                                        scripts/realdomain_report.py (video-level)
```

Artifacts (all non-redistributable media live under the gitignored
`datasets/external-cache/`):

* `docs/research/realdomain-sources.yaml` — 20 documented source ids (the label
  provenance for the 7 "known" videos is `real-footage-vlm-labels.json`).
* `docs/research/realdomain-rc2-{test,val}.json` — RC2 video-level report.
* `docs/research/realdomain-rc1-{test,val}.json` — RC1 video-level report.
* `docs/research/release-diagnostic-rc2/` — frame-level metrics from the audited
  `tv reliability evaluate` (diagnostic policy, all gates optional).

## What was acquired

`tv acquire` tried 20 documented ids with a 20 s cap at 1 fps and produced
**19 sources / 364 frames**. Failures: `squUJUnyW6I` (media unavailable) — the
other four recovered on a re-run after the `extract_frames` boundary fix below.

## Labels are weak and were noisy

The VLM (`qwen/qwen3-vl-8b`) labelled single frames at 256×144. It was slow
(~21 min for 364 frames) and **unreliable at the video level**: it returned
`uncertain` for most frames, and several clips whose titles say "fail" came back
`normal`, while some normal clips came back `failing`. Example video-level
majorities:

| source | title label | VLM majority | VLM counts (f/n/u) |
|---|---|---|---|
| 87gsL-Luvis | failing | failing | 20/0/0 |
| mGH3p3AF1YY | failing | failing | 13/2/2 |
| uzKMV_O42SY | failing | failing | 15/2/3 |
| 6FoHkBggrWI | failing | **normal** | 0/7/13 |
| Ts-lEqkJv0A | failing | **normal** | 4/16/0 |
| 0TY6LVvMpUo | normal | normal | 0/1/19 |
| 188yo59MbbU | failing | uncertain | 6/6/8 |

Because video-title (weak) labels are a video-level signal and single-frame VLM
is not, the pipeline was corrected to **prefer the title label and use the VLM
only as a fallback** (`tv/data/streams.py`, documented in the code). The VLM
step remains valuable as an uncertain-fallback and for evidential review, but it
is not trustworthy as the primary in-domain teacher at this sampling rate.

Final weak labels: **12 normal / 7 failing** streams.

## Splits (source-isolated, class-balanced)

| split | streams | frames | normal | failing |
|---|---|---|---|---|
| train | 13 | 256 | 156 | 100 |
| val | 3 | 51 | 31 | 20 |
| test | 3 | 57 | 40 | 17 |

Held-out streams:

* val: `0TY6LVvMpUo`, `LE3PO8pwwR8` (normal), `188yo59MbbU` (failing)
* test: `nbkf372s_cA`, `38gnsbGqcx4` (normal), **`mGH3p3AF1YY` (failing, one of
  the two known real-footage failing videos)**

`87gsL-Luvis` (the other known failing video) landed in train.

## Training

`tv train --config training/configs/planned/rc2_realdomain.yaml`
(ShuffleNetV2-x1.0, ImageNet-pretrained, 30 epochs max, early stop). Best
`val_loss` 0.267 at epoch 18 (val frame acc 0.843); ~15 min on Apple M1 Pro.
`class_to_idx = {failing: 0, normal: 1}` (correct positive-class convention).

## Results

### Video-level (decision = median frame score ≥ 0.5) — `scripts/realdomain_report.py`

| eval set | model | video FAR | video detection |
|---|---|---|---|
| **test** | RC1 | **1.00** (2/2 normals alarmed) | 0.00 |
| **test** | RC2 | **0.00** (0/2) | 0.00 |
| val | RC1 | 1.00 | 1.00 |
| val | RC2 | 0.00 | 1.00 (188yo, median 0.88) |

RC2 per-stream medians: test normals 0.017 / 0.020, test failing 0.078; val
normals 0.013 / 0.050, val failing 0.878.

### Frame-level (audited `tv reliability evaluate`, threshold 0.5)

| surface | model | FPR | FNR | precision | recall | F1 | AUROC |
|---|---|---|---|---|---|---|---|
| dev_val (n=51) | RC2 | 0.194 | 0.100 | 0.750 | 0.900 | 0.818 | **0.971** |
| dev_val (n=51) | RC1 | 0.807 | 0.200 | 0.390 | 0.800 | 0.525 | 0.366 |
| locked_test (n=57) | RC2 | 0.000 | 1.000 | n/a | 0.000 | n/a | **0.966** |
| locked_test (n=57) | RC1 | 1.000 | 0.529 | 0.167 | 0.471 | 0.246 | 0.260 |

## Interpretation (honest)

* **The false-alarm failure that blocked RC1 is gone.** On every held-out real
  normal stream, RC1 fires (video FAR 1.0); RC2 does not (video FAR 0.0). RC2's
  frame-level AUROC on real footage is 0.97 (val) / 0.97 (test) versus RC1's
  **0.37 / 0.26** — i.e. on this data RC1's scores are essentially
  anti-correlated with the truth, while RC2 ranks correctly.
* **RC2 is not yet usable as a detector: it is mis-calibrated, not blind.** Its
  ranking is strong (AUROC 0.97) but the fixed 0.5 threshold sits above the
  failing stream's scores (test failing median 0.078, above the normal medians
  0.017/0.020). A threshold between the normal ceiling and the failing scores
  would detect it — but note the gap is thin on this tiny sample.
* **Success criteria (from the design):**
  * *video FAR ≤ 0.2 on held-out normal (RC1 = 1.0)* → **met** (0.0 on 4 held-out
    normal streams).
  * *detect ≥ 1 of the 2 known failing videos* → **not met at 0.5** (the held-out
    known failing, `mGH3p3AF1YY`, is missed; the other was in train).

## Limitations (must be stated wherever these numbers are used)

* **n is tiny:** 19 streams, 364 frames; the test split is 3 streams. Single
  streams swing every rate; these are not benchmark-grade.
* **Labels are weak/VLM, not human-verified.** The normal holdout is not
  human-verified, so the release `normal_holdout` gate stays INCONCLUSIVE.
* **Single-frame VLM was too noisy** to be the primary teacher at 20 frames /
  256 px; it was demoted to a fallback.
* **Frames are 256×144** (format 160). Higher-resolution acquisition is untested.
* **No target-hardware benchmark**; no ONNX/INT8 parity for RC2 (the diagnostic
  run evaluated the PyTorch checkpoint only).
* Compilation-style sources were excluded, but several clips still mix prints,
  weakening the "one stream = one print" assumption.

## Engineering defects found and fixed during the run

1. `extract_frames` aborted a whole source when a request landed past the clip
   end; now clamps below duration and skips individual frame failures.
2. `acquire` aborted the whole batch on one bad source; now records per-source
   failures and continues (19/20 succeeded).
3. `streams` emitted the manifest's relative frame paths, which the loader then
   double-prefixed; now emits absolute paths.
4. Label precedence corrected (video-title weak label first, single-frame VLM as
   fallback) with the evidence above.
5. `split_streams` made class-balanced (per-label sharding) so val/test see both
   classes; previously val had no positives.

All five are covered by tests; full suite: **438 passed, 17 skipped**.

## Recommended next steps

1. **Calibrate the operating point** on a val surface that contains a failing
   stream (and fix the evaluator's threshold-selection INCONCLUSIVE shown for
   this suite).
2. **Scale acquisition** to dozens of sources per class and **human-verify a
   small gold set** so the normal/failing holdouts become gate-usable.
3. Re-check the VLM teacher at higher frame counts/resolution before relying on
   it, or keep it as an advisory layer only.
4. Only then re-run the RC2 domain-robust configuration and the release gate.
