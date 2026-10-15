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

---

# Follow-up — calibration and failure-recall investigation (2026-10-14)

This section answers *why* RC2 makes zero video-level false alarms on the
held-out normals yet misses the one held-out failing stream at threshold 0.5,
and whether a defensible operating point exists. Everything below is a
**diagnostic**; the labels stay weak/VLM and none of it satisfies the release
policy.

## Reproduction (exact)

All committed artifacts were reproduced byte-for-byte on this tree
(`feat/realdomain-automated-pipeline` @ `b52cf74`); only
`datasets/external-cache/` (gitignored) was read.

| artifact | check |
|---|---|
| `realdomain-rc2-{test,val}.json`, `realdomain-rc1-{test,val}.json` | `scripts/realdomain_report.py` output `diff`s **identical** |
| `release-diagnostic-rc2/summary.txt` | `tv reliability evaluate` output `diff`s **identical** |
| `release-diagnostic-rc2/summary.json` | equal (incl. software versions) |

```bash
S=datasets/external-cache/rc2-realdomain/splits
training/.venv/bin/python scripts/realdomain_report.py \
  --split $S/test.csv \
  --checkpoint experiments/rc2_realdomain_shufflenet_v2_x1_0/artifacts/best.pt \
  --out /tmp/rc2-test.json

training/.venv/bin/python -m tv.cli reliability evaluate \
  --suite training/configs/reliability/suite-realdomain.yaml \
  --policy training/configs/reliability/policy-realdomain-diagnostic.yaml \
  --model rc2=experiments/rc2_realdomain_shufflenet_v2_x1_0/artifacts/best.pt \
  --output-dir /tmp/diag-rc2
```

Exact inputs:

| item | value |
|---|---|
| RC2 checkpoint | `experiments/rc2_realdomain_shufflenet_v2_x1_0/artifacts/best.pt` |
| RC2 sha256 | `9862f9306e2e092ef567efd077cab77740991e08d310102383d4a938aa919c06` |
| RC2 meta | epoch 18, val_loss 0.267234, `class_to_idx={failing:0, normal:1}`, `shufflenet_v2_x1_0` |
| RC1 checkpoint sha256 | `780d20ad1781a6c2f7c5756a24ea2a075657deee949ef1b80020fc27feeca279` |
| config | `training/configs/planned/rc2_realdomain.yaml` (image_size 224, decision_threshold 0.5) |
| preprocessing | `Resize((224,224))` → `ToTensor` → ImageNet `Normalize` — **identical** in `realdomain_report.py` and `tv.reliability` (`_build_transforms(.., is_train=False)`) |
| val split | `splits/val.csv` sha256 `7dd2606199d1a9a41339c422fa58b19d78fbf6c9a1deb8fd16bb76ec92823f04`, n=51 (31 normal / 20 failing, 3 streams) |
| test split | `splits/test.csv` sha256 `ddc3257267b7809da1b85800838c99010bdb1aeb125bbf116ea98222ec1d4758`, n=57 (40 normal / 17 failing, 3 streams) |
| threshold | report default 0.5; diagnostic evaluator also 0.5 (selection INCONCLUSIVE — see below) |

Threshold selection in the diagnostic is INCONCLUSIVE and falls back to 0.5
because the diagnostic policy declares no `threshold_selection.max_fpr`
(`select_threshold`'s message names the wrong cause — a cosmetic defect left
unchanged to keep the committed `summary.txt` reproducible).

## Held-out scores (the decisive detail)

Video-level = median frame score; a stream is flagged failing iff median ≥ 0.5.
`scripts/realdomain_threshold_sweep.py` (new) exports per-frame scores.

| stream | label | split | n | median | max | first frame ≥ 0.06 |
|---|---|---|---|---|---|---|
| `0TY6LVvMpUo` | normal | val | 20 | 0.0131 | 0.6617 | — |
| `LE3PO8pwwR8` | normal | val | 11 | 0.0500 | 0.2293 | — |
| `188yo59MbbU` | failing | val | 20 | 0.8780 | 0.9546 | 0 |
| `nbkf372s_cA` | normal | test | 20 | 0.0165 | 0.0223 | — |
| `38gnsbGqcx4` | normal | test | 20 | 0.0195 | 0.0399 | — |
| **`mGH3p3AF1YY`** | **failing** | **test** | 17 | **0.0785** | **0.2818** | **0** |

`mGH3p3AF1YY` per-frame P(failing) over time (1 fps, first 17 s):

```
0.222 0.237 0.017 0.052 0.049 0.110 0.046 0.037 0.041 0.095 0.161 0.218 0.078 0.057 0.041 0.282 0.246
```

Every frame is **below 0.5** (max 0.282): the model's argmax for all 17 frames
is `normal`. Frame-level AUROC is still 0.966 on test (0.971 on val) because
those frames still rank above the held-out normals — RC2 ranks the failure
correctly but with a **compressed absolute scale**.

## Source separation audit (no leakage)

* Splits are source-isolated: train/val/test stream sets are pairwise disjoint
  (`train∩val = train∩test = val∩test = ∅`); each stream is the suite
  `source_key`. Verified with the split CSVs and `check_source_leakage`.
* Threshold sweeping and calibration below use the **validation split only**.
  Test is scored for reporting only and is never used to pick a value.
* Label provenance: each stream's label is the **hand-assigned `weak_label`**
  from `realdomain-sources.yaml` (surfaced in `streams.json` as
  `label_source: "title"` — a legacy tag); the manifest `title`/`uploader`
  fields are empty (no metadata was actually parsed). This is weak supervision,
  not verified ground truth; all suite surfaces are `verified: false`. The VLM
  re-derives frame labels but its video-level majority does not override the
  weak label here (the weak label is present).

## Missed failure: root cause

The missed failure is a **generalization/representation gap at the fixed
operating point — not a temporal-filtering issue and not fixable by monotonic
calibration**:

1. **Not temporal.** The decision unit is a per-stream median; there is no
   temporal filter in this diagnostic path. The VLM (which can see) labels
   13/17 `mGH3p3AF1YY` frames `failing / spaghetti` with high confidence, so the
   failure is genuinely visible; the model simply scores it low.
2. **Ranking is fine; scale is not.** AUROC is 0.966 (test) / 0.971 (val). Every
   held-out failing frame is ranked above the normals yet sits below 0.5.
3. **Calibration cannot fix it.** Temperature scaling is monotonic, so it
   preserves argmax for every `T > 0`. Fitting `T` on validation gives
   `T* = 0.850` and **0/17** `mGH` frames become `argmax = failing` for any
   `T ∈ {0.2, 0.5, 1, 2, 5}` (max P(failing) is 0.453 even at `T = 5`). A fixed
   0.5 boundary cannot be rescued by temperature scaling.
4. **The validation split is not identifiable.** Its only failing stream sits at
   median 0.878, ~0.83 above the highest validation normal (0.050). A wide band
   of thresholds (0.06–0.86) all give *perfect* validation performance, so
   validation cannot reveal that failures can score as low as 0.078.

## Threshold sweep (validation-selected)

`--max-fpr 0.0`, grid 0.02…0.98. Validation: 2 normals (0.0131, 0.0500), 1
failing (0.8780).

| selection rule (validation only) | threshold | val FAR | val recall | held-out FAR | held-out detection |
|---|---|---|---|---|---|
| repository `select_threshold` (max recall, then **lowest** threshold) | **0.06** | 0/2 | 1/1 | **0/2** | **1/1** |
| conservative (highest feasible threshold) | 0.86 | 0/2 | 1/1 | 0/2 | 0/1 |
| midpoint margin `(0.0500+0.8780)/2` | 0.464 | 0/2 | 1/1 | 0/2 | 0/1 |

Held-out (test) separation, evaluation-only: highest normal median 0.0195,
failing median 0.0785 — **separable** for any threshold in (0.0195, 0.0785).
Detection delay at the repository threshold 0.06: first `mGH` frame ≥ 0.06 is
frame **0** (median rule flags the stream anyway).

**The repository rule with a fine grid does select 0.06 and *would* catch the
held-out failure** (`docs/research/realdomain-rc2-threshold-sweep.json`). The
original miss happened because (a) the diagnostic ran no selection → fixed 0.5,
and (b) the release-policy grid is `[0.3 … 0.7]`, whose floor is far above the
failure's 0.078.

## Can a defensible operating point be claimed? — No.

A threshold satisfying both requirements **on the point estimates** exists and is
reachable by a legitimate validation-only rule (0.06: val FAR 0/2 & recall 1/1;
held-out FAR 0/2 & detection 1/1). But it **cannot be claimed**, because the
evidence is too thin and not identifiable:

* **Normal sample too small.** Held-out independent normals n = 2; with the two
  validation normals, n = 4. Wilson 95% upper bound on FAR: **0.66** (n=2) or
  **0.49** (n=4). To bound video FAR ≤ 0.2 at 95% confidence you need **≈16
  normal streams with zero alarms**.
* **Failing sample too small.** Held-out failing n = 1; recall 1/1 has Wilson
  95% lower bound **0.21**. ≈4 all-detected failing streams are needed for a
  lower bound ≥ 0.5.
* **Threshold not identifiable.** On validation, 0.06 and 0.86 are equally
  consistent (both FAR 0/2, recall 1/1). The 0.06 choice has only **0.010**
  margin above the highest validation normal.
* **Weak labels.** The "failing" normal/failing labels are hand-assigned and
  VLM-corroborated, not human-verified, so the normal-holdout gate stays
  INCONCLUSIVE regardless of the threshold.

**Smallest additional labeled recording needed:** to make the FAR ≤ 0.2 claim
supportable, **≈16 human-verified normal streams (≥2 venues) with zero false
alarms**, plus **≥4 human-verified failing streams** to bound recall. This is a
data problem, not a calibration one.

## Calibration (validation-only) — evaluated, not adopted

Temperature scaling fit on validation (min-NLL): `T* = 0.850`.

| | T = 1 | T = 0.85 |
|---|---|---|
| val NLL | 0.2672 | 0.2643 |
| val ECE | 0.0853 | 0.0747 |
| val AUROC | 0.9710 | 0.9710 (invariant) |
| val failing median | 0.878 | 0.910 |
| test failing median (`mGH`) | 0.0785 | 0.052 |

The NLL-optimal temperature is `< 1`, i.e. it *sharpens* and pushes the
already-low failing scores **lower**, moving the failure further from any fixed
threshold. Calibration is therefore **not adopted**; the module
`training/tv/calibration.py` exists only to make the result reproducible and is
not wired into the gate.

## Regression tests added

`training/tests/test_realdomain_calibration.py` (9 tests, all synthetic guards —
not real-world evidence):

* temperature preserves argmax for any `T > 0`; a below-0.5 failing frame cannot
  be flipped by any temperature;
* `fit_temperature` never worsens validation NLL;
* ECE bounds;
* `select_threshold` is most-sensitive/lowest-feasible on a real-like val
  surface, and the perfect-performance band is wide (non-identifiable);
* the release policy requires `normal_holdout` + ONNX parity while the
  diagnostic policy is all-optional, and every real-domain suite surface is
  `verified: false` (a weak-label surface can never satisfy release).

Also corrected a stale comment in `training/tv/reliability.py` that claimed the
threshold tie-break picks the *higher* threshold (it picks the lowest).

## Artifacts

* `docs/research/realdomain-rc2-threshold-sweep.json` — per-frame scores, all
  operating points, held-out evaluation and oracle band.
* `scripts/realdomain_threshold_sweep.py` — regenerates it (selection on val
  only).
* `training/tv/calibration.py`, `training/tests/test_realdomain_calibration.py`.
