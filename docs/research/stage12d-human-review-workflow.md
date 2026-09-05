# Stage 12D — Human Review Workflow for External Pilot Footage

**Date:** 2026-09-05
**Status:** Built a ready-for-humans review workflow (manifest + contact sheets) for the six external pilot sources. **No labeling performed by this agent; no training.**

---

## 1. Purpose and scope

Stage 12C established that only **36 provisional, human-UNVERIFIED frames** (21 normal + 15 failing) exist across the six acquired videos — far too little to justify Experiment 2. The analysis driving this stage concluded:

> "Stage 12C did not produce enough trustworthy external data to justify Experiment 2... The immediate blocker is not code. It's labels."

Stage 12D therefore builds the **human-verification workflow** that turns a subset of the existing six videos into **human-verified ground truth**, rather than downloading more footage or training prematurely.

**Hard constraints honored:**
- **No training.**
- **No inference / no auto-labeling.** This agent has no image/vision capability; the manifest is produced with **empty (null) decisions** for a human to complete. Nothing is labeled by the machine.
- **No modification of `datasets/processed/printing-errors/` or the frozen official test set.**
- **No additional downloads.** Only the six acquired sources are reviewed.
- **Review units are deterministic and reproducible** from the stored Stage 12B structure analysis.
- **No fabricated ground truth** — the output `verified_labels.json` is written only when a completed, validated manifest is merged.

---

## 2. Why sequences, not isolated frames

For a printing failure, **10 consecutive frames are often far easier to classify than one arbitrary frame** (a growing blob / curling edge / missing layer is obvious in motion, ambiguous in a still). Therefore:

- **Continuous stable footage → SEQUENCE units**: a contact-sheet JPG of ~10 consecutive 1-FPS frames with timestamps, reviewed as a unit.
- **Unstable / cut-heavy regions → FRAME units**: a single isolated frame, used where a "sequence" is meaningless (a scene cut every second).

The reviewer answers per unit: `normal` | `failing` | `uncertain`, plus optional `failure_type` and optional `transition_start` / `transition_end`.

---

## 3. Method (reproducible)

- **Module:** `training/tv/review.py` (`load_analysis`, `is_unstable`, `stable_runs`, `define_review_units`, `sequence_sheet`, `build_manifest`, `merge_verified`).
- **CLI:** `scripts/run_human_review.py` with `build` / `summary` / `merge` subcommands.
- **Stability:** reused from the stored Stage 12B analysis (`pilot_footage_analysis.json`, per-sample `cut_flag`). A sample is **unstable** when `cut_flag` is true OR `frac_changed > 0.30` AND `diff_bbox_frac > 0.50` (camera pan / scene change) — identical to the Stage 12B definition. The t=0 sample (no predecessor) is treated as stable.
- **Stable runs:** maximal contiguous spans of stable seconds.
- **Review units:** a stable run of ≥ `MIN_SEQUENCE_SECONDS=6` seconds becomes SEQUENCE units chunked into blocks of `SEQUENCE_CHUNK_SECONDS=10` seconds. Every remaining second becomes a FRAME unit. **Every second of every video is reviewable exactly once** (no gaps, no double-coverage).
- **Contact sheets:** one JPG per sequence unit, tiles downscaled to 256px with `t=<k>.0s` labels, saved to `labels/sheets/`.
- **Manifest:** `labels/review_manifest.json`, all `decision`/`failure_type`/`transition_*` = null.
- **Merge:** `merge_verified` validates every decision against the vocabulary and, only when the human has completed it, writes `labels/verified_labels.json` with `provenance: human_reviewed`. No auto-assigned labels.

---

## 4. Review units produced (no decisions filled yet)

| Source | Dur (s) | Units | Sequence | Frame | Seq-reviewable secs | Stable secs | Unstable secs |
|---|---|---:|---:|---:|---:|---:|---:|
| NORMAL001 | 20.1 | 3 | 3 | 0 | 21 | 21 | 0 |
| SP001 | 25.5 | 6 | 3 | 3 | 23 | 24 | 2 |
| SP002 | 55.0 | 55 | 0 | 55 | **0** | 14 | 41 |
| STR001 | 645.1 | 520 | 19 | 501 | 144 | 312 | 333 |
| UE001 | 519.6 | 311 | 28 | 283 | 237 | 342 | 178 |
| UE003 | 367.6 | 111 | 33 | 78 | 290 | 308 | 60 |
| **TOTAL** | | **1,006** | **86** | **920** | **715** | | |

86 contact sheets rendered under `labels/sheets/`.

---

## 5. Readable synthesis of the review workload

This is the real evidence the stage sought: **how many usable human-verified frames can the six videos realistically yield, and at what cost?**

- **SP001 (23 seq secs, 3 sequence units + 3 frames):** the strongest failure candidate. 23 continuous seconds form a compact, high-yield review (the same run Stage 12B flagged for object growth t=0–22, plus the hard cut at t=23 as a frame unit). **Highest value-per-effort.**
- **NORMAL001 (21 seq secs, 3 sequence units):** fully stable; 21 seconds of a possible normal print. High value if confirmed as depicting a real successful print.
- **UE003 (290 seq secs, 33 sequence units + 78 frames):** the largest single continuous-run yield (t=148–256 = 108 s stable, long UE001 window). Moderate-to-good value; dominated by one or two long stable runs.
- **UE001 (237 seq secs, 28 sequence units + 283 frames):** 88 s and 27 s stable windows exist, **but Stage 12B found a persistent on-screen troubleshooting checklist** — i.e. much of it is tutorial framing, not stable printer footage. Reviewable seconds are lower than raw stable seconds and many will come back `uncertain`.
- **STR001 (144 seq secs, 19 sequence units + 501 frames):** mostly PrusaSlicer screen-captures (Stage 12B `STRINGING_SOURCE_FAILED`). 501 isolated frame units are low-value; the 19 sequence units may capture brief printer footage but are unlikely to yield many usable frames. **Lowest value-per-effort.**
- **SP002 (0 seq secs, 55 frame units):** 41 cuts in 55 s, no stable run ≥ 3 s. All 55 seconds are isolated-frame units; essentially no sequence review possible. **Effectively non-contributing.**

**Bottom line:** only **SP001, NORMAL001, and (partially) UE003** present enough *continuous* footage to be worth sequence-level human review in a short session. Reviewing every one of the 1,006 units is unnecessary to answer the experiment question — see the recommendation below.

---

## 6. Recommendation: continuous-footage-first acquisition criterion

A second, independently important finding of the pilot is that the original acquisition strategy was too optimistic: **videos titled "how to fix stringing" are potentially terrible training data**, whereas "3D printer camera — print fails due to stringing" with 10 minutes of continuous footage is potentially excellent.

**Revise the acquisition criterion from "failure type first" to "continuous footage first, failure type second":**

> Prefer long, continuous, stable printer-monitoring feeds (live-cam / printer-cam channels) over tutorial / montage videos, regardless of the failure keyword in the title. A stable feed that may or may not contain a specific failure is more valuable than a heavily-edited video that claims a failure but has minable footage.

This is recorded as guidance for any future Stage 12E acquisition and reflected in the per-source usefulness ranking above (Section 5). No new sources are downloaded in this stage.

---

## 7. Deliverables

- `training/tv/review.py` — human-review module (manifest, contact sheets, stable runs, merge).
- `scripts/run_human_review.py` — CLI (`build` / `summary` / `merge`).
- `training/tests/test_review.py` — 10 tests (stability, stable runs, unit splitting/coverage, deterministic unit IDs + null decisions, merge validation incl. partial merge). Full suite passing.
- `datasets/external/video-frames/labels/review_manifest.json` — 1,006 units, **all decisions null** (ready for human completion).
- `datasets/external/video-frames/labels/sheets/*.jpg` — 86 sequence contact sheets.
- `datasets/external/video-frames/labels/verified_labels.json` — **not created** (only written by merge after a human completes the manifest).
- `docs/research/stage12d-human-review-workflow.md` / `.json` — this report.

---

## 8. How a reviewer completes it

1. **Review the high-value sources first** (SP001, NORMAL001, UE003) via the sequence contact sheets.
2. For each sequence unit in `review_manifest.json`, set `decision` (`normal`/`failing`/`uncertain`), optionally `failure_type`, and transition times where a visible transition exists.
3. For isolated frame units (e.g. the SP001 t=23 cut), decide from the single frame or mark `uncertain`.
4. Save the manifest, then run:
   `python scripts/run_human_review.py merge` (or `--allow-partial` to merge only the units decided so far).
5. `merge_verified` validates the vocabulary and writes `verified_labels.json` (`provenance: human_reviewed`). **No training step runs here.**

---

## 9. Critical limitations

1. **No decisions are filled yet (0/1006).** The manifest is a ready-for-humans artifact; this stage does not claim any verified frames.
2. **This agent has no vision capability** and cannot itself confirm any failure; only a human (or a future vision reviewer) completes the manifest.
3. **715 sequence-reviewable seconds is an upper bound, not a guaranteed yield.** Much of UE001 and STR001 will reasonably come back `uncertain`; real usable count may be far lower and concentrated in SP001/NORMAL001/UE003.
4. **SP002 contributes nothing** (no stable run ≥ 3 s). **STR001 contributes very little** (slicer screens).
5. **Uncertain is an allowed decision**, never silently forced to normal/failing.

---

## 10. Next step (not executed)

Proceed directly to the human review of the highlighted high-value sources (SP001, NORMAL001, UE003) and merge the decisions. Then the single decisive question can be answered:

- **Enough verified external frames (order of several hundred across ≥ a few independent sessions) → run the deliberate Experiment 2** (Kaggle + verified external train, same ShuffleNetV2, same methodology, frozen official test), or
- **Not enough → acquire continuous-footage-first sources before Experiment 2.**

No further elaborate methodology stages are warranted unless the evidence demands them.

**Stage stops here. No training initiated; no verified labels produced by the machine.**