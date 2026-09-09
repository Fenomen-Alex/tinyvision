# Release-Candidate Dataset Build

Status: **staged; VLM proposals complete (2,931/2,931, proposals only — never labels); source-annotated release training dataset built (release-rc1, 2026-09-09); rc1 trained and evaluated — see [`rc1-release-candidate.md`](rc1-release-candidate.md); human verification of Trier failure environments still pending for a future RELEASE gate**

This document records the build of the release-candidate dataset staged on
2026-09-08 using the environment-centric staging pipeline
([`dataset-staging-pipeline.md`](dataset-staging-pipeline.md)).

Every environment is the atomic unit, one environment maps to exactly one split,
and no label is treated as verified until a human has reviewed it. No training,
no model-architecture change, no threshold tuning, and no synthetic data were
performed in this build.

## 1. Acquisition

### Trier FFF-QC v2 (primary new source)

Dataset: *Quality Control in Additive Manufacturing: Image Dataset for Defect
Detection in Fused Filament Fabrication* — Zenodo record 18628777, v2.0.0,
published 2026-02-13, license **CC BY 4.0**.
Authors: Bast, Scherer, Wahl, Naumann, Dartmann, Didas.

| file                     | size (bytes)   | MD5 (verified locally)              |
|--------------------------|----------------|-------------------------------------|
| `base_data.zip`          | 19,632,181,220 | `b5b3354970c0a229ea63334b1346da07`  |
| `additional_defect_data.zip` | 16,129,011,133 | `11ea08432cd0306b93ab880f8d0493be` |
| `test_data.zip`          | 12,630,419,031 | `3d6e8295621bd033d193a196f11025e3`  |

All three MD5s were recomputed on the local files and match the values returned
by the Zenodo API at retrieval time. Records are in
`datasets/staging/trier/metadata/characterization.json` and
`datasets/staging/trier/metadata/index.json`.

Source structure (from zip listings, no fabrication):

* `base_data.zip` — `/base/{full_res,cropped}/{extrusion_error_print,layershift_error_print,no_error_print}/camera{1,2,3}`
  (~4.7k-8.9k frames per dir).
* `additional_defect_data.zip` — `/additional_defect_data/{stringing_error,no_stringing_error}/full_res/camera{1,2,3}`
  (5,000 frames per dir).
* `test_data.zip` — `/test/{full_res,cropped}/{extrusion_error_print,layershift_error_print,no_error_print}/camera{1,2,3}`
  (~2.1k-2.4k frames per dir).

## 2. Extraction (selective, reproducible)

Full expansion of all three zips (>90 GB) would exceed available disk, so a
deterministic subsampled full-resolution extraction was performed per the plan
in `datasets/staging/trier/extraction_plan.json` via
`scripts/extract_trier_selective.py` (stride per zip: base 5, additional 5,
test 2). Result: **27,154 frames, 10 GB** under
`datasets/raw/trier-fffqc-v2/extracted/` (raw zips kept under `datasets/raw/`).

The original per-dir sizes are preserved in the zip listings
(`datasets/staging/trier/listing_*.json`), so the staged subset is traceable to
the full source.

## 3. Environments (24)

`scripts/build_trier_staging.py` derives one environment per (source root,
print type, camera). Physical camera angles are **not** claimed (source states
only "three different camera angles"); view is recorded as
`cameraN of <print> (<root>); physical angle unstated`. Environment keys are
unique at the (source, contributor, camera, view+source-root) level.

| source root             | train/failure | train/normal | holdout/failure | holdout/normal |
|-------------------------|---------------|--------------|-----------------|----------------|
| `base_data.zip`         | 6 (extrusion 3, layer_shift 3) | 3 | – | – |
| `additional_defect_data.zip` | 3 (stringing 3) | 3 | – | – |
| `test_data.zip`         | – | – | 6 (extrusion 3, layer_shift 3) | 3 |
| **total Trier**         | **9**         | **6**        | **6**           | **3**          |

Split assignments follow source intent: `base_data` + `additional_defect_data`
→ `train`; `test_data` → `external_holdout` (kept out of training by design).
Frame counts per environment: train 933-1777, holdout 1069-1186. The two
normal train environments exceed the nominal 1600-frame cap by ~7-11%
(1748-1777); recorded, not treated as a blocker.

Samples: 27,154 total; `source_label`/`source_failure_type` come from the
source directory name; **all verification is `pending`** and `tinyvision_label`
is `null` until a human reviews. No label is bid as verified.

## 4. Bambu cleanup

The Bambu staging (3 environments, 135 recordings) is unchanged in content:
only the real-time continuous-camera contributor (`Borillion`, 135 recordings)
is assigned `train`; the two timelapse-only contributors
(`Similar-Ad-1223`, `v2thegreat`) remain unassigned **candidate_only**, now with
an explicit note not to promote timelapses to training. No further Bambu work
was warranted this cycle.

## 5. Human review units (prepared, not completed)

`tv review-prep` produced **2,931 review units** (2,881 sequence contact-sheet
units + 50 frame units) from the Trier staged samples, with **no decisions
pre-filled**. Priorities follow the task ordering:

| priority | coverage | units |
|----------|----------|-------|
| P1 | new failure environments | 992 |
| P2 | new normal train environments | 929 |
| P3 | new normal external holdouts | 323 |
| P4 | new failure external holdouts | 687 |

Units live in `datasets/staging/trier/review/review_units.json`; contact sheets
in `datasets/staging/trier/review/sheets/*.jpg`. Missing/blank tiles mean the
frame file is unavailable (blank tile, never a guess). A reviewer marks each
unit `normal`, `failing` (+ `failure_type`), or `uncertain`; `uncertain` can
never become verified. `tv review-merge --dir datasets/staging/trier` applies
completed decisions with provenance `human:release-review` through the envdata
state machine (verified/rejected/uncertain). **No review session has run yet.**

## 6. VLM review pass (proposals only)

The existing local VLM review pipeline (`training/tv/vlm_review.py`,
`scripts/run_vlm_review.py`) was extended **backwards-compatibly** so the Stage
12D defaults are untouched and the same transport/prompt/validation path is
reused for staged review sets:

- `--env-staging <DIR>`: reads `<DIR>/review/review_units.json`, resolves
  evidence to the rendered contact sheets (sequence units) or the extracted
  frames (frame units), enables the extended failure vocabulary
  (`EXTENDED_FAILURE_TYPES`), and upserts proposals to
  `<DIR>/review/vlm_proposals.json`.
- Processing order follows the task instruction: P1 train failure → P2 train
  normal → P3 holdout failure → P4 holdout normal (sequence units before frame
  units within each class).
- VLM output is **proposal only**: `review_units.json`, samples and verified
  labels are never modified by the review module or driver. Proposals carry
  `environment_id`, `recording_id`, `provenance: vlm:qwen/qwen3-vl-8b`,
  decision, failure_type, confidence, visual_evidence, model, and timestamp.
- `--evidence-max-side 1024` caps the contact-sheet longest side before VLM
  encoding (~2.5× throughput; no training-data impact). The review run is
  resumable via `--skip-existing` (idempotent per-unit upsert).
- `scripts/aggregate_vlm_env.py` aggregates proposals by environment and writes
  `env_evidence.json`, `env_ranking.json`, and `human_candidates.json`
  (representative units per candidate environment for surgical human
  confirmation).
- `tv review-confirm` records a **single human decision** on one review unit
  (surgical verification) without touching anything else; `tv review-merge`
  later stamps the samples with `human:release-review` provenance. This is the
  only step that can turn a VLM-backed candidate into a verified environment.

`tv env-readiness` (baseline, 2026-09-08, pre-VLM):

| gate | required | current | met |
|------|----------|---------|-----|
| training environments (train split) | 20 | 27 | yes |
| new human-verified failure environments | 5 | 0 | **no** |
| new normal holdout environments | 2 | 3 | yes |
| new failure holdout environments | 2 | 6 | yes |

**Training proceeded on 2026-09-09 via source annotations** (`release-rc1`), per
the release-candidate decision record. The 5-human-verified-failure-environment
requirement is a release-class acquisition heuristic (data-strategy §9.7,
readiness gate G-4) and intentionally does not block training a research/
release-candidate run whose labels carry `source_annotation` provenance; it
still gates a public **release** readiness claim. Full results in
[`rc1-release-candidate.md`](rc1-release-candidate.md).

Failure-type coverage (by environment capability, new + holdings):

| type | current | target | met |
|------|---------|--------|-----|
| stringing | 3 | 2 | yes |
| layer_shift | 6 | 1 | yes |
| extrusion_error | 6 | 1 | yes |
| spaghetti | 2 | 3 | no |
| under_extrusion | 0 | 3 | no |
| bed_adhesion | 0 | 2 | no |
| warping | 0 | 2 | no |

## 8. Outcome and next actions

- **VLM review (2026-09-09, complete)**: Qwen3-VL-8B (`qwen/qwen3-vl-8b` via
  LM Studio at `http://127.0.0.1:1234`, evidence capped at longest-side 1024 px)
  processed all 2,931 Trier review units in priority order
  (P1 train failure → P2 train normal → P3 holdout failure → P4 holdout normal).
  **2,931 proposals, 2,931 success, 0 errors.** Decisions: normal 2,776,
  failing 155, uncertain 0; confidence mean 0.988 (min 0.95). Failure-type
  labels: spaghetti 101, stringing 53, bed_adhesion 1 — treated as candidate
  taxonomy only (VLM fine-grained failure-type labels are unreliable and must
  be adjudicated by a human).
- **Candidate ranking** (`datasets/staging/trier/review/env_ranking.json`):
  7 environments carry ≥1 VLM-failing proposal; the strongest are
  `extrusion_error_print` camera views (e.g. camera1/base: 73 failing / 34
  normal, mean confidence 0.95). One false positive (`no_error_print` camera2,
  1 failing hit). All 9 layer_shift/stringing environments were *not* flagged —
  the montage evidence hides thin defects (stringing) and modest layer shifts,
  so absence-of-flag is **not** evidence of absence.
- **Human review (2026-09-09)**: the first five units selected from the raw
  ranking (early-print, e.g. `...__seq016`) were manually inspected and found
  *uncertain* — the 256 px per-tile default sheets do not show visible failure.
  The candidate set was re-ranked on temporal persistence, cross-camera
  wall-clock consensus, and episode coherence; the re-ranked set and its input
  definitions live in
  `datasets/staging/trier/review/candidate_definitions.json` /
  `human_candidates.json`. For every candidate, `scripts/build_human_evidence.py`
  re-renders the underlying **full-resolution source frames** at 768 px per tile
  into `datasets/staging/trier/review/evidence/<env>/<unit>.jpg` (no more 256 px
  montages). All 9 layer_shift/stringing environments were *not* flagged — the
  montage evidence hides thin defects (stringing) and modest layer shifts, so
  absence-of-flag is **not** evidence of absence; those environments still need
  a direct human look before any negative conclusion.
- **Missing-type search**: the VLM pass found no credible Trier examples of
  `spaghetti` (its 'spaghetti' labels appear to re-label extrusion-error
  footage), `under_extrusion`, `bed_adhesion`, or `warping`. These four types
  remain coverage gaps for a later source; they are **not** readiness-gate
  conditions.
- **Human verification (surgical)**: confirm the representative units of the
  5 strongest failure environments with `tv review-confirm`, then
  `tv review-merge --dir datasets/staging/trier`. This is the only step that
  records `human:*` provenance and flips the
  `new_failure_envs_human_verified_min` gate. It is inherently manual and
  cannot be completed inside this automated cycle.
- Re-run `tv env-readiness` after each review/merge tranche.

## 9. Reproducibility

| artifact | path |
|----------|------|
| downloads / zips | `datasets/raw/trier-fffqc-v2/*.zip` (git-ignored) |
| extraction plan | `datasets/staging/trier/extraction_plan.json` |
| extraction tool + listings | `scripts/extract_trier_selective.py`, `datasets/staging/trier/listing_*.json` |
| staging builder | `scripts/build_trier_staging.py` |
| staged manifest | `datasets/staging/trier/manifests/env_manifest.json` |
| review units + sheets | `datasets/staging/trier/review/` |
| readiness report | `tv env-readiness` (reproducible) |
| tests | `training/tests/test_envdata.py`, `training/tests/test_review_units.py` |