# Dataset Staging + Contributor Intake Pipeline

Status: **research/planning deliverable — infrastructure only. No model
training, no model or threshold changes, no synthetic data generation, no
blind bulk downloads, no commits.**

This document specifies how new camera data enters TinyVision. It implements
the recommendation of
[docs/research/tinyvision-data-strategy.md](./tinyvision-data-strategy.md):
the independent unit of a dataset is the **camera environment**, not the
frame, recording, or video. All splitting, readiness accounting, and review
operate on environments.

## Motivation

The experimental phase proved that the model generalises poorly to novel
camera environments. The only cures are (a) more genuinely different
environments and (b) holding out entire environments for evaluation. Frames
from the same environment are correlated and must never straddle a
train/evaluation boundary. This pipeline makes that structurally impossible.

## 1. Trier FFF-QC characterization

* Dataset: *Quality Control in Additive Manufacturing: Image Dataset for
  Defect Detection in Fused Filament Fabrication* (Trier University of Applied
  Sciences), Zenodo record [18628777](https://zenodo.org/records/18628777).
* License **CC BY 4.0** (verified against the Zenodo API on 2026-09-07).
* Files (~48.4 GB, md5 verified from the API record): `base_data.zip`
  19.6 GB, `additional_defect_data.zip` 16.1 GB, `test_data.zip` 12.6 GB.
* Captured **from three different camera angles during production**; defects
  include extrusion errors, layer shifts and stringing, plus defect-free
  prints; high-resolution and cropped images.
* **Environment units are NOT yet verified.** The three angles may encode one
  to several environments. Frame/printer counts require inspecting the
  extracted tree.

Deliverables:

* `datasets/staging/trier/metadata/zenodo_record_v2.json` — raw API record.
* `datasets/staging/trier/metadata/characterization.json` — metadata-based
  characterization (explicitly marked `environment_verified: false`).
* `tv env-index --kind trier [--root PATH]` — indexes an extracted local copy
  when one exists (counts per category/view, `structure_verified`), still
  without copying media.

**No blind download:** the 48 GB should only be fetched after a decision to
license-verify and stage it; before that, the characterization stands on
metadata only.

## 2. Bambu environment verification

* Dataset: [Bambu Timelapse Dataset](https://huggingface.co/datasets/v2thegreat/bambu-timelapse-dataset),
  **CC BY 4.0** (verified), split `train/timelapses` and `train/videos`.
* Metadata CSVs (fetched 2026-09-07, cached under
  `datasets/staging/bambu/metadata/`):

| split      | rows | contributors                                        |
|------------|------|-----------------------------------------------------|
| timelapses | 405  | Similar-Ad-1223 (198), v2thegreat (120), Borillion (87) |
| videos     | 135  | Borillion (135)                                     |

* The camera environment is a contributor's print-chamber camera, so the
  dataset yields **3 candidate normal environments**.
* Only **Borillion** (135 full-length videos, real-time ~30 fps) is usable
  for ~1 fps per-environment training. The timelapses are skimmed
  (a few encoded frames per print) and are kept only as robustness-probe
  material.
* Environment inventory (without downloading media) is produced by
  `tv env-index --kind bambu` → `datasets/staging/bambu/metadata/inventory.json`,
  and the staged manifest is built offline by
  `scripts/build_bambu_staging.py`.

## 3. Environment manifest schema

Canonical manifest: `manifests/env_manifest.json` (schema
`tinyvision-env-manifest`, version 1). Fields:

* `dataset`: id, name, license(+URL), provenance, acquisition method,
  retrieval date, URL, and explicit booleans
  `redistribution_allowed` / `training_allowed` / `evaluation_allowed`.
* `environments[]`:
  * `environment_id` — persistent `env-<uuid4>`, never regenerated.
  * `environment_key` — stable human-readable
    `env_<source>_<contributor>_<camera_identity>_<view>`; unique per manifest.
  * `source_dataset`, `contributor`, `camera` (identity + description),
    `view` (vocabulary: front/side/overhead/bed_mount/inside_chamber/unknown),
    `printer`, `environment_group`, `split_assignment` (train/validation/
    external_holdout or null), lighting/background notes, free metadata
    (e.g. `normal_available`, `failure_type`).
* `recordings[]`: `recording_id`, `environment_id`, `source`, duration/fps/
  resolution, `continuous_realtime`, `temporal_note`, `md5`.
* `license_consents[]`: per-contributor consent records
  (training/evaluation/redistribution), consent date, preferred credit.

Frames are stored as per-recording JSONL under `samples/<recording_id>.jsonl`;
each sample carries its `environment_id`, a `source_label` (never auto-trusted)
and the `tinyvision_label` + `verification` state that TinyVision itself
trusts only after human review.

Example template: `datasets/staging/env_manifest.example.json`.

## 4. Environment-level splitting

`tv env-split --dir datasets/staging/<dataset> [--assign key=split,... |
--ratios R,R,R]`

* One environment → exactly one of `train`, `validation`, `external_holdout`.
* Recordings can reference one environment only; environment keys are unique;
  every recording environment must exist (`check_environment_leakage`).
* Deterministic seeded allocation over environment IDs only (never
  frames/recordings); manual assignment by environment key is the primary
  path.
* Unassigned environments are legitimate *candidates*: they contribute to no
  split and appear as warnings, not failures, in `env-validate`.

Existing historical splits from the experiment phase are *not* retrofitted;
they are recorded as frozen holdings (section 11) and never re-split.

## 5. Human-review integration

The pipeline deliberately reuses the project's review discipline
(`tv/review.py` Stage-12D flow; `scripts/run_human_review.py`):

* `source_label` from an external dataset or contributor is **never**
  auto-promoted.
* Each sample's `tinyvision_label` requires `verification == verified` with a
  provenance string whose prefix is `human:*` (e.g. `human:stage12d`) or an
  approved pipeline stage; `source_auto`, `source`, `vlm_only` are rejected
  outright.
* State machine in `tv/envdata/state.py`:
  `pending → verified | rejected | uncertain`; `uncertain → verified |
  rejected`; verified/rejected are immutable without an explicit reasoned
  override.
* A failure environment only counts toward the training gate once **at least
  one** of its samples carries `verified` + `failing` + `human:*` provenance
  (`readiness`).

## 6. Contributor intake kit

See `docs/contributors/`:

* `README.md` — the program, what is asked, what is done with the data.
* `recording-guide.md` — fixed camera, continuous real-time footage, lighting/
  background guidance, metadata fields, normal + failure shots.
* `contributor-consent.md` — consent form granting training/evaluation/
  redistribution rights per contributor (contributors release the insurer
  and the project from liability; no claim to *or guarantee of* licence
  correctness — see the file for the exact wording).

A contributor dataset is only valid when every represented environment has a
matching `license_consents` entry; `env-validate` enforces this
(`missing_consent`).

## 7. Intake validation

`tv env-validate --dir datasets/staging/<dataset>`

* Structural checks: schema fields, unique env IDs, unique env keys,
  dangling recordings, split vocabulary, usage-rights booleans vs license,
  retrieval date format.
* Consent checks for contributor datasets (`missing_consent`).
* Physical checks when recordings are local: video open/probe (fps,
  resolution, duration), exact md5 duplicates (`exact_duplicate`), obvious
  near-duplicate recordings in the same environment, and a heuristic
  "possible timelapse / edited cadence" flag (`possible_timelapse`) — the
  latter is a warning only: semantic quality judgement is explicitly out of
  scope for now.
* Exit code nonzero on any `error`; warnings are reported for the human.

## 8. Immutable environment identity

`tv/envdata/identity.py`

* `new_environment_id()` → random `env-<uuid4>`, never regenerated.
* `EnvironmentRegistry` binds `environment_key → environment_id`; re-binding
  a key to a different ID raises. Datasets built with a registry therefore
  produce the same ID for the same logical environment across runs.
* Two recordings belong to the same environment **only** if they share the
  same physical setup, camera identity, viewpoint, lighting and visible
  printer appearance. Splitting uses these IDs, making cross-environment
  frame leakage impossible by construction.

## 9. Staging area

```
datasets/staging/
    env_manifest.example.json        # template
    readiness/canonical_holdings.json # already-owned frozen assets
    bambu/                           # staged Bambu dataset
        environments/<env_id>.json   # derived views
        recordings/<rec_id>.json
        manifests/env_manifest.json  # canonical
        metadata/                    # CSVs + inventory.json
        review/
        samples/<recording_id>.jsonl
    trier/                           # characterization only (no download)
        metadata/zenodo_record_v2.json
        metadata/characterization.json
    contrib/                         # future contributor datasets
```

Per-record environment/JSON views under `environments/` and `recordings/` are
derived from the canonical manifest (`dump_manifest`), kept for humans and
tooling that wants single-file access.

## 10. Training-readiness gate

`tv env-readiness` prints a full report of all counts **regardless** of
outcome:

* environments per split (train / validation / external_holdout), normal vs
  failure;
* human-verified failure environments in staged train splits;
* failure-type coverage vs the strategy targets
  (spaghetti ≥3, under-extrusion ≥3, stringing ≥2, bed-adhesion ≥2,
  warping ≥2, layer-shift ≥1, extrusion-error ≥1) — counted across staged and
  holdings;
* a `training_ready` boolean and a recovery plan listing the exact per-gate
  deltas.

## 11. Next-training gate (do not train before this passes)

Combines the staging manifests with `datasets/staging/readiness/canonical_holdings.json`
(the frozen assets already owned: printing-errors venue, stage-13/14 splits,
seven-video holdout, SKZ probe, external pilot). The four conditions:

| gate | required |
|------|----------|
| training environments (train split, holdings + staged) | ≥ 20 |
| new human-verified failure environments (staged after this pipeline) | ≥ 5 |
| new normal holdout environments (staged `external_holdout`) | ≥ 2 |
| new failure holdout environments (staged `external_holdout`) | ≥ 2 |

**Current state (2026-09-07):** holdings contribute 11 training environments
(8 normal + 3 failure), so gate 1 is at 12/20 (+ the 1 staged Bambu normal).
Staged verified failure environments: 0/5. New holdout environments: 0/2 and
0/2. **The next serious training run must not happen until this gate passes.**

## 12. Documentation

* This file (human-readable spec).
* `docs/research/dataset-staging-pipeline.json` (machine-readable spec).
* `docs/contributors/*` (intake kit).
* Every staged dataset carries its provenance in `metadata/` (license,
  retrieval date, known limitations).

## 13. Tests

`training/tests/test_envdata.py` (24 tests):

* identity stability/rebinding rules;
* environment-level splitting: no leakage, small-pool train preservation,
  dangling/multi-home recording detection, key collisions;
* manifest + consent + sample validation;
* review-state machine (verified requires `human:*`; immutable states);
* readiness gate deltas and verified-failure-env counting;
* Trier index/characterization and Bambu inventory counts;
* exact-duplicate md5 detection.

Full suite: `uv run --project training python -m pytest` (target: no new
failures over the pre-existing local baseline — the 5 `test_skz_experiment`
failures reference git-ignored SKZ probe images not present in this checkout
and predate this work).

## 14. Proposed structure (final layout)

The final repository layout keeps training, staging and release separate:

```
datasets/staging/            environment-manifest data + provenance (this task)
docs/contributors/           intake kit + consent templates
training/tv/envdata/         staging/validation/readiness code + CLI
docs/research/dataset-staging-pipeline.{md,json}
docs/research/tinyvision-data-strategy.{md,json}
```

## 15. Report on investigation outcomes (answer set)

1. **Trier environment count** — unknown/UNVERIFIED until a local extraction
   exists; three camera angles at one production site claimed. Mid-risk gap:
   Trier may contribute as few as one to a few environment units.
2. **Bambu environment count** — exactly 3 candidate environments (one per
   contributor), of which 1 (Borillion) has continuous real-time video usable
   for training; 135 full-length video recordings catalogued.
3. **Schema** — `tinyvision-env-manifest` v1 implemented with immutable
   environment identity, consents, samples with explicit verification states.
4. **Intake** — implemented: `tv env-validate`, contributor consent
   enforcement, video probing, duplicate detection, timelapse heuristic.
5. **Splitting** — implemented at environment granularity with leakage
   checks; existing historical splits frozen and excluded.
6. **Readiness gate** — implemented: `tv env-readiness` reporting all counts
   and per-gate deltas; `training_ready` says NO today because the next-training
   gate has not been met.
7. **Current qualifying environments** — 12 training environments (holdings 11
   + Bambu 1), 0 new verified failure environments, 0 new holdout environments.
8. **Remaining gap to the gate** — +8 training environments, +5 new
   human-verified failure environments, +2 normal and +2 failure new holdout
   environments. Failure-type coverage beyond holdings (spaghetti 2, other 1)
   still needs, by target: under-extrusion ×3, stringing ×2, bed-adhesion ×2,
   warping ×2, layer-shift ×1, extrusion-error ×1, spaghetti ×1.