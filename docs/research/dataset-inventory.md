# Dataset inventory and usability classification

Facts in this document are generated from the on-disk tree and recorded
manifests. The machine-readable companion is
[`dataset-inventory.json`](dataset-inventory.json), regenerated with:

```
python scripts/dataset_inventory.py
```

Licensing verdicts and usability classifications are **editorial** and must be
read together with the per-dataset provenance notes under
[`datasets/`](datasets/README.md) and the summaries in
[`../../ML.md`](../../ML.md). Where a licence is unverified we do not use the
data (AGENTS.md dataset policy).

## 1. On-disk inventory (what is actually present)

`git_tracked_files` counts tracked paths even when the worktree copy is absent
(the external holdout/label files are intentionally not committed and are
deleted in this checkout — see §4).

| Path | Size (MB) | Files | Images | Videos | Tracked |
|---|---:|---:|---:|---:|---:|
| `datasets/Printing_Errors` | 1581.3 | 19485 | 19478 | 0 | 0 |
| `datasets/processed` | 1313.8 | 16318 | 16287 | 0 | 0 |
| `datasets/staging` | 667.9 | 2978 | 2898 | 0 | 93 |
| `datasets/external` | 0.01 | 1 | 0 | 0 | 183 |
| `datasets/raw` | 0.01 | 1 | 0 | 0 | 0 |

Total ≈ 3.6 GB. `models/`, `benchmarks/`, `cli/` are empty directories.

`datasets/processed/` contains the derived, environment-split training data:
`printing-errors/` (with `splits/{val,test}.csv`), `release-rc1/`
(`derived_manifest.json`), and the augmentation sets
`external-aug-exp2|exp3-normal|exp4-balanced`, `skz-aux-expB`.

## 2. Classification of assets

### Used by RC1 (training / evaluation)

| Asset | Role | Licence | Verdict |
|---|---|---|---|
| `Printing_Errors` (NilsHagenBeyer "3d printing errors") | Exp1 binary-classification training source | Apache 2.0 | **Usable** (used) |
| `processed/printing-errors` | Environment-split derived frames; `val`/`test` CSVs | inherits Apache 2.0 | **Usable** (used) |
| Trier FFF-QC v2 (`datasets/staging/trier`) | Strong independent holdout; 24 envs / 27,154 frames staged | **CC BY 4.0** (verified against Zenodo API 2026-09-07) | **Usable** (used) |

### Candidate / staged (not yet used for a release)

| Asset | Role | Licence | Verdict |
|---|---|---|---|
| Bambu Timelapse (`staging/bambu`) | Candidate-only timelapse environments | **CC BY 4.0** | Usable with caveat (cadence/lighting; timelapse, not continuous) |
| SKZ Nozzle-Cam | External generalization + aux training (probe) | **CC BY-SA 4.0 (HF) / CC BY 4.0 (Zenodo), unresolved** | Used only as a test-split probe; **blocked for release** until licence resolved |
| wengmhu FDM defect | Failure supplement (multi-class) | MIT | Usable |
| Roboflow/seprinder | Detection experiments | CC BY 4.0 (provenance caution) | Usable with caveat |
| Roboflow/syLucauc | Detection experiments | CC BY 4.0 (provenance caution) | Usable with caveat |

### Not suitable / do-not-use

| Asset | Role | Licence | Verdict |
|---|---|---|---|
| Caxton (Cambridge) | — | CC BY 4.0 | **Not suitable** (continuous params only, not images/labels) |
| Kaggle 3D-Printer Defected | Binary classification | unverified | **Do not use** |
| HuggingFace "Javiai" | Detection | unknown | **Do not use** |

### Own footage (needs acquisition)

| Asset | Role | Verdict |
|---|---|---|
| Own normal collection | Deployment-time normal class | **Must collect** (per ML.md) |
| Own failure collection | Failure class (+ injection) | **Must collect** (per ML.md) |

## 3. Recorded frozen holdings

`datasets/staging/readiness/canonical_holdings.json` is the project's own
summary of already-owned frozen assets used by `tv env-readiness`:

| Holding | Kind | Envs (normal/failure) | Note |
|---|---|---|---|
| `printing-errors-frozen` | train | 1 / 0 | single ELP camera venue |
| `stage13-train` | train | 7 / 0 | 333 frames, Standard YouTube License, pending human review |
| `stage14-train` | train | 0 / 2 | spaghetti 1 + other 1, 100 frames, pending human review |
| `external-pilot-verified-failing` | train | 0 / 1 | 44 human-verified failing frames |
| `seven-video-holdout` | holdout | 7 / 0 | frozen 753-frame holdout |
| `stage13-eval` | holdout | 2 / 0 | 120 frames, Standard YouTube License |
| `skz-eval` | holdout | 1 / 0 | SKZ nozzle-cam probe |
| `external-pilot-normal-eval` | holdout | 1 / 0 | UE003 normal footage |

`tv env-readiness` currently exits **1** (not ready): 0/5 human-verified failure
environments and failure-type coverage gaps (spaghetti 1 short, under_extrusion
3, bed_adhesion 2, warping 2). This is a release-class acquisition gate.

## 4. Absent-but-tracked data (this checkout)

`datasets/external/**` label manifests are **tracked in git but deleted in this
worktree**; the frame images are git-ignored and were never committed. The
183 `git status` deletions are exactly these files. Consequences:

* The independent normal holdout used for the RC1 **FPR = 1.000** finding cannot
  be re-executed here (source footage is non-redistributable YouTube
  `LY3jlcO34f4`, `zkJgLanCSUw`). This is recorded as CONFIRMED-by-artifact, not
  independently reproducible.
* `training/tests/test_exp2_external.py` skips (does not fail) when the external
  labels are absent.

See [`../../docs/research/reliability-audit.md`](reliability-audit.md) and
[`../reliability-evaluation.md`](../reliability-evaluation.md).

## 5. Actions required to make data usable

1. Resolve SKZ licensing (CC BY-SA vs CC BY) before any release use.
2. Acquire ≥5 human-verified failure environments and close the failure-type
   coverage gaps to clear `tv env-readiness`.
3. Collect own normal + failure footage on target hardware/cameras.
4. Do not reinstate the deleted `datasets/external/**` files to make tests pass.
