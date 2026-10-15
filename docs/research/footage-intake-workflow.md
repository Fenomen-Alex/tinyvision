# Footage intake workflow

**Status:** implemented (thin layer over existing commands). This document describes how real-world 3D-print footage enters the pipeline for a **per-printer, printer-disjoint** evaluation.

## Objective

Register real video sessions (one complete print per session) with printer/camera/label/onset/split metadata, validate independence and completeness (fail-closed), and connect intake → `tv poc` → `tv calibrate` → locked hold-out evaluation → release-gate inputs. No training. No changes to `tv calibrate`/`tv poc` semantics or the release gate policy.

## Design (Option A)

The objective's three splits map to existing tooling without modifying calibration logic:

| Intake split | Purpose | How it is used |
|--------------|---------|----------------|
| `calibration` | calibrate the operating point for the declared printer/camera | becomes calibration clips for `tv calibrate` |
| `validation` | choose/report an operating point tier that is **never** used to select the final threshold | mapped to `calibrate`'s `holdout` (reported, never used for selection) |
| `holdout` | **locked holdout**; independent stream, excluded from calibration | **not** included in the calibrate spec; reserved for held-out evaluation (release-gate input) |

Uncertain sessions (`label: uncertain`) are validated but **excluded** from any calibrate spec and from held-out reports. Any attempt to "hide" contamination of the locked holdout fails closed.

## Session sheet

A session sheet is YAML or JSON describing clips/sessions for a single printer/camera. Each item represents one complete print/session (stream). The sheet may be:

- an object `{sessions: [...] , printer: ..., camera: ..., license: ..., ...}` (inherited defaults)
- or a bare list of sessions

Session fields (representative):

| Field | Required | Notes |
|------|----------|-------|
| `id` | yes | unique within sheet |
| `label` | yes | one of `normal`, `failing`, `uncertain` |
| `split` | no (default `calibration`) | one of `calibration`, `validation`, `holdout` (aliases `locked_holdout`, `locked` → `holdout`) |
| `source` | conditional | local video path or directory of frames; required **unless** `scores` are precomputed. |
| `scores` | conditional | list `[{timestamp_s, score}, ...]` where `score = P(failing)` (0..1); allowed when source not provided for fast unit tests. |
| `failure_type` | for failing | must be in `tv.envdata.schema.FAILURE_TYPE_VOCAB`; unknown → error; `normal` with failure_type → warning. |
| `onset_s` | recommended | onset time; missing for failing → warning; `normal` with onset → warning; negative or > duration → error. |
| `duration_s` | recommended | session duration; used for onset range check. |
| `printer`/`camera`/`license`/`provenance`/`redistributable`/`notes` | optional | inherited to sessions if present on sheet/root. |
| `md5` | optional | declarative byte hash; used for duplicate detection (across/within splits). |
| `fps`,`width`,`height` | optional | metadata (not required for validation). |

The canonical vocabulary comes from `tv.envdata.schema` (labels/splits and `FAILURE_TYPE_VOCAB`). The intake layer reuses `tv.envdata.validate` for video probing, media hashing and duplicate/timelapse heuristics.

## Validation rules (fail-closed)

`tv.sessions.validate_sessions(sheet, probe=False)` emits flags with severity `error`/`warning`/`info`. The intake **fails closed** (exit 1) if any error is present; it **passes** (exit 0) if only warnings/info exist.

Key checks:

- **Structural:** missing id/duplicate id; invalid label/split; missing `source` *and* missing `scores`; `bad_scores`.
- **License/provenance basics:** missing `license` → error; `external_source` → info (still allowed but flagged).
- **Uncertain:** `uncertain_label` → warning; uncertain sessions are **excluded** from calibrate specs and from held-out reports (they cannot contaminate the locked holdout).
- **Failure semantics:** failing missing `failure_type` → warning; unknown failure_type → error; `normal_has_failure_type` / `normal_has_onset` → warning; failing missing `onset_s` → warning; `onset_out_of_range` (negative or > duration) → error.
- **Media:** with `probe=True`, local `source` is probed. `missing_file`/`unreadable_video` → error. `metadata_mismatch` (declared vs probed duration/fps/width/height) → warning. `file_md5` is recorded when probed; declared `md5` mismatches produce `metadata_mismatch` (warning) and do not override detection.
- **Duplicates/independence:** `duplicate_media_across_splits` → **error** (byte-identical media found in different splits) — this is leakage and must block intake. `duplicate_media` (same split) → warning. `near_duplicate` (same md5 or heuristic) → warning.
- **Split-class presence (gate):** `split_missing_class` (calibration has only one class) → error. `no_validation_split` → info (operating point will be reported on locked holdout only). `validation_missing_class` → warning. **`missing_locked_holdout`** → **error** (fail closed — cannot validate on an independent held-out stream). **`contaminated_holdout`** → **error** (locked holdout contains non-normal/failing, e.g. uncertain). **`holdout_missing_class`** (holdout has only normal or only failing) → error.
- **Printer/camera:** `missing_printer` → warning; `multi_printer` → **error** (all sessions must be for the declared printer/camera setup; cross-printer must not be mixed into one sheet); `multi_camera` → warning.
- **Environment leakage:** the sheet-level validator focuses on media/semantics; `envdata.split.check_environment_leakage` remains the environment-level check when working with environment manifests (separate tooling, unchanged).

## Conversion → calibrate spec

`build_calibrate_spec(sheet)` produces a `tv.calibrate`-compatible spec (the same shape `load_clip_spec` expects):

- calibration → `split: calibration` (unchanged)
- validation → `split: holdout` (**mapped**, never used for selection; reported as the "holdout tier" in calibration output)
- locked `holdout` → **excluded entirely**
- `uncertain` → **excluded entirely**

The emitted spec is intentionally minimal (drops fields not needed for scoring/aggregation). `write_calibrate_spec` supports YAML/JSON. The spec is round-trippable through `tv.calibrate.load_clip_spec` and runnable with `tv.calibrate.calibrate`.

## Held-out evaluation on the locked holdout

`heldout_report(sheet, scored_sessions, threshold, confirm)` scores only sessions with `split == "holdout"` (locked). For each such session it uses `scores` if provided, else computes via model; decisions are stream-level (median score ≥ threshold; alarm only if `confirm` consecutive frames ≥ threshold — same convention as `tv.calibrate._aggregate`). Outputs include `n_normal`, `n_failing`, `detection_rate`, `false_alarm_rate_per_print`, per-session rows, and counts of usable/held-out sessions.

`emit_holdout_csv(sheet, path)` writes a minimal CSV of locked-holdout sessions (`session,source,label,failure_type,onset_s,printer,camera,md5,license`) for release-gate artifact collection.

## Artifacts & locations

Everything written by intake automation is **outside Git**. Typical structure under `--data-dir`:

| Path | Contents |
|------|----------|
| `intake_report.json` | validation report (errors/warnings/info, counts, codes/messages) **plus** `sheet`: the full session inventory (id, label, split, role, evidence, md5, provenance) |
| `calibration-spec.yaml` | calibrate spec emitted from the sheet (calibration + validation→holdout; locked holdout excluded) |
| `calibration.json` | full `tv.calibrate` output. **Its `holdout` tier is the intake *validation* split** (calibrate's naming); threshold selection (`config.select_on == "calibration"`) uses calibration clips only. |
| `validation_report.json` | schema `tinyvision-session-validation`, role `independent_reporting_only` — the clearly-labelled, independent validation result. **Not** the locked holdout. |
| `holdout_eval.json` | schema `tinyvision-session-holdout`, role `locked_holdout` — the **final** held-out result at the fixed threshold (with model identity, settings and per-session roles) |
| `holdout_sessions.csv` | locked holdout sessions (release-gate input) |
| `run_manifest.json` | run-level identity/settings/provenance: spec path, checkpoint, model info, settings, full session inventory, validation vs locked-holdout summaries, environment (python/torch/git HEAD), artifact list |
| `poc/` | per-video `tv poc` reports (one JSON per session with a local `source`), **calibration/validation only** — the locked holdout is never opened here |

The `run_session_intake.py` script defaults/forces writing under `--data-dir`; never attempts to write into the repository. Checkpoints, videos, caches and reports must remain under the data area (ignored by Git).

### Inspect the final report

```bash
# The final held-out result (this is the number that matters):
python3 -c "import json,sys; r=json.load(open('OUT/holdout_eval.json')); \
print('locked holdout @', r['threshold'], '->', r['sessions']); \
print('detection', r['detection_rate'], 'false-alarms/print', r['false_alarm_rate_per_print'], \
'roles', [s['role'] for s in r['sessions']])"

# Everything about the run (identity, settings, provenance, separation):
python3 -m json.tool OUT/run_manifest.json
```

### Validation vs the locked final evaluation

These are **different splits and must not be conflated**:

- The **validation** split is read during calibration *only to be reported*; it never influences threshold selection. It appears as `validation_report.json` (and, confusingly, as the `holdout` tier inside `calibration.json` — that is `tv calibrate`'s name for "the independent tier", not the locked holdout).
- The **locked holdout** is excluded from the calibrate spec and from `poc/`, and is opened **exactly once**, after the threshold is fixed, producing `holdout_eval.json`. This is the final held-out result.

The automation relabels calibrate's `holdout` tier as `[validation]` in its console output and writes a separate `validation_report.json` so the two can never be mistaken for one another.

## CLI

Module: `tv.cli`.

```text
tv sessions validate --spec SHEET.yaml [--probe] [--output-json PATH]
  - exits 2 if SessionSheetError (bad load/format)
  - exits 1 if any validation errors exist
  - exits 0 if passes or only warnings/info

tv sessions build --spec SHEET.yaml --out CALIB_SPEC.yaml [--checkpoint CKPT] [--probe]
  - validates first (fail-closed) → exit 1 if not pass, no spec written on failure
  - builds calibrate spec, optionally injects checkpoint, writes YAML (or JSON if .json)
  - exits 2 on write error
```

For `tv sessions validate|build`, the `--probe` flag forces media probing and MD5 computation; without it only declared metadata is checked (use `--probe` for real footage). The `run_session_intake.py` automation probes by default (`--no-probe` to disable).

## End-to-end automation

`scripts/run_session_intake.py` runs the full path deterministically:

1. `load_session_sheet` + `validate_sessions(sheet, probe=...)` → **fail closed**, write `intake_report.json` (with the session inventory). Probing is **on by default** (disable with `--no-probe`).
2. `build_calibrate_spec` + `write_calibrate_spec` → `calibration-spec.yaml` (locked holdout excluded).
3. Optional `tv poc` per session with a `source` **for calibration/validation only** (skippable with `--skip-poc`) → `poc/*.json`. The locked holdout is never opened here.
4. `tv calibrate` on the loaded spec → `calibration.json` (threshold selected on **calibration clips only**); the console summary relabels the calibrate `holdout` tier as `[validation]`, and a clearly-labelled `validation_report.json` is written.
5. Only now, with the threshold fixed, the **locked holdout** is scored (reusing precomputed `scores` or the model) and `heldout_report` → `holdout_eval.json` + `holdout_sessions.csv`.

A `run_manifest.json` records identity, settings, roles and provenance for the whole run.

**Ordering guarantee:** the locked holdout is not opened (no `poc`, no scoring) until after the threshold is fixed. If no threshold can be selected, the locked holdout is *not* evaluated (`[5/5] ... locked holdout NOT evaluated`).

**Exit codes (fail-closed).** The script returns non-zero and writes no downstream artifacts on any failure:

- `1` — intake validation failed (missing files/unreadable media, invalid labels, insufficient class coverage, cross-split leakage, missing/contaminated locked holdout, …).
- `2` — unparseable sheet, bad `--thresholds`, or failed inference/scoring (e.g. a video that cannot be decoded, a missing `--checkpoint` when needed).
- `0` — the run completed; inspect `holdout_eval.json` and `run_manifest.json`.

Important: the script does **not** change gate policy. It produces release-gate inputs; the release gate itself (`tv.cli reliability evaluate`) remains unchanged and must be re-run with those inputs only when policy/data change.

## Real-data handoff

### Minimum required footage

One sheet is **one printer/camera setup** (mixing printers is a hard error). At minimum a useful evaluation needs:

- **calibration**: ≥2 normal sessions and ≥2 failing sessions (fewer works but the threshold and its error bars are unreliable; ≥1 normal and ≥1 failing is the hard floor).
- **validation** (recommended): ≥1 normal and ≥1 failing independent session.
- **locked holdout** (required, fail-closed): ≥1 normal and ≥1 failing independent session.

Each **session = one complete print/print job** recorded start to finish (do not split one print into sessions, do not mix prints into one session). Sessions must not share media (byte-identical media across splits is a hard error). Record `duration_s` (or let `--probe` measure it) and, for failing sessions, an approximate `onset_s`.

### Labelling instructions

- `label`: `normal` for a print that completed correctly; `failing` for a print that failed; `uncertain` if a human cannot decide (uncertain sessions are kept but excluded from every split).
- `failure_type` (failing only): one of the `tv.envdata.schema.FAILURE_TYPE_VOCAB` values (e.g. `spaghetti`, `stringing`, `under_extrusion`, `warping`, `layer_shift`, `bed_adhesion`).
- `onset_s` (failing): approximate seconds from session start to the first visible failure. If unknown, omit it (a warning is recorded).
- Labels must be human-verified; `provenance`/`license` must be recorded. Never infer a label from the model being evaluated.

### Exact commands

```bash
# 0. (optional) pre-flight the sheet, printing a readable report
training/.venv/bin/python -m tv.cli sessions validate \
  --spec footage/sessions.yaml --probe --output-json ~/tinyvision-data/session-intake/intake_report.json

# 1. build the calibration spec only (fail-closed; writes nothing on failure)
training/.venv/bin/python -m tv.cli sessions build \
  --spec footage/sessions.yaml --out ~/tinyvision-data/session-intake/calibration-spec.yaml \
  --checkpoint experiments/rc1_release_candidate_shufflenet_v2_x1_0/artifacts/best.pt --probe

# 2. the full workflow (validate -> poc -> calibrate -> validation -> locked holdout)
training/.venv/bin/python scripts/run_session_intake.py \
  --spec footage/sessions.yaml \
  --data-dir ~/tinyvision-data/session-intake \
  --checkpoint experiments/rc1_release_candidate_shufflenet_v2_x1_0/artifacts/best.pt
```

All generated artifacts land under `--data-dir` (**outside Git**). Probing is on by default in the automation; add `--no-probe` to skip it. If every session already carries precomputed `scores`, `--checkpoint` is not required.

## Examples (minimal)

Valid sheet with real media (one complete print per session):

```yaml
dataset_id: my_printer_2026
printer: X1C-01
camera: rpi-cam-v3
license: own footage
provenance: human-labelled 2026-10
sessions:
  - {id: c_n1, label: normal,    split: calibration, source: footage/c_n1.mp4}
  - {id: c_n2, label: normal,    split: calibration, source: footage/c_n2.mp4}
  - {id: c_f1, label: failing,   split: calibration, source: footage/c_f1.mp4,
     failure_type: stringing, onset_s: 143.0, duration_s: 600.0}
  - {id: v_n1, label: normal,    split: validation,  source: footage/v_n1.mp4}
  - {id: v_f1, label: failing,   split: validation,  source: footage/v_f1.mp4,
     failure_type: spaghetti, onset_s: 88.0, duration_s: 540.0}
  - {id: h_n1, label: normal,    split: holdout,     source: footage/h_n1.mp4}
  - {id: h_f1, label: failing,   split: holdout,     source: footage/h_f1.mp4,
     failure_type: spaghetti, onset_s: 61.0, duration_s: 480.0}
```

Equivalent sheet with precomputed per-frame scores (no media needed; used in tests):

```yaml
printer: P1
camera: C0
license: own footage
sessions:
  - {id: c_n1, label: normal,  split: calibration, scores: [{timestamp_s: 0, score: 0.05}]}
  - {id: c_f1, label: failing, split: calibration, failure_type: stringing, onset_s: 1, duration_s: 5, scores: [{timestamp_s: 0, score: 0.9}]}
  - {id: v_n1, label: normal,  split: validation,  scores: [{timestamp_s: 0, score: 0.04}]}
  - {id: v_f1, label: failing, split: validation,  failure_type: stringing, onset_s: 1, duration_s: 5, scores: [{timestamp_s: 0, score: 0.85}]}
  - {id: h_n1, label: normal,  split: holdout,     scores: [{timestamp_s: 0, score: 0.02}]}
  - {id: h_f1, label: failing, split: holdout,     failure_type: spaghetti, onset_s: 2, duration_s: 5, scores: [{timestamp_s: 0, score: 0.8}]}
```

Run:

```bash
training/.venv/bin/python scripts/run_session_intake.py \
  --spec footage/sessions.yaml \
  --data-dir ~/tinyvision-data/session-intake \
  --checkpoint experiments/rc1_release_candidate_shufflenet_v2_x1_0/artifacts/best.pt
```

## Safety notes

- **Per-printer only.** A calibrated threshold for one printer/camera does not establish cross-printer generalization. The intake enforces single-printer in one sheet (`multi_printer` → error).
- **Independence is mandatory.** Byte-identical media across splits is a hard error (`duplicate_media_across_splits`). Never share print time ranges between splits.
- **Locked holdout is sacrosanct.** Missing or contaminated locked holdout fails closed; validation sessions never leak into the locked holdout and are reported separately as the calibrate holdout tier.
- **No weakening of gates.** Intake produces inputs; it never changes `reliability` policy, evaluation criteria, or thresholds. The release gate must be re-run with criteria unchanged.
- **No vision/VLM, no fabrication.** The agent cannot visually verify footage; probe is structural/readability (frames count, md5, basic sanity). Labels and independence remain human responsibility. Never fabricate results.
- **Outside Git.** Large media/reports/checkpoints stay outside the repository; `datasets/external/**` deletions remain preserved (read-only via git show).
- **Reuses existing code.** Intake is a thin layer: `sessions.py` delegates to `tv.envdata.{schema,validate}` and `tv.calibrate` semantics; no modifications to calibration/aggregation/report schema.
