# Stage 14 — Target-domain Failure Dataset

## Purpose

A small, source-grouped, VLM-verified target-domain **failure** image set used
to balance the Stage-13 normal augmentation in Experiment 4. The goal was to test
whether adding ~100-300 real target-domain failure frames restores failure
sensitivity lost by the Stage-13 normal bias, while retaining the large normal-FP
improvement.

## Sources

Two independent fixed-camera YouTube videos (sub-second-dense frames within
confirmed failure windows):

| source_id | title | channel | printer | failure type | frames |
|-----------|-------|---------|---------|--------------|--------|
| `ZYfXW0JN2k4` | 3d Print Fail Caught On Camera | Inside The Mind Of Matt | unknown (blue filament) | spaghetti / blob | 70 |
| `3zXQrw0E-Ag` | Bambu A1 Blob of Death: What Happened & How I Fixed It | Diaz Creative Studio | Bambu Lab A1 | blob of death | 30 |
| **total** | | | | | **100** |

- Spaghetti is required (≥1 source): provided by `ZYfXW0JN2k4`.
- Both sources use a **fixed / mostly-fixed camera** viewing the whole print
  (partial view), distinct from the nozzle-closeup clips that were rejected.
- The two printers are different (unknown blue-filament printer vs Bambu Lab A1),
  satisfying the source-diversity requirement.

## Provenance attributes

| attribute | value |
|-----------|-------|
| name | stage14-failure |
| source | YouTube (2 independent channels) |
| URL | `https://www.youtube.com/watch?v=ZYfXW0JN2k4`, `https://www.youtube.com/watch?v=3zXQrw0E-Ag` |
| license | **unknown** (Standard YouTube License; not redistributable) |
| image count | 100 |
| annotation format | per-frame failure type + magnitude in `manifest.json`; `pending_human_review` |
| classes | failing (spaghetti, blob) |
| resolution | 1280x720 (both) |
| collection methodology | fixed-camera, continuous real-time footage |
| known limitations | see below |
| intended usage | Stage-14 Exp4 training augmentation (target balancing) |
| retrieval date | 2026-09-07 |

## Label authority

Every frame's label was produced by a **neutral per-frame Qwen3-VL review**
(`qwen/qwen3-vl-8b`) during window verification, and only frames in a contiguous
window where the reviewer consistently saw failure are included. Boundary frames
verified as *healthy* / *idle* / *no-print-visible* were **excluded** (not left
mislabeled). Status is `pending_human_review`: a human pass is mandatory before
any release use.

### Clean contiguous failure windows (VLM-verified)

| source | window | seconds | frames |
|--------|--------|---------|--------|
| `ZYfXW0JN2k4` | main | t=98-156 | 59 (spaghetti→blob development) |
| `ZYfXW0JN2k4` | final | t=221-231 | 11 (spaghetti re-episode) |
| `3zXQrw0E-Ag` | blob | t=142-171 | 30 (blob of death) |

Frames are sampled 1-2 s apart within windows, so near-duplicates are **intentional**
(to train on continuous failure evolution), documented as a caveat.

## Rejected candidates (screening)

| source | reason |
|--------|--------|
| `24dukPzuBAM` (GXS 3D benchy "FAILED") | no visible failure anywhere; normal print through t=2260; same rig as other GXS clips |
| `4N1XUyW39YY` (GXS 3D bridge fail) | nozzle close-up view; no failure; same rig as other GXS clips |
| `kWK3pUjccrQ` | nozzle close-up; no visible failure; very low fps (possibly screen recording) |
| `O7dEPsU6-pQ` | nozzle close-up (static); only mild stringing/mixed |
| `4k980nMcRys` | explainer/edited, no whole-print failure footage |
| `X-dQ2r0YI4Y` | screen recording of detection UI; no usable footage |
| `vroX8UuUkRw` | vertical Short, multi-scene edited timelapse |
| `A-B2A8SXqwA` | 80-min repair tutorial, one blob frame, mostly talking |
| `RRSRCzolhmM` | plugin review, zero failure footage |

## Files

- `raw/<id>/` — source video + `info.json` provenance
- `screening/candidate_meta.json` — per-candidate adjacent-frame-diff stats
- `screening/candidate_vlm_proposals.json` — coarse 12-frame-per-video screening
- `screening/window_verification*.json` — dense per-frame failing/healthy boundary checks
- `frames/<id>/*.jpg` — extracted frames, sub-second spaced
- `manifest.json` — per-frame catalog with `pending_human_review`
