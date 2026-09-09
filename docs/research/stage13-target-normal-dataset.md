# Stage 13: Target-Domain Normal-Print Attention Dataset

Status: acquisition complete; labels pending human review
Created: 2026-09-07
Machine-readable record: [stage13-target-normal-dataset.json](stage13-target-normal-dataset.json)
Related: [stage13-experiment.md](stage13-experiment.md),
[exp3-target-normal-design.md](exp3-target-normal-design.md)

## Purpose

A small, source-grouped augmentation set of verified-normal FDM printing frames
to test whether the frozen TinyVision binary classifier's catastrophic
false-positive rate on independent normal footage is caused by insufficient
diversity of normal target-domain training data (Experiment 3).

## Sources (9 independent continuous real-time prints)

| source | printer (when known) | role | resolution/fps | frames | channel |
|--------|----------------------|------|----------------|--------|---------|
| DHpo4Za2--M | Ender 3 Pro | train | 1920x1080/30 | 89 | Szabó Dénes |
| LwHaHzVV_iE | Creality K1 | train | 1920x1080/25 | 46 | Creality (official) |
| PscyiwiZ9Mc | unknown | train | 1280x720/30 | 60 | XIMBRIUM |
| R0xnULlObrs | Anycubic Chiron | train | 1920x1080/30 | 40 | Nawar |
| kZ06bpskj9I | Creality K1 Max | train | 1920x1080/30 | 55 | Shy |
| uCCXJzvYxKA | Prusa CORE ONE L | train | 1142x720/15 | 26 | BMAN_PRINTS |
| zFX80aMaYFg | unknown (Logitech C270) | train | 640x480/30 | 17 | Evel-Knievel |
| LY3jlcO34f4 | unknown | eval holdout | 1280x720/30 | 60 | The Civilized Barbarian |
| zkJgLanCSUw | unknown | eval holdout | 1920x1080/30 | 60 | Martinez |

Total: **333 train frames (7 sources) + 120 held-out eval frames (2 sources)**.

Excluded after screening: `EnhYvvWLozE` (Prusa, mostly idle print bed),
`xbZC4e6LY48` (too dark/blurry), `Bl3ivR13SYs` (VLM uncertain 48/48).

## Acquisition and validation protocol

1. Source selection: unedited, fixed-camera, real-time webcam/live streams of
   FDM prints (no timelapses, no edits, no review/narration, print region
   visible).
2. Downloads: `yt-dlp -f "bv*[height<=720]/b" --write-info-json`, 2026-09-07,
   into `datasets/external/stage13-normal/raw/<id>/`.
3. Continuity screening: adjacent-frame structural difference over sampled
   windows confirms every source is continuous real-time footage (no jump
   edits; all `jumpfrac=0.00`).
4. Per-frame sampling: two offset passes (35 s and 50 s offsets, one sample
   every ~30 s thereafter), capped at 60 frames/source; samples stored
   `screening/samples/<id>/{pass_a,b}/sNN.jpg` with structural coverage.
5. Per-frame VLM review: local `qwen/qwen3-vl-8b`, strict neutral single-frame
   prompt, one call per frame. A frame is selected only if
   `decision==normal AND part_visible==True`. `uncertain`/`failing`/part-not-
   visible frames are excluded. Proposal-only storage in
   `screening/vlm_proposals.json` (nothing merged into any verified labels).
6. Source-grouping: every source belongs entirely to one role (7 train / 2
   held-out eval). No frame-level splitting, so no leakage.
7. Human verification: recorded as `pending_human_review`. Frame labels are
   VLM-screened, not human-verified; a human pass
   (via `scripts/run_human_review.py`) must complete before release use.

## Licensing

All sources are Standard YouTube License; **not redistributable**. Batch
metadata (title, channel, upload date, resolution, fps) captured by yt-dlp at
retrieval time and stored in `candidates/<id>.json`. For evaluation and
experiment use only.

## Files

* `manifest.json` — full per-frame manifest (provenance, time, pass, coverage,
  decision, part_visible, source, role).
* `stage13-normal-dataset.md` (this document) + `.json` (machine-readable).
* `screening/samples/<id>/{pass_a,b}/sNN.jpg` — sampled frames.
* `screening/vlm_proposals.json` — VLM proposals (opinions, not ground truth).
* `screening/manifest.json`, `continuity -- source_screening.json`.
* Derived split (Exp3): `datasets/processed/external-aug-exp3-normal/`
  — train.csv = frozen printing-errors train + 333 normal frames (absolute
  paths); val/test are byte-identical frozen copies.

## Known limitations

* VLM-screened, pending human review.
* 24-source-to-frame compression: sources with unknown printers or wide
  webcam views (zFX80aMaYFg) contribute thin but still valid normal frames.
* Only fixed-camera DLT side views; no nozzle-cam normal frames.
* Not redistributable (Standard YouTube License).
* Held-out eval sources represent 2 printers unknown; per-source FPR spread is
  large (0.10 vs 0.42) — see experiment report.