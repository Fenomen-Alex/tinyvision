# Stage 13: Target-Domain Normal-Print Training Augmentation Dataset

Created: 2026-09-07

## Goal

Augment the frozen `printing-errors` training set with a small, high-quality, *source-grouped* set of verified-normal target-domain frames. This dataset is used for Experiment 3: testing whether the model's persistent normal-print false positive problem on independent live footage is caused by insufficient diversity of normal training data.

## Selection criteria

- Source independence: every video/source is a single, undivided unit. No frames from one video are split across train and eval.
- Source-grouping: two strong, distinct sources held out entirely for external normal evaluation.
- Temporal spacing: samples are 30-35 s apart; near-duplicate consecutive frames are not included.
- Per-frame screening: every sampled frame reviewed individually with a strict, neutral, single-frame VLM prompt (qwen3-vl-8b). Frames accepted only if `decision==normal AND part_visible==True`.
- Human verification status: `pending_human_review` for all frames. A human pass must be completed before any public-facing release.
- License: Standard YouTube License (not redistributable); eval-only / internal experiment use only.

## Source summary

| source_id | role | printer | frames | pass | license |
|-----------|------|---------|--------|------|---------|
| DHpo4Za2--M | train | Ender 3 Pro | 89 | YouTube |
| LwHaHzVV_iE | train | Creality K1 | 46 | YouTube |
| PscyiwiZ9Mc | train | unknown | 60 | YouTube |
| R0xnULlObrs | train | Anycubic Chiron | 40 | YouTube |
| kZ06bpskj9I | train | Creality K1 Max | 55 | YouTube |
| uCCXJzvYxKA | train | Prusa Core One L | 26 | YouTube |
| zFX80aMaYFg | train | unknown (Logitech C270 webcam) | 17 | YouTube |
| LY3jlcO34f4 | eval_holdout | unknown | 60 | YouTube |
| zkJgLanCSUw | eval_holdout | unknown | 60 | YouTube |

## Frame counts

- **Train frames:** 333 (from 7 sources)
- **Eval holdout frames:** 120 (from 2 sources)
- **Selection rule:** decision==normal AND part_visible==True

## Per-source train frame counts

- DHpo4Za2--M: 89
- LwHaHzVV_iE: 46
- PscyiwiZ9Mc: 60
- R0xnULlObrs: 40
- kZ06bpskj9I: 55
- uCCXJzvYxKA: 26
- zFX80aMaYFg: 17

## Human verification status

All frames in this dataset are VLM-screened only (`pending_human_review`). The multi-signal screening pipeline is:

1. Source design: unedited, fixed-camera, real-time webcam/live streams of FDM prints.
2. Continuity analysis: adjacent-frame structural diff confirms continuous real-time footage (no timelapses or edits).
3. Per-frame VLM review: strict, neutral prompt requiring `decision==normal AND part_visible==True`; ambiguous frames excluded.
4. Selection: only frames passing all three criteria are included.

A human pass via `scripts/run_human_review.py build|review|merge` must be run before any release.

## Known limitations

- YouTube Standard License; not redistributable.
- All sources are from the same general FDM printer ecosystem; printer diversity is moderate (Ender, Creality K1, Anycubic, Prusa, plus two unknown-printer webcam streams).
- zFX80aMaYFg frames often show the print indirectly via a wide Logitech C270 webcam; `part_visible` is true but the part is small in the frame.
- Frame counts per source range from 23 to 99; some sources contribute relatively few frames.

## Files

- `manifest.json`: full per-frame manifest with provenance.
- `stage13-normal-dataset.md`: this document.
- `screening/samples/<id>/{pass_a,b}/`: source sample frames.
- `screening/vlm_proposals.json`: VLM verdicts for all sampled frames.
- `candidates/*.json`: source metadata (title, channel, resolution).

## Derived splits for Experiment 3

- `datasets/processed/external-aug-exp3-normal/splits/train.csv`
- `datasets/processed/external-aug-exp3-normal/splits/val.csv`
- `datasets/processed/external-aug-exp3-normal/splits/test.csv`
- `datasets/processed/external-aug-exp3-normal/derived_manifest.json`

Train.csv = frozen printing-errors train (5,863 failing + 5,391 normal) + 333 verified-normal target-domain frames (absolute paths). Val/test are byte-identical copies of the frozen printing-errors splits.
