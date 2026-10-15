# Dataset: Kaggle Early detection of 3D printing issues

**Source:** [early-detection-of-3d-printing-issues](https://www.kaggle.com/competitions/early-detection-of-3d-printing-issues)
**Dataset name:** early-detection-of-3d-printing-issues (competition)
**License:** Subject to Competition Rules (not open). Treat as **research-only**, **non-redistributable**. (`noncommercial=True`, `redistributable=False`)
**Attribution:** Kenneth Jiang / Obico — "Early detection of 3D printing issues" Kaggle competition
**Retrieved:** 2026-10-14
**Auth:** Kaggle CLI 2.2.4 (OAuth), user mygames8888; competition rules accepted (`userHasEntered: True`)

## Data description
Label: print-level binary `has_under_extrusion` (1 = failure/under-extrusion, 0 = good/normal print). All snapshots within a print share the same label. No print has mixed labels. A `print_id` is a single print session.

Files:
- `train.csv` (81,060 rows): `img_path, printer_id, print_id, has_under_extrusion`. `img_path` format: `<printer_id>/<print_id>/<timestamp>.jpg`.
- `test.csv` (25,279 rows): `img_path, printer_id, print_id` (unlabeled).
- `sample_submission.csv` (25,279 rows): `img_path, has_under_extrusion`.
- `images/` (~12.29 GB, 106,339 jpgs): `images/<printer>/<print>/<timestamp>.jpg`.

Printers: 7 total (021,022,023,101,102,103,104). Train uses 022,101,102,103,104 (5). 021,023 appear only in test (withheld). Test also includes withheld prints from 101–104.

## Train split (labeled)
52 prints total:
- normal (0): 27 prints
- failing (1, under-extrusion): 25 prints

Per printer (train):
- 022: 17 prints (10 normal, 7 failing), 7,492 imgs (env val)
- 101: 9 prints (4 normal, 5 failing), 20,205 imgs (train)
- 102: 6 prints (2 normal, 4 failing), 22,670 imgs (train)
- 103: 8 prints (4 normal, 4 failing), 9,927 imgs (test env)
- 104: 12 prints (7 normal, 5 failing), 20,766 imgs (train)

Frames/print: min 107, max 4310, median 1227.5.

## Leakage safety (critical)
- **Print-level consistency:** no print has mixed labels → grouping by `print_id` is leak-safe.
- **Independent streams:** each print_id = one independent stream (snapshots ~0.5s apart; adjacent clips from one print do not count as independent). Different prints = independent.
- **Environment-held-out splits:** grouped by `printer_id` (environment). No printer appears in >1 split:
  - train: 101,102,104 (27 streams: 13 normal, 14 failing? 101:4N/5F,102:2N/4F,104:7N/5F => 13N,14F; 63,641 frames)
  - val: 022 (17 streams: 10 normal, 7 failing; 7,492 frames)
  - test: 103 (8 streams: 4 normal, 4 failing; 9,927 frames)

## Domain / limitations
- Nozzle-camera snapshots (first-layer-like view). Distinct from bed-camera (EmptyDeck).
- Single failure mode represented (under-extrusion). No normal class in image annotations (print-level only).
- Test prints are unlabeled (competition holdout); cannot be used as verified normal/failing evidence for FAR estimation.

## Registered manifest (acquired 2026-10-14)
- Path: `/Users/alex/tinyvision-data/kaggle-early-detection/manifest.json`
- Splits: `/Users/alex/tinyvision-data/kaggle-early-detection/splits/{splits.json,train.csv,val.csv,test.csv}`
- Index: `/Users/alex/tinyvision-data/kaggle-early-detection/index.csv`
- Metadata: `/Users/alex/tinyvision-data/kaggle-early-detection/kaggle-early-detection-metadata.json`
- Streams: 52 labeled (27 normal, 25 failing) from 5 printers; environment-held-out (no printer spans splits).
- Validation: `validate_manifest` OK (0 warnings); `build_streams` OK; environment disjointness verified.
