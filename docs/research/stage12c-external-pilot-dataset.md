# Stage 12C — External Pilot Dataset Build

**Date:** 2026-09-05
**Status:** Built a provisional external pilot dataset by converting the six acquired videos into deterministic frames with provisional labels, deduplication, and source-level splits. **Research only. No training.**

---

## 1. Objective

Convert the six videos acquired in Stage 12B (SP001, SP002, NORMAL001, UE001, UE003, STR001) into a clean, reproducible external **pilot** dataset under `datasets/external/video-frames/`:

1. Deterministic 1–2 FPS frame extraction (integer-second grid on the Stage 12B label grid).
2. Provisional per-frame labels (`normal` / `failing` / `uncertain`) via the documented `LABEL_MAP`.
3. Per-frame metadata + provenance.
4. Deduplication that removes **only exact byte-identical duplicates**; phash near-duplicates are a **diagnostic report** (never auto-removed), so temporal coverage of continuous footage is preserved.
5. Source-level (atomic) train/validation split by deterministic `sha256(source_id) % 100 < 20`.
6. Statistics and a diversity analysis vs the Kaggle reference data.

**Hard constraints honored (no violation):**
- No training run.
- No modification of `datasets/processed/printing-errors/` (Kaggle reference, used read-only).
- No modification of the frozen official test set.
- No additional downloads beyond the six acquired videos.
- No external test set is constructed in this stage.
- Source video = minimum split unit (no frame of a source appears in both splits).
- No fabricated visual claims; labels are provisional and not human-verified ground truth.

**Critical capability limitation (must be stated prominently):** this agent has **no image-input (vision) capability**. It cannot visually confirm a failure type in any frame. All labels below are derived from **Stage 12B deterministic structure analysis** (frame-difference, cut detection, OCR, foreground-growth trend) and are therefore **provisional and NOT human-verified ground truth**. Human review of the contact-sheet montages remains mandatory before any pilot label is treated as true.

---

## 2. Method (reproducible)

- **Module:** `training/tv/external_pilot.py` (`probe_media`, `frame_grid`, `extract_sequence`, `build_frame_records`, `map_label`, `phash`, `hamming`, `find_exact_duplicates`, `near_duplicate_report`, `assign_split`, `compute_statistics`, `build_external_pilot`).
- **CLI:** `scripts/build_external_pilot.py`, `scripts/analyze_external_pilot.py`.
- **Extraction:** single-frame ffmpeg `-ss <t> -frames:v 1` seek per integer timestamp `t` in the 1–2 FPS grid. Frame `k` ↔ timestamp `t = grid[k]` seconds. This exactly matches the Stage 12B label grid (integer seconds), avoiding the ffmpeg rate-filter dropping the final partial second of a clip. `source_frame_index = round(t * source_fps)`; `segment_id = "<sid>__s0"`; canonical labeled name `f"{sid}__t{t:07.3f}__f{src_idx:06d}.jpg"`.
- **FPS per source:** 1 FPS for all six (kept strictly on the Stage 12B labeled grid).
- **Labels:** `map_label` applied through `LABEL_MAP` — `normal_provisional`→`normal`, `suspect_growth`→`failing`, `uncertain` and `scene_transition_or_final_state`→`uncertain`. Uncertain is never silently converted to a usable class; uncertain frames are preserved in metadata but excluded from the training-ready `labeled/` folders.
- **Deduplication:** `find_exact_duplicates` removes an instance iff an earlier kept record shares the **same md5** (first occurrence kept). `near_duplicate_report` computes phash Hamming distance (threshold 4) for **diagnostics only** and does not remove anything.
- **Split:** `assign_split` — `sha256(source_id) % 100 < 20` → validation, source-atomic.

---

## 3. Extraction results

| Source | FPS | Extracted frames | Kept (post-md5-dedup) | Exact-md5 removed |
|---|---|---|---|---|
| NORMAL001 | 1 | 21 | 21 | 0 |
| SP001 | 1 | 26 | 26 | 0 |
| SP002 | 1 | 55 | 53 | 2 |
| STR001 | 1 | 646 | 642 | 4 |
| UE001 | 1 | 520 | 437 | 83 |
| UE003 | 1 | 368 | 367 | 1 |
| **Total** | | **1,636** | **1,546** | **90** |

Raw frames live in `extracted/<sid>/sequence/`; training-ready labeled copies in `labeled/normal/` and `labeled/failing/`.

---

## 4. Labels

Applies the Stage 12B provisional labels (only SP001 and NORMAL001 carry frame labels) through the documented mapping.

| Label | Count | Sources |
|---|---|---|
| `normal` | 21 | NORMAL001 (t=0–20) |
| `failing` | 15 | SP001 (t=8–22) |
| `uncertain` | 1,510 | all others (unlabeled sources / SP001 uncertain and scene-transition) |
| **Total kept** | **1,546** | |

**Failure-type inventory (preserved per source, all unverified):**

| Failure type | Sessions | Frames (kept) |
|---|---|---|
| normal | 1 | 21 |
| spaghetti | 2 | 79 |
| stringing | 1 | 642 |
| under_extrusion | 2 | 804 |

Per-frame `provenance.verification = "provisional_unverified"` throughout. Frames from sources without frame labels (SP002, STR001, UE001, UE003) are `uncertain` with no usable copies.

---

## 5. Deduplication

- **Removed:** **90 exact-md5 duplicates across 29 groups.** These are repeated/frozen frames within a source — e.g. UE001's 83 repeats (frozen/tutorial stills), STR001's 4, SP002's 2, UE003's 1. Removing them prevents duplicate-frame inflation in downstream statistics.
- **Not removed (per instruction):** phash near-duplicates that are **within-video temporal neighbors** are preserved to avoid destroying temporal coverage. The diagnostic found **3,547 within-source near-duplicate pairs** (expected for continuous footage) and **23 cross-source pairs** — all involving near-black / end-card / slicer-screen frames in the *unusable/uncertain* sources (SP002 t=2,52; STR001 t=175,243,641; UE001 t=7,29; UE003 t=366). No cross-source pair involves any labeled normal/failing frame, and none are exact duplicates. **Policy: diagnostic only; no removal.**

Full detail in `metadata/dedup_removed.json`.

---

## 6. Source-level split

Deterministic `sha256(source_id) % 100 < 20 → validation`:

- **external_train:** `[NORMAL001, SP001, UE003]`
- **external_validation:** `[SP002, STR001, UE001]`

**Critical limitation (must be documented):** this split is **degenerate for validation**. All labeled sources (SP001, NORMAL001) land in train; the validation split contains **zero labeled frames** — it only holds uncertain/unusable footage (SP002, STR001, UE001). It therefore provides **no evaluation signal**. Per instruction, no external test set is constructed in this stage. A meaningful external validation set is deferred until ≥5 independent labeled sources exist. Until then, external pilot frames are best used only as an additional `external_train` diversity source, never as a validated test signal.

---

## 7. Statistics

`metadata/external_pilot_statistics.json` and build summary:

- Total kept frames: **1,546**
- Labeled normal: **21** · Labeled failing: **15** · Uncertain: **1,510**
- Exact duplicates removed: **90**
- Split assignment per source as in Section 6.

---

## 8. Diversity vs the Kaggle reference

Comparison on the common 256×256 grid (Kaggle already 256×256; external resized), 400-image stratified Kaggle sample vs the 36 labeled external frames (`metadata/external_pilot_diversity.json`).

| Metric | Kaggle | External (labeled) |
|---|---|---|
| Sample size | 400 | 36 |
| Mean brightness | 129.4 | 89.7 (darker) |
| Brightness std | 34.8 | 27.9 |
| Mean contrast | 78.7 | 64.0 |
| Mean saturation | 67.0 | 77.1 |
| Dominant dimensions | 256×256 | 720×1280, 1080×1920 |

**Per-class brightness (external pilot vs Kaggle):**

| Class | Kaggle mean gray | External pilot mean gray |
|---|---|---|
| normal | 139.4 | 66.4 (pilot is notably darker) |
| failing | 119.0 | 122.2 (comparable) |

**phash Hamming distance** (64-bit): intra-Kaggle mean **28.9** vs external-vs-Kaggle mean **32.1**. The higher external-vs-Kaggle distance (≈ +3 bits) indicates the external pilot occupies a **visually different domain** (darker scenes, portrait orientations, different cameras) than the Kaggle training data.

**Interpretation (honest):** the external pilot is measurably distinct from the Kaggle data in brightness, orientation, and perceptual hash distance. This is **desirable for generalization testing but only if/when the pilot gains verified labels**; the current 36 usable frames (21 normal + 15 failing) are far too few to draw any accuracy conclusion. The diversity gap also means a naive external->Kaggle fine-tune may face a domain shift — motivating **external-data fine-tuning with frozen/light augmentation** rather than expecting zero-shot transfer. **No accuracy or generalization claim is made** — none was measured.

---

## 9. Repository structure

```
datasets/external/video-frames/
  raw/<sid>/            originals + *.info.json (source of truth)
  extracted/<sid>/sequence/   deterministic 1-FPS frames (1,546 kept)
  labeled/normal/       21 training-ready normal copies
  labeled/failing/      15 training-ready failing copies
  labels/pilot_labels.json   Stage 12B provisional labels
  metadata/
    external_pilot_dataset.json         full frame records (incl. phash)
    external_pilot_statistics.json
    external_pilot_build_summary.json
    external_pilot_diversity.json
    dedup_removed.json
    pilot_*.json                        Stage 12B analysis artifacts
  splits/external_train.json
  splits/external_val.json
  montage_*.jpg          contact sheets for mandatory human review
```

---

## 10. Deliverables

- `training/tv/external_pilot.py` — deterministic build/diversity module.
- `scripts/build_external_pilot.py`, `scripts/analyze_external_pilot.py` — reproducible CLI.
- `training/tests/test_external_pilot.py` — 11 tests (label mapping, grid/naming, phash/hamming determinism & sensitivity, exact-md5 dedup, diagnostic-only near-dup, source-atomic deterministic split). Full suite **219 passed**.
- `datasets/external/video-frames/` — the pilot dataset tree (Section 9).
- `docs/research/stage12c-external-pilot-dataset.md` / `.json` — this report.

---

## 11. Critical limitations (must be read)

1. **Labels are provisional and NOT ground truth.** No image/vision capability; no human reviewer ran during this stage. `normal` = Stage 12B `normal_provisional`; `failing` = Stage 12B `suspect_growth` (SP001). The growing object in SP001 is **not visually confirmed to be a spaghetti failure** — human review is required.
2. **36 usable frames is tiny.** 21 normal (one source) + 15 failing (one source). This is a **pilot**, not a training dataset. No accuracy/generalization claim is made.
3. **Degenerate validation split.** Validation has zero labeled frames. No external test set was built (per instruction). Meaningful external validation is deferred until ≥5 labeled independent sources exist.
4. **Source-atomicity honored:** no frame of a source appears in both splits.
5. **Uncertain frames are preserved in metadata but excluded** from training-ready copies and usable statistics; they are never silently relabeled.
6. **Diversity is measured, not a guarantee**: the domain-shift numbers (Section 8) argue for a **simple fine-tune** once labels are verified — not for assuming zero-shot transfer.

---

## 12. Recommendation / next step (not executed)

- **Do not train on this pilot.** The 36 usable frames are insufficient and unverified.
- **Mandatory human visual review** of the six montages and the SP001/NORMAL001 labeled frames to confirm failure type before any training integration.
- **Expand labeled sources to ≥5** independent videos (train and validation) with verified labels before treating external data as an evaluation signal.
- Keep `datasets/processed/printing-errors/` and the frozen official test set untouched.

**Stage stops here. No training initiated.**
