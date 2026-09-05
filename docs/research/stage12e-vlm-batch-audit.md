# Stage 12E — VLM Batch Audit for Human Review Preparation

**Date:** 2026-09-05
**Status:** Post-batch audit document prepared for the human review pass. **No VLM rerun, no label changes, no merges. VLM predictions remain proposals and are kept out of the human labeling flow (blind review).**

---

## 1. Purpose and scope

The Stage 12D-extra `--batch` run processed all **120 high-value units** (NORMAL001, SP001, UE003) through the local Qwen3-VL-8B in 120/120 successes. This document audits that run and prepares the **human** Stage 12E labeling pass.

This is **preparation and analysis only**:

- The VLM is **not** run again.
- The Stage 12D-extra implementation is **not** modified.
- `review_manifest.json` is **not** modified; no human-review decisions are touched.
- `verified_labels.json` is **not** created or modified.
- No VLM prediction is merged into any label artifact.

---

## 2. VLM batch outcome (120/120)

| Source | Total | normal | failing | uncertain |
|---|---:|---:|---:|---:|
| NORMAL001 | 3 | 3 | 0 | 0 |
| SP001 | 6 | 3 | 3 | 0 |
| UE003 | 111 | 104 | 1 | 6 |
| **TOTAL** | **120** | **110** | **4** | **6** |

- failure_type: **stringing 4**, **spaghetti 0**, **under_extrusion 0**, null 116.
- unit types: 39 sequence + 81 frame.
- confidence: min 0.0, max 1.0, mean 0.9372 (distribution: `1.0`×41, `0.98`×65, `0.99`×4, `0.95`×4, `0.0`×6).
- runtime: 1h 52m 21s (14:27:48Z → 16:20:09Z), 197 total API requests.

---

## 3. VLM failing units

All four proposed `failure_type=stringing` at confidence 0.95:

| Unit | Type | failure_type | Confidence | Attempts |
|---|---|---:|---:|---:|
| SP001__seq000 | sequence | stringing | 0.95 | 1 |
| SP001__seq001 | sequence | stringing | 0.95 | 1 |
| SP001__seq002 | sequence | stringing | 0.95 | 1 |
| UE003__frm0027 | frame | stringing | 0.95 | 2 |

UE003__frm0027 is the **only** failing unit outside SP001 and the only UE003 non-normal unit that is not uncertain.

---

## 4. VLM uncertain units

| Unit | Type | Confidence | Attempts |
|---|---|---:|---:|
| UE003__seq010 | sequence | 0.0 | 1 |
| UE003__seq015 | sequence | 0.0 | 1 |
| UE003__seq016 | sequence | 0.0 | 1 |
| UE003__seq024 | sequence | 0.0 | 1 |
| UE003__seq028 | sequence | 0.0 | 1 |
| UE003__seq032 | sequence | 0.0 | 1 |

Model-reported visual evidence for these units: hand interacting with the extruder/print head, person operating a control panel or touchscreen, or a black screen with text — i.e. **no printed object or print bed visible**. The model abstained at confidence 0.0. This is consistent for tutorial / UI-heavy footage and does not, by itself, imply a print state.

---

## 5. Important disagreement — SP001: `spaghetti` (provisional) vs `stringing` (VLM)

The three **SP001 sequence units** carry an unresolved disagreement between the two independent signals:

- **Stage 12B structural analysis (provisional, determinant-based):** SP001 was the strongest structural failure candidate — a growing foreground blob (object growth trend) mapped to a **provisional `spaghetti` (suspect_growth)** label.
- **Stage 12D-extra VLM (visual, independent):** all three SP001 sequence units proposed **`stringing`** at confidence 0.95, citing visible thin filament strands.

The prompts sent to the VLM were **neutral**: no source ID, unit ID, filename, Stage 12B label, or documented failure type was passed to the model, so its `stringing` reading is its own visual judgment, not a copy of prior state.

**Neither classification is ground truth.** Structural blob growth and a text-reading of thin strands are both *hypotheses*. The human reviewer resolves this on the actual contact sheets, and must do so **without being influenced by either prior claim**. This is the single highest-value question of the Stage 12E pass.

---

## 6. Retry observation

- **43 units** succeeded on attempt 1.
- **77 units** succeeded on attempt 2 (one retry each), mostly UE003 frame units.
- **0 final errors.**

The current proposal schema (`schema_version: 1`) records the **final attempt count** and final status but **not the transient retry cause** (e.g. timeout vs response-validation failure on the first attempt). Retry causes are therefore not reportable from the stored artifact. This is a documented schema limitation, not an error.

---

## 7. Human-review integrity (blindness requirements)

Stage 12E must remain **blind to VLM predictions during the initial human-labeling pass**:

1. The reviewer inspects the **visual evidence** (Stage 12D contact sheets for sequence units, extracted frame for frame units) and assigns `normal` / `failing` / `uncertain` **independently**.
2. VLM predictions are **not** added to `review_manifest.json`.
3. VLM predictions are **not** added to the human-review UI or to the contact sheets (`labels/sheets/*.jpg`).
4. The existing Stage 12D review tooling (`training/tv/review.py`, `scripts/run_human_review.py`) is **not modified**.
5. This audit may flag units of interest (Section 3, 4, 5) for a *later* label-vs-proposal comparison, but the reviewer must not consult those predictions while assigning initial labels.

---

## 8. Review priority (recommended order)

1. **NORMAL001** — 3 sequence units; likely-clean print baseline; fastest to confirm.
2. **SP001** — 6 units (3 sequence + 3 frame); resolves the spaghetti/stringing disagreement (Section 5); highest information value.
3. **UE003** — 111 units (33 sequence + 78 frame); the bulk of the work; the 6 uncertain units (Section 4) will likely be confirmed as non-printer footage or marked uncertain.

Within each source, follow the existing Stage 12D workflow (review sequence sheets, set `decision`, then `python scripts/run_human_review.py merge`).

---

## 9. Validation performed

- Full test suite: **268 passed** (unchanged from the batch state).
- `review_manifest.json`: byte-for-byte unchanged (sha256 `fbb7a711…`, mtime 15:12:20).
- `verified_labels.json`: absent.
- `vlm_review_proposals.json`: unchanged by this task (hash recorded before and after).
- No VLM requests were made during this task.

---

## 10. Next step (not executed)

The human reviewer performs the Stage 12E labeling pass (NORMAL001 → SP001 → UE003) using the existing Stage 12D workflow, then merges via `run_human_review.py merge`. Only after the human labels exist may the VLM proposals be compared against them.

**Stage stops here. No Stage 12E labeling was started; no training; no merges; VLM predictions were not used to change any labels.**