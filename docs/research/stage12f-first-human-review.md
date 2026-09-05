# Stage 12F — First Human Review Tranche and VLM Comparison

**Date:** 2026-09-05
**Status:** Documentation/audit only. The first six-unit human-review tranche from Stage 12D is complete, merged, and compared to the existing Stage 12D-extra VLM proposals. **Finalized: manual human review is stopped at 24 reviewed units (see §8). No human decisions were altered, no training, no VLM rerun, no new VLM requests.**

---

## 1. Purpose and scope

Stage 12E prepared the human labeling pass and explicitly deferred any label-vs-proposal comparison until human labels existed. The first human-review tranche is now complete. This document:

- Records the six reviewed units and their human labels.
- Compares them against the already-existing VLM proposals.
- Interprets the comparison **carefully**, explicitly as an audit/calibration sample.

This is a documentation task only. Nothing here is a training run and no VLM call was made.

---

## 2. Human-reviewed units and labels

The first tranche contains exactly these six **sequence** units, all with human decisions recorded in `datasets/external/video-frames/labels/review_manifest.json` and merged into `datasets/external/video-frames/labels/verified_labels.json` (`provenance = human_reviewed`):

| Unit | Source | Type | Human decision | Human failure type |
|---|---|---|---|---|
| NORMAL001__seq000 | NORMAL001 | sequence | failing | stringing |
| NORMAL001__seq001 | NORMAL001 | sequence | failing | stringing |
| NORMAL001__seq002 | NORMAL001 | sequence | failing | stringing |
| SP001__seq000 | SP001 | sequence | failing | spaghetti |
| SP001__seq001 | SP001 | sequence | failing | spaghetti |
| SP001__seq002 | SP001 | sequence | failing | spaghetti |

These are the actual reviewed labels; they were **not altered** by this task.

---

## 3. VLM proposals for the same units

From `datasets/external/video-frames/labels/vlm_review_proposals.json` (model `qwen/qwen3-vl-8b`, local LM Studio endpoint, schema_version 1; proposals only — never ground truth):

| Unit | VLM decision | VLM failure type | VLM confidence |
|---|---|:---:|---:|
| NORMAL001__seq000 | normal | — | 0.98 |
| NORMAL001__seq001 | normal | — | 0.98 |
| NORMAL001__seq002 | normal | — | 0.99 |
| SP001__seq000 | failing | stringing | 0.95 |
| SP001__seq001 | failing | stringing | 0.95 |
| SP001__seq002 | failing | stringing | 0.95 |

---

## 4. Comparison

```
human/VLM agreement:   0/6
human/VLM disagreement: 6/6
```

Disagreement breakdown:

- **NORMAL001 (3 units):** human = `failing / stringing`; VLM = `normal`. Disagreement is in **decision AND failure type** (VLM had no failure type at all, as it did not flag failure).
- **SP001 (3 units):** human = `failing / spaghetti`; VLM = `failing / stringing`. Both flags agree on **decision (failing)**; disagreement is in **failure type**.

Notably, the three NORMAL001 proposals were among the VLM's **most confident** outputs in the batch (0.98/0.98/0.99), yet the human reviewer found visible stringing in all three. The three SP001 proposals were flagged failing at 0.95 with the wrong failure type.

---

## 5. Interpretation (with caution)

**This six-unit tranche is an audit/calibration sample. It is insufficient to establish general VLM accuracy.** No accuracy, recall, or precision number should be derived from six units, and no conclusion about the VLM's overall value may be drawn from it.

The result is nevertheless important:

1. **The human reviewer found visible failures in all six units** (stringing in all three NORMAL001 sequence units, spaghetti in all three SP001 sequence units).
2. **The VLM classified all three NORMAL001 units as normal** at high confidence — an observed false-negative tendency on this sample for units the human called failing/stringing.
3. **The VLM detected failure in SP001 but assigned a different failure type** (`stringing` instead of `spaghetti`).
4. **The earlier Stage 12B provisional `spaghetti` interpretation for SP001 is consistent with the human labels** for all three SP001 sequence units. Stage 12B's structural `suspect_growth` hypothesis for the SP001 print-area growth now aligns with the human visual finding, though it remains a hypothesis-to-label match, not proven ground truth.
5. **The VLM `stringing` result for SP001 therefore remains a disagreement requiring further analysis.** It is one unit-level claim against the human label and the Stage 12B structural analysis; it is not resolved by this tranche.

**Careful wording requirement:** the human labels are human-reviewed labels from this tranche. They are **not** objectively proven ground truth — no notion of "the human is correct" should be asserted as fact. The comparison describes agreement/disagreement between two independent signals (a human visual review and a local VLM proposal).

Additional factual observations, not conclusions:

- The source **NORMAL001 was named and prioritized (Stage 12E) as a likely-clean baseline**; the human tranche instead found stringing in all three of its sequence units. Source names describe video-channel origin, not verified content.
- Only the three **sequence** units of SP001 are in this tranche. The three SP001 frame units remain unreviewed.
- The remaining 1,000 manifest units had no human decision at the time of the tranche (final state: **982** — see §8).

---

## 6. Integrity verification

| Check | Result |
|---|---|
| `review_manifest.json` unit count | 1,006 |
| Units with a human decision | 6 in the tranche above; **24 in the final state** (see §8) |
| `verified_labels.json` verified units | exactly 6 |
| `verified_labels.json` provenance | `human_reviewed` (stage `stage12d`) |
| `vlm_review_proposals.json` | unchanged (sha256 `6457afdd02150a90891614140296d3b76e4d832536adbcb090ad99de6f683b03`) |
| VLM source code (`training/tv/vlm_review.py`, `scripts/run_vlm_review.py`) | unchanged |
| Training performed | none |
| New VLM requests during this task | none |

**Not modified by this task:** `training/tv/vlm_review.py`, `scripts/run_vlm_review.py`, `datasets/external/video-frames/labels/vlm_review_proposals.json`, and no human decisions were altered.

---

## 7. Validation performed (finalization)

- Full test suite: **290 passed** at finalization (see report for the run).
- `review_manifest.json` sha256 `72b4ee89…` (final human-reviewed manifest, 24 decided units; recorded before and after finalization).
- `verified_labels.json` sha256 `27038df1…` (the six-unit merged file; unchanged).
- `vlm_review_proposals.json` sha256 `6457afdd…` (unchanged before and after).
- Earlier 12F note: one VLM-integrity test (`test_manifest_and_verified_untouched`) was updated to assert a review op leaves an existing verified file byte-for-byte unchanged instead of asserting repo-wide absence (invalid once `merge` creates it).

---

## 8. Finalization — manual human review stopped

Manual human review is **stopped at the current point**, per operator decision. No decisions were altered during finalization. Final review state:

- **1,006** manifest units; **24** with human decisions; **982** undecided.
- Final decision mix (by source / type / decision):

| Group | Units | Type | Decision / failure type |
|---|---|---:|---|
| NORMAL001 | seq000–002 (3) | sequence | `failing / stringing` |
| SP001 | seq000–002 (3) | sequence | `failing / spaghetti` |
| UE003 | seq000–002 (3) | sequence | `normal` |
| UE003 | frm0000–0004 (5) | frame | `normal` |
| UE003 | seq003–011, seq020 (10) | sequence | `uncertain` |

- **UE003 contains substantial setup/non-print footage.** Human notes record, e.g., `Printer is being setting up, no print progress or printed object visible` (UE003__seq003). This is consistent with the Stage 12E VLM-uncertain evidence (hands/extruder interaction, control panels, black-screen-with-text frames). **Exhaustive manual review of UE003 was stopped because it has low information value** for a printed-object failure detector.
- `verified_labels.json` is unchanged from the six-unit merger: **6** verified units, `provenance = human_reviewed`, containing exactly the NORMAL001 + SP001 sequence units (§2). The **18 additional UE003 decisions remain in the manifest only**; **no further merge was performed** during finalization.
- The human/VLM disagreement recorded in §4/§5 (0/6 agreement on the overlapped six units) remains an **audit/calibration observation** — not an accuracy metric and not a ground-truth claim.

---

## 9. Next steps (not executed here)

1. Decide separately whether the 18 additionally reviewed UE003 decisions (or a selected subset) should be merged into verified labels via `merge --allow-partial` — not done during finalization.
2. Extend the human/VLM comparison only if/when human review resumes and more calibrated labels exist.
3. Investigate the SP001 failure-type disagreement (`spaghetti` human + Stage 12B vs `stringing` VLM) with further evidence.
4. The 24 human-reviewed units are a small calibration/validation signal; they are **not** training ground truth at the current size.

**Stage 12F closes here: manual review stopped; no label changes; no training; no VLM calls.**