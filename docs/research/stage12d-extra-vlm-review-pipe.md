# Stage 12D-extra — Local VLM Review Pipe (Proposals Only)

**Date:** 2026-09-05
**Status:** Built + tested + smoke-tested against the real local LM Studio endpoint for a single unit. **No batch run; no verified labels produced by the machine; no training.**

---

## 1. Purpose and scope

Stage 12D produced a **human-review manifest** (`review_manifest.json`, 1,006 units, 0 decisions filled) plus 86 contact sheets. Its limitation is throughput: the human reviewer is free, and this agent has **no image-input (vision) capability**, so it cannot itself judge a single frame.

This stage builds a **local, offline, opinion-only pluralizer**: a CLI that asks a small **vision-language model (Qwen3-VL-8B) running on this machine via LM Studio** to propose a classification for each review unit. The result is a **proposal file** — explicitly **not ground truth**, never merged into labels.

> The VLM proposal is a *second opinion* shown to the human reviewer and a *pre-screening* signal for the Experiment 2 decision. It is authoritative about nothing.

**Hard requirements honored:**
- `review_manifest.json` is **read-only** — never modified, added to, or re-serialized.
- `verified_labels.json` is **never created or modified** by this pipe.
- Output goes exclusively to `labels/vlm_review_proposals.json`, an idempotent upsert store.
- **Neutral prompt** — the model is never sent a source ID, unit ID, filename, Stage 12B label, documented failure type, or prior prediction. Metadata is used only to resolve images and to label the *proposal* it belongs to.
- **No fabricated labels** — `uncertain` is allowed; any API/validation failure produces an explicit error proposal, never a guessed classification.
- No new dependencies (stdlib `urllib`); LM Studio server settings untouched (Context 16,384, Temperature 0.15).

---

## 2. Architecture

```
review_manifest.json
        │  (read-only, 1,006 units)
        ▼
scripts/run_vlm_review.py  ──►  training/tv/vlm_review.py
        │  selects units                     │
        │   resolves visual evidence         │
        │   builds neutral multimodal prompt │
        ▼                                    ▼
LM Studio  http://127.0.0.1:1234  POST /v1/chat/completions
        │  qwen/qwen3-vl-8b
        ▼
strict JSON validation (bounded retries)
        ▼
labels/vlm_review_proposals.json   (idempotent upsert by review_unit_id)
```

- **Selection guard:** only the three **high-value sources** are eligible: `NORMAL001` (3 units), `SP001` (6 units), `UE003` (111 units) → **120 units total (39 sequence + 81 frame, 334 sequence-reviewable seconds)**. Selecting UE001/STR001/SP002 is refused; running with no selection is an argparse error; the full 120-unit run requires the explicit `--batch` flag.
- **Evidence mapping:**
  - SEQUENCE unit → its contact sheet `labels/sheets/<review_unit_id>.jpg` (single image).
  - FRAME unit → the target extracted frame (`extracted/<sid>/sequence/frame_%04d.jpg`) plus, by default, the immediate neighbours (k−1, k+1) clearly marked `[TARGET]` vs `[CONTEXT]`.
- **Transport:** OpenAI-compatible `POST /v1/chat/completions`, images as `data:image/jpeg;base64,…` parts; stdlib `urllib` (isolated in `_http_post_json` so tests can mock it).

---

## 3. Prompt discipline (the part that matters)

The classification prompt is **application-owned and stable** (not free-form). It asks the model to classify the *visible print state* only:

- decision ∈ `normal` | `failing` | `uncertain`
- if `failing`, failure_type ∈ `spaghetti` | `stringing` | `under_extrusion` | `null`
- a 0–1 confidence, `visual_evidence` (free text the model actually saw)

Strictly **excluded** from the model request: source id, unit id, file paths/names, the Stage 12B provisional labels, the documented failure-type of the source, and any prior proposal. The model cannot tell you *whose* video it is looking at.

---

## 4. Response validation & error handling

All model responses are validated before being stored:

- Content is extracted as a JSON object (a single balanced `{…}` block inside prose is accepted; prose alone, malformed JSON, or multiple embedded objects are rejected).
- `decision` must be in the vocabulary; `failure_type` must be `null`/`None`/`""` or in the vocabulary; `confidence` must be a real number in [0,1]; `visual_evidence` is required.
- **Bounded retries** (`--retries`, default 2; timeout default 60 s). HTTP/connection/timeout failures and validation failures are retried, then converted into an explicit **error proposal** (`status: "error"`, `error_type`, `error`, `attempts`) — an error is never classified as `normal`/`failing`/`uncertain`.
- Missing visual evidence (e.g. a missing sheet/frame) short-circuits to an error proposal without any API call.

---

## 5. CLI

```
python3 scripts/run_vlm_review.py --unit NORMAL001__seq000   # single unit
python3 scripts/run_vlm_review.py --source NORMAL001         # one source
python3 scripts/run_vlm_review.py --unit X --dry-run         # plan only, no API/writes
python3 scripts/run_vlm_review.py --batch                    # EXPLICIT opt-in: all 120
```

Options: `--unit`, `--source`, `--batch`, `--dry-run`, `--endpoint` (default `http://127.0.0.1:1234`), `--model` (default `qwen/qwen3-vl-8b`), `--output`, `--retries` (default 2), `--timeout` (default 60.0), `--no-frame-context`.

**Idempotency:** re-running a unit replaces only that unit's proposal; proposals for untouched units are preserved. `--dry-run` prints the selection + evidence plan and writes nothing.

---

## 6. Proposal schema (`schema_version: 1`)

Success proposal fields: `review_unit_id`, `source_id`, `unit_type`, `decision`, `failure_type`, `confidence`, `visual_evidence`, `model`, `endpoint`, `status: "success"`, `attempts`, `generated_at`.

Error proposal fields: `review_unit_id`, `source_id`, `unit_type`, `status: "error"`, `error_type` (`timeout`/`http`/`connection`/`envelope`/`validation`/`missing_evidence`), `error`, `attempts`, `generated_at`.

`source_id`/`unit_type`/`model`/`endpoint` in the proposal are **provenance**, not model input.

---

## 7. Tests

`training/tests/test_vlm_review.py` — **39 tests**, all mocking the HTTP transport (no LM Studio needed):

- **Manifest / selection:** load; select by unit and by source; no-selection error; excluded-source refusal; unknown unit; `--batch` = exactly the 120 high-value units and refuses combination with `--unit`.
- **Evidence:** sheet resolution; frame target+context; optional context; missing-image → error.
- **Validation (11 cases):** valid normal/failing/uncertain; all failure types; null normalization; invalid decision / failure type / confidence; malformed JSON; prose-wrapped extraction; double-object rejection; missing fields.
- **API (mocked):** success; timeout → retry → success; HTTP failure → error; validation-retry → error; bounded final failure; non-JSON body.
- **Integrity:** after a real review-pipe run, `review_manifest.json` is **byte-for-byte unchanged** and `verified_labels.json` never exists.
- **Output:** schema valid; re-run produces no duplicate proposals; unrelated proposals preserved; errors explicit; `--dry-run` writes nothing.

Full suite: **268 passed** (229 prior + 39 new).

---

## 8. Real smoke test (single unit, against live LM Studio)

Command: `python3 scripts/run_vlm_review.py --unit NORMAL001__seq000`

Result: **1 success, 0 errors, 1 attempt**, UTC `2026-09-05T13:34:33Z`. Stored proposal:

```json
{
  "review_unit_id": "NORMAL001__seq000",
  "source_id": "NORMAL001",
  "unit_type": "sequence",
  "decision": "normal",
  "failure_type": null,
  "confidence": 1.0,
  "visual_evidence": "The 3D print is progressing layer by layer with consistent, smooth extrusion and no visible defects such as stringing, spaghetti, or under-extrusion. The print bed and nozzle are moving as expected without any signs of failure.",
  "status": "success",
  "attempts": 1
}
```

**Integrity after the smoke test:** `review_manifest.json` mtime unchanged (15:12:20, the Stage 12D build) → file untouched; `verified_labels.json` absent; only `vlm_review_proposals.json` was created.

**Only this single unit was processed.** The remaining 119 high-value units were intentionally left unprocessed.

---

## 9. Limitations & decisions to review before batch

1. **Proposals are uncalibrated opinions.** The smoke-test `confidence: 1.0` is not trustworthy confidence; VLM confidence ≠ a calibrated probability, and the model has seen one contact sheet, not the print. Human review remains the ground-truth authority.
2. **Frame-unit live path unexercised.** Evidence resolution (target + neighbours) was verified by dry-run; only a *sequence* (contact sheet) request has been made to the live server.
3. **Validation retries are cheap but bounded.** With server-side temperature 0.15, repeated calls can return the same invalid JSON, exhausting retries quickly — by design, this degrades to an explicit error proposal rather than endless looping.
4. **`--batch` is 120 sequential requests** (one per unit, up to retries each); no concurrency or rate limiting. Runtime depends on local GPU/CPU.
5. **JPEG-only input** (both sheets and frames are JPG; fixed `image/jpeg` data-URI).
6. **Upsert never purges.** Re-running replaces the touched unit; stale proposals for other units are preserved and would need explicit cleanup if the vocabulary changes.
7. **The doc names in the manifest** have known cosmetic drift (fractional times for 23.976 fps sources vs integer sheet labels); frame identity (`frames[k] == round(times[k])`) is consistent, so evidence maps correctly.

---

## 10. Next step (not executed)

1. Decide with the human reviewer whether the 120-unit VLM batch is wanted; if so, run `python3 scripts/run_vlm_review.py --batch` (explicit opt-in, ~120 requests).
2. Independently, the **human** completes `review_manifest.json` for the high-value units (the Stage 12D workflow) and merges via `run_human_review.py merge` → produces human-verified labels.
3. Once both exist, compare human labels vs VLM proposals as a sanity signal before the Experiment 2 decision.

**Stage stops here. No batch VLM run, no training, no verified labels produced by the machine.**