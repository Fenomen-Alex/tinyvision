"""Stage 12D-extra: local VLM review proposal pipeline (Qwen3-VL-8B via LM Studio).

Architecture
------------
The VLM is a PROPOSAL GENERATOR, never a ground-truth authority.

    review_manifest.json  ->  run_vlm_review.py  ->  Qwen3-VL-8B (LM Studio)
                                         |                |
                                         +-----> validate model response
                                                         |
                                                         v
                                           vlm_review_proposals.json

Guarantees (enforced here, do not break):
  1. `review_manifest.json` is only ever READ, never written.
  2. `verified_labels.json` is never created or modified by this module.
  3. Proposals are written ONLY to the proposals file (idempotent upsert by
     review_unit_id: re-running the same unit replaces that unit's entry and
     leaves unrelated proposals untouched).
  4. The VLM prompt is application-owned and neutral: source IDs, unit IDs,
     filenames, Stage 12B labels, documented failure types and other metadata
     are used internally only to locate evidence and associate results. They
     are NEVER sent to the model.
  5. Failures are recorded explicitly; a failed request never becomes a
     classification.
  6. Only the high-value pilot sources (SP001, NORMAL001, UE003) are selectable
     by default. Selecting UE001/STR001/SP002 is refused.

Transport is stdlib `urllib` only (no new dependency). The HTTP request is
isolated in `_http_post_json` so tests can mock it without LM Studio running.
"""
from __future__ import annotations

import base64
import datetime as _dt
import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "datasets" / "external" / "video-frames"

MANIFEST = EXT / "labels" / "review_manifest.json"
PROPOSALS_OUT = EXT / "labels" / "vlm_review_proposals.json"
VERIFIED = EXT / "labels" / "verified_labels.json"  # never touched by this module
SHEETS_DIR = EXT / "labels" / "sheets"
EXTRACTED_DIR = EXT / "extracted"

SCHEMA_VERSION = 1
DEFAULT_ENDPOINT = "http://127.0.0.1:1234"
DEFAULT_MODEL = "qwen/qwen3-vl-8b"
DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 2

# Only these pilot sources are eligible for VLM review by default.
HIGH_VALUE_SOURCES = {"SP001", "NORMAL001", "UE003"}

DECISIONS = {"normal", "failing", "uncertain"}
FAILURE_TYPES = {"spaghetti", "stringing", "under_extrusion"}

# Application-owned neutral prompt. Keep these semantics; do not inject
# filename / source / label / metadata information into it.
REVIEW_PROMPT = (
    "Review this 3D-printing footage for a TinyVision review proposal.\n"
    "Classify the visible result as exactly one of:\n"
    "- normal\n"
    "- failing\n"
    "- uncertain\n"
    "If and only if the footage visibly shows a failure, optionally classify the "
    "failure type as:\n"
    "- spaghetti\n"
    "- stringing\n"
    "- under_extrusion\n"
    "Base the judgment only on what is visibly shown.\n"
    "Do not infer information from filenames, source names, metadata, or previous labels.\n"
    'Use "uncertain" when the visual evidence is insufficient.\n'
    "Return only valid JSON with:\n"
    '{"decision": "normal | failing | uncertain", '
    '"failure_type": "spaghetti | stringing | under_extrusion | null", '
    '"confidence": 0.0, '
    '"visual_evidence": "brief description of visible evidence"}'
)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------
@dataclass
class ReviewUnit:
    review_unit_id: str
    source_id: str
    unit_type: str  # "sequence" | "frame"
    frames: list[int]
    times: list[float]


@dataclass
class Evidence:
    unit: ReviewUnit
    kind: str  # "sheet" | "frame"
    primary: Path      # image the model must classify
    context: list[Path]  # neighbouring frames for frame units (reference only)


class MissingEvidenceError(Exception):
    def __init__(self, unit_id: str, path: Path):
        super().__init__(f"missing evidence for {unit_id}: {path}")
        self.unit_id = unit_id
        self.path = path


class ResponseValidationError(Exception):
    """Model returned content that failed JSON/schema validation."""


class RetryableApiError(Exception):
    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type


# ---------------------------------------------------------------------------
# Manifest loading / selection
# ---------------------------------------------------------------------------
def load_manifest(path: Path = MANIFEST) -> dict:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or "units" not in data:
        raise ValueError(f"manifest at {path} is not a valid Stage 12D manifest")
    return data


def _unit_from_dict(u: dict) -> ReviewUnit:
    return ReviewUnit(
        review_unit_id=str(u["review_unit_id"]),
        source_id=str(u["source_id"]),
        unit_type=str(u["type"]),
        frames=[int(x) for x in u["frames"]],
        times=[float(x) for x in u["times"]],
    )


def select_units(manifest: dict, unit_id: str | None = None,
                 source: str | None = None, batch: bool = False,
                 high_value: frozenset[str] | None = None) -> list[ReviewUnit]:
    """Select review units deterministically (order preserved from manifest).

    At least one of unit_id / source / batch must be provided. Unknown units,
    empty selections, and non-high-value sources are refused.
    """
    hv = frozenset(high_value or HIGH_VALUE_SOURCES)
    if not (unit_id or source or batch):
        raise ValueError(
            "no selection: provide --unit, --source, or --batch "
            "(batch requires explicit opt-in)")
    if source is not None and source not in hv:
        raise ValueError(f"source {source!r} is not in the high-value review set {sorted(hv)}")
    if batch and (unit_id or source):
        raise ValueError("--batch cannot be combined with --unit or --source")

    units = [_unit_from_dict(u) for u in manifest.get("units", [])]
    selected: list[ReviewUnit] = []
    seen: set[str] = set()
    for u in units:
        if unit_id is not None and u.review_unit_id != unit_id:
            continue
        if source is not None and u.source_id != source:
            continue
        if batch and u.source_id not in hv:
            continue
        # a unit explicitly named must belong to a high-value source
        if u.review_unit_id == unit_id and u.source_id not in hv:
            raise ValueError(
                f"unit {u.review_unit_id!r} belongs to non-high-value source {u.source_id!r}")
        if u.review_unit_id in seen:
            continue
        seen.add(u.review_unit_id)
        selected.append(u)

    if unit_id is not None and not any(u.review_unit_id == unit_id for u in selected):
        # could be an unknown id in a high-value source
        if not any(u.review_unit_id == unit_id for u in units):
            raise ValueError(f"unknown review unit {unit_id!r}")
    if not selected:
        raise ValueError("selection produced no review units")
    return selected


# ---------------------------------------------------------------------------
# Visual evidence resolution
# ---------------------------------------------------------------------------
def frame_path(source_id: str, frame_index: int) -> Path:
    return EXTRACTED_DIR / source_id / "sequence" / f"frame_{frame_index:04d}.jpg"


def resolve_evidence(unit: ReviewUnit,
                     include_context: bool = True) -> Evidence:
    """Resolve the visual evidence for a unit.

    Sequence units -> the contact sheet in `labels/sheets/`.
    Frame units    -> the target frame; optionally the immediately adjacent
                      frames as clearly-marked context (never as targets).
    """
    if unit.unit_type == "sequence":
        primary = SHEETS_DIR / f"{unit.review_unit_id}.jpg"
        if not primary.exists():
            raise MissingEvidenceError(unit.review_unit_id, primary)
        return Evidence(unit=unit, kind="sheet", primary=primary, context=[])

    if not unit.frames:
        raise MissingEvidenceError(unit.review_unit_id, frame_path(unit.source_id, -1))
    target = unit.frames[0]
    primary = frame_path(unit.source_id, target)
    if not primary.exists():
        raise MissingEvidenceError(unit.review_unit_id, primary)
    context: list[Path] = []
    if include_context:
        for neighbor in (target - 1, target + 1):
            p = frame_path(unit.source_id, neighbor)
            if p.exists():
                context.append(p)
    return Evidence(unit=unit, kind="frame", primary=primary, context=context)


def image_to_data_url(path: Path) -> str:
    b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------------
# Prompt / message building (neutral; identifies target vs context)
# ---------------------------------------------------------------------------
def build_messages(prompt: str, data_urls: list[str], unit_type: str) -> list[dict]:
    text = prompt
    if unit_type == "frame" and len(data_urls) > 1:
        text = (
            "The first image is the [TARGET] frame you must classify.\n"
            "The remaining images are [CONTEXT] neighbouring frames, "
            "for reference only.\n"
            + prompt
        )
    content: list[dict] = [{"type": "text", "text": text}]
    content += [{"type": "image_url", "image_url": {"url": url}} for url in data_urls]
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# HTTP transport (stdlib only, isolated for test mocking)
# ---------------------------------------------------------------------------
def _http_post_json(url: str, payload: dict, timeout: float) -> tuple[int, str]:
    """POST JSON and return (status_code, response_text).

    Raises RetryableApiError for connection-level failures (never for HTTP
    error statuses, which are returned to the caller).
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except urllib.error.URLError as e:
        raise RetryableApiError("connection", f"connection failed: {e}")
    except (TimeoutError, socket.timeout) as e:
        raise RetryableApiError("timeout", f"request timed out: {e}")
    except OSError as e:
        raise RetryableApiError("connection", f"transport error: {e}")


def call_lmstudio(endpoint: str, model: str, prompt: str, data_urls: list[str],
                  timeout: float = DEFAULT_TIMEOUT) -> str:
    """Call the OpenAI-compatible chat completions endpoint; return assistant text.

    Raises RetryableApiError on connection/timeout/HTTP/envelope problems.
    """
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": build_messages(prompt, data_urls, "frame" if len(data_urls) > 1 else "sequence"),
    }
    status, text = _http_post_json(url, payload, timeout)
    if status >= 400:
        raise RetryableApiError("http", f"HTTP {status}: {text[:300]}")
    try:
        envelope = json.loads(text)
    except json.JSONDecodeError as e:
        raise RetryableApiError("envelope", f"non-JSON response body: {e}")
    if not isinstance(envelope, dict):
        raise RetryableApiError("envelope", "response envelope is not an object")
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RetryableApiError("envelope", "response has no choices")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RetryableApiError("envelope", "assistant content missing or empty")
    return content


# ---------------------------------------------------------------------------
# Response validation (strict)
# ---------------------------------------------------------------------------
def extract_json_object(text: str) -> dict:
    """Safely extract a single JSON object from model text.

    If a single brace-delimited object cannot be isolated and parsed, raises
    ResponseValidationError. Returns a dict only.
    """
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ResponseValidationError("no JSON object in model response")
    candidate = text[start:end + 1]
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ResponseValidationError(f"invalid JSON object: {e}")
    if not isinstance(obj, dict):
        raise ResponseValidationError("extracted JSON is not an object")
    return obj


def validate_response(obj: dict) -> tuple[str, str | None, float, str]:
    """Validate a parsed model response into (decision, failure_type, confidence, evidence).

    Raises ResponseValidationError on any violation. failure_type is normalized
    to None for null/missing.
    """
    decision = obj.get("decision")
    if not isinstance(decision, str) or decision.strip().lower() not in DECISIONS:
        raise ResponseValidationError(f"invalid decision: {decision!r}")

    raw_ft = obj.get("failure_type", None)
    failure_type: str | None
    if raw_ft is None:
        failure_type = None
    else:
        ft = str(raw_ft).strip().lower()
        if ft in ("null", "none", ""):
            failure_type = None
        elif ft not in FAILURE_TYPES:
            raise ResponseValidationError(f"invalid failure_type: {raw_ft!r}")
        else:
            failure_type = ft

    conf = obj.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        raise ResponseValidationError(f"invalid confidence: {conf!r}")
    if not (0.0 <= float(conf) <= 1.0):
        raise ResponseValidationError(f"confidence out of range: {conf!r}")

    evidence = obj.get("visual_evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ResponseValidationError("visual_evidence missing or empty")

    return decision.strip().lower(), failure_type, float(conf), evidence.strip()


# ---------------------------------------------------------------------------
# Proposal production
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _success_proposal(unit: ReviewUnit, model: str, endpoint: str, result: tuple,
                      attempts: int) -> dict:
    decision, failure_type, confidence, evidence = result
    return {
        "review_unit_id": unit.review_unit_id,
        "source_id": unit.source_id,
        "unit_type": unit.unit_type,
        "decision": decision,
        "failure_type": failure_type,
        "confidence": confidence,
        "visual_evidence": evidence,
        "model": model,
        "endpoint": endpoint,
        "status": "success",
        "attempts": attempts,
        "generated_at": _now_iso(),
    }


def _error_proposal(unit: ReviewUnit, model: str, endpoint: str,
                    error_type: str, message: str, attempts: int) -> dict:
    return {
        "review_unit_id": unit.review_unit_id,
        "source_id": unit.source_id,
        "unit_type": unit.unit_type,
        "model": model,
        "endpoint": endpoint,
        "status": "error",
        "error_type": error_type,
        "error": message,
        "attempts": attempts,
        "generated_at": _now_iso(),
    }


def propose_unit(unit: ReviewUnit, endpoint: str = DEFAULT_ENDPOINT,
                 model: str = DEFAULT_MODEL, include_context: bool = True,
                 timeout: float = DEFAULT_TIMEOUT, retries: int = DEFAULT_RETRIES,
                 dry_run: bool = False) -> dict:
    """Produce a proposal for one unit. Never writes to disk; never touches the
    manifest or verified labels."""
    try:
        evidence = resolve_evidence(unit, include_context=include_context)
    except MissingEvidenceError as e:
        return _error_proposal(unit, model, endpoint, "missing_evidence",
                               str(e), attempts=0)

    if dry_run:
        return {
            "review_unit_id": unit.review_unit_id,
            "source_id": unit.source_id,
            "unit_type": unit.unit_type,
            "kind": evidence.kind,
            "primary": str(evidence.primary),
            "context": [str(p) for p in evidence.context],
            "status": "dry_run",
        }

    data_urls = [image_to_data_url(evidence.primary)]
    data_urls += [image_to_data_url(p) for p in evidence.context]

    last_error: RetryableApiError | None = None
    last_validation: ResponseValidationError | None = None
    attempts = 0
    for attempt in range(retries + 1):
        attempts = attempt + 1
        try:
            content = call_lmstudio(endpoint, model, REVIEW_PROMPT, data_urls,
                                    timeout=timeout)
            obj = extract_json_object(content)
            result = validate_response(obj)
        except RetryableApiError as e:
            last_error = e
            continue
        except ResponseValidationError as e:
            last_validation = e
            continue
        return _success_proposal(unit, model, endpoint, result, attempts)

    if last_error is not None:
        return _error_proposal(unit, model, endpoint, last_error.error_type,
                               str(last_error), attempts)
    if last_validation is not None:
        return _error_proposal(unit, model, endpoint, "validation",
                               str(last_validation), attempts)
    return _error_proposal(unit, model, endpoint, "unknown", "unknown failure", attempts)


# ---------------------------------------------------------------------------
# Proposal store (versioned, idempotent upsert)
# ---------------------------------------------------------------------------
def load_proposals(path: Path = PROPOSALS_OUT) -> dict:
    if not Path(path).exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _now_iso(),
            "model": DEFAULT_MODEL,
            "endpoint": DEFAULT_ENDPOINT,
            "proposals": [],
        }
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or "proposals" not in data:
        raise ValueError(f"proposal file {path} is malformed")
    return data


def upsert_proposals(store: dict, new_proposals: list[dict]) -> dict:
    """Replace proposals by review_unit_id; leave unrelated entries untouched.

    Deterministic, single-write ordering: kept unrelated first (original order),
    then upserted entries for the touched units.
    """
    touched: set[str] = set()
    for p in new_proposals:
        touched.add(p["review_unit_id"])
    kept = [p for p in store.get("proposals", []) if p["review_unit_id"] not in touched]
    # new proposals keep their own order (deduped by review_unit_id)
    merged = kept + [p for p in new_proposals]
    store["proposals"] = merged
    return store


def write_proposals(store: dict, path: Path = PROPOSALS_OUT) -> Path:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(store, indent=2, ensure_ascii=False))
    return Path(path)


def run_review(units: list[ReviewUnit], endpoint: str = DEFAULT_ENDPOINT,
               model: str = DEFAULT_MODEL, include_context: bool = True,
               timeout: float = DEFAULT_TIMEOUT, retries: int = DEFAULT_RETRIES,
               dry_run: bool = False, output: Path = PROPOSALS_OUT) -> dict:
    """Run the review for selected units and write the upserted proposal store."""
    proposals = [propose_unit(u, endpoint, model, include_context, timeout,
                              retries, dry_run) for u in units]
    if dry_run:
        return {"dry_run": proposals}
    store = load_proposals(output)
    store = upsert_proposals(store, proposals)
    store["generated_at"] = _now_iso()
    store["model"] = model
    store["endpoint"] = endpoint
    write_proposals(store, output)
    return store