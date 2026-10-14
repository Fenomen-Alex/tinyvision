"""VLM-assisted frame labelling (a proposal, never ground truth).

Every frame is labelled ``failing`` / ``normal`` / ``uncertain``. VLM output is
treated as a *proposal*: ``uncertain`` is preserved and never silently coerced
into a class. A source's ``video_label`` is the majority of non-uncertain frame
labels (ties resolve to uncertain).
"""
from __future__ import annotations

from pathlib import Path

from tv.vlm_review import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    ResponseValidationError,
    call_lmstudio,
    extract_json_object,
    image_to_data_url_scaled,
)

LABELS = ("failing", "normal", "uncertain")

LABEL_PROMPT = (
    "You are inspecting a single frame of a 3D printing process recorded from a "
    "camera pointed at the printer bed. Decide whether the print is currently "
    "FAILING (spaghetti, blob, detachment, layer shift, warping, no extrusion) or "
    "NORMAL (the print looks as expected). If the frame is ambiguous, off-topic, "
    "too dark, or you cannot see the print, answer 'uncertain'.\n"
    "Reply with ONLY a JSON object: "
    '{"label": "failing"|"normal"|"uncertain", "failure_type": <string|null>, '
    '"confidence": <0..1>, "reason": <short string>}.'
)


def build_prompt(extra: str = "") -> str:
    return LABEL_PROMPT + (f"\n{extra}" if extra else "")


def parse_label_response(text: str) -> dict:
    """Parse model text into a proposal dict; never raises, defaults to uncertain."""
    fallback = {"label": "uncertain", "failure_type": None,
                "confidence": 0.0, "reason": "unparsed_or_invalid"}
    try:
        obj = extract_json_object(text)
    except ResponseValidationError:
        return fallback
    label = obj.get("label")
    if not isinstance(label, str) or label.strip().lower() not in LABELS:
        return fallback
    conf = obj.get("confidence")
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0
    conf = min(max(conf, 0.0), 1.0)
    ft = obj.get("failure_type")
    reason = obj.get("reason")
    return {"label": label.strip().lower(),
            "failure_type": ft if isinstance(ft, str) else None,
            "confidence": conf,
            "reason": reason if isinstance(reason, str) else ""}


def label_frame(path: Path, endpoint: str = DEFAULT_ENDPOINT,
                model: str = DEFAULT_MODEL, caller=call_lmstudio,
                timeout: float = DEFAULT_TIMEOUT, max_side: int = 512) -> dict:
    try:
        data_url = image_to_data_url_scaled(Path(path), max_side)
        text = caller(endpoint, model, build_prompt(), [data_url], timeout)
    except Exception as e:  # unreadable frame / network / HTTP -> uncertain proposal
        return {"frame": Path(path).name, "path": str(path), "label": "uncertain",
                "failure_type": None, "confidence": 0.0, "reason": f"error: {e}"}
    parsed = parse_label_response(text)
    parsed.update({"frame": Path(path).name, "path": str(path)})
    return parsed


def _majority(counts: dict) -> str:
    f, n = counts.get("failing", 0), counts.get("normal", 0)
    if f > n:
        return "failing"
    if n > f:
        return "normal"
    return "uncertain"


def autolabel_source(source_id: str, frames: list[Path],
                     endpoint: str = DEFAULT_ENDPOINT, model: str = DEFAULT_MODEL,
                     caller=call_lmstudio, timeout: float = DEFAULT_TIMEOUT,
                     max_side: int = 512, sleep=None) -> dict:
    counts = {k: 0 for k in LABELS}
    out = []
    for i, path in enumerate(frames):
        if i and sleep is not None:
            sleep(0.0)
        proposal = label_frame(path, endpoint=endpoint, model=model, caller=caller,
                               timeout=timeout, max_side=max_side)
        counts[proposal["label"]] += 1
        out.append(proposal)
    return {"id": source_id, "model": model, "endpoint": endpoint,
            "video_label": _majority(counts), "counts": counts, "frames": out}
