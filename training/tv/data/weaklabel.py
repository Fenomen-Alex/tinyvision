"""Video-level weak labelling from titles/descriptions.

Video-level only. Frame-level labels come from autolabel. ``uncertain`` is never
silently resolved; it is preserved for review.
"""
from __future__ import annotations

FAILING_KEYWORDS: tuple[str, ...] = (
    "spaghetti", "fail", "failed", "failure", "blob", "layer shift",
    "layershift", "warp", "warping", "adhesion", "stringing", "clog",
    "clogged", "under extrusion", "under-extrusion", "messed up", "disaster",
    "detached", "knocked over", "broken print", "print fail",
)
NORMAL_KEYWORDS: tuple[str, ...] = (
    "timelapse", "time-lapse", "time lapse", "success", "successful",
    "finished", "complete", "asmr", "perfect print", "print test",
)


def weak_label(title: str, description: str = "") -> str:
    """Return ``"failing" | "normal" | "uncertain"`` from video text."""
    text = f"{title} {description}".lower()
    if any(k in text for k in FAILING_KEYWORDS):
        return "failing"
    if any(k in text for k in NORMAL_KEYWORDS):
        return "normal"
    return "uncertain"
