"""Verification state tracking for manifest samples.

A source annotation (``source_label`` from an external dataset or a contributor)
is never automatically promoted to a TinyVision ``tinyvision_label``. The label
TinyVision actually trusts can be set only by transitioning a sample to
``verified`` with an acceptable provenance (a human review stage such as
``human:stage12d``, or an approved pipeline stage).

State machine::

    pending  ->  verified | rejected | uncertain
    uncertain -> verified | rejected
    verified  -> (immutable; explicit override with reason)
    rejected  -> (immutable; explicit override with reason)

All transitions require a provenance string; ``verified`` additionally rejects
banned provenance values (``source_auto``, ``source``, ``vlm_only``).
"""

from __future__ import annotations

from tv.envdata.schema import (
    ALLOWED_VERIFIED_PROVENANCE,
    BANNED_VERIFICATION_PROVENANCE,
    FAILURE_TYPE_VOCAB,
    TINYVISION_LABELS,
    VERIFICATION_STATES,
    Sample,
)

_TRANSITIONS = {
    "pending": {"verified", "rejected", "uncertain"},
    "uncertain": {"verified", "rejected"},
    "verified": set(),
    "rejected": set(),
}


class TransitionError(ValueError):
    pass


def allowed_transitions(state: str) -> set[str]:
    if state not in VERIFICATION_STATES:
        raise ValueError(f"unknown verification state {state!r}")
    return set(_TRANSITIONS[state])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TransitionError(message)


def _check_provenance(provenance: str | None, to: str) -> None:
    _require(provenance and provenance.strip(), f"{to} requires a provenance string")
    provenance = provenance.strip()
    if to == "verified":
        _require(
            provenance not in BANNED_VERIFICATION_PROVENANCE,
            f"provenance {provenance!r} is not allowed to set a verified label",
        )
        _require(
            provenance.split(":")[0] in ALLOWED_VERIFIED_PROVENANCE,
            f"provenance {provenance!r} must start with one of "
            f"{sorted(ALLOWED_VERIFIED_PROVENANCE)}",
        )


def set_verification(
    sample: Sample,
    to: str,
    provenance: str,
    *,
    tinyvision_label: str | None = None,
    tinyvision_failure_type: str | None = None,
) -> None:
    """Transition a sample's verification state (in-place, mutating ``sample``)."""
    if to not in VERIFICATION_STATES:
        raise TransitionError(f"unknown target state {to!r}")
    if to not in allowed_transitions(sample.verification):
        raise TransitionError(
            f"cannot move sample {sample.sample_id} verification "
            f"{sample.verification} -> {to}; allowed: {sorted(allowed_transitions(sample.verification))}"
        )
    _check_provenance(provenance, to)
    if to == "verified":
        _require(
            tinyvision_label in TINYVISION_LABELS,
            "verified samples require tinyvision_label in "
            f"{TINYVISION_LABELS}, got {tinyvision_label!r}",
        )
        if tinyvision_failure_type is not None:
            _require(
                tinyvision_failure_type in FAILURE_TYPE_VOCAB,
                f"unknown failure type {tinyvision_failure_type!r}",
            )
        sample.tinyvision_label = tinyvision_label
        sample.tinyvision_failure_type = tinyvision_failure_type
    else:
        _require(
            tinyvision_label is None and tinyvision_failure_type is None,
            "only 'verified' may set tinyvision_label / tinyvision_failure_type",
        )
    sample.verification = to
    sample.verification_provenance = provenance


def force_verification(
    sample: Sample,
    to: str,
    provenance: str,
    reason: str,
    *,
    tinyvision_label: str | None = None,
    tinyvision_failure_type: str | None = None,
) -> None:
    """Override an otherwise-immutable state (verified/rejected). Must supply a
    reason. Still subject to provenance/label rules for ``verified``."""
    _require(reason and reason.strip(), "force_verification requires a reason")
    if to not in VERIFICATION_STATES:
        raise TransitionError(f"unknown target state {to!r}")
    _check_provenance(provenance, to)
    if to == "verified":
        _require(
            tinyvision_label in TINYVISION_LABELS,
            "verified samples require tinyvision_label",
        )
        if tinyvision_failure_type is not None:
            _require(
                tinyvision_failure_type in FAILURE_TYPE_VOCAB,
                f"unknown failure type {tinyvision_failure_type!r}",
            )
        sample.tinyvision_label = tinyvision_label
        sample.tinyvision_failure_type = tinyvision_failure_type
    sample.verification = to
    sample.verification_provenance = provenance