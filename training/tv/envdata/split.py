"""Environment-level dataset splitting.

Splitting operates on *environments* only: every environment belongs to exactly
one split (``train``, ``validation`` or ``external_holdout``) and no recording
can live in more than one environment. This makes frame-level leakage between
training and evaluation impossible for observations from the same camera setup.

Rules enforced here:

* one environment -> exactly one split;
* one environment key -> exactly one environment ID;
* environment IDs and recording IDs are unique;
* every recording references an environment that exists;
* assignments are explicit and validated; deterministic auto-allocation is
  available via a seeded RNG over environment IDs only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from tv.envdata.schema import (
    SPLITS,
    Environment,
    Manifest,
    Recording,
)

SPLIT_VOCAB = SPLITS
ASSIGNABLE_SPLITS = ("train", "validation", "external_holdout")


@dataclass
class LeakageIssue:
    code: str
    message: str
    environment_id: str | None = None
    recording_id: str | None = None


@dataclass
class LeakageReport:
    issues: list[LeakageIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def add(self, code: str, message: str, **kw: str | None) -> None:
        self.issues.append(LeakageIssue(code=code, message=message, **kw))

    def codes(self) -> list[str]:
        return [i.code for i in self.issues]


def check_environment_leakage(manifest: Manifest) -> LeakageReport:
    """Verify every environment integrity invariant. Returns report with no
    issues on success."""
    report = LeakageReport()
    envs = manifest.env_index()
    recs = manifest.recording_index()

    for env in manifest.environments:
        if env.split_assignment is not None and env.split_assignment not in SPLIT_VOCAB:
            report.add(
                "bad_split",
                f"environment {env.environment_id} has unknown split "
                f"{env.split_assignment!r}",
                environment_id=env.environment_id,
            )

    keys: dict[str, str] = {}
    for env in manifest.environments:
        prev = keys.get(env.environment_key)
        if prev is not None and prev != env.environment_id:
            report.add(
                "key_collision",
                f"environment_key {env.environment_key!r} maps to both "
                f"{prev} and {env.environment_id}",
                environment_id=env.environment_id,
            )
        keys[env.environment_key] = env.environment_id

    rec_env: dict[str, str] = {}
    for rec in manifest.recordings:
        if rec.environment_id not in envs:
            report.add(
                "dangling_recording",
                f"recording {rec.recording_id} references missing environment "
                f"{rec.environment_id}",
                recording_id=rec.recording_id,
            )
            continue
        prev = rec_env.get(rec.recording_id)
        if prev is not None and prev != rec.environment_id:
            report.add(
                "recording_multi_env",
                f"recording {rec.recording_id} referenced by multiple "
                f"environments ({prev}, {rec.environment_id})",
                recording_id=rec.recording_id,
            )
        rec_env[rec.recording_id] = rec.environment_id

    unassigned = [e.environment_id for e in manifest.environments if e.split_assignment is None]
    if unassigned:
        report.add(
            "partial_assignment",
            f"{len(unassigned)} environment(s) have no split assignment: "
            f"{', '.join(unassigned[:5])}{'...' if len(unassigned) > 5 else ''}",
        )
    return report


@dataclass
class SplitPlan:
    env_count: dict[str, int] = field(default_factory=lambda: {s: 0 for s in SPLIT_VOCAB})
    allocations: dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        parts = ", ".join(f"{s}={n}" for s, n in self.env_count.items() if n)
        return f"environments per split: {parts}"


def allocate_environments(
    environment_ids: list[str],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> SplitPlan:
    """Deterministically allocate environments to train/validation/holdout.

    Allocation is over environment IDs only (never frames/recordings). With
    fewer than ``1/ratio`` environments the training split wins the remainder so
    tiny pools still produce a training set; callers are expected to review the
    plan for small pools.
    """
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must be a triple summing to 1.0")
    n = len(environment_ids)
    ids = list(environment_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_train = max(1, round(n * ratios[0]))
    n_val = round(n * ratios[1])
    n_val = max(0, min(n_val, n - n_train))
    plan = SplitPlan()
    plan.allocations = {ec: "train" for ec in ids[:n_train]}
    plan.allocations.update({ec: "validation" for ec in ids[n_train : n_train + n_val]})
    plan.allocations.update({ec: "external_holdout" for ec in ids[n_train + n_val :]})
    for s in plan.allocations.values():
        plan.env_count[s] += 1
    return plan


def apply_plan(manifest: Manifest, plan: SplitPlan) -> list[str]:
    """Apply an allocation to the manifest, returning the list of changed env IDs."""
    changed: list[str] = []
    envs = manifest.env_index()
    for env_id, split in plan.allocations.items():
        if env_id not in envs:
            raise KeyError(f"allocation references unknown environment {env_id}")
        if split not in SPLIT_VOCAB:
            raise ValueError(f"bad split {split!r} for {env_id}")
        if envs[env_id].split_assignment != split:
            envs[env_id].split_assignment = split
            changed.append(env_id)
    return changed


def drop_assignments(manifest: Manifest, environment_ids: list[str] | None = None) -> list[str]:
    """Clear split assignments for the given environments (or all)."""
    changed: list[str] = []
    for env in manifest.environments:
        if environment_ids is not None and env.environment_id not in environment_ids:
            continue
        if env.split_assignment is not None:
            env.split_assignment = None
            changed.append(env.environment_id)
    return changed


def group_recordings_by_environment(manifest: Manifest) -> dict[str, list[Recording]]:
    grouped: dict[str, list[Recording]] = {}
    for rec in manifest.recordings:
        grouped.setdefault(rec.environment_id, []).append(rec)
    return grouped


def environment_recording_counts(manifest: Manifest) -> dict[str, int]:
    return {env_id: len(recs) for env_id, recs in group_recordings_by_environment(manifest).items()}