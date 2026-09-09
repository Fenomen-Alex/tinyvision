"""Training-readiness reporting and the next-training gate.

Combines the *canonical holdings* snapshot (already-owned frozen assets such as
the stage-13/14 splits, seven-video holdout and SKZ eval) with the staged
dataset manifests to answer: is this enough to run the next serious training
run?

Gate conditions (from the data-strategy research, §9.7/§12):

1. at least 20 training environments (train split) in total;
2. at least 5 *new* human-verified failure environments (staged after this
   pipeline) — historical holdings do not count; failure environments are
   identified by ``metadata.failure_type`` and are only counted as
   human-verified once at least one of their samples carries a verified
   failing label with ``human:*`` provenance;
3. at least 2 new normal holdout environments (external_holdout split);
4. at least 2 new failure holdout environments.

A report is produced regardless; ``training_ready`` is True only when all four
gates pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tv.envdata.schema import FAILURE_TYPE_VOCAB, Manifest, Sample

HOLDING_KINDS = ("train", "holdout")

# strategy target coverage for failure types across the combined pool
FAILURE_TYPE_TARGETS = {
    "spaghetti": 3,
    "under_extrusion": 3,
    "stringing": 2,
    "bed_adhesion": 2,
    "warping": 2,
    "layer_shift": 1,
    "extrusion_error": 1,
}

GATE_CONDITIONS = {
    "training_environments_min": (20, "training environments (train split)"),
    "new_failure_envs_human_verified_min": (5, "new human-verified failure environments"),
    "new_normal_holdout_envs_min": (2, "new normal holdout environments"),
    "new_failure_holdout_envs_min": (2, "new failure holdout environments"),
}


@dataclass
class HoldingAsset:
    name: str
    kind: str
    normal_envs: int = 0
    failure_envs: int = 0
    failure_types: dict[str, int] = field(default_factory=dict)
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "normal_envs": self.normal_envs,
            "failure_envs": self.failure_envs,
            "failure_types": dict(self.failure_types),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HoldingAsset":
        return cls(
            name=data["name"],
            kind=data["kind"],
            normal_envs=int(data.get("normal_envs", 0)),
            failure_envs=int(data.get("failure_envs", 0)),
            failure_types=dict(data.get("failure_types") or {}),
            note=data.get("note"),
        )


@dataclass
class HoldingsSnapshot:
    updated: str
    assets: list[HoldingAsset] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "tinyvision-canonical-holdings",
            "updated": self.updated,
            "assets": [a.to_dict() for a in self.assets],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HoldingsSnapshot":
        return cls(
            updated=data["updated"],
            assets=[HoldingAsset.from_dict(a) for a in data.get("assets", [])],
        )

    @classmethod
    def load(cls, path: str | Path) -> "HoldingsSnapshot":
        import json
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def train_envs(self) -> tuple[int, int]:
        normal = sum(a.normal_envs for a in self.assets if a.kind == "train")
        failure = sum(a.failure_envs for a in self.assets if a.kind == "train")
        return normal, failure

    def holdout_envs(self) -> tuple[int, int]:
        normal = sum(a.normal_envs for a in self.assets if a.kind == "holdout")
        failure = sum(a.failure_envs for a in self.assets if a.kind == "holdout")
        return normal, failure


def _sample_counts_by_env(samples: list[Sample]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for s in samples:
        entry = counts.setdefault(s.environment_id, {"verified_failing": 0, "verified": 0, "pending": 0})
        if s.verification == "verified":
            entry["verified"] += 1
            if s.tinyvision_label == "failing" and str(s.verification_provenance).startswith("human"):
                entry["verified_failing"] += 1
        elif s.verification == "pending":
            entry["pending"] += 1
    return counts


@dataclass
class StagedManifestView:
    manifest: Manifest
    samples: list[Sample]

    def env_counts(self) -> dict[str, dict[str, int]]:
        env_counts: dict[str, dict[str, int]] = {}
        sample_counts = _sample_counts_by_env(self.samples)
        for env in self.manifest.environments:
            split = env.split_assignment or "unassigned"
            failure = env.has_failure_type()
            sc = sample_counts.get(env.environment_id, {"verified_failing": 0, "verified": 0, "pending": 0})
            entry = env_counts.setdefault(
                split, {"normal": 0, "failure": 0, "verified_failing_envs": 0}
            )
            if failure:
                entry["failure"] += 1
                if sc["verified_failing"] > 0:
                    entry["verified_failing_envs"] += 1
            else:
                entry["normal"] += 1
        return env_counts

    def failure_type_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for env in self.manifest.environments:
            ft = env.metadata.get("failure_type")
            if isinstance(ft, str) and ft not in (None, "", "none"):
                dist[ft] = dist.get(ft, 0) + 1
        return dist


def _gate(current: int, required: int, label: str) -> dict[str, Any]:
    return {
        "required": required,
        "current": current,
        "met": current >= required,
        "delta": max(0, required - current),
        "label": label,
    }


def compute_readiness(
    holdings: HoldingsSnapshot,
    staged: list[StagedManifestView],
) -> dict[str, Any]:
    hold_train_n, hold_train_f = holdings.train_envs()
    hold_hold_n, hold_hold_f = holdings.holdout_envs()

    staged_train_n = 0
    staged_train_f = 0
    staged_verified_f_envs = 0
    staged_hold_n = 0
    staged_hold_f = 0
    staged_hold_verified_f_envs = 0
    failure_types: dict[str, int] = {}
    for view in staged:
        counts = view.env_counts()
        train = counts.get("train", {"normal": 0, "failure": 0, "verified_failing_envs": 0})
        hold = counts.get(
            "external_holdout", {"normal": 0, "failure": 0, "verified_failing_envs": 0}
        )
        staged_train_n += train["normal"]
        staged_train_f += train["failure"]
        staged_verified_f_envs += train["verified_failing_envs"]
        staged_hold_n += hold["normal"]
        staged_hold_f += hold["failure"]
        staged_hold_verified_f_envs += hold["verified_failing_envs"]
        for ft, n in view.failure_type_distribution().items():
            failure_types[ft] = failure_types.get(ft, 0) + n

    total_train_envs = hold_train_n + hold_train_f + staged_train_n + staged_train_f
    new_failure_envs_human_verified = staged_verified_f_envs + staged_hold_verified_f_envs
    new_hold_n = staged_hold_n
    new_hold_f = staged_hold_f

    holdings_failure_types: dict[str, int] = {}
    for asset in holdings.assets:
        for ft, n in asset.failure_types.items():
            holdings_failure_types[ft] = holdings_failure_types.get(ft, 0) + n

    gate_values = {
        "training_environments_min": total_train_envs,
        "new_failure_envs_human_verified_min": new_failure_envs_human_verified,
        "new_normal_holdout_envs_min": new_hold_n,
        "new_failure_holdout_envs_min": new_hold_f,
    }
    gates = {
        name: _gate(gate_values[name], req, label)
        for name, (req, label) in GATE_CONDITIONS.items()
    }

    coverage = {}
    for ft in FAILURE_TYPE_VOCAB:
        current = failure_types.get(ft, 0) + holdings_failure_types.get(ft, 0)
        target = FAILURE_TYPE_TARGETS.get(ft, 0)
        coverage[ft] = {"current": current, "target": target, "met": current >= target}
    met_all = all(c["met"] for c in coverage.values())
    coverage_gaps = [ft for ft, c in coverage.items() if not c["met"]]

    overall = all(g["met"] for g in gates.values())

    return {
        "schema": "tinyvision-readiness-report",
        "training_ready": overall,
        "gates": gates,
        "breakdown": {
            "holdings": {
                "train": {"normal": hold_train_n, "failure": hold_train_f},
                "holdout": {"normal": hold_hold_n, "failure": hold_hold_f},
            },
            "staged": {
                "train": {"normal": staged_train_n, "failure": staged_train_f},
                "holdout": {"normal": staged_hold_n, "failure": staged_hold_f},
                "verified_failure_envs": staged_verified_f_envs,
                "holdout_verified_failure_envs": staged_hold_verified_f_envs,
            },
            "failure_type_coverage": coverage,
            "failure_type_fully_covered": met_all,
            "missing_failure_types": coverage_gaps,
        },
    }


def generate_booster_text(report: dict[str, Any]) -> str:
    b = report["breakdown"]
    gaps = [
        f"{c['target'] - c['current']}× {ft} (target {c['target']}, have {c['current']})"
        for ft, c in b["failure_type_coverage"].items()
        if not c["met"]
    ]
    lines = []
    for name, g in report["gates"].items():
        if not g["met"]:
            lines.append(f"  - {name}: need {g['delta']} more ({g['label']})")
    ft_line = "none" if not gaps else ", ".join(gaps)
    lines.append(f"  - failure-type coverage gaps: {ft_line}")
    return "Recovery plan:\n" + "\n".join(lines)


def default_holdings_path() -> Path:
    return Path("datasets/staging/readiness/canonical_holdings.json")