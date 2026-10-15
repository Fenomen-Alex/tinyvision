"""Repeatable, auditable reliability evaluation and release acceptance gates.

Design goals (see docs/reliability-evaluation.md):

* One entry point (``tv reliability evaluate``) over named evaluation suites.
* Explicit score convention: the **failing** class is positive (class index 0),
  so ``score = P(failing)`` and a frame is predicted ``failing`` when
  ``score >= threshold``.
* Undefined metrics are reported as ``null`` with a reason — never silently
  coerced to 0.0 (e.g. FPR on a surface with no normal samples).
* Group/environment-aware metrics; leakage checks across source groups.
* Threshold comparison, with selection restricted to a validation surface.
* Versioned acceptance policy producing PASS / WARN / FAIL / INCONCLUSIVE,
  where missing or insufficient evidence is INCONCLUSIVE (never PASS).
"""
from __future__ import annotations

import hashlib
import json
import math
import gc
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Failing is the positive class across TinyVision (class_to_idx sorted
# alphabetically: failing=0, normal=1).
DEFAULT_POSITIVE_CLASS = 0
FAIL_IDX = 0
NORMAL_IDX = 1

# Surface roles. Only dev_validation may be used for threshold/model selection.
ROLE_DEV_VALIDATION = "dev_validation"
ROLE_NORMAL_HOLDOUT = "normal_holdout"
ROLE_FAILURE_HOLDOUT = "failure_holdout"
ROLE_CROSS_ENVIRONMENT = "cross_environment"
ROLE_LOCKED_TEST = "locked_test"
INDEPENDENT_ROLES = {
    ROLE_NORMAL_HOLDOUT,
    ROLE_FAILURE_HOLDOUT,
    ROLE_CROSS_ENVIRONMENT,
    ROLE_LOCKED_TEST,
}

STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"
STATUS_INCONCLUSIVE = "INCONCLUSIVE"

# ONNX/INT8 parity comparisons run on at most this many samples per surface
# (bounded so fixed-batch=1 export formats stay cheap; the comparison measures
# format drift, which needs only a representative subset).
PARITY_MAX_SAMPLES = 512


class MissingDataError(RuntimeError):
    """Raised when required evaluation data is absent.

    Missing data must never produce a passing evaluation.
    """


class SuiteError(ValueError):
    """Raised for malformed suite definitions."""


# ---------------------------------------------------------------------------
# Hashing / resolution helpers
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _data_path_candidates(path: str, dataset_root: str | Path | None) -> list[Path]:
    p = Path(path)
    if p.is_absolute():
        return [p]
    candidates = []
    if dataset_root is not None:
        candidates.append(Path(dataset_root) / p)
    candidates.append(Path.cwd() / p)
    candidates.append(p)
    return candidates


def _resolve_data_path(path: str, dataset_root: str | Path | None) -> Path:
    """Resolve a possibly-relative dataset path against candidate roots."""
    for c in _data_path_candidates(path, dataset_root):
        if c.exists():
            return c
    # Return the primary candidate so callers can show a useful path.
    return _data_path_candidates(path, dataset_root)[0]


# ---------------------------------------------------------------------------
# Samples and surfaces
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    image_path: str
    label: int
    source_id: str | None = None
    environment_id: str | None = None
    surface_id: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class Surface:
    id: str
    role: str
    format: str
    path: str
    label_map: dict[str, int]
    image_key: str | None = None
    label_key: str | None = None
    source_key: str | None = None
    environment_key: str | None = None
    entries_key: str = "entries"
    filter_field: str | None = None
    filter_equals: object | None = None
    provenance: str = "unknown"
    verified: bool = False
    license: str = "unknown"
    notes: str = ""

    def resolved(self, dataset_root: str | Path | None) -> Path:
        return _resolve_data_path(self.path, dataset_root)


@dataclass
class Suite:
    name: str
    version: int
    surfaces: list[Surface]
    positive_class: int = DEFAULT_POSITIVE_CLASS
    dataset_root: str | None = None
    path: str | None = None

    def surface(self, sid: str) -> Surface:
        for s in self.surfaces:
            if s.id == sid:
                return s
        raise SuiteError(f"no surface with id {sid!r} in suite {self.name!r}")


def load_suite(path: str | Path, dataset_root: str | Path | None = None) -> Suite:
    import yaml

    p = Path(path)
    raw = yaml.safe_load(p.read_text()) or {}
    surfaces_raw = raw.get("surfaces")
    if not surfaces_raw:
        raise SuiteError(f"suite {p} defines no surfaces")
    surfaces = []
    seen: set[str] = set()
    for s in surfaces_raw:
        sid = s.get("id")
        if not sid:
            raise SuiteError(f"suite {p}: surface missing 'id'")
        if sid in seen:
            raise SuiteError(f"suite {p}: duplicate surface id {sid!r}")
        seen.add(sid)
        role = s.get("role")
        if role is None:
            raise SuiteError(f"surface {sid!r}: missing 'role'")
        fmt = s.get("format", "csv")
        if fmt not in ("csv", "manifest_json"):
            raise SuiteError(f"surface {sid!r}: unknown format {fmt!r}")
        if "path" not in s:
            raise SuiteError(f"surface {sid!r}: missing 'path'")
        flt = s.get("filter") or {}
        surfaces.append(Surface(
            id=sid,
            role=role,
            format=fmt,
            path=str(s["path"]),
            label_map={str(k): int(v) for k, v in (s.get("label_map") or {}).items()},
            image_key=s.get("image_key"),
            label_key=s.get("label_key"),
            source_key=s.get("source_key"),
            environment_key=s.get("environment_key"),
            entries_key=s.get("entries_key", "entries"),
            filter_field=flt.get("field"),
            filter_equals=flt.get("equals"),
            provenance=s.get("provenance", "unknown"),
            verified=bool(s.get("verified", False)),
            license=s.get("license", "unknown"),
            notes=s.get("notes", ""),
        ))
    root = dataset_root if dataset_root is not None else raw.get("dataset_root")
    return Suite(
        name=raw.get("name", Path(path).stem),
        version=int(raw.get("version", 1)),
        surfaces=surfaces,
        positive_class=int(raw.get("positive_class", DEFAULT_POSITIVE_CLASS)),
        dataset_root=root,
        path=str(p),
    )


def _map_label(value: object, label_map: dict[str, int], surface_id: str) -> int:
    key = str(value)
    if key in label_map:
        return label_map[key]
    # Allow numeric labels already in {0,1}.
    try:
        n = int(key)
    except (TypeError, ValueError):
        raise SuiteError(
            f"surface {surface_id!r}: label {value!r} not in label_map {label_map}"
        )
    return n


def load_samples(surface: Surface, dataset_root: str | Path | None) -> list[Sample]:
    """Load samples for a surface. Raises MissingDataError when absent.

    Never silently returns an empty list for missing files.
    """
    import csv

    resolved = surface.resolved(dataset_root)
    if not resolved.exists():
        tried = ", ".join(str(c) for c in _data_path_candidates(surface.path, dataset_root))
        raise MissingDataError(
            f"surface {surface.id!r}: data not found (tried: {tried}). "
            f"Set the suite's dataset_root or pass --dataset-root pointing at "
            f"the directory that contains {surface.path!r}. "
            f"(External/holdout datasets are intentionally not committed.)"
        )

    def _image(img: str) -> str:
        # Relative image paths are anchored at the dataset root (e.g. a CSV in
        # splits/ referencing images under the data dir), not at the CSV itself.
        if Path(img).is_absolute():
            return img
        base = Path(dataset_root) if dataset_root is not None else resolved.parent
        return str(base / img)

    samples: list[Sample] = []
    if surface.format == "csv":
        label_key = surface.label_key or "label"
        image_key = surface.image_key or "path"
        with open(resolved) as f:
            for row in csv.DictReader(f):
                if surface.filter_field and str(row.get(surface.filter_field)) != str(surface.filter_equals):
                    continue
                samples.append(Sample(
                    image_path=_image(row[image_key]),
                    label=_map_label(row[label_key], surface.label_map, surface.id),
                    source_id=row.get(surface.source_key) if surface.source_key else None,
                    environment_id=row.get(surface.environment_key) if surface.environment_key else None,
                    surface_id=surface.id,
                    extra=dict(row),
                ))
    elif surface.format == "manifest_json":
        data = json.loads(resolved.read_text())
        entries = data.get(surface.entries_key, data if isinstance(data, list) else [])
        if not isinstance(entries, list):
            raise SuiteError(f"surface {surface.id!r}: entries must be a list")
        image_key = surface.image_key or "image_path"
        label_key = surface.label_key or "class"
        for e in entries:
            if surface.filter_field and str(e.get(surface.filter_field)) != str(surface.filter_equals):
                continue
            img = _image(e[image_key])
            samples.append(Sample(
                image_path=img,
                label=_map_label(e[label_key], surface.label_map, surface.id),
                source_id=e.get(surface.source_key) if surface.source_key else None,
                environment_id=e.get(surface.environment_key) if surface.environment_key else None,
                surface_id=surface.id,
                extra=dict(e),
            ))
    else:  # pragma: no cover - guarded in load_suite
        raise SuiteError(f"surface {surface.id!r}: unknown format {surface.format!r}")

    if not samples:
        raise MissingDataError(
            f"surface {surface.id!r}: file {resolved} exists but contains 0 usable "
            f"samples (check filter/label_map)."
        )
    return samples


# ---------------------------------------------------------------------------
# Metrics (undefined metrics are None, never 0.0)
# ---------------------------------------------------------------------------

def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    """Wilson score interval for a proportion. None when total == 0."""
    if total <= 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)) / denom
    return [round(max(0.0, centre - half), 6), round(min(1.0, centre + half), 6)]


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
    positive: int = DEFAULT_POSITIVE_CLASS,
) -> dict:
    """Binary metrics with a configurable positive class.

    ``positive`` is the integer label of the positive class (failing=0 by
    default). Returns ``None`` (with an ``undefined`` reason) for metrics that
    are mathematically undefined on the given sample, rather than 0.0.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    negative = 1 - positive
    n = int(len(y_true))

    tp = int(np.sum((y_true == positive) & (y_pred == positive)))
    tn = int(np.sum((y_true == negative) & (y_pred == negative)))
    fp = int(np.sum((y_true == negative) & (y_pred == positive)))
    fn = int(np.sum((y_true == positive) & (y_pred == negative)))

    n_pos = tp + fn
    n_neg = tn + fp

    undefined: list[str] = []

    def _ratio(num: int, den: int, name: str):
        if den == 0:
            undefined.append(name)
            return None
        return round(num / den, 6)

    recall = _ratio(tp, n_pos, "recall")
    fpr = _ratio(fp, n_neg, "fpr")
    fnr = _ratio(fn, n_pos, "fnr")
    precision = _ratio(tp, tp + fp, "precision")
    accuracy = round((tp + tn) / n, 6) if n else None
    if n == 0:
        undefined.append("accuracy")
    if precision is not None and recall is not None:
        f1 = round(2 * precision * recall / (precision + recall), 6) if (precision + recall) else 0.0
    else:
        f1 = None
        undefined.append("f1")

    auroc = None
    if scores is not None and n_pos > 0 and n_neg > 0:
        from sklearn.metrics import roc_auc_score

        # sklearn ranks the greater label (1) as positive; orient the score so
        # the configured positive class is ranked highest.
        oriented = np.asarray(scores, dtype=float)
        if positive != 1:
            oriented = -oriented
        try:
            auroc = round(float(roc_auc_score(y_true, oriented)), 6)
        except ValueError:
            auroc = None
            undefined.append("auroc")
    else:
        undefined.append("auroc")

    return {
        "positive_class": int(positive),
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fpr": fpr,
        "fnr": fnr,
        "auroc": auroc,
        "fpr_ci95": wilson_interval(fp, n_neg),
        "fnr_ci95": wilson_interval(fn, n_pos),
        "undefined": sorted(set(undefined)),
        "confusion_matrix": [[tn, fp], [fn, tp]],  # rows true [neg,pos], cols pred
    }


def predictions_from_scores(scores: np.ndarray, threshold: float,
                            positive: int = DEFAULT_POSITIVE_CLASS) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    return np.where(scores >= threshold, positive, 1 - positive)


# ---------------------------------------------------------------------------
# Leakage checks
# ---------------------------------------------------------------------------

@dataclass
class Leak:
    code: str
    message: str
    surfaces: list[str] = field(default_factory=list)


def check_source_leakage(samples_by_surface: dict[str, list[Sample]],
                         roles: dict[str, str] | None = None) -> list[Leak]:
    """Detect identical samples and source-group overlap across surfaces.

    Independent surfaces (holdouts / locked test) must not share a source group
    or an identical image with any other surface. This catches the classic
    frame-level leakage that looks like generalization.
    """
    roles = roles or {}
    leaks: list[Leak] = []

    # Exact duplicate image paths across different surfaces.
    path_to_surfaces: dict[str, set[str]] = {}
    for sid, samples in samples_by_surface.items():
        for s in samples:
            key = str(Path(s.image_path))
            path_to_surfaces.setdefault(key, set()).add(sid)
    for key, sids in path_to_surfaces.items():
        if len(sids) > 1:
            leaks.append(Leak(
                code="duplicate_image",
                message=f"identical image appears in surfaces {sorted(sids)}: {key}",
                surfaces=sorted(sids),
            ))

    # Source-group overlap.
    source_to_surfaces: dict[str, set[str]] = {}
    for sid, samples in samples_by_surface.items():
        for s in samples:
            if s.source_id is None:
                continue
            source_to_surfaces.setdefault(str(s.source_id), set()).add(sid)
    for source, sids in source_to_surfaces.items():
        if len(sids) > 1:
            # Only a leak if at least one of the surfaces is independent.
            if any(roles.get(x) in INDEPENDENT_ROLES for x in sids):
                leaks.append(Leak(
                    code="source_group_leak",
                    message=(
                        f"source/env group {source!r} appears in surfaces {sorted(sids)}; "
                        f"independent surfaces must not share source groups"
                    ),
                    surfaces=sorted(sids),
                ))
    return leaks


# ---------------------------------------------------------------------------
# Threshold sweep and selection
# ---------------------------------------------------------------------------

def default_threshold_grid() -> list[float]:
    return [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95


def threshold_sweep(y_true: np.ndarray, scores: np.ndarray, thresholds: list[float],
                    positive: int = DEFAULT_POSITIVE_CLASS) -> list[dict]:
    out = []
    for t in thresholds:
        preds = predictions_from_scores(scores, t, positive)
        m = compute_binary_metrics(y_true, preds, scores, positive)
        m["threshold"] = round(float(t), 6)
        out.append(m)
    return out


def select_threshold(y_true: np.ndarray, scores: np.ndarray,
                     objective: str = "max_recall_subject_to_max_fpr",
                     max_fpr: float | None = None,
                     grid: list[float] | None = None,
                     min_samples: int = 1,
                     positive: int = DEFAULT_POSITIVE_CLASS) -> dict:
    """Choose a threshold using validation data only.

    Returns a dict with ``threshold`` (may be None), ``status`` and ``reason``.
    If the validation surface has no usable normal samples for the FPR
    constraint, selection is INCONCLUSIVE and the caller must fall back to a
    fixed threshold without claiming it was tuned.
    """
    grid = grid or default_threshold_grid()
    y_true = np.asarray(y_true).astype(int)
    n_neg = int(np.sum(y_true != positive))
    n_pos = int(np.sum(y_true == positive))

    if len(y_true) < min_samples:
        return {"threshold": None, "status": STATUS_INCONCLUSIVE,
                "reason": f"validation surface has {len(y_true)} samples < min {min_samples}"}
    if objective == "max_recall_subject_to_max_fpr":
        if max_fpr is None or n_neg == 0 or n_pos == 0:
            return {"threshold": None, "status": STATUS_INCONCLUSIVE,
                    "reason": "objective requires both normal and failing samples on validation data"}
        sweep = threshold_sweep(y_true, scores, grid, positive)
        feasible = [s for s in sweep if s["fpr"] is not None and s["fpr"] <= max_fpr]
        if not feasible:
            return {"threshold": None, "status": STATUS_INCONCLUSIVE,
                    "reason": f"no threshold in grid satisfies FPR <= {max_fpr}"}
        # Highest recall; among those, the LOWEST threshold (most sensitive).
        # The FPR ceiling is already enforced by the feasible filter, so the
        # tie-break maximises sensitivity, matching the project priority that
        # false negatives matter most for failure detection.
        best = sorted(feasible, key=lambda s: (-(s["recall"] or 0.0), s["threshold"]))[0]
        return {"threshold": best["threshold"], "status": STATUS_PASS,
                "reason": f"max recall with FPR <= {max_fpr} on validation data",
                "validation_metrics": best}
    if objective == "max_f1":
        sweep = threshold_sweep(y_true, scores, grid, positive)
        scored = [s for s in sweep if s["f1"] is not None]
        if not scored:
            return {"threshold": None, "status": STATUS_INCONCLUSIVE,
                    "reason": "F1 undefined across grid"}
        best = max(scored, key=lambda s: s["f1"])
        return {"threshold": best["threshold"], "status": STATUS_PASS,
                "reason": "max F1 on validation data", "validation_metrics": best}
    if objective == "fixed":
        return {"threshold": None, "status": STATUS_PASS, "reason": "fixed threshold requested"}
    return {"threshold": None, "status": STATUS_INCONCLUSIVE,
            "reason": f"unknown objective {objective!r}"}


# ---------------------------------------------------------------------------
# Acceptance policy and gates
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    gate: str
    status: str
    reasons: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"gate": self.gate, "status": self.status,
                "reasons": self.reasons, "details": self.details}


def load_policy(path: str | Path) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text()) or {}


def _insufficient(details: dict, min_samples: int | None, min_sources: int | None,
                  n_samples: int, n_sources: int, gate: str) -> GateResult | None:
    reasons = []
    if min_samples is not None and n_samples < min_samples:
        reasons.append(f"{gate}: {n_samples} samples < required {min_samples}")
    if min_sources is not None and n_sources < min_sources:
        reasons.append(f"{gate}: {n_sources} sources < required {min_sources}")
    if reasons:
        return GateResult(gate=gate, status=STATUS_INCONCLUSIVE, reasons=reasons,
                          details=details)
    return None


def evaluate_gates(surface_results: dict[str, dict], policy: dict,
                   extra: dict | None = None) -> dict:
    """Evaluate all gates in ``policy`` against surface results.

    ``surface_results`` maps surface role -> result dict containing at least
    ``metrics`` (from compute_binary_metrics), ``n_sources`` and ``n_samples``.
    ``extra`` carries parity/deployment inputs.
    """
    extra = extra or {}
    gates_cfg = policy.get("gates", {})
    results: list[GateResult] = []

    for role, cfg in gates_cfg.items():
        if role == "parity":
            results.append(_gate_parity(cfg, extra.get("parity")))
            continue
        if role == "deployment":
            results.append(_gate_deployment(cfg, extra.get("deployment")))
            continue
        results.append(_gate_surface(role, cfg, surface_results.get(role)))

    mandatory_fail = [g for g in results if g.status == STATUS_FAIL and _required(gates_cfg, g.gate)]
    mandatory_inc = [g for g in results if g.status == STATUS_INCONCLUSIVE and _required(gates_cfg, g.gate)]
    optional_fail = [g for g in results if g.status == STATUS_FAIL and not _required(gates_cfg, g.gate)]

    if mandatory_fail:
        overall = STATUS_FAIL
    elif mandatory_inc:
        overall = STATUS_INCONCLUSIVE
    elif optional_fail:
        overall = STATUS_WARN
    else:
        overall = STATUS_PASS
    return {
        "overall_status": overall,
        "gates": [g.to_dict() for g in results],
        "blocks_release": overall in (STATUS_FAIL, STATUS_INCONCLUSIVE),
    }


def _required(gates_cfg: dict, gate: str) -> bool:
    cfg = gates_cfg.get(gate, {})
    return bool(cfg.get("required", True))


def _gate_surface(role: str, cfg: dict, result: dict | None) -> GateResult:
    if result is None:
        required = cfg.get("required", True)
        return GateResult(gate=role, status=(
            STATUS_INCONCLUSIVE if required else STATUS_WARN),
            reasons=[f"{role}: {'required' if required else 'optional'} "
                     f"evaluation surface is missing or could not run"])
    m = result.get("metrics", {})
    n_samples = result.get("n_samples", m.get("n", 0))
    n_sources = result.get("n_sources", 0)
    details = {"n_samples": n_samples, "n_sources": n_sources,
               "fpr": m.get("fpr"), "fnr": m.get("fnr"),
               "recall": m.get("recall"), "precision": m.get("precision")}

    insuff = _insufficient(details, cfg.get("min_samples"), cfg.get("min_sources"),
                           n_samples, n_sources, role)
    if insuff is not None:
        return insuff

    reasons = []
    if "max_fpr" in cfg:
        if m.get("fpr") is None:
            return GateResult(gate=role, status=STATUS_INCONCLUSIVE,
                              reasons=[f"{role}: FPR undefined (no normal samples)"],
                              details=details)
        if m["fpr"] > cfg["max_fpr"]:
            reasons.append(f"{role}: FPR {m['fpr']:.4f} > max {cfg['max_fpr']}")
    if "max_fnr" in cfg:
        if m.get("fnr") is None:
            return GateResult(gate=role, status=STATUS_INCONCLUSIVE,
                              reasons=[f"{role}: FNR undefined (no failing samples)"],
                              details=details)
        if m["fnr"] > cfg["max_fnr"]:
            reasons.append(f"{role}: FNR {m['fnr']:.4f} > max {cfg['max_fnr']}")
    if "min_recall" in cfg:
        if m.get("recall") is None:
            return GateResult(gate=role, status=STATUS_INCONCLUSIVE,
                              reasons=[f"{role}: recall undefined"],
                              details=details)
        if m["recall"] < cfg["min_recall"]:
            reasons.append(f"{role}: recall {m['recall']:.4f} < min {cfg['min_recall']}")
    if "min_precision" in cfg:
        if m.get("precision") is None:
            return GateResult(gate=role, status=STATUS_INCONCLUSIVE,
                              reasons=[f"{role}: precision undefined"],
                              details=details)
        if m["precision"] < cfg["min_precision"]:
            reasons.append(f"{role}: precision {m['precision']:.4f} < min {cfg['min_precision']}")
    if "min_environments" in cfg:
        n_env = result.get("n_environments", 0)
        details["n_environments"] = n_env
        if n_env < cfg["min_environments"]:
            return GateResult(gate=role, status=STATUS_INCONCLUSIVE,
                              reasons=[f"{role}: {n_env} environments < required "
                                       f"{cfg['min_environments']}"], details=details)

    status = STATUS_FAIL if reasons else STATUS_PASS
    return GateResult(gate=role, status=status, reasons=reasons, details=details)


def _gate_parity(cfg: dict, parity: dict | None) -> GateResult:
    if parity is None:
        return GateResult(gate="parity", status=(
            STATUS_INCONCLUSIVE if cfg.get("required", True) else STATUS_WARN),
            reasons=["parity: no ONNX/INT8 artifacts were evaluated against the framework model"])
    reasons = []
    details = {}
    for fmt, prefix in (("onnx", "onnx"), ("int8", "int8")):
        block = parity.get(fmt)
        if block is None:
            # A format that is not supplied is only INCONCLUSIVE when the policy
            # explicitly requires that specific format; otherwise it is skipped.
            if cfg.get(f"require_{prefix}"):
                return GateResult(gate="parity", status=STATUS_INCONCLUSIVE,
                                  reasons=[f"parity: required {fmt} artifacts not provided"])
            continue
        details[fmt] = {
            "prediction_disagreement": block.get("prediction_disagreement"),
            "max_score_abs_drift": block.get("max_score_abs_drift"),
        }
        max_dis = cfg.get(f"max_{prefix}_prediction_disagreement",
                          cfg.get("max_prediction_disagreement"))
        if max_dis is not None and block.get("prediction_disagreement", 0.0) > max_dis:
            reasons.append(f"parity: {fmt} prediction disagreement "
                           f"{block['prediction_disagreement']:.4f} > {max_dis}")
        max_drift = cfg.get(f"max_{prefix}_score_abs_drift", cfg.get("max_score_abs_drift"))
        if max_drift is not None and block.get("max_score_abs_drift", 0.0) > max_drift:
            reasons.append(f"parity: {fmt} score abs drift "
                           f"{block['max_score_abs_drift']:.4f} > {max_drift}")
    status = STATUS_FAIL if reasons else STATUS_PASS
    return GateResult(gate="parity", status=status, reasons=reasons, details=details)


def _gate_deployment(cfg: dict, dep: dict | None) -> GateResult:
    if dep is None:
        return GateResult(gate="deployment", status=(
            STATUS_INCONCLUSIVE if cfg.get("required", True) else STATUS_WARN),
            reasons=["deployment: no model size / latency artifacts"])
    reasons = []
    details = dict(dep)
    if "max_model_size_mb" in cfg and dep.get("model_size_mb") is not None:
        if dep["model_size_mb"] > cfg["max_model_size_mb"]:
            reasons.append(f"deployment: model {dep['model_size_mb']:.2f} MB > "
                           f"max {cfg['max_model_size_mb']} MB")
    if "max_latency_ms" in cfg and dep.get("max_latency_ms") is not None:
        if dep["max_latency_ms"] > cfg["max_latency_ms"]:
            reasons.append(f"deployment: latency {dep['max_latency_ms']:.1f} ms > "
                           f"max {cfg['max_latency_ms']} ms")
    status = STATUS_FAIL if reasons else STATUS_PASS
    return GateResult(gate="deployment", status=status, reasons=reasons, details=details)


# ---------------------------------------------------------------------------
# Inference (framework / ONNX / INT8) — identical preprocessing contract
# ---------------------------------------------------------------------------

def discover_artifacts(framework_path: str | Path) -> dict[str, str | None]:
    """Find sibling ONNX/INT8 exports next to a checkpoint.

    The export step writes ``model.onnx`` and ``model_int8.onnx`` beside
    ``best.pt``; if they are absent, parity gates simply report nothing.
    """
    framework_path = Path(framework_path)
    out: dict[str, str | None] = {"framework": str(framework_path)}
    for key, name in (("onnx", "model.onnx"), ("int8", "model_int8.onnx")):
        candidate = framework_path.parent / name
        out[key] = str(candidate) if candidate.exists() else None
    return out


def _eval_transform(image_size: int, mean=None, std=None):
    from .data.loaders import _build_transforms

    aug = {}
    if mean and std:
        aug["normalize_mean"] = list(mean)
        aug["normalize_std"] = list(std)
    return _build_transforms(image_size, augmentation=aug, is_train=False)


def _tensor_batches(image_paths: list[str], image_size: int, batch_size: int,
                    mean=None, std=None):
    import torch
    from PIL import Image

    tfm = _eval_transform(image_size, mean, std)
    for i in range(0, len(image_paths), batch_size):
        chunk = image_paths[i:i + batch_size]
        yield torch.stack([tfm(Image.open(p).convert("RGB")) for p in chunk])


def score_framework(checkpoint_path: str | Path, samples: list[Sample],
                    image_size: int = 224, batch_size: int = 32) -> np.ndarray:
    """Return P(failing) per sample from a PyTorch checkpoint (CPU, deterministic)."""
    import torch

    from .models.factory import build_classifier

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    num_classes = int(ckpt.get("num_classes", 2))
    model_name = ckpt.get("config_model", "shufflenet_v2_x1_0")
    model = build_classifier(model_name, num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Reuse the exact normalisation the checkpoint was trained with; ``None``
    # falls back to ImageNet statistics (identical to the legacy behaviour).
    mean = ckpt.get("normalize_mean")
    std = ckpt.get("normalize_std")

    scores: list[float] = []
    with torch.no_grad():
        for batch in _tensor_batches([s.image_path for s in samples], image_size,
                                     batch_size, mean=mean, std=std):
            probs = torch.softmax(model(batch), dim=1).numpy()
            scores.extend(float(p[FAIL_IDX]) for p in probs)
    return np.asarray(scores, dtype=float)


def score_onnx(onnx_path: str | Path, samples: list[Sample],
               image_size: int = 224, batch_size: int = 32, int8: bool = False) -> np.ndarray:
    """Return P(failing) per sample from an ONNX (optionally INT8) model.

    Modes exported with a fixed batch size (e.g. 1) are detected from the input
    shape and scored at that batch size automatically.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    # Deterministic, low-thread sessions: reduces the native thread-pool
    # teardown surface that can abort when ORT shares a process with PyTorch,
    # and keeps scoring reproducible across machines.
    opts.intra_op_num_threads = int(os.environ.get("TV_ORT_INTRA_THREADS", "1"))
    opts.inter_op_num_threads = int(os.environ.get("TV_ORT_INTER_THREADS", "1"))
    sess = ort.InferenceSession(str(onnx_path), opts)
    inp = sess.get_inputs()[0]
    fixed_batch = inp.shape[0]
    if isinstance(fixed_batch, int) and not isinstance(fixed_batch, bool):
        batch_size = fixed_batch
    input_name = inp.name
    scores: list[float] = []
    for batch in _tensor_batches([s.image_path for s in samples], image_size, batch_size):
        arr = batch.numpy().astype(np.float32)
        logits = sess.run(None, {input_name: arr})[0]
        probs = _softmax(logits)
        scores.extend(float(p[FAIL_IDX]) for p in probs)
    # Release the native session before returning so its thread pool is torn
    # down deterministically rather than during interpreter finalization.
    del sess
    gc.collect()
    return np.asarray(scores, dtype=float)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Surface evaluation and report assembly
# ---------------------------------------------------------------------------

def evaluate_surface(samples: list[Sample], scores: np.ndarray, threshold: float,
                     positive: int = DEFAULT_POSITIVE_CLASS) -> dict:
    y_true = np.asarray([s.label for s in samples])
    preds = predictions_from_scores(scores, threshold, positive)
    metrics = compute_binary_metrics(y_true, preds, scores, positive)

    def _group(key: str | None) -> dict:
        groups: dict[str, list[int]] = {}
        for i, s in enumerate(samples):
            g = getattr(s, key) if key else None
            if g is None:
                continue
            groups.setdefault(str(g), []).append(i)
        out = {}
        for g, idx in sorted(groups.items()):
            idx_arr = np.asarray(idx)
            out[g] = {
                "n": int(len(idx)),
                **compute_binary_metrics(y_true[idx_arr], preds[idx_arr], scores[idx_arr], positive),
            }
        return out

    sources = {str(s.source_id) for s in samples if s.source_id is not None}
    environments = {str(s.environment_id) for s in samples if s.environment_id is not None}
    return {
        "threshold": round(float(threshold), 6),
        "positive_class": int(positive),
        "n_samples": len(samples),
        "n_sources": len(sources),
        "n_environments": len(environments),
        "sources": sorted(sources),
        "environments": sorted(environments),
        "metrics": metrics,
        "per_source": _group("source_id"),
        "per_environment": _group("environment_id"),
        "score_stats": {
            "failing_true_mean": _safe_mean(scores[y_true == positive]),
            "normal_true_mean": _safe_mean(scores[y_true != positive]),
        },
    }


def _safe_mean(a: np.ndarray) -> float | None:
    return round(float(np.mean(a)), 6) if len(a) else None


def compare_formats(framework_scores: np.ndarray, other_scores: np.ndarray,
                    y_true: np.ndarray, threshold: float,
                    positive: int = DEFAULT_POSITIVE_CLASS) -> dict:
    framework_scores = np.asarray(framework_scores, dtype=float)
    other_scores = np.asarray(other_scores, dtype=float)
    fw_pred = predictions_from_scores(framework_scores, threshold, positive)
    ot_pred = predictions_from_scores(other_scores, threshold, positive)
    disagreement = float(np.mean(fw_pred != ot_pred)) if len(fw_pred) else None
    abs_drift = float(np.max(np.abs(framework_scores - other_scores))) if len(framework_scores) else None
    fw_m = compute_binary_metrics(y_true, fw_pred, framework_scores, positive)
    ot_m = compute_binary_metrics(y_true, ot_pred, other_scores, positive)
    return {
        "prediction_disagreement": _round(disagreement),
        "max_score_abs_drift": _round(abs_drift),
        "mean_score_abs_drift": _round(
            float(np.mean(np.abs(framework_scores - other_scores)))
            if len(framework_scores) else None),
        "framework_metrics": fw_m,
        "other_metrics": ot_m,
        "metric_degradation": {
            "fpr_delta": _delta(fw_m["fpr"], ot_m["fpr"]),
            "fnr_delta": _delta(fw_m["fnr"], ot_m["fnr"]),
            "recall_delta": _delta(ot_m["recall"], fw_m["recall"]),
        },
    }


def _round(v):
    return None if v is None else round(float(v), 6)


def _delta(a, b):
    if a is None or b is None:
        return None
    return round(float(b) - float(a), 6)


def evaluate(
    suite: Suite,
    models: dict[str, dict[str, str]],
    policy: dict | None = None,
    dataset_root: str | Path | None = None,
    thresholds: list[float] | None = None,
    default_threshold: float = 0.5,
    output_dir: str | Path | None = None,
    config_path: str | None = None,
    policy_path: str | None = None,
    image_size: int = 224,
    batch_size: int = 32,
) -> dict:
    """Run the reliability evaluation across all suite surfaces and models.

    ``models`` maps model name -> {"framework": path, "onnx": path, "int8": path}
    (framework required; onnx/int8 optional).
    """
    thresholds = thresholds or default_threshold_grid()
    dataset_root = dataset_root if dataset_root is not None else suite.dataset_root
    positive = suite.positive_class

    # Load all surfaces; missing data is recorded, never silently skipped.
    samples_by_surface: dict[str, list[Sample]] = {}
    surface_errors: dict[str, str] = {}
    for surface in suite.surfaces:
        try:
            samples_by_surface[surface.id] = load_samples(surface, dataset_root)
        except MissingDataError as exc:
            surface_errors[surface.id] = str(exc)
            samples_by_surface[surface.id] = []

    roles = {s.id: s.role for s in suite.surfaces}
    leaks = check_source_leakage(samples_by_surface, roles)

    model_reports: dict[str, dict] = {}
    for name, paths in models.items():
        model_reports[name] = _evaluate_model(
            name, paths, suite, samples_by_surface,
            policy, thresholds, default_threshold, positive, image_size, batch_size,
        )

    report = {
        "schema": "tinyvision-reliability-eval",
        "schema_version": 1,
        "suite": {
            "name": suite.name,
            "version": suite.version,
            "path": suite.path,
            "sha256": sha256_file(suite.path) if suite.path and Path(suite.path).exists() else None,
            "positive_class": positive,
            "score_convention": "score = P(failing); predicted failing iff score >= threshold",
        },
        "policy": {
            "id": (policy or {}).get("id"),
            "policy_version": (policy or {}).get("policy_version"),
            "status": (policy or {}).get("status"),
            "path": policy_path,
            "sha256": sha256_file(policy_path) if policy_path and Path(policy_path).exists() else None,
        } if policy else None,
        "config": {
            "path": config_path,
            "sha256": sha256_file(config_path) if config_path and Path(config_path).exists() else None,
        } if config_path else None,
        "software": _software_versions(),
        "surfaces": [
            {
                "id": s.id, "role": s.role, "provenance": s.provenance,
                "verified": s.verified, "license": s.license,
                "n_loaded": len(samples_by_surface.get(s.id, [])),
                "error": surface_errors.get(s.id),
            }
            for s in suite.surfaces
        ],
        "leakage": [leak.__dict__ for leak in leaks],
        "models": model_reports,
    }
    if output_dir is not None:
        write_report(report, output_dir)
    return report


def _evaluate_model(name, paths, suite, samples_by_surface,
                    policy, thresholds, default_threshold, positive,
                    image_size, batch_size) -> dict:
    framework_path = paths.get("framework")
    if not framework_path or not Path(framework_path).exists():
        return {"model": name, "error": f"framework artifact not found: {framework_path}"}

    framework_scores: dict[str, np.ndarray] = {}
    for surface in suite.surfaces:
        samples = samples_by_surface.get(surface.id, [])
        if not samples:
            continue
        framework_scores[surface.id] = score_framework(
            framework_path, samples, image_size, batch_size)

    # Threshold selection uses dev_validation data ONLY.
    selection = {"threshold": default_threshold, "status": STATUS_INCONCLUSIVE,
                 "reason": "no dev_validation surface available; using fixed default"}
    sel_cfg = (policy or {}).get("threshold_selection") or {}
    dev_surface = next((s for s in suite.surfaces if s.role == ROLE_DEV_VALIDATION), None)
    if dev_surface is not None and framework_scores.get(dev_surface.id) is not None:
        samples = samples_by_surface[dev_surface.id]
        y_dev = np.asarray([s.label for s in samples])
        selection = select_threshold(
            y_dev, framework_scores[dev_surface.id],
            objective=sel_cfg.get("objective", "max_recall_subject_to_max_fpr"),
            max_fpr=sel_cfg.get("max_fpr"),
            grid=sel_cfg.get("grid") or thresholds,
            min_samples=int(sel_cfg.get("min_samples", 1)),
            positive=positive,
        )
    selected_threshold = selection.get("threshold") or default_threshold

    surfaces_out: dict[str, dict] = {}
    for surface in suite.surfaces:
        samples = samples_by_surface.get(surface.id, [])
        if not samples or surface.id not in framework_scores:
            continue
        y_true = np.asarray([s.label for s in samples])
        scores = framework_scores[surface.id]
        surfaces_out[surface.id] = {
            "role": surface.role,
            "at_selected_threshold": evaluate_surface(samples, scores, selected_threshold, positive),
            "at_fixed_threshold": evaluate_surface(samples, scores, default_threshold, positive),
            "threshold_sweep": threshold_sweep(y_true, scores, thresholds, positive),
            "predictions": [
                {
                    "image_path": s.image_path,
                    "label": int(s.label),
                    "score": round(float(sc), 6),
                    "pred": int(p),
                    "source_id": s.source_id,
                    "environment_id": s.environment_id,
                }
                for s, sc, p in zip(samples, scores,
                                    predictions_from_scores(scores, selected_threshold, positive))
            ],
        }

    # Format parity on every surface that has data.
    parity = {}
    for fmt, key in (("onnx", "onnx"), ("int8", "int8")):
        path = paths.get(key)
        if not path or not Path(path).exists():
            continue
        # Use the largest surface for a stable comparison, bounded subset.
        target = max(framework_scores, key=lambda sid: len(framework_scores[sid]), default=None)
        if target is None:
            continue
        samples = samples_by_surface[target][:PARITY_MAX_SAMPLES]
        other = score_onnx(path, samples, image_size, batch_size, int8=(fmt == "int8"))
        parity[fmt] = {
            "surface": target,
            "n_samples": len(samples),
            **compare_formats(framework_scores[target][:PARITY_MAX_SAMPLES], other,
                              np.asarray([s.label for s in samples]), selected_threshold, positive),
        }

    deployment = {
        "model_size_mb": round(Path(framework_path).stat().st_size / (1024 * 1024), 4),
        "artifacts": {k: (str(v) if v else None) for k, v in paths.items()},
    }

    gates = None
    if policy:
        gate_surfaces = {role: surfaces_out[sid]["at_selected_threshold"]
                         for sid, role in ((s.id, s.role) for s in suite.surfaces)
                         if sid in surfaces_out}
        gates = evaluate_gates(gate_surfaces, policy,
                               extra={"parity": parity or None, "deployment": deployment})

    return {
        "model": name,
        "artifacts": {k: (str(v) if v else None) for k, v in paths.items()},
        "artifact_sha256": {
            k: sha256_file(v) for k, v in paths.items()
            if v and Path(v).exists() and Path(v).is_file()
        },
        "threshold_selection": selection,
        "selected_threshold": selected_threshold,
        "surfaces": surfaces_out,
        "parity": parity,
        "deployment": deployment,
        "gates": gates,
    }


def _software_versions() -> dict:
    import sys
    versions = {"python": sys.version.split()[0]}
    for mod in ("torch", "torchvision", "numpy", "sklearn", "onnx", "onnxruntime"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:  # pragma: no cover - optional
            versions[mod] = "not-installed"
    return versions


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(report: dict, output_dir: str | Path) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    text_path = out / "summary.txt"
    text_path.write_text(format_summary(report) + "\n")
    return {"summary_json": str(summary_path), "summary_txt": str(text_path)}


def format_summary(report: dict) -> str:
    lines = [f"TinyVision reliability evaluation — suite '{report['suite']['name']}'"]
    if report.get("leakage"):
        lines.append(f"LEAKAGE WARNINGS: {len(report['leakage'])}")
        for leak in report["leakage"]:
            lines.append(f"  - {leak['code']}: {leak['message']}")
    for name, model in report.get("models", {}).items():
        lines.append("")
        lines.append(f"Model: {name}")
        if model.get("error"):
            lines.append(f"  ERROR: {model['error']}")
            continue
        sel = model.get("threshold_selection", {})
        lines.append(f"  threshold selection: {sel.get('status')} "
                     f"(threshold={model.get('selected_threshold')}) — {sel.get('reason')}")
        for sid, surface in model.get("surfaces", {}).items():
            m = surface["at_selected_threshold"]["metrics"]
            lines.append(
                f"  [{sid}] role={surface['role']} n={m['n']} "
                f"FPR={_fmt(m['fpr'])} FNR={_fmt(m['fnr'])} "
                f"P={_fmt(m['precision'])} R={_fmt(m['recall'])} "
                f"F1={_fmt(m['f1'])} AUROC={_fmt(m['auroc'])}"
                + (f" undefined={m['undefined']}" if m["undefined"] else "")
            )
        for fmt, block in model.get("parity", {}).items():
            scope = ""
            if block.get("surface") is not None:
                scope = f" on {block['surface']} n={block.get('n_samples')}"
            lines.append(f"  parity[{fmt}]{scope} disagreement={block['prediction_disagreement']} "
                         f"max|dscore|={block['max_score_abs_drift']}")
        gates = model.get("gates")
        if gates:
            lines.append(f"  RELEASE GATE: {gates['overall_status']} "
                         f"(blocks_release={gates['blocks_release']})")
            for g in gates["gates"]:
                lines.append(f"    - {g['gate']}: {g['status']}"
                             + (f" — {'; '.join(g['reasons'])}" if g["reasons"] else ""))
    return "\n".join(lines)


def _fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


def overall_exit_code(report: dict) -> int:
    """0 = all mandatory gates passed; 2 = FAIL; 3 = INCONCLUSIVE.

    A report with no gate decisions at all (e.g. produced without an acceptance
    policy) is INCONCLUSIVE, never a pass: acceptance requires criteria to have
    been applied. Leakage is always a FAIL.
    """
    statuses = []
    for model in report.get("models", {}).values():
        gates = model.get("gates")
        if gates:
            statuses.append(gates["overall_status"])
    if report.get("leakage"):
        return 2
    if any(s == STATUS_FAIL for s in statuses):
        return 2
    if not statuses:
        return 3
    if any(s == STATUS_INCONCLUSIVE for s in statuses):
        return 3
    return 0
