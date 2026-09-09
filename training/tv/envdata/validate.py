"""Intake validation for staged datasets.

Validation is structural and conservative: we do NOT judge image semantics or
label correctness (that is the review stage). We check manifest schema,
identity/leakage integrity (via the splitter), licensing/consent, and physical
attributes of recordings when local files are present.

Severity model: ``error`` flags mean the dataset must not advance; ``warning``
flags are recorded for human attention (e.g. possible timelapse cadence).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from tv.envdata.schema import (
    SPLITS,
    TINYVISION_LABELS,
    VERIFICATION_STATES,
    Manifest,
)
from tv.envdata.split import check_environment_leakage

SEVERITIES = ("error", "warning", "info")


@dataclass
class Flag:
    severity: str
    code: str
    message: str
    environment_id: str | None = None
    recording_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "environment_id": self.environment_id,
            "recording_id": self.recording_id,
        }


class Validator:
    def __init__(self) -> None:
        self.flags: list[Flag] = []

    def error(self, code: str, message: str, **kw: str | None) -> None:
        self._add("error", code, message, **kw)

    def warning(self, code: str, message: str, **kw: str | None) -> None:
        self._add("warning", code, message, **kw)

    def info(self, code: str, message: str, **kw: str | None) -> None:
        self._add("info", code, message, **kw)

    def _add(self, severity: str, code: str, message: str, **kw: str | None) -> None:
        if severity not in SEVERITIES:
            raise ValueError(severity)
        self.flags.append(
            Flag(severity=severity, code=code, message=message,
                 environment_id=kw.get("environment_id"),
                 recording_id=kw.get("recording_id"))
        )

    def errors(self) -> list[Flag]:
        return [f for f in self.flags if f.severity == "error"]

    def passes(self) -> bool:
        return not self.errors()

    def report(self) -> dict:
        return {
            "ok": self.passes(),
            "n_errors": len(self.errors()),
            "n_warnings": sum(1 for f in self.flags if f.severity == "warning"),
            "flags": [f.to_dict() for f in self.flags],
        }


def validate_manifest(
    manifest: Manifest,
    staging_root: str | Path | None = None,
) -> Validator:
    v = Validator()
    ds = manifest.dataset

    if not ds.dataset_id.strip():
        v.error("missing_dataset_id", "dataset.dataset_id is empty")
    if not ds.name.strip():
        v.error("missing_dataset_name", "dataset.name is empty")
    if not ds.license.strip():
        v.error("missing_license", "dataset.license is empty")
    else:
        if not (ds.redistribution_allowed or ds.training_allowed or ds.evaluation_allowed):
            v.error(
                "no_usage_rights",
                f"dataset {ds.dataset_id} declares no training, evaluation or "
                "redistribution permission",
            )
        if ds.training_allowed and "nc" in ds.license.lower().replace("-", " ").replace(" ", "_") and not ds.evaluation_allowed:
            v.warning(
                "nc_training_only",
                f"license {ds.license} is non-commercial; training use requires "
                "internal-only handling",
            )
    if ds.retrieval_date:
        _check_date_format(v, "retrieval_date", ds.retrieval_date)

    env_ids = set()
    for env in manifest.environments:
        if env.environment_id in env_ids:
            v.error("duplicate_env_id", f"duplicate environment ID {env.environment_id}",
                    environment_id=env.environment_id)
        env_ids.add(env.environment_id)
        if not env.environment_key.strip():
            v.error("missing_env_key", "environment has empty environment_key",
                    environment_id=env.environment_id)
        if env.split_assignment is not None and env.split_assignment not in SPLITS:
            v.error("bad_split", f"unknown split {env.split_assignment!r}",
                    environment_id=env.environment_id)
        if env.source_dataset != ds.dataset_id:
            v.error(
                "env_source_mismatch",
                f"environment source_dataset {env.source_dataset!r} does not match "
                f"dataset {ds.dataset_id!r}",
                environment_id=env.environment_id,
            )

    for rec in manifest.recordings:
        if rec.environment_id not in env_ids:
            v.error("dangling_recording",
                    f"recording {rec.recording_id} -> missing environment {rec.environment_id}",
                    recording_id=rec.recording_id)
        for field, val in (("fps", rec.fps), ("width", rec.width), ("height", rec.height),
                           ("duration_s", rec.duration_s)):
            if val is not None and val < 0:
                v.error("negative_attr", f"recording has negative {field}={val}",
                        recording_id=rec.recording_id)
        if rec.continuous_realtime is False and not rec.temporal_note:
            v.warning("no_temporal_note",
                      "recording is non-continuous (e.g. timelapse) but has no temporal_note",
                      recording_id=rec.recording_id)

    for consent in manifest.consents:
        if not consent.training_allowed and ds.training_allowed:
            pass  # consent can still be missing if dataset-level license covers it
        if not consent.contributor.strip():
            v.error("missing_consent_contributor", "consent has empty contributor")

    if "contrib" in ds.dataset_id or ds.acquisition_method == "contributor":
        names = {c.contributor for c in manifest.consents}
        authors = {e.contributor for e in manifest.environments if e.contributor}
        missing = authors - names
        if missing:
            v.error(
                "missing_consent",
                f"contributor dataset has environments without consent: "
                f"{', '.join(sorted(missing))}",
            )

    leakage = check_environment_leakage(manifest)
    for issue in leakage.issues:
        if issue.code == "partial_assignment":
            # Candidate environments may legitimately have no split yet; they
            # simply do not contribute to any split. Everything else blocks.
            v.warning(issue.code, issue.message, environment_id=issue.environment_id,
                      recording_id=issue.recording_id)
        else:
            v.error(issue.code, issue.message, environment_id=issue.environment_id,
                    recording_id=issue.recording_id)

    if staging_root is not None:
        _check_staging_files(v, staging_root, manifest)
    return v


def _check_date_format(v: Validator, field: str, value: str) -> None:
    import re
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        v.error(f"bad_{field}", f"{field} must be YYYY-MM-DD, got {value!r}")


def validate_samples(
    manifest: Manifest,
    samples_dir: str | Path | None = None,
    samples: list | None = None,
) -> Validator:
    v = Validator()
    rec_ids = {r.recording_id for r in manifest.recordings}
    env_ids = {e.environment_id for e in manifest.environments}
    loaded = _load_samples(samples_dir) if samples_dir else []
    if samples is not None:
        loaded = list(samples)
    if not loaded:
        v.info("no_samples", "no sample files loaded; sample-level checks skipped")
        return v

    seen = set()
    for s in loaded:
        if s.sample_id in seen:
            v.error("duplicate_sample_id", f"duplicate sample {s.sample_id}")
        seen.add(s.sample_id)
        if s.recording_id not in rec_ids:
            v.error("dangling_sample",
                    f"sample {s.sample_id} -> missing recording {s.recording_id}")
        if s.environment_id not in env_ids:
            v.error("dangling_sample_env",
                    f"sample {s.sample_id} -> missing environment {s.environment_id}")
        if s.verification not in VERIFICATION_STATES:
            v.error("bad_verification", f"sample {s.sample_id} has state {s.verification!r}")
        if s.verification == "verified" and (s.tinyvision_label is None
                                             or s.tinyvision_label not in TINYVISION_LABELS):
            v.error("verified_without_label",
                    f"verified sample {s.sample_id} lacks a valid tinyvision_label")
    return v


def _load_samples(samples_dir: str | Path) -> list:
    from tv.envdata.schema import Sample
    out = []
    root = Path(samples_dir)
    if not root.exists():
        return out
    for path in sorted(root.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(Sample.from_dict(json.loads(line)))
    return out


def _check_staging_files(v: Validator, staging_root: str | Path, manifest: Manifest) -> None:
    root = Path(staging_root)
    if not (root / "manifests" / "env_manifest.json").exists():
        v.error("missing_manifest_files", "manifests/env_manifest.json not found")
    for ext in ("environments", "recordings", "metadata", "review", "samples"):
        if not (root / ext).is_dir():
            v.warning("missing_dir", f"staging directory {ext}/ missing")
    for rec in manifest.recordings:
        if rec.source.startswith(("file:", "http://", "https://")):
            continue  # external reference, cannot check locally
        p = Path(rec.source)
        if p.exists() and not p.is_file():
            v.error("source_not_file", f"recording source {rec.source} is not a file",
                    recording_id=rec.recording_id)


# ---------------------------------------------------------------------------
# Video-level intake checks (physical media; no semantic judgement)
# ---------------------------------------------------------------------------

def file_md5(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk_ := f.read(chunk):
            h.update(chunk_)
    return h.hexdigest()


def probe_video(path: str | Path) -> dict:
    """Return physical attributes of a video via OpenCV; throws on failure."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _ = cap.read()
    if not ok:
        raise ValueError(f"cannot read a frame from {path}")
    cap.release()
    duration_s = (frames / fps) if fps and frames and fps > 0 else None
    return {
        "fps": fps,
        "frame_count": frames,
        "width": width,
        "height": height,
        "duration_s": duration_s,
    }


def flag_possible_timelapse(
    v: Validator,
    path: str | Path,
    recording_id: str,
    sample_every: int = 24,
    max_pairs: int = 64,
) -> None:
    """Heuristic-only check: if reported fps looks real-time but consecutive
    frames differ a lot, the recording is likely skimmed/edited. Flags a
    warning; never an error."""
    import cv2

    try:
        import numpy as np
    except Exception:
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        v.warning("video_unreadable", f"could not open video {path}",
                  recording_id=recording_id)
        return
    prev = None
    diffs = []
    idx = 0
    while idx < max_pairs * sample_every:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                diffs.append(float(np.mean(np.abs(gray.astype(int) - prev.astype(int)))))
            prev = gray
        idx += 1
    cap.release()
    if not diffs:
        return
    import statistics
    med = statistics.median(diffs)
    if med > 25.0:
        v.warning(
            "possible_timelapse",
            f"median inter-frame diff {med:.1f} at nominal fps suggests "
            "skimmed/edited cadence (timelapse)",
            recording_id=recording_id,
        )


def flag_obvious_duplicates(
    v: Validator,
    recordings: list[tuple[str, str, str, dict]],  # (recording_id, environment_id, path, probe)
) -> None:
    """Exact md5 dedup plus obvious accidental duplicates: same environment,
    same duration/fps/resolution within tolerance."""
    md5_seen: dict[str, str] = {}
    for rec_id, env_id, path, probe in recordings:
        try:
            digest = file_md5(path)
        except Exception:
            continue
        prev = md5_seen.get(digest)
        if prev is not None:
            v.error("exact_duplicate", f"recording {rec_id} is a byte-identical "
                    f"duplicate of {prev}", recording_id=rec_id)
        else:
            md5_seen[digest] = rec_id

    by_env: dict[str, list[tuple[str, dict]]] = {}
    for rec_id, env_id, path, probe in recordings:
        if probe.get("duration_s") is None:
            continue
        by_env.setdefault(env_id, []).append((rec_id, probe))
    for env, items in by_env.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a = items[i][1]
                b = items[j][1]
                if (a["width"] == b["width"] and a["height"] == b["height"]
                        and abs((a.get("duration_s") or 0) - (b.get("duration_s") or 0)) < 0.5
                        and abs((a.get("fps") or 0) - (b.get("fps") or 0)) < 0.1):
                    v.warning("near_duplicate",
                              f"recordings {items[i][0]} and {items[j][0]} look like "
                              "the same capture in the same environment",
                              recording_id=items[j][0])