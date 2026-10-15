"""Real-world footage session intake: register, validate, and convert footage.

A *session* is one complete print recorded by one printer/camera setup. Sessions
are the atomic unit of the target-domain evaluation: frames are never split, and
each session belongs to exactly one of three splits:

* ``calibration`` — used by :mod:`tv.calibrate` to select the per-printer
  threshold;
* ``validation`` — independent sessions reported by ``tv calibrate`` (mapped onto
  its ``holdout`` split) but never used to select a threshold;
* ``holdout`` — the **locked** holdout: reserved for held-out evaluation and
  release-gate inputs; it is never seen by threshold selection.

This module is intake-only. It does not train, does not touch a locked test set,
and does not change any evaluation criterion. It reuses :mod:`tv.calibrate`
(:func:`build_calibrate_spec` emits its clips spec) and the video-probing helpers
in :mod:`tv.envdata.validate`.

Validation is structural and conservative, mirroring ``tv env-validate``: we do
NOT judge image semantics or whether a human label is correct (that is review).
``error`` flags mean the sheet must not advance; ``warning`` flags are recorded
for human attention. The locked holdout fails closed: a missing or contaminated
holdout is an ``error``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tv.envdata.schema import FAILURE_TYPE_VOCAB
from tv.envdata.validate import file_md5, probe_video

SESSION_SPLITS = ("calibration", "validation", "holdout")
SESSION_LABELS = ("normal", "failing", "uncertain")
# Accepted spellings for the locked holdout (unambiguous aliases only).
_SPLIT_ALIASES = {"locked_holdout": "holdout", "locked": "holdout"}

SEVERITIES = ("error", "warning", "info")


class SessionSheetError(Exception):
    """Raised on an unparseable or structurally invalid session sheet."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """One complete print/session of real footage with print-level metadata."""

    id: str
    label: str
    split: str = "calibration"
    source: str | None = None
    # Optional precomputed per-frame scores (P(failing)); when present no model
    # or media is required for the sheet to validate or for tv calibrate.
    scores: list | None = None
    printer: str | None = None
    camera: str | None = None
    failure_type: str | None = None
    onset_s: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    md5: str | None = None
    provenance: str | None = None
    license: str | None = None
    redistributable: bool | None = None
    notes: str | None = None

    @property
    def usable(self) -> bool:
        """A session counts toward a split only with a normal/failing label."""
        return self.label in ("normal", "failing")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "split": self.split,
            "source": self.source,
            "scores": self.scores,
            "printer": self.printer,
            "camera": self.camera,
            "failure_type": self.failure_type,
            "onset_s": self.onset_s,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "duration_s": self.duration_s,
            "md5": self.md5,
            "provenance": self.provenance,
            "license": self.license,
            "redistributable": self.redistributable,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        if "id" not in data:
            raise SessionSheetError("session entry is missing 'id'")
        if "label" not in data:
            raise SessionSheetError(f"session {data['id']!r} is missing 'label'")
        split = str(data.get("split", "calibration")).strip().lower()
        split = _SPLIT_ALIASES.get(split, split)
        return cls(
            id=str(data["id"]),
            label=str(data["label"]).strip().lower(),
            split=split,
            source=data.get("source"),
            scores=data.get("scores"),
            printer=data.get("printer"),
            camera=data.get("camera"),
            failure_type=data.get("failure_type"),
            onset_s=data.get("onset_s"),
            fps=data.get("fps"),
            width=data.get("width"),
            height=data.get("height"),
            duration_s=data.get("duration_s"),
            md5=data.get("md5"),
            provenance=data.get("provenance"),
            license=data.get("license"),
            redistributable=data.get("redistributable"),
            notes=data.get("notes"),
        )


@dataclass
class SessionSheet:
    """A registered collection of footage sessions for one printer/camera."""

    sessions: list[Session] = field(default_factory=list)
    dataset_id: str | None = None
    printer: str | None = None
    camera: str | None = None
    checkpoint: str | None = None
    image_size: int | None = None
    license: str | None = None
    provenance: str | None = None
    notes: str | None = None
    path: str | None = None

    def ids(self) -> list[str]:
        return [s.id for s in self.sessions]

    def by_split(self, split: str) -> list[Session]:
        return [s for s in self.sessions if s.split == split and s.usable]

    def usable_count(self, split: str) -> tuple[int, int]:
        """(normal, failing) usable session counts for one split."""
        group = self.by_split(split)
        return (sum(1 for s in group if s.label == "normal"),
                sum(1 for s in group if s.label == "failing"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "tinyvision-session-sheet",
            "schema_version": 1,
            "dataset_id": self.dataset_id,
            "printer": self.printer,
            "camera": self.camera,
            "checkpoint": self.checkpoint,
            "image_size": self.image_size,
            "license": self.license,
            "provenance": self.provenance,
            "notes": self.notes,
            "sessions": [s.to_dict() for s in self.sessions],
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _parse_sheet_payload(data: Any, path: str | None) -> SessionSheet:
    if isinstance(data, list):
        data = {"sessions": data}
    if not isinstance(data, dict) or "sessions" not in data:
        raise SessionSheetError(
            "session sheet must be a mapping with a 'sessions' list "
            "(or a bare list of sessions)"
        )
    raw_sessions = data["sessions"]
    if not isinstance(raw_sessions, list) or not raw_sessions:
        raise SessionSheetError("session sheet contains no sessions")
    sheet = SessionSheet(
        sessions=[],
        dataset_id=data.get("dataset_id"),
        printer=data.get("printer"),
        camera=data.get("camera"),
        checkpoint=data.get("checkpoint"),
        image_size=data.get("image_size"),
        license=data.get("license"),
        provenance=data.get("provenance"),
        notes=data.get("notes"),
        path=path,
    )
    for i, raw in enumerate(raw_sessions):
        if not isinstance(raw, dict):
            raise SessionSheetError(f"session #{i} is not a mapping")
        try:
            session = Session.from_dict(raw)
        except SessionSheetError as exc:
            raise SessionSheetError(f"session #{i}: {exc}") from exc
        # Inherit sheet-level defaults.
        session.printer = session.printer or sheet.printer
        session.camera = session.camera or sheet.camera
        session.license = session.license or sheet.license
        session.provenance = session.provenance or sheet.provenance
        sheet.sessions.append(session)
    return sheet


def load_session_sheet(path: str | Path) -> SessionSheet:
    """Load a session sheet from a YAML or JSON file.

    Raises :class:`SessionSheetError` on a missing file or malformed content.
    """
    p = Path(path)
    if not p.exists():
        raise SessionSheetError(f"session sheet not found: {p}")
    text = p.read_text()
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise SessionSheetError(f"could not parse YAML sheet {p}: {exc}") from exc
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SessionSheetError(f"could not parse JSON sheet {p}: {exc}") from exc
    return _parse_sheet_payload(data, str(p))


def write_session_sheet(sheet: SessionSheet, path: str | Path) -> None:
    """Write a sheet as YAML (``.yaml``/``.yml``) or JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = sheet.to_dict()
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml

        p.write_text(yaml.safe_dump(payload, sort_keys=False))
    else:
        p.write_text(json.dumps(payload, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class SessionFlag:
    severity: str
    code: str
    message: str
    session_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "session_id": self.session_id,
        }


class SessionValidator:
    def __init__(self) -> None:
        self.flags: list[SessionFlag] = []

    def _add(self, severity: str, code: str, message: str,
             session_id: str | None = None) -> None:
        if severity not in SEVERITIES:
            raise ValueError(severity)
        self.flags.append(SessionFlag(severity, code, message, session_id))

    def error(self, code: str, message: str, session_id: str | None = None) -> None:
        self._add("error", code, message, session_id)

    def warning(self, code: str, message: str, session_id: str | None = None) -> None:
        self._add("warning", code, message, session_id)

    def info(self, code: str, message: str, session_id: str | None = None) -> None:
        self._add("info", code, message, session_id)

    def errors(self) -> list[SessionFlag]:
        return [f for f in self.flags if f.severity == "error"]

    def warnings(self) -> list[SessionFlag]:
        return [f for f in self.flags if f.severity == "warning"]

    def passes(self) -> bool:
        return not self.errors()

    def report(self) -> dict:
        return {
            "ok": self.passes(),
            "n_errors": len(self.errors()),
            "n_warnings": len(self.warnings()),
            "flags": [f.to_dict() for f in self.flags],
        }


def _is_local_path(source: str) -> bool:
    return not source.startswith(("http://", "https://", "file:"))


def _probe_source(source: str) -> tuple[str, dict | None, str | None]:
    """Best-effort physical probe of a local source.

    Returns ``(kind, probe, md5)`` where ``kind`` is one of
    ``file`` / ``dir`` / ``glob`` / ``external`` / ``missing``. ``probe`` is the
    cv2 probe dict for a file, else None.
    """
    if source.startswith(("http://", "https://", "file:")):
        return "external", None, None
    if any(ch in source for ch in "*?["):
        import glob as _glob

        matches = sorted(p for p in _glob.glob(source) if Path(p).is_file())
        if not matches:
            return "glob", None, None
        source = matches[0]
    p = Path(source)
    if not p.exists():
        return "missing", None, None
    if p.is_dir():
        return "dir", None, None
    try:
        probe = probe_video(p)
    except Exception:
        return "file", None, None
    try:
        digest = file_md5(p)
    except Exception:
        digest = None
    return "file", probe, digest


def validate_sessions(
    sheet: SessionSheet,
    *,
    probe: bool = False,
) -> SessionValidator:
    """Validate a session sheet structurally (and optionally probe media).

    ``probe=True`` opens local video files to check readability and physical
    attributes and to detect byte-identical duplicates. With ``probe=False`` the
    checks are structural only (and use any declared ``md5``/duration).
    """
    v = SessionValidator()

    if not sheet.license:
        v.error(
            "missing_license",
            "no license/permission declared for the footage; record it at the "
            "sheet or session level (unlicensed data cannot be used)",
        )
    if not sheet.printer:
        v.warning("missing_printer",
                  "no 'printer' declared; the calibration report cannot assert "
                  "which printer it applies to")

    # --- per-session structural checks ------------------------------------
    seen_ids: set[str] = set()
    printers: dict[str, str] = {}
    cameras: dict[str, str] = {}
    for s in sheet.sessions:
        sid = s.id
        if not sid.strip():
            v.error("missing_session_id", "session has an empty id", sid)
        elif sid in seen_ids:
            v.error("duplicate_session_id", f"duplicate session id {sid!r}", sid)
        seen_ids.add(sid)

        if s.source is None and s.scores is None:
            v.error("missing_source",
                    f"session {sid!r} has neither 'source' nor 'scores'", sid)
        if s.scores is not None and not isinstance(s.scores, list):
            v.error("bad_scores", f"session {sid!r} 'scores' must be a list", sid)

        if s.label not in SESSION_LABELS:
            v.error("invalid_label",
                    f"session {sid!r} label must be one of {SESSION_LABELS}, "
                    f"got {s.label!r}", sid)
        elif s.label == "uncertain":
            v.warning("uncertain_label",
                      f"session {sid!r} is labelled uncertain and will be "
                      "excluded from every split", sid)

        if s.split not in SESSION_SPLITS:
            v.error("invalid_split",
                    f"session {sid!r} split must be one of {SESSION_SPLITS}, "
                    f"got {s.split!r} (never 'test'/'locked')", sid)

        if s.printer:
            printers.setdefault(s.printer, sid)
        if s.camera:
            cameras.setdefault(s.camera, sid)

        # failure_type / onset consistency
        if s.label == "failing":
            if not s.failure_type:
                v.warning("missing_failure_type",
                          f"failing session {sid!r} has no failure_type", sid)
            elif s.failure_type not in FAILURE_TYPE_VOCAB:
                v.error("unknown_failure_type",
                        f"session {sid!r} failure_type {s.failure_type!r} not in "
                        f"{FAILURE_TYPE_VOCAB}", sid)
            if s.onset_s is None:
                v.warning("missing_onset",
                          f"failing session {sid!r} has no onset_s (approximate "
                          "onset time)", sid)
        elif s.label == "normal":
            if s.failure_type and s.failure_type != "none":
                v.warning("normal_has_failure_type",
                          f"normal session {sid!r} declares failure_type "
                          f"{s.failure_type!r}", sid)
            if s.onset_s is not None:
                v.warning("normal_has_onset",
                          f"normal session {sid!r} declares onset_s", sid)

        if s.onset_s is not None:
            if s.onset_s < 0:
                v.error("onset_out_of_range",
                        f"session {sid!r} onset_s={s.onset_s} is negative", sid)
            elif s.duration_s is not None and s.onset_s > s.duration_s + 0.5:
                v.error("onset_out_of_range",
                        f"session {sid!r} onset_s={s.onset_s} exceeds declared "
                        f"duration {s.duration_s}", sid)

        # --- physical media checks ----------------------------------------
        if probe and s.source and _is_local_path(s.source):
            kind, probed, digest = _probe_source(s.source)
            if kind == "missing":
                v.error("missing_file",
                        f"session {sid!r} source not found: {s.source}", sid)
            elif kind == "glob":
                v.error("missing_file",
                        f"session {sid!r} source glob matched no files: "
                        f"{s.source}", sid)
            elif kind == "dir":
                exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
                frames = [p for p in Path(s.source).iterdir()
                          if p.suffix.lower() in exts]
                if not frames:
                    v.error("unreadable_video",
                            f"session {sid!r} source directory has no image "
                            f"frames: {s.source}", sid)
            elif kind == "file":
                if probed is None:
                    v.error("unreadable_video",
                            f"session {sid!r} could not be decoded: {s.source}",
                            sid)
                else:
                    _check_probe(v, s, probed, digest)
        if probe and s.source and not _is_local_path(s.source):
            v.info("external_source",
                   f"session {sid!r} source is external; media checks skipped",
                   sid)

    # --- physical detection of duplicates ---------------------------------
    _flag_duplicates(v, sheet, probe=probe)

    # --- split independence / class presence ------------------------------
    _check_splits(v, sheet)

    # --- one setup per sheet ----------------------------------------------
    if len(printers) > 1:
        v.error(
            "multi_printer",
            f"sessions declare {len(printers)} printers ({sorted(printers)}); "
            "tv calibrate is per-printer — write one sheet per printer/camera",
        )
    if len(cameras) > 1:
        v.warning(
            "multi_camera",
            f"sessions declare {len(cameras)} cameras ({sorted(cameras)}); a "
            "camera change is an environment change",
        )
    return v


def _check_probe(v: SessionValidator, s: Session, probed: dict,
                 digest: str | None) -> None:
    """Compare declared physical attributes against the probed video."""
    s.md5 = s.md5 or digest
    for field_name, declared, actual in (
        ("fps", s.fps, probed.get("fps")),
        ("width", s.width, probed.get("width")),
        ("height", s.height, probed.get("height")),
        ("duration_s", s.duration_s, probed.get("duration_s")),
    ):
        if declared is None or actual is None:
            continue
        tol = max(0.5, 0.01 * float(actual)) if field_name in ("fps", "duration_s") else 0
        if abs(float(declared) - float(actual)) > tol:
            v.warning(
                "metadata_mismatch",
                f"session {s.id!r} declares {field_name}={declared} but the "
                f"video reports {round(float(actual), 4)}",
                s.id,
            )
    if s.duration_s is None:
        s.duration_s = probed.get("duration_s")
    if s.fps is None:
        s.fps = probed.get("fps")


def _flag_duplicates(v: SessionValidator, sheet: SessionSheet,
                     *, probe: bool) -> None:
    """Byte-identical duplicates (always) and near-duplicates (probe only)."""
    by_md5: dict[str, list[Session]] = {}
    for s in sheet.sessions:
        if s.md5:
            by_md5.setdefault(s.md5, []).append(s)
    for digest, group in by_md5.items():
        if len(group) < 2:
            continue
        splits = {g.split for g in group}
        ids = ", ".join(g.id for g in group)
        if len(splits) > 1:
            for g in group:
                v.error(
                    "duplicate_media_across_splits",
                    f"session {g.id!r} is byte-identical to another session in a "
                    f"different split ({ids}); this is leakage",
                    g.id,
                )
        else:
            for g in group[1:]:
                v.warning("duplicate_media",
                          f"session {g.id!r} is a byte-identical duplicate of "
                          f"{group[0].id!r} within the same split", g.id)

    if not probe:
        return
    by_env: dict[tuple, list[Session]] = {}
    for s in sheet.sessions:
        if s.width and s.height and s.duration_s is not None:
            by_env.setdefault((s.printer, s.camera, s.width, s.height), []).append(s)
    for group in by_env.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if abs((a.duration_s or 0) - (b.duration_s or 0)) < 0.5 and \
                        abs((a.fps or 0) - (b.fps or 0)) < 0.1:
                    v.warning(
                        "near_duplicate",
                        f"sessions {a.id!r} and {b.id!r} look like the same "
                        "capture (same printer/camera/resolution/duration)",
                        b.id,
                    )


def _check_splits(v: SessionValidator, sheet: SessionSheet) -> None:
    """Required class presence per split and the locked-holdout fail-closed gate."""
    n_cal_n, n_cal_f = sheet.usable_count("calibration")
    if n_cal_n == 0 or n_cal_f == 0:
        v.error(
            "split_missing_class",
            "calibration split needs at least one normal and one failing session "
            f"(have normal={n_cal_n}, failing={n_cal_f}); no threshold can be "
            "selected",
        )

    n_val_n, n_val_f = sheet.usable_count("validation")
    if (n_val_n + n_val_f) == 0:
        v.info("no_validation_split",
               "no validation sessions; the operating point will be reported on "
               "the locked holdout only")
    elif n_val_n == 0 or n_val_f == 0:
        v.warning(
            "validation_missing_class",
            "validation split has a single class "
            f"(normal={n_val_n}, failing={n_val_f}); it cannot check both "
            "false-alarm and detection behaviour",
        )

    holdout = [s for s in sheet.sessions if s.split == "holdout"]
    usable_holdout = [s for s in holdout if s.usable]
    if not holdout:
        v.error(
            "missing_locked_holdout",
            "no locked holdout sessions; the operating point cannot be validated "
            "on independent sessions (fail closed)",
        )
    else:
        for s in holdout:
            if not s.usable:
                v.error(
                    "contaminated_holdout",
                    f"locked holdout session {s.id!r} has label {s.label!r}; the "
                    "holdout must contain only normal/failing sessions",
                    s.id,
                )
        n_h_n, n_h_f = sheet.usable_count("holdout")
        if usable_holdout and (n_h_n == 0 or n_h_f == 0):
            v.error(
                "holdout_missing_class",
                "locked holdout needs at least one normal and one failing session "
                f"(have normal={n_h_n}, failing={n_h_f})",
            )


# ---------------------------------------------------------------------------
# Conversion to existing commands
# ---------------------------------------------------------------------------

def build_calibrate_spec(sheet: SessionSheet) -> dict:
    """Convert a sheet into a :mod:`tv.calibrate` clips spec.

    ``calibration`` sessions map to the calibrate ``calibration`` split and
    ``validation`` sessions map to its ``holdout`` split (reported, never used
    for selection). The **locked holdout** and any uncertain sessions are
    deliberately excluded so they can never influence threshold selection.
    """
    clips: list[dict] = []
    for s in sheet.sessions:
        if not s.usable:
            continue
        if s.split == "calibration":
            csplit = "calibration"
        elif s.split == "validation":
            csplit = "holdout"
        else:  # locked holdout is reserved for held-out / release-gate evaluation
            continue
        clip: dict[str, Any] = {"id": s.id, "label": s.label, "split": csplit}
        if s.source:
            clip["source"] = s.source
        if s.scores is not None:
            clip["scores"] = s.scores
        if s.printer is not None:
            clip["printer"] = s.printer
        if s.camera is not None:
            clip["camera"] = s.camera
        if s.failure_type:
            clip["failure_type"] = s.failure_type
        if s.onset_s is not None:
            clip["onset_s"] = s.onset_s
        clips.append(clip)
    return {
        "printer": sheet.printer,
        "camera": sheet.camera,
        "checkpoint": sheet.checkpoint,
        "image_size": sheet.image_size,
        "clips": clips,
    }


def write_calibrate_spec(spec: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml

        p.write_text(yaml.safe_dump(spec, sort_keys=False))
    else:
        p.write_text(json.dumps(spec, indent=2) + "\n")


def describe_sheet(sheet: SessionSheet) -> dict:
    """Inventory of a sheet preserving ids, split roles, and provenance."""
    return {
        "dataset_id": sheet.dataset_id,
        "printer": sheet.printer,
        "camera": sheet.camera,
        "checkpoint": sheet.checkpoint,
        "image_size": sheet.image_size,
        "license": sheet.license,
        "provenance": sheet.provenance,
        "notes": sheet.notes,
        "n_sessions": len(sheet.sessions),
        "sessions": [
            {
                "id": s.id,
                "label": s.label,
                "split": s.split,
                "role": "locked_holdout" if s.split == "holdout" else s.split,
                "usable": s.usable,
                "source": s.source,
                "md5": s.md5,
                "failure_type": s.failure_type,
                "onset_s": s.onset_s,
                "printer": s.printer,
                "camera": s.camera,
                "license": s.license,
                "provenance": s.provenance,
            }
            for s in sheet.sessions
        ],
    }


def heldout_report(sheet: SessionSheet, scored: dict[str, list[dict]],
                   threshold: float, confirm: int = 1,
                   meta: dict | None = None) -> dict:
    """Session-level decisions on the **locked holdout** at ``threshold``.

    Reuses :mod:`tv.calibrate`'s aggregation so the numbers agree exactly with
    the calibration report. ``scored`` maps session id -> ``[{"timestamp_s",
    "score"}]``. Held-out labels are never used to choose ``threshold``.

    Only ``split == "holdout"`` sessions are scored: the locked holdout, never
    the validation split. The report is explicitly labelled so it cannot be
    confused with the validation tier of the calibration report.
    """
    from tv.calibrate import CalibrationClip, _aggregate

    hold_sessions = [s for s in sheet.sessions
                     if s.split == "holdout" and s.usable]
    hold = [
        CalibrationClip(id=s.id, label=s.label, source=s.source or "",
                        printer=s.printer, camera=s.camera, split="holdout",
                        scores=s.scores)
        for s in hold_sessions
    ]
    report = _aggregate(hold, scored, threshold, confirm)
    report["schema"] = "tinyvision-session-holdout"
    report["schema_version"] = 1
    report["role"] = "locked_holdout"
    report["intake_split"] = "holdout"
    report["printer"] = sheet.printer
    report["camera"] = sheet.camera
    report["threshold"] = round(float(threshold), 6)
    report["confirm"] = confirm
    report["sessions"] = [
        {"id": s.id, "label": s.label, "role": "locked_holdout",
         "intake_split": "holdout", "split": "holdout",
         "failure_type": s.failure_type, "onset_s": s.onset_s}
        for s in hold_sessions
    ]
    if meta:
        report["meta"] = meta
    return report


def validation_report(cal: dict, sheet: SessionSheet,
                      meta: dict | None = None) -> dict:
    """Relabel ``tv calibrate``'s ``holdout`` tier as the intake **validation** split.

    ``tv calibrate`` names the tier it reports (but never selects on) ``holdout``;
    in the intake mapping that tier is the *validation* split, which is distinct
    from the locked holdout. This returns a clearly-labelled, self-describing view
    so the two are never confused in user-facing output.
    """
    block = cal.get("holdout") if isinstance(cal, dict) else None
    classification = (cal or {}).get("selection") or {}
    config = (cal or {}).get("config") or {}
    report: dict[str, Any] = {
        "schema": "tinyvision-session-validation",
        "schema_version": 1,
        "intake_split": "validation",
        "role": "independent_reporting_only",
        "printer": (cal or {}).get("printer"),
        "camera": (cal or {}).get("camera"),
        "threshold": classification.get("selected_threshold"),
        "confirm": config.get("confirm"),
        "model": (cal or {}).get("model"),
        "evaluated": block is not None,
        "note": (
            "Validation sessions are reported for information only. They do NOT "
            "select the calibration threshold and they are NOT the locked holdout."
        ),
    }
    if block is not None:
        report.update(block)
    report["sessions"] = [
        {"id": s.id, "label": s.label, "role": "validation",
         "intake_split": "validation", "split": "validation",
         "failure_type": s.failure_type, "onset_s": s.onset_s}
        for s in sheet.sessions if s.split == "validation" and s.usable
    ]
    if meta:
        report["meta"] = meta
    return report


def emit_holdout_csv(sheet: SessionSheet, path: str | Path) -> int:
    """Write the locked holdout sessions as a CSV for held-out/gate plumbing.

    Columns: ``session,source,label,failure_type,onset_s``. One row per session
    (session-level, not frame-level). Returns the number of rows written.
    """
    import csv

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "source", "label", "failure_type", "onset_s"])
        for s in sheet.sessions:
            if s.split != "holdout" or not s.usable:
                continue
            w.writerow([s.id, s.source or "", s.label, s.failure_type or "",
                        "" if s.onset_s is None else s.onset_s])
            rows += 1
    return rows
