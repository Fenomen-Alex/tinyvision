#!/usr/bin/env python3
"""Automate the real-footage intake -> evaluation path.

Given a *session sheet* (one complete print per session, tagged with a printer,
camera, label, failure type, onset and split), this runs the whole path:

    intake (validate, fail closed)
        -> tv poc per video (auditable per-video report, calibration/validation only)
        -> tv calibrate (select a per-printer threshold on calibration clips)
        -> validation reported independently (never used to select the threshold)
        -> held-out evaluation on the LOCKED holdout at the selected threshold
        -> release-gate inputs (holdout sessions CSV + held-out report)

Ordering guarantee: the **locked holdout** is not opened (no inference, no
scoring) until *after* the threshold has been fixed on calibration clips only.
The per-video ``tv poc`` step deliberately excludes locked-holdout sessions.

Nothing here trains a model, changes any gate policy or evaluation criterion, or
touches Git. Every artifact is written under ``--data-dir`` (keep it outside the
repository). The numbers are only meaningful for the declared printer/camera;
per-printer calibration does NOT establish cross-printer generalization.

Example::

    training/.venv/bin/python scripts/run_session_intake.py \
        --spec footage/sessions.yaml \
        --data-dir ~/tinyvision-data/session-intake \
        --checkpoint experiments/rc1_release_candidate_shufflenet_v2_x1_0/artifacts/best.pt
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

# Sessions whose media may be read before the threshold is fixed. The locked
# holdout is intentionally absent: it is only read in the final held-out step.
_PRE_THRESHOLD_SPLITS = ("calibration", "validation")


def _load_or_raise(path: str):
    from tv.sessions import SessionSheetError, load_session_sheet

    try:
        return load_session_sheet(path)
    except SessionSheetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)


def _score_sessions(sheet, split, checkpoint, image_size, sample_fps, max_frames):
    """Return {session_id: [{"timestamp_s", "score"}]} for one split.

    Uses each session's precomputed ``scores`` when present, otherwise the frozen
    model via the canonical POC preprocessing.
    """
    from tv.calibrate import CalibrationClip, CalibrationError, score_clip

    targets = [s for s in sheet.sessions if s.split == split and s.usable]
    need_model = any(s.scores is None for s in targets)
    model = transform = device = info = None
    if need_model:
        from tv.poc import PocError, load_model

        if not checkpoint:
            print("error: scoring requires --checkpoint (some sessions have no "
                  "precomputed scores)", file=sys.stderr)
            raise SystemExit(2)
        try:
            model, transform, device, info = load_model(checkpoint, image_size, "auto")
        except PocError as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise SystemExit(2)

    out: dict[str, list] = {}
    for s in targets:
        clip = CalibrationClip(id=s.id, label=s.label, source=s.source or "",
                               printer=s.printer, camera=s.camera, split=split,
                               scores=s.scores)
        if s.scores is not None:
            out[s.id] = s.scores
        else:
            try:
                out[s.id] = score_clip(clip, model, transform, device, info.fail_idx,
                                       sample_fps, max_frames)
            except CalibrationError as exc:
                print(f"error: {exc}", file=sys.stderr)
                raise SystemExit(2)
    return out


def _run_poc(sheet, data_dir, checkpoint, image_size, sample_fps, max_frames):
    """Auditable per-video POC report, for pre-threshold splits only.

    The locked holdout is never opened here; it is scored once, later, after the
    threshold is fixed.
    """
    from tv.poc import PocConfig, PocError, run_poc

    poc_dir = Path(data_dir) / "poc"
    poc_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for s in sheet.sessions:
        if s.split not in _PRE_THRESHOLD_SPLITS or not s.source:
            continue
        if not checkpoint:
            print(f"  poc: skipped {s.id}: no checkpoint", file=sys.stderr)
            skipped += 1
            continue
        try:
            config = PocConfig(source=s.source, checkpoint=checkpoint,
                               image_size=image_size, sample_fps=sample_fps,
                               max_frames=max_frames)
            result = run_poc(config)
        except PocError as exc:
            print(f"  poc: skipped {s.id}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        with open(poc_dir / f"{s.id}.json", "w") as f:
            json.dump(result, f, indent=2)
        written += 1
    return written, skipped


def _environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        env["torch"] = torch.__version__
    except Exception:
        pass
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=Path(__file__).resolve().parents[1])
        if head.returncode == 0:
            env["git_head"] = head.stdout.strip()
    except Exception:
        pass
    return env


def _run_meta(cal, sheet, checkpoint, args):
    """Identity/settings/provenance block shared by the run and held-out reports."""
    return {
        "checkpoint": str(checkpoint) if checkpoint else None,
        "model": cal.get("model"),
        "config": {
            "image_size": args.image_size,
            "sample_fps": cal["config"].get("sample_fps"),
            "max_frames": args.max_frames,
            "confirm": args.confirm,
            "fpr_ceiling": args.fpr_ceiling,
            "thresholds": cal["config"].get("thresholds"),
        },
        "sheet": {
            "dataset_id": sheet.dataset_id,
            "printer": sheet.printer,
            "camera": sheet.camera,
            "license": sheet.license,
            "provenance": sheet.provenance,
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="Session sheet (YAML/JSON)")
    ap.add_argument("--data-dir", required=True,
                    help="Output directory for every artifact (keep it outside Git)")
    ap.add_argument("--checkpoint", default=None,
                    help="Frozen best.pt (required if any session lacks scores)")
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--sample-fps", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--thresholds", default=None,
                    help="Comma-separated threshold grid for calibration")
    ap.add_argument("--confirm", type=int, default=1,
                    help="Consecutive frames above threshold to alarm a session")
    ap.add_argument("--fpr-ceiling", type=float, default=0.1,
                    help="Max false-alarm rate per normal print allowed in selection")
    ap.add_argument("--no-probe", dest="probe", action="store_false",
                    help="Skip probing local media during validation (default: probe)")
    ap.add_argument("--skip-poc", action="store_true",
                    help="Skip the per-video `tv poc` reports")
    ap.set_defaults(probe=True)
    args = ap.parse_args(argv)

    from tv.calibrate import CalibrationError, calibrate, format_summary, load_clip_spec
    from tv.sessions import (
        build_calibrate_spec,
        describe_sheet,
        emit_holdout_csv,
        heldout_report,
        validate_sessions,
        validation_report,
        write_calibrate_spec,
    )

    data_dir = Path(args.data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    sheet = _load_or_raise(args.spec)
    checkpoint = args.checkpoint or sheet.checkpoint

    # 1. Intake validation — fail closed.
    v = validate_sessions(sheet, probe=args.probe)
    report = v.report()
    report["sheet"] = describe_sheet(sheet)
    (data_dir / "intake_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[1/5] intake validation: ok={report['ok']} "
          f"errors={report['n_errors']} warnings={report['n_warnings']} "
          f"(probe={'on' if args.probe else 'off'})")
    for f in v.flags:
        print(f"      {f.severity}: {f.code}: {f.message}")
    if not v.passes():
        print("\nintake FAILED (fail closed); no downstream artifacts written.",
              file=sys.stderr)
        return 1

    # 2. Emit the calibrate spec (calibration + validation->holdout; locked holdout excluded).
    spec = build_calibrate_spec(sheet)
    if checkpoint:
        spec["checkpoint"] = checkpoint
    spec_path = data_dir / "calibration-spec.yaml"
    write_calibrate_spec(spec, spec_path)
    print(f"[2/5] wrote calibrate spec: {spec_path} "
          f"({len(spec['clips'])} clips; locked holdout excluded)")

    # 3. Per-video POC reports (calibration/validation only; locked holdout untouched).
    if args.skip_poc:
        print("[3/5] poc per video: skipped")
    else:
        n, skipped = _run_poc(sheet, data_dir, checkpoint, args.image_size,
                              args.sample_fps, args.max_frames)
        extra = f" ({skipped} skipped)" if skipped else ""
        print(f"[3/5] poc per video: {n} report(s) under {data_dir / 'poc'}{extra} "
              "[calibration/validation only]")

    # 4. Calibrate the per-printer threshold on calibration clips only.
    thresholds = None
    if args.thresholds:
        try:
            thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
        except ValueError:
            print(f"error: --thresholds must be comma-separated numbers, "
                  f"got {args.thresholds!r}", file=sys.stderr)
            return 2
    try:
        loaded = load_clip_spec(spec_path)
        cal = calibrate(loaded,
                        checkpoint=checkpoint,
                        image_size=args.image_size,
                        sample_fps=args.sample_fps,
                        thresholds=thresholds,
                        confirm=args.confirm,
                        fpr_ceiling=args.fpr_ceiling,
                        max_frames=args.max_frames)
    except CalibrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    (data_dir / "calibration.json").write_text(json.dumps(cal, indent=2) + "\n")
    print("[4/5] calibration (threshold selected on calibration clips only):")
    for line in format_summary(cal).splitlines():
        # tv calibrate names the independent tier "holdout"; in the intake mapping
        # that tier is the *validation* split. Relabel for clarity.
        line = line.replace("[holdout]", "[validation]").replace(", holdout {",
                                                                    ", validation {")
        print(f"      {line}")
    # Independent validation report (never used to select the threshold).
    meta = _run_meta(cal, sheet, checkpoint, args)
    val = validation_report(cal, sheet, meta=meta)
    (data_dir / "validation_report.json").write_text(json.dumps(val, indent=2) + "\n")
    if val["evaluated"]:
        print(f"      validation (independent, NOT the locked holdout): "
              f"n_normal={val['n_normal']} n_failing={val['n_failing']} "
              f"detection={val['detection_rate']} "
              f"far/print={val['false_alarm_rate_per_print']}")

    # 5. Held-out evaluation on the LOCKED holdout at the selected threshold.
    threshold = cal["selection"].get("selected_threshold")
    hold_path = data_dir / "holdout_eval.json"
    rows = emit_holdout_csv(sheet, data_dir / "holdout_sessions.csv")
    hold_eval = None
    if threshold is None:
        print("[5/5] locked holdout: no threshold selected "
              "(calibration insufficient); locked holdout NOT evaluated")
    else:
        # Locked holdout is read only NOW, after the threshold is fixed.
        scored = _score_sessions(sheet, "holdout", checkpoint, args.image_size,
                                 args.sample_fps, args.max_frames)
        hold_eval = heldout_report(sheet, scored, threshold, args.confirm, meta=meta)
        hold_path.write_text(json.dumps(hold_eval, indent=2) + "\n")
        print(f"[5/5] locked holdout @ threshold {threshold}: "
              f"detection={hold_eval['detection_rate']} "
              f"far/print={hold_eval['false_alarm_rate_per_print']} "
              f"(n_normal={hold_eval['n_normal']}, n_failing={hold_eval['n_failing']})")

    # Run manifest: identity, settings, provenance, and artifact list.
    artifacts = ["intake_report.json", "calibration-spec.yaml", "calibration.json",
                 "validation_report.json", "holdout_sessions.csv"]
    if hold_eval is not None:
        artifacts.append("holdout_eval.json")
    if not args.skip_poc:
        artifacts.append("poc/")
    manifest = {
        "schema": "tinyvision-session-intake-run",
        "schema_version": 1,
        "spec": str(Path(args.spec).expanduser()),
        "data_dir": str(data_dir),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "settings": {
            "probe": args.probe,
            "skip_poc": args.skip_poc,
            "image_size": args.image_size,
            "sample_fps": args.sample_fps,
            "max_frames": args.max_frames,
            "thresholds": thresholds,
            "confirm": args.confirm,
            "fpr_ceiling": args.fpr_ceiling,
        },
        "sheet": describe_sheet(sheet),
        "calibration": {
            "schema": cal.get("schema"),
            "status": cal.get("status"),
            "select_on": cal["config"].get("select_on"),
            "selected_threshold": threshold,
        },
        "validation": {
            "evaluated": val["evaluated"],
            "role": "independent_reporting_only",
            "n_normal": val.get("n_normal"),
            "n_failing": val.get("n_failing"),
            "detection_rate": val.get("detection_rate"),
            "false_alarm_rate_per_print": val.get("false_alarm_rate_per_print"),
        },
        "locked_holdout": {
            "evaluated": hold_eval is not None,
            "n_normal": (hold_eval or {}).get("n_normal"),
            "n_failing": (hold_eval or {}).get("n_failing"),
            "detection_rate": (hold_eval or {}).get("detection_rate"),
            "false_alarm_rate_per_print": (hold_eval or {}).get("false_alarm_rate_per_print"),
        },
        "model": cal.get("model"),
        "environment": _environment(),
        "artifacts": artifacts,
        "provenance": sheet.provenance,
        "license": sheet.license,
    }
    (data_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\nArtifacts:")
    for name in ("intake_report.json", "calibration-spec.yaml", "calibration.json",
                 "validation_report.json", "run_manifest.json", "holdout_sessions.csv"):
        print(f"  {data_dir / name}")
    if hold_eval is not None:
        print(f"  {hold_path}")
    print(f"  {rows} locked-holdout session(s) recorded (release-gate input)")
    print("\nInterpretation:")
    print("  - calibration.json        : threshold selected on CALIBRATION clips only")
    print("  - validation_report.json  : independent validation (NOT the locked holdout; "
          "never selected the threshold)")
    print("  - holdout_eval.json       : LOCKED holdout, evaluated once at the fixed "
          "threshold - the final held-out result")
    print("  - run_manifest.json       : identity, settings, roles, provenance")
    print("\nNOTE: calibrated for the declared printer/camera only. A per-printer "
          "threshold does not establish cross-printer generalization, and none of "
          "these numbers change the release gate policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
