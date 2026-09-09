"""TinyVision training pipeline CLI.

Usage:
    tv train       --config CONFIG_PATH
    tv evaluate    --config CONFIG_PATH [--checkpoint PATH]
    tv export      --config CONFIG_PATH [--checkpoint PATH]
    tv split       --data-dir DIR --output-dir DIR [--ratios 0.7,0.15,0.15] [--seed 42]
    tv holdout     --data-dir DIR [--output-dir DIR]
    tv capture     --output-dir DIR --class normal|failing [--session-id ID] [--duration SECS] [--fps N]
    tv list-models [--task classification|detection]
    tv env-validate --dir STAGING_DIR
    tv env-split --dir STAGING_DIR [--assign key=split,... | --ratios R,R,R] [--seed N]
    tv env-readiness [--holdings PATH] [--datasets-dir DIR]
    tv env-index --kind trier|bambu [--root PATH] [--metadata JSON]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_train(args: argparse.Namespace) -> None:
    from tv.config import load_config
    from tv.train import train

    config = load_config(Path(args.config))
    print(f"Training {config.model} on {config.dataset.name} ({config.task})")
    result = train(config, base_dir=args.experiments_dir)
    print("\nTraining complete.")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


def cmd_evaluate(args: argparse.Namespace) -> None:
    from tv.config import load_config
    from tv.evaluate import evaluate

    config = load_config(Path(args.config))
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    result = evaluate(config, checkpoint_path=ckpt, base_dir=args.experiments_dir)
    print("\nEvaluation results:")
    print(json.dumps(result, indent=2))


def cmd_export(args: argparse.Namespace) -> None:
    from tv.config import load_config
    from tv.export import export

    config = load_config(Path(args.config))
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    result = export(config, checkpoint_path=ckpt, base_dir=args.experiments_dir)
    print("\nExport results:")
    print(json.dumps(result, indent=2))


def cmd_split(args: argparse.Namespace) -> None:
    from tv.data.splitter import split_dataset

    ratios = tuple(float(x) for x in args.ratios.split(","))
    if len(ratios) != 3:
        print("Error: --ratios must have exactly 3 comma-separated values")
        sys.exit(1)

    output_dir = args.output_dir
    if output_dir is None:
        # Default to <data-dir>/splits so the training loader finds it.
        output_dir = str(Path(args.data_dir) / "splits")

    result = split_dataset(
        data_dir=args.data_dir,
        output_dir=output_dir,
        ratios=ratios,
        seed=args.seed,
    )
    print("Split complete:")
    print(json.dumps(result.stats, indent=2))
    print(f"\nSplit manifest written to {output_dir}/splits.json")


def cmd_holdout(args: argparse.Namespace) -> None:
    from tv.data.splitter import build_holdout

    result = build_holdout(args.data_dir, output_dir=args.output_dir)
    print("Holdout test manifest complete (all sessions -> one held-out test set):")
    print(json.dumps(result.stats, indent=2))
    print(f"\nWritten to {args.output_dir or (Path(args.data_dir) / 'splits')}")


def cmd_capture(args: argparse.Namespace) -> None:
    from tv.data.capture import CAPTURE_CLASSES, capture_session

    session_class = args.session_class
    if session_class is None:
        session_class = _prompt_class()
    if session_class not in CAPTURE_CLASSES:
        print(f"Error: class must be one of {CAPTURE_CLASSES}")
        sys.exit(1)

    result = capture_session(
        output_dir=args.output_dir,
        session_class=session_class,
        session_id=args.session_id,
        duration=args.duration,
        fps=args.fps,
        camera=args.camera,
        printer=args.printer,
        object=args.object,
        failure_type=args.failure_type,
        notes=args.notes,
    )
    print("\nCapture complete:")
    print(json.dumps(result, indent=2))


def _prompt_class() -> str:
    """Interactively and strictly confirm the session class before recording.

    Re-prompts until ``normal`` or ``failing`` (also accepts ``n``/``f``) is
    chosen. This makes it hard to accidentally label a session incorrectly.
    """
    while True:
        ans = input("Session class? [normal/failing, n/f]: ").strip().lower()
        if ans in ("normal", "n"):
            return "normal"
        if ans in ("failing", "f"):
            return "failing"
        print(f"  Invalid choice {ans!r}; enter 'normal' or 'failing'.")


def cmd_eval_deep(args: argparse.Namespace) -> None:
    from tv.config import load_config
    from tv.deepeval import deep_evaluate, print_report

    config = load_config(Path(args.config))
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    report = deep_evaluate(
        config,
        checkpoint_path=ckpt,
        base_dir=args.experiments_dir,
        split=args.split,
        output_json=args.output_json,
        metadata_dir=args.metadata_dir,
    )
    print_report(report)


def cmd_robustness(args: argparse.Namespace) -> None:
    from tv.config import load_config
    from tv.robustness import evaluate_robustness, print_report

    config = load_config(Path(args.config))
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    perturbations = args.perturbations.split(",") if args.perturbations else None
    report = evaluate_robustness(
        config,
        checkpoint_path=ckpt,
        base_dir=args.experiments_dir,
        split=args.split,
        output_json=args.output_json,
        perturbations=perturbations,
        seed=args.seed,
    )
    print_report(report)


def cmd_temporal(args: argparse.Namespace) -> None:
    from tv.config import load_config
    from tv.temporaleval import run_frozen_inference, evaluate_split

    config = load_config(Path(args.config))
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    ckpt = ckpt or (
        Path(args.experiments_dir) / config.experiment_id / "artifacts" / "best.pt"
    )
    cache = Path(args.cache)

    print(f"Running frozen inference on split '{args.split}' -> {cache}")
    result = run_frozen_inference(
        config, ckpt, args.split, output_json=cache if args.write_cache else None
    )

    variants = {
        "raw": {"threshold": 0.5},
        "ewma": {"alpha": 0.15, "threshold": 0.5},
        "fsm": {
            "alpha": 0.15,
            "use_delta": False,
            "fail_threshold_abs": 0.65,
            "fail_threshold": 0.30,
            "confirm": 3,
            "warmup": 30,
            "suspect_threshold": 0.05,
            "recover_threshold": 0.01,
        },
    }
    if args.variant and args.variant != "all":
        variants = {args.variant: variants[args.variant]}

    for name, params in variants.items():
        r = evaluate_split(result, params, name)
        print(
            f"[{name}] det={r['sequence_detection_rate']} missed={r['missed_failure_events']} "
            f"seqFA={r['sequence_false_alarm_rate']} nFA={r['n_false_trigger_seq']} "
            f"meanFNR={r['mean_seq_fnr']} meanFPR={r['mean_seq_fpr']} "
            f"delay_frames={r['proxy_delay_frames_median']}"
        )


def cmd_audit_dataset(args: argparse.Namespace) -> None:
    from tv.data.audit import audit_dataset, print_report

    report = audit_dataset(
        data_dir=Path(args.data_dir),
        min_width=args.min_width,
        min_height=args.min_height,
    )
    print_report(report)

    if args.output_json:
        import json as jsonmod
        with open(args.output_json, "w") as f:
            jsonmod.dump(report.to_dict(), f, indent=2)
        print(f"\nJSON report written to {args.output_json}")


def cmd_skz_eval(args: argparse.Namespace) -> None:
    from tv.skz_experiment import evaluate_probe

    result = evaluate_probe(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        subset=args.subset,
        train_fraction=args.train_fraction,
        seed=args.seed,
        image_size=args.image_size,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )
    print(json.dumps(result, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nJSON report written to {args.output_json}")


def cmd_skz_prep(args: argparse.Namespace) -> None:
    from tv.skz_experiment import prepare_expB_splits

    manifest = prepare_expB_splits(
        printing_errors_dir=args.printing_errors_dir,
        out_dir=args.out_dir,
        manifest_path=args.manifest,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


def cmd_exp2_prep(args: argparse.Namespace) -> None:
    from tv.exp2_external import prepare_exp2_splits

    manifest = prepare_exp2_splits(
        printing_errors_dir=args.printing_errors_dir,
        out_dir=args.out_dir,
    )
    print(json.dumps(manifest, indent=2))


def cmd_exp2_eval(args: argparse.Namespace) -> None:
    from tv.exp2_external import evaluate_external

    result = evaluate_external(
        checkpoint_path=args.checkpoint,
        subset=args.subset,
        image_size=args.image_size,
        batch_size=args.batch_size,
        threshold=args.threshold,
        kind=args.kind,
    )
    print(json.dumps(result, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nJSON report written to {args.output_json}")


def cmd_exp3_eval(args: argparse.Namespace) -> None:
    from tv.config import load_config
    from tv.exp3_external import compare

    result = compare(
        baseline_checkpoint=args.baseline,
        exp3_checkpoint=args.exp3,
        config=load_config(Path(args.config)),
    )
    print(json.dumps(result, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nJSON report written to {args.output_json}")


def cmd_exp4_eval(args: argparse.Namespace) -> None:
    from tv.config import load_config
    from tv.exp4_external import compare_three

    result = compare_three(
        baseline_checkpoint=args.baseline,
        exp3_checkpoint=args.exp3,
        exp4_checkpoint=args.exp4,
        config=load_config(Path(args.config)),
    )
    print(json.dumps(result, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nJSON report written to {args.output_json}")


def cmd_poc(args: argparse.Namespace) -> None:
    from tv.poc import PocConfig, run_poc

    config = PocConfig(
        source=args.input if args.input else args.camera,
        checkpoint=args.checkpoint,
        image_size=args.image_size,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
        threshold=args.threshold,
        annotated_output=args.annotated_output,
        alpha=args.alpha,
        baseline_window=args.baseline_window,
        warmup=args.warmup,
        suspect_threshold=args.suspect_threshold,
        fail_threshold=args.fail_threshold,
        recover_threshold=args.recover_threshold,
        fail_threshold_abs=args.fail_threshold_abs,
        confirm=args.confirm,
    )
    result = run_poc(config)
    print(json.dumps(result, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nJSON report written to {args.output_json}")


def cmd_env_validate(args: argparse.Namespace) -> None:
    from tv.envdata.schema import load_manifest
    from tv.envdata.validate import validate_manifest, validate_samples

    from pathlib import Path as _Path
    root = _Path(args.dir)
    manifest = load_manifest(root)
    v = validate_manifest(manifest, staging_root=root)
    samples_dir = root / "samples"
    if samples_dir.is_dir():
        v2 = validate_samples(manifest, samples_dir=samples_dir)
        v.flags.extend(v2.flags)
    print(json.dumps(v.report(), indent=2))
    if not v.passes():
        sys.exit(1)


def cmd_env_split(args: argparse.Namespace) -> None:
    from tv.envdata.schema import load_manifest, dump_manifest
    from tv.envdata.split import (
        allocate_environments,
        apply_plan,
        check_environment_leakage,
        drop_assignments,
    )

    root = args.dir
    manifest = load_manifest(root)
    if args.assign:
        plan_entries = {}
        for pair in args.assign.split(","):
            if "=" not in pair:
                raise SystemExit(f"expected key=split, got {pair!r}")
            key, split = pair.split("=", 1)
            split = split.strip()
            if split not in ("train", "validation", "external_holdout"):
                raise SystemExit(f"unknown split {split!r}")
            envs = [e for e in manifest.environments
                    if e.environment_key == key or e.environment_id == key]
            if not envs:
                raise SystemExit(f"no environment matching {key!r}")
            if len(envs) > 1:
                raise SystemExit(f"key {key!r} matches multiple environments")
            plan_entries[envs[0].environment_id] = split
        from tv.envdata.split import SplitPlan
        plan = SplitPlan(allocations=plan_entries)
        changed = apply_plan(manifest, plan)
    elif args.ratios:
        ids = [e.environment_id for e in manifest.environments
               if e.split_assignment is None]
        if not ids:
            print("No unassigned environments; nothing to allocate.")
            return
        ratios = tuple(float(x) for x in args.ratios.split(","))
        from tv.envdata.split import SplitPlan
        plan = allocate_environments(ids, ratios=ratios, seed=args.seed)
        changed = apply_plan(manifest, plan)
    else:
        from tv.envdata.split import SplitPlan
        plan = SplitPlan(allocations={})
        changed = []
        print("No assignment provided; only validating current assignments.")

    dump_manifest(manifest, root)
    report = check_environment_leakage(manifest)
    for issue in report.issues:
        print(f"  leak: {issue.code}: {issue.message}")
    print(f"Assigned {len(changed)} environment(s); manifest written to {root}/manifests/env_manifest.json")
    if not report.ok:
        sys.exit(1)
    print("Environment-level leakage check: OK")


def cmd_env_readiness(args: argparse.Namespace) -> None:
    from tv.envdata.schema import load_manifest, load_samples
    from tv.envdata.readiness import (
        HoldingsSnapshot,
        StagedManifestView,
        compute_readiness,
        default_holdings_path,
        generate_booster_text,
    )

    import glob
    holdings = HoldingsSnapshot.load(args.holdings or default_holdings_path())
    staged_roots = [p for p in sorted(glob.glob(str(args.datasets_dir) + "/*")) if (Path(p) / "manifests" / "env_manifest.json").exists()]
    views = []
    for root in staged_roots:
        manifest = load_manifest(root)
        views.append(StagedManifestView(manifest, load_samples(root)))
    report = compute_readiness(holdings, views)
    print(json.dumps(report, indent=2))
    if not report["training_ready"]:
        print("\n" + generate_booster_text(report))
        sys.exit(1)


def cmd_env_index(args: argparse.Namespace) -> None:
    from tv.envdata import trier, bambu

    import urllib.error
    if args.kind == "trier":
        target = Path("datasets/staging/trier")
        meta_dir = target / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        if args.metadata:
            with open(args.metadata) as f:
                record = json.load(f)
            characterization = trier.characterize_from_zenodo(record)
            with open(meta_dir / "characterization.json", "w") as f:
                json.dump(characterization, f, indent=2)
            print("Wrote characterization.json from Zenodo metadata.")
        if args.root:
            index = trier.index_trier_tree(args.root)
            with open(meta_dir / "index.json", "w") as f:
                json.dump(index, f, indent=2)
            print(json.dumps(index, indent=2))
        else:
            print(f"NOTE: no --root given; tree indexing skipped. Place extracted "
                  f"Trier data on disk and re-run with --root.")
    elif args.kind == "bambu":
        target = Path("datasets/staging/bambu")
        meta_dir = target / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        try:
            csvs = bambu.fetch_metadata_csvs(meta_dir)
        except urllib.error.URLError as exc:
            print(f"network unavailable ({exc}); using cached CSVs if present")
            csvs = {
                "timelapses": meta_dir / "timelapses_metadata.csv",
                "videos": meta_dir / "videos_metadata.csv",
            }
        if not (csvs["timelapses"].exists() and csvs["videos"].exists()):
            print("metadata CSVs unavailable; run `tv env-index --kind bambu` "
                  "with network, or drop the CSVs into datasets/staging/bambu/metadata/")
            sys.exit(1)
        inventory = bambu.build_inventory(csvs["timelapses"], csvs["videos"])
        with open(meta_dir / "inventory.json", "w") as f:
            json.dump(inventory, f, indent=2)
        print(json.dumps(inventory, indent=2))
    else:
        raise SystemExit("unknown --kind (trier|bambu)")


def cmd_review_prep(args: argparse.Namespace) -> None:
    from collections import defaultdict
    from tv.envdata.schema import load_manifest, load_samples
    from tv.envdata.review_units import (
        assign_priorities,
        build_review_units,
        render_contact_sheet,
        write_review_units,
    )

    root = Path(args.dir)
    manifest = load_manifest(root)
    samples = load_samples(root)
    out_path = root / "review" / (args.review_file or "review_units.json")
    existing = {}
    if out_path.exists():
        from tv.envdata.review_units import load_review_units
        existing = {u.review_unit_id: u for u in load_review_units(out_path)}
    rec_meta = {rec.recording_id: rec.metadata for rec in manifest.recordings}
    max_gap = max(
        (m.get("extraction_stride", 1) for m in rec_meta.values()), default=1,
    )
    units = build_review_units(manifest, samples, max_gap=max_gap)
    for u in units:
        prev = existing.get(u.review_unit_id)
        if prev is not None:
            u.decision = prev.decision
            u.failure_type = prev.failure_type
            u.notes = prev.notes
    assign_priorities(units, manifest)
    frame_root = Path(args.frame_root) if args.frame_root else None
    resolver = None
    if frame_root is not None:
        frame_map: dict[tuple[str, int], str] = {}
        for s in samples:
            file = (s.metadata or {}).get("file")
            if file is not None:
                frame_map[(s.recording_id, s.frame_index)] = file
        def _res(rec_id, idx):  # noqa: E306
            return frame_root / frame_map[(rec_id, idx)] if (rec_id, idx) in frame_map else None
        resolver = _res
    n_sheets = 0
    if frame_root is not None:
        sheets_dir = root / "review" / "sheets"
        for u in units:
            if u.type == "sequence":
                sheet = sheets_dir / f"{u.review_unit_id}.jpg"
                if not sheet.exists() or args.force:
                    render_contact_sheet(frame_root, u, sheet, resolver=resolver)
                    n_sheets += 1
    payload = write_review_units(out_path, units)
    print(json.dumps(payload, indent=2))
    print(f"review units -> {out_path} (sheets rendered: {n_sheets})")


def cmd_review_confirm(args: argparse.Namespace) -> None:
    from tv.envdata.review_units import load_review_units, write_review_units

    root = Path(args.dir)
    review_path = root / "review" / (args.review_file or "review_units.json")
    units = load_review_units(review_path)
    target = [u for u in units if u.review_unit_id == args.unit]
    if not target:
        raise SystemExit(f"unknown review unit {args.unit!r} in {review_path}")
    u = target[0]
    decision = args.decision.lower()
    if decision not in {"normal", "failing", "uncertain"}:
        raise SystemExit(f"invalid decision {args.decision!r}")
    if decision == "failing" and not args.failure_type:
        raise SystemExit("--failure-type required when decision=failing")
    if decision != "failing":
        args.failure_type = None
    u.decision = decision
    u.failure_type = args.failure_type
    if args.note:
        u.notes = args.note
    payload = write_review_units(review_path, units)
    print(f"recorded human decision {decision} on {u.review_unit_id} -> {review_path}")


def cmd_review_merge(args: argparse.Namespace) -> None:
    from tv.envdata.schema import load_manifest, load_samples, write_samples
    from tv.envdata.review_units import merge_completed

    root = Path(args.dir)
    manifest = load_manifest(root)
    samples = load_samples(root)
    review_path = root / "review" / (args.review_file or "review_units.json")
    result = merge_completed(
        review_path, manifest, samples,
        require_all=args.require_all,
    )
    write_samples(root, samples)
    print(json.dumps(result, indent=2))
    n_verified = sum(1 for s in samples if s.verification == "verified")
    n_failing = sum(
        1 for s in samples
        if s.verification == "verified" and s.tinyvision_label == "failing"
    )
    print(f"verified samples: {n_verified} (failing: {n_failing})")


def cmd_list_models(args: argparse.Namespace) -> None:
    from tv.models.factory import get_model_spec, list_models

    for name in list_models(task=args.task):
        spec = get_model_spec(name)
        print(f"  {name:25s}  task={spec.task:15s}  license={spec.license}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tv",
        description="TinyVision training pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--config", required=True, help="Experiment config YAML")
    p_train.add_argument(
        "--experiments-dir", default="experiments", help="Base dir for experiments"
    )
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate a trained model")
    p_eval.add_argument("--config", required=True, help="Experiment config YAML")
    p_eval.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    p_eval.add_argument(
        "--experiments-dir", default="experiments", help="Base dir for experiments"
    )
    p_eval.set_defaults(func=cmd_evaluate)

    p_export = sub.add_parser("export", help="Export model to ONNX / INT8")
    p_export.add_argument("--config", required=True, help="Experiment config YAML")
    p_export.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    p_export.add_argument(
        "--experiments-dir", default="experiments", help="Base dir for experiments"
    )
    p_export.set_defaults(func=cmd_export)

    p_split = sub.add_parser("split", help="Generate leakage-free train/val/test splits")
    p_split.add_argument("--data-dir", required=True, help="Root data directory")
    p_split.add_argument(
        "--output-dir",
        default=None,
        help="Output dir for splits (default: <data-dir>/splits)",
    )
    p_split.add_argument(
        "--ratios", default="0.7,0.15,0.15", help="Train,val,test ratios (comma-sep)"
    )
    p_split.add_argument("--seed", type=int, default=42, help="Random seed")
    p_split.set_defaults(func=cmd_split)

    p_holdout = sub.add_parser(
        "holdout", help="Build a single held-out test manifest (frozen-model eval)"
    )
    p_holdout.add_argument("--data-dir", required=True, help="Root data directory (Layout A)")
    p_holdout.add_argument(
        "--output-dir",
        default=None,
        help="Output dir for the manifest (default: <data-dir>/splits)",
    )
    p_holdout.set_defaults(func=cmd_holdout)

    p_capture = sub.add_parser("capture", help="Capture a labelled webcam session (Layout A)")
    p_capture.add_argument(
        "--output-dir",
        required=True,
        help="Root dataset dir (e.g. datasets/target-camera); session saved under <class>/<session_id>/",
    )
    p_capture.add_argument(
        "--class",
        dest="session_class",
        default=None,
        help="Session class: normal or failing (prompted interactively if omitted)",
    )
    p_capture.add_argument("--session-id", default=None, help="Session ID (default: next session_XXX within class)")
    p_capture.add_argument(
        "--duration", type=float, default=10.0, help="Capture duration in seconds (0=until Ctrl+C)"
    )
    p_capture.add_argument("--fps", type=float, default=2.0, help="Target frames per second")
    p_capture.add_argument("--camera", type=int, default=0, help="Camera device index")
    p_capture.add_argument("--printer", default=None, help="Printer identifier (recorded in metadata)")
    p_capture.add_argument("--object", default=None, help="Print/object identifier (recorded in metadata)")
    p_capture.add_argument("--failure-type", default=None, help="Failure type e.g. stringing, under_extrusion (failing only)")
    p_capture.add_argument("--notes", default=None, help="Free-text notes (recorded in metadata)")
    p_capture.set_defaults(func=cmd_capture)

    p_audit = sub.add_parser("audit-dataset", help="Audit a dataset for integrity and statistics")
    p_audit.add_argument("--data-dir", required=True, help="Root data directory (Layout A or B)")
    p_audit.add_argument("--min-width", type=int, default=32, help="Minimum image width")
    p_audit.add_argument("--min-height", type=int, default=32, help="Minimum image height")
    p_audit.add_argument("--output-json", default=None, help="Write JSON report to file")
    p_audit.set_defaults(func=cmd_audit_dataset)

    p_env_validate = sub.add_parser(
        "env-validate",
        help="Validate a staged dataset manifest (environment integrity, licensing, staging layout)",
    )
    p_env_validate.add_argument(
        "--dir",
        required=True,
        help="Staging root (contains manifests/env_manifest.json), e.g. datasets/staging/bambu",
    )
    p_env_validate.set_defaults(func=cmd_env_validate)

    p_env_split = sub.add_parser(
        "env-split",
        help="Assign/validate environment-level splits (one environment -> exactly one split)",
    )
    p_env_split.add_argument(
        "--dir",
        required=True,
        help="Staging root (contains manifests/env_manifest.json)",
    )
    p_env_split.add_argument(
        "--assign",
        default=None,
        help="Comma-separated key=split, e.g. env_bambu_borillion_bonnp-ipcam_inside_chamber=external_holdout",
    )
    p_env_split.add_argument(
        "--ratios", default=None,
        help="Auto-allocate unassigned environments with train,val,holdout ratios like 0.6,0.2,0.2",
    )
    p_env_split.add_argument("--seed", type=int, default=42, help="Allocation seed")
    p_env_split.set_defaults(func=cmd_env_split)

    p_env_readiness = sub.add_parser(
        "env-readiness",
        help="Report whether staged + holdings satisfy the next-training gate",
    )
    p_env_readiness.add_argument(
        "--holdings",
        default=None,
        help="Path to canonical_holdings.json (default datasets/staging/readiness/canonical_holdings.json)",
    )
    p_env_readiness.add_argument(
        "--datasets-dir",
        default="datasets/staging",
        help="Where staged datasets live (default: datasets/staging)",
    )
    p_env_readiness.set_defaults(func=cmd_env_readiness)

    p_env_index = sub.add_parser(
        "env-index",
        help="Build environment inventories from dataset metadata (trier | bambu)",
    )
    p_env_index.add_argument("--kind", required=True, choices=["trier", "bambu"])
    p_env_index.add_argument(
        "--root", default=None,
        help="Trier: path to an extracted local copy of the dataset",
    )
    p_env_index.add_argument(
        "--metadata", default=None,
        help="Trier: path to a downloaded Zenodo API record JSON (characterization)",
    )
    p_env_index.set_defaults(func=cmd_env_index)

    p_review_prep = sub.add_parser(
        "review-prep",
        help="Build human review units + contact sheets from a staged manifest",
    )
    p_review_prep.add_argument("--dir", required=True, help="Staging dir with manifests/env_manifest.json")
    p_review_prep.add_argument(
        "--frame-root", default=None,
        help="Extracted 1-fps frame root: <root>/<recording_id>/frame_<NNNN>.jpg",
    )
    p_review_prep.add_argument("--review-file", default="review_units.json")
    p_review_prep.add_argument(
        "--force", action="store_true", help="Re-render contact sheets even if present"
    )
    p_review_prep.set_defaults(func=cmd_review_prep)

    p_review_confirm = sub.add_parser(
        "review-confirm",
        help="Record a single human decision on one review unit (surgical verification)",
    )
    p_review_confirm.add_argument("--dir", required=True, help="Staging dir with manifests/env_manifest.json")
    p_review_confirm.add_argument("--unit", required=True, help="Review unit id")
    p_review_confirm.add_argument(
        "--decision", required=True, choices=["normal", "failing", "uncertain"]
    )
    p_review_confirm.add_argument("--failure-type", default=None)
    p_review_confirm.add_argument("--note", default=None)
    p_review_confirm.add_argument("--review-file", default="review_units.json")
    p_review_confirm.set_defaults(func=cmd_review_confirm)

    p_review_merge = sub.add_parser(
        "review-merge",
        help="Merge completed human review decisions back into staged samples",
    )
    p_review_merge.add_argument("--dir", required=True, help="Staging dir with manifests/env_manifest.json")
    p_review_merge.add_argument("--review-file", default="review_units.json")
    p_review_merge.add_argument(
        "--require-all", action="store_true",
        help="Error out if any unit is missing a decision",
    )
    p_review_merge.set_defaults(func=cmd_review_merge)

    p_deep = sub.add_parser("eval-deep", help="Deep error analysis on a split")
    p_deep.add_argument("--config", required=True, help="Experiment config YAML")
    p_deep.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    p_deep.add_argument(
        "--experiments-dir", default="experiments", help="Base dir for experiments"
    )
    p_deep.add_argument(
        "--split", default="test", choices=["train", "val", "test"], help="Split to evaluate"
    )
    p_deep.add_argument("--output-json", default=None, help="Write JSON report to file")
    p_deep.add_argument(
        "--metadata-dir",
        default=None,
        help="Target-camera root with <class>/<session>/metadata.json for provenance + failure-type grouping",
    )
    p_deep.set_defaults(func=cmd_eval_deep)

    p_rob = sub.add_parser(
        "robustness",
        help="Offline robustness/generalization eval of a frozen model "
             "(perturbation severity sweep, no training)",
    )
    p_rob.add_argument("--config", required=True, help="Experiment config YAML")
    p_rob.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    p_rob.add_argument(
        "--experiments-dir", default="experiments", help="Base dir for experiments"
    )
    p_rob.add_argument(
        "--split", default="test", choices=["train", "val", "test"], help="Split to evaluate"
    )
    p_rob.add_argument("--output-json", default=None, help="Write JSON report to file")
    p_rob.add_argument(
        "--perturbations",
        default=None,
        help="Comma-separated list of perturbations (default: all)",
    )
    p_rob.add_argument("--seed", type=int, default=0, help="Seed for stochastic perturbations")
    p_rob.set_defaults(func=cmd_robustness)

    p_temp = sub.add_parser(
        "temporal",
        help="Offline temporal decision-system eval (raw/EWMA/FSM) on a frozen model",
    )
    p_temp.add_argument("--config", required=True, help="Experiment config YAML")
    p_temp.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    p_temp.add_argument(
        "--experiments-dir", default="experiments", help="Base dir for experiments"
    )
    p_temp.add_argument(
        "--split", default="test", choices=["train", "val", "test"], help="Split to evaluate"
    )
    p_temp.add_argument("--cache", default=None, help="Cache JSON path")
    p_temp.add_argument(
        "--write-cache", action="store_true", help="Write the frozen-inference cache to --cache"
    )
    p_temp.add_argument(
        "--variant",
        default="all",
        choices=["all", "raw", "ewma", "fsm"],
        help="Decision variant(s) to evaluate",
    )
    p_temp.set_defaults(func=cmd_temporal)

    p_list = sub.add_parser("list-models", help="List available model architectures")
    p_list.add_argument("--task", default=None, help="Filter by task (classification/detection)")
    p_list.set_defaults(func=cmd_list_models)

    p_skz_eval = sub.add_parser(
        "skz-eval",
        help="Evaluate a frozen binary checkpoint against the SKZ Nozzle-Cam probe "
             "(external generalization, Experiment A; no training)",
    )
    p_skz_eval.add_argument("--checkpoint", required=True, help="Path to model checkpoint (best.pt)")
    p_skz_eval.add_argument(
        "--manifest",
        default=None,
        help="Path to probe manifest.json (default: datasets/external/skz-probe/manifest.json)",
    )
    p_skz_eval.add_argument(
        "--subset",
        default="all",
        choices=["all", "skz_eval", "skz_aux"],
        help="Token subset to evaluate: all (default), held-out skz_eval tokens, or skz_aux tokens",
    )
    p_skz_eval.add_argument("--train-fraction", type=float, default=None,
                            help="Token split train fraction (default 0.8)")
    p_skz_eval.add_argument("--seed", type=int, default=42, help="Token split seed")
    p_skz_eval.add_argument("--image-size", type=int, default=224, help="Resize target (default 224)")
    p_skz_eval.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    p_skz_eval.add_argument("--threshold", type=float, default=0.5,
                            help="Decision threshold on P(failing)")
    p_skz_eval.add_argument("--output-json", default=None, help="Write full report to JSON")
    p_skz_eval.set_defaults(func=cmd_skz_eval)

    p_skz_prep = sub.add_parser(
        "skz-prep",
        help="Build the Experiment B derived split dir: printing-errors train + SKZ-aux "
             "(group-split), with unchanged printing-errors val/test",
    )
    p_skz_prep.add_argument(
        "--printing-errors-dir",
        required=True,
        help="Root printing-errors dataset dir (has splits/train.csv etc.)",
    )
    p_skz_prep.add_argument(
        "--out-dir", required=True, help="Where to write derived splits/ + derived_manifest.json"
    )
    p_skz_prep.add_argument(
        "--manifest",
        default=None,
        help="Path to probe manifest.json (default: datasets/external/skz-probe/manifest.json)",
    )
    p_skz_prep.add_argument("--train-fraction", type=float, default=None,
                            help="Token split train fraction (default 0.8)")
    p_skz_prep.add_argument("--seed", type=int, default=42, help="Token split seed")
    p_skz_prep.set_defaults(func=cmd_skz_prep)

    p_exp2_prep = sub.add_parser(
        "exp2-prep",
        help="Build the Experiment-2A derived split dir: printing-errors train + verified "
             "external failing frames (NORMAL001/SP001), unchanged printing-errors val/test",
    )
    p_exp2_prep.add_argument(
        "--printing-errors-dir",
        required=True,
        help="Root printing-errors dataset dir (has splits/train.csv etc.)",
    )
    p_exp2_prep.add_argument(
        "--out-dir", required=True,
        help="Where to write derived splits/ + derived_manifest.json",
    )
    p_exp2_prep.set_defaults(func=cmd_exp2_prep)

    p_exp2_eval = sub.add_parser(
        "exp2-eval",
        help="Evaluate a frozen binary checkpoint over verified external video frames "
             "(UE003 normal eval / NORMAL001+SP001 failing)",
    )
    p_exp2_eval.add_argument("--checkpoint", required=True, help="Path to model checkpoint (best.pt)")
    p_exp2_eval.add_argument(
        "--subset", default="normal", choices=["normal", "failing", "all"],
        help="External frames to score: normal (UE003, default), failing (NORMAL001/SP001), or all",
    )
    p_exp2_eval.add_argument(
        "--kind", default="generalization",
        choices=["generalization", "training_fit"],
        help="Label the result generalization (unseen) or training_fit (frames in train set)",
    )
    p_exp2_eval.add_argument("--image-size", type=int, default=224, help="Resize target (default 224)")
    p_exp2_eval.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    p_exp2_eval.add_argument("--threshold", type=float, default=0.5,
                             help="Decision threshold on P(failing)")
    p_exp2_eval.add_argument("--output-json", default=None, help="Write full report to JSON")
    p_exp2_eval.set_defaults(func=cmd_exp2_eval)

    p_exp3_eval = sub.add_parser(
        "exp3-eval",
        help="Compare baseline vs Experiment-3 checkpoints on frozen original test "
             "and held-out target-domain normal sources (stage13)",
    )
    p_exp3_eval.add_argument("--baseline", required=True,
                             help="Path to baseline checkpoint (best.pt)")
    p_exp3_eval.add_argument("--exp3", required=True,
                             help="Path to Experiment-3 checkpoint (best.pt)")
    p_exp3_eval.add_argument(
        "--config", default="training/configs/exp3_target_normal_aug.yaml",
        help="Experiment config (used for thresholds/geometry, frozen-test data_dir overridden)",
    )
    p_exp3_eval.add_argument("--output-json", default=None,
                             help="Write full comparison report to JSON")
    p_exp3_eval.set_defaults(func=cmd_exp3_eval)

    p_exp4_eval = sub.add_parser(
        "exp4-eval",
        help="Compare baseline / Exp3 / Exp4 checkpoints on frozen original test "
             "and held-out target-domain normal sources (stage14)",
    )
    p_exp4_eval.add_argument("--baseline", required=True,
                             help="Path to baseline checkpoint (best.pt)")
    p_exp4_eval.add_argument("--exp3", required=True,
                             help="Path to Experiment-3 checkpoint (best.pt)")
    p_exp4_eval.add_argument("--exp4", required=True,
                             help="Path to Experiment-4 checkpoint (best.pt)")
    p_exp4_eval.add_argument(
        "--config", default="training/configs/exp4_target_balanced.yaml",
        help="Experiment config (used for thresholds/geometry, frozen-test data_dir overridden)",
    )
    p_exp4_eval.add_argument("--output-json", default=None,
                             help="Write full comparison report to JSON")
    p_exp4_eval.set_defaults(func=cmd_exp4_eval)

    p_poc = sub.add_parser(
        "poc",
        help="End-to-end inference POC: run the frozen classifier + temporal "
             "decision logic over a video file or webcam (no training)",
    )
    p_poc.add_argument(
        "--input", default=None,
        help="Input video file path (or glob) — mutually exclusive with --camera",
    )
    p_poc.add_argument(
        "--camera", type=int, default=None,
        help="Webcam device index (e.g. 0) — mutually exclusive with --input",
    )
    p_poc.add_argument(
        "--checkpoint",
        default="experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/best.pt",
        help="Path to a frozen best.pt checkpoint",
    )
    p_poc.add_argument("--image-size", type=int, default=224, help="Resize target")
    p_poc.add_argument(
        "--sample-fps", type=float, default=None,
        help="Sampling cadence (process every Nth frame to emulate this FPS)",
    )
    p_poc.add_argument("--max-frames", type=int, default=None,
                       help="Stop after processing this many (sampled) frames")
    p_poc.add_argument("--threshold", type=float, default=0.5,
                       help="Raw per-frame decision threshold on P(failing)")
    p_poc.add_argument("--annotated-output", default=None,
                       help="Write an annotated video (overlay of state) to this path")
    p_poc.add_argument("--alpha", type=float, default=0.15, help="EWMA smoothing factor")
    p_poc.add_argument("--baseline-window", type=int, default=60, help="EWMA baseline window")
    p_poc.add_argument("--warmup", type=int, default=30, help="FSM warm-up frames (OBSERVING)")
    p_poc.add_argument("--suspect-threshold", type=float, default=0.05,
                       help="FSM SUSPECT entry/exit threshold (signal/delta)")
    p_poc.add_argument("--fail-threshold", type=float, default=0.30,
                       help="FSM confirmed-fail threshold (signal/delta)")
    p_poc.add_argument("--recover-threshold", type=float, default=0.01,
                       help="FSM recovery threshold (FAILING -> PAUSED)")
    p_poc.add_argument("--fail-threshold-abs", type=float, default=0.65,
                       help="Absolute P(failing) hard-latch for immediate FAILING")
    p_poc.add_argument("--confirm", type=int, default=3,
                       help="Consecutive frames at/above fail threshold to confirm")
    p_poc.add_argument("--output-json", default=None, help="Write full report to JSON")
    p_poc.set_defaults(func=cmd_poc)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
