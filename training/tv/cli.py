"""TinyVision training pipeline CLI.

Usage:
    tv train       --config CONFIG_PATH
    tv evaluate    --config CONFIG_PATH [--checkpoint PATH]
    tv export      --config CONFIG_PATH [--checkpoint PATH]
    tv split       --data-dir DIR --output-dir DIR [--ratios 0.7,0.15,0.15] [--seed 42]
    tv holdout     --data-dir DIR [--output-dir DIR]
    tv capture     --output-dir DIR --class normal|failing [--session-id ID] [--duration SECS] [--fps N]
    tv list-models [--task classification|detection]
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

    p_list = sub.add_parser("list-models", help="List available model architectures")
    p_list.add_argument("--task", default=None, help="Filter by task (classification/detection)")
    p_list.set_defaults(func=cmd_list_models)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
