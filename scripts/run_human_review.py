#!/usr/bin/env python3
"""Stage 12D: human-review workflow CLI for the external pilot footage.

Subcommands:
  build     (re)generate the review manifest + sequence contact sheets.
  summary   print per-source reviewable-frame counts (sequence vs frame).
  review    interactive human labeling (opens each unit's contact sheet /
            frame in the default viewer; saves the manifest after every unit).
  merge     validate a completed manifest into verified_labels.json.

The `review` command shows ONLY visual evidence and neutral manifest metadata.
It is blind to the Stage 12D-extra VLM proposals: it never reads
`vlm_review_proposals.json`, displays none of the VLM predictions, and never
writes `verified_labels.json` (that is `merge`'s job).

Examples:
  python scripts/run_human_review.py build
  python scripts/run_human_review.py build --force   # wipe + regenerate decisions
  python scripts/run_human_review.py summary
  python scripts/run_human_review.py review                       # undecided sequences only
  python scripts/run_human_review.py review --source NORMAL001
  python scripts/run_human_review.py review --unit NORMAL001__seq000
  python scripts/run_human_review.py merge           # requires completed manifest
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from tv.review import (  # noqa: E402
    MANIFEST,
    VERIFIED,
    build_manifest,
    completed_count,
    interactive_review,
    merge_verified,
    save_manifest_atomic,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="(re)generate manifest + contact sheets")
    b.add_argument("--force", action="store_true",
                   help="wipe existing decisions and regenerate all units")
    sub.add_parser("summary", help="print per-source reviewable-frame counts")
    r = sub.add_parser("review",
                       help="interactive human labeling (opens contact sheets / frames)")
    r.add_argument("--unit", action="append", metavar="REVIEW_UNIT_ID",
                   help="review exactly this unit (repeatable; command-line order preserved)")
    r.add_argument("--source", action="append", metavar="SOURCE_ID",
                   help="review all units of this source (repeatable; manifest order)")
    r.add_argument("--sequence-only", action="store_true",
                   help="exclude FRAME units from the selection")
    m = sub.add_parser("merge", help="validate a completed manifest into verified labels")
    m.add_argument("--manifest", default=str(MANIFEST))
    m.add_argument("--out", default=str(VERIFIED))
    m.add_argument("--allow-partial", action="store_true",
                   help="allow merging units that are already decided, leaving rest null")
    args = ap.parse_args()

    if args.cmd == "build":
        manifest = build_manifest(force=args.force)
        done = completed_count(manifest)
        n_seq = sum(1 for u in manifest["units"] if u["type"] == "sequence")
        n_frm = sum(1 for u in manifest["units"] if u["type"] == "frame")
        print(f"manifest: {MANIFEST}")
        print(f"units: {len(manifest['units'])} (sequence={n_seq}, frame={n_frm})")
        print(f"decisions already filled: {done}")
        print(f"contact sheets: {MANIFEST.parent / 'sheets'}")
        return 0

    if args.cmd == "summary":
        manifest = json.loads(MANIFEST.read_text())
        print(f"{'source':<10} {'dur_s':<7} {'units':<6} {'seq':<5} {'frm':<5} "
              f"{'seq_secs':<9} {'stable_s':<9} {'unstable_s':<10} done")
        tot = {"units": 0, "seq": 0, "frm": 0, "seq_secs": 0, "done": 0}
        for sid, info in sorted(manifest["per_source"].items()):
            s = info
            print(f"{sid:<10} {s['duration_s']:<7.1f} {s['n_sequence_units'] + s['n_frame_units']:<6} "
                  f"{s['n_sequence_units']:<5} {s['n_frame_units']:<5} "
                  f"{s['sequence_reviewable_seconds']:<9} {s['stable_seconds']:<9} "
                  f"{s['unstable_seconds']:<10}")
            tot["units"] += s["n_sequence_units"] + s["n_frame_units"]
            tot["seq"] += s["n_sequence_units"]
            tot["frm"] += s["n_frame_units"]
            tot["seq_secs"] += s["sequence_reviewable_seconds"]
        tot["done"] = completed_count(manifest)
        print(f"{'TOTAL':<10} {'':<7} {tot['units']:<6} {tot['seq']:<5} {tot['frm']:<5} "
              f"{tot['seq_secs']:<9}")
        print(f"decisions filled: {tot['done']}/{tot['units']}")
        return 0

    if args.cmd == "review":
        manifest = json.loads(MANIFEST.read_text())
        try:
            interactive_review(
                manifest,
                unit_ids=args.unit,
                sources=args.source,
                sequence_only=args.sequence_only,
                manifest_path=MANIFEST,
                save_fn=lambda m: save_manifest_atomic(m, MANIFEST),
            )
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        return 0

    if args.cmd == "merge":
        try:
            result = merge_verified(Path(args.manifest), Path(args.out),
                                    require_all=not args.allow_partial)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        print(f"verified {result['n_units_verified']} units -> {args.out}")
        print(f"  sequence={result['n_sequence_units']} frame={result['n_frame_units']}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
