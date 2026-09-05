#!/usr/bin/env python3
"""Stage 12D-extra: local VLM review CLI (Qwen3-VL-8B via LM Studio).

The VLM only generates PROPOSALS. It never modifies the human-review manifest
and never creates verified labels. Proposals go to
`datasets/external/video-frames/labels/vlm_review_proposals.json`.

Selection (at least one required unless --batch):
  --unit   <REVIEW_UNIT_ID>   e.g. NORMAL001__seq000
  --source <SOURCE_ID>        e.g. NORMAL001 (SP001/NORMAL001/UE003 only)
  --batch                     explicit opt-in to process ALL high-value units

Safety
  * Selecting UE001/STR001/SP002 is refused (non high-value sources).
  * --dry-run prints the selection + evidence plan without calling the API or
    writing anything.
  * Re-running a unit replaces only that unit's proposal (idempotent upsert);
    unrelated proposals are preserved.

Examples
  python3 scripts/run_vlm_review.py --unit  NORMAL001__seq000
  python3 scripts/run_vlm_review.py --source NORMAL001
  python3 scripts/run_vlm_review.py --unit NORMAL001__seq000 --dry-run
  python3 scripts/run_vlm_review.py --batch
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from tv.vlm_review import (  # noqa: E402
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    PROPOSALS_OUT,
    HIGH_VALUE_SOURCES,
    load_manifest,
    propose_unit,
    run_review,
    select_units,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="run_vlm_review.py",
        description="Local VLM review proposal generator (proposals-only; never ground truth).",
    )
    sel = ap.add_argument_group("selection")
    sel.add_argument("--unit", metavar="REVIEW_UNIT_ID",
                     help="single review unit (e.g. NORMAL001__seq000)")
    sel.add_argument("--source", metavar="SOURCE_ID",
                     help=f"all units of one source ({sorted(HIGH_VALUE_SOURCES)})")
    sel.add_argument("--batch", action="store_true",
                     help="EXPLICIT opt-in: process all high-value units")

    cfg = ap.add_argument_group("configuration")
    cfg.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                     help=f"LM Studio OpenAI-compatible base URL (default {DEFAULT_ENDPOINT})")
    cfg.add_argument("--model", default=DEFAULT_MODEL,
                     help=f"model id (default {DEFAULT_MODEL})")
    cfg.add_argument("--output", default=str(PROPOSALS_OUT),
                     help=f"proposal output file (default {PROPOSALS_OUT})")
    cfg.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                     help=f"bounded retries per unit request (default {DEFAULT_RETRIES})")
    cfg.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                     help=f"per-request timeout seconds (default {DEFAULT_TIMEOUT})")
    cfg.add_argument("--no-frame-context", action="store_false", dest="include_context",
                     help="do not attach neighbouring frames for FRAME units")
    cfg.add_argument("--dry-run", action="store_true",
                     help="print selection + evidence plan; no API call, no writes")

    args = ap.parse_args()

    if not (args.unit or args.source or args.batch):
        ap.error("provide --unit, --source, or --batch (no automatic full run)")

    manifest = load_manifest()
    try:
        units = select_units(manifest, unit_id=args.unit, source=args.source,
                             batch=args.batch)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"dry-run: {len(units)} unit(s) selected")
        n_seq = sum(1 for u in units if u.unit_type == "sequence")
        n_frm = sum(1 for u in units if u.unit_type == "frame")
        print(f"  sequence units: {n_seq}  frame units: {n_frm}")
        for u in units:
            proposal = propose_unit(u, endpoint=args.endpoint, model=args.model,
                                    include_context=args.include_context,
                                    timeout=args.timeout, retries=args.retries,
                                    dry_run=True)
            print(f"  {proposal['review_unit_id']:<28} kind={proposal['kind']:<6} "
                  f"primary={proposal['primary']}")
            for c in proposal["context"]:
                print(f"  {'':<28} context={c}")
        return 0

    print(f"processing {len(units)} unit(s) via {args.model} @ {args.endpoint}")
    try:
        store = run_review(units, endpoint=args.endpoint, model=args.model,
                           include_context=args.include_context,
                           timeout=args.timeout, retries=args.retries,
                           output=Path(args.output))
    except (ValueError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    successes = [p for p in store["proposals"]
                 if p["review_unit_id"] in {u.review_unit_id for u in units}
                 and p.get("status") == "success"]
    errors = [p for p in store["proposals"]
              if p["review_unit_id"] in {u.review_unit_id for u in units}
              and p.get("status") == "error"]
    print(f"written: {args.output}")
    print(f"  success: {len(successes)}  error: {len(errors)}")
    for p in errors:
        print(f"  ERROR {p['review_unit_id']}: [{p.get('error_type')}] {p.get('error')}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())