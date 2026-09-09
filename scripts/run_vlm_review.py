#!/usr/bin/env python3
"""Stage 12D-extra: local VLM review CLI (Qwen3-VL-8B via LM Studio).

The VLM only generates PROPOSALS. It never modifies the human-review manifest
and never creates verified labels. Proposals go to
`datasets/external/video-frames/labels/vlm_review_proposals.json`.

The same CLI driver also runs environment-staged review sets (e.g. Trier) via
`--env-staging DIR`. In that mode units are read from
`<DIR>/review/review_units.json`, evidence is the rendered contact sheets under
`<DIR>/review/sheets/` (plus the extracted frames for FRAME units), the extended
failure vocabulary is enabled, and proposals are upserted to
`<DIR>/review/vlm_proposals.json` with environment/recording identity fields.
The VLM remains a proposals-only cursor: nothing is written to the review units
file and no verified labels are produced.

Stage 12D selection (at least one required unless --env-staging):
  --unit   <REVIEW_UNIT_ID>   e.g. NORMAL001__seq000
  --source <SOURCE_ID>        e.g. NORMAL001 (SP001/NORMAL001/UE003 only)
  --batch                     explicit opt-in to process ALL high-value units

Envdata selection:
  --env-staging <DIR>         envdata staging root (review/review_units.json)
  --priority <P1,P2,...>      processing order classes:
                                P1 train failure  P2 train normal
                                P3 holdout failure  P4 holdout normal
                              (default P1,P2,P3,P4)
  --limit <N>                 process at most N units in this invocation
  --skip-existing             skip units that already have a successful proposal
                              (default on; --no-skip-existing forces re-run)

Safety
  * Selecting UE001/STR001/SP002 is refused (non high-value sources).
  * --dry-run prints the selection + evidence plan without calling the API or
    writing anything.
  * Re-running a unit replaces only that unit's proposal (idempotent upsert);
    unrelated proposals are preserved.

Examples
  python3 scripts/run_vlm_review.py --unit  NORMAL001__seq000
  python3 scripts/run_vlm_review.py --source NORMAL001
  python3 scripts/run_vlm_review.py --batch
  python3 scripts/run_vlm_review.py --env-staging datasets/staging/trier \
      --priority P1,P2 --limit 30 --dry-run
  python3 scripts/run_vlm_review.py --env-staging datasets/staging/trier \
      --limit 200
"""
import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from tv.envdata.schema import load_manifest, load_samples  # noqa: E402
from tv.vlm_review import (  # noqa: E402
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    EXTENDED_FAILURE_TYPES,
    EXTENDED_REVIEW_PROMPT,
    PROPOSALS_OUT,
    HIGH_VALUE_SOURCES,
    Evidence,
    ReviewUnit,
    load_manifest as load_stage12_manifest,
    load_proposals,
    propose_unit,
    run_review,
    select_units,
    upsert_proposals,
    write_proposals,
)

PRIORITY_RANK = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


FAILURE_GROUPS = {
    "extrusion_error_print": "extrusion_error",
    "layershift_error_print": "layer_shift",
    "stringing_error": "stringing",
}
NORMAL_GROUPS = {"no_error_print", "no_stringing_error"}


def _is_failure_group(group: str) -> bool:
    if group in NORMAL_GROUPS:
        return False
    if group in FAILURE_GROUPS:
        return True
    raise ValueError(f"unknown Trier environment group {group!r}")


def _envdata_order_class(split: str, group: str) -> int:
    """P1 train failure, P2 train normal, P3 holdout failure, P4 holdout normal."""
    is_failure = _is_failure_group(group)
    if split == "train":
        return 0 if is_failure else 1
    return 2 if is_failure else 3


def load_envdata_units(staging_root: Path, frames_root: Path | None = None) -> dict:
    """Return {review_unit_id: (unit_dict, review_unit, evidence, extra)}."""
    units_meta = json.loads(
        (staging_root / "review" / "review_units.json").read_text())
    if not isinstance(units_meta, dict) or "units" not in units_meta:
        raise ValueError(f"invalid envdata review file at {staging_root}/review")

    manifest = load_manifest(staging_root)
    samples = load_samples(staging_root)

    if frames_root is None:
        index_path = staging_root / "metadata" / "index.json"
        if index_path.exists():
            idx = json.loads(index_path.read_text())
            candidates = [idx.get("root"), idx.get("frame_root")]
            for c in candidates:
                if isinstance(c, str) and c:
                    frames_root = Path(c)
                    break
    if frames_root is None or not frames_root.is_dir():
        raise ValueError(
            "no frame root: pass --frame-root or add root/frame_root to "
            "metadata/index.json")

    env_to_group = {e.environment_id: e.environment_group for e in manifest.environments}
    env_to_split = {e.environment_id: e.split_assignment for e in manifest.environments}

    frame_by_rec = {}
    for s in samples:
        frame_by_rec.setdefault(s.recording_id, {})[s.frame_index] = s.metadata.get("file")

    sheets_dir = staging_root / "review" / "sheets"
    out = {}
    for u in units_meta["units"]:
        uid = str(u["review_unit_id"])
        unit_type = "sequence" if str(u["type"]) == "sequence" else "frame"
        unit = ReviewUnit(
            review_unit_id=uid,
            source_id=str(u["recording_id"]),
            unit_type=unit_type,
            frames=[int(x) for x in u["frames"]],
            times=[float(x) for x in u["times"]],
        )
        if unit_type == "sequence":
            sheet = sheets_dir / f"{uid}.jpg"
            if not sheet.exists():
                continue
            evidence = Evidence(unit=unit, kind="sheet", primary=sheet, context=[])
        else:
            rec_id = str(u["recording_id"])
            files = frame_by_rec.get(rec_id, {})
            target = int(u["frames"][0])
            rel = files.get(target)
            if rel is None:
                continue
            primary = frames_root / str(rel)
            if not primary.exists():
                continue
            context_parts = []
            for neighbor in (target - 1, target + 1):
                nrel = files.get(neighbor)
                if nrel is None:
                    continue
                np_ = frames_root / str(nrel)
                if np_.exists():
                    context_parts.append(np_)
            evidence = Evidence(unit=unit, kind="frame", primary=primary,
                                context=context_parts)

        env_id = str(u["environment_id"])
        extra = {
            "environment_id": env_id,
            "recording_id": str(u["recording_id"]),
            "provenance": "vlm:qwen/qwen3-vl-8b",
        }
        order = _envdata_order_class(env_to_split[env_id], env_to_group[env_id])
        out[uid] = {"priority": order, "unit": unit, "evidence": evidence,
                    "extra": extra}
    return out


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

    env = ap.add_argument_group("envdata selection")
    env.add_argument("--env-staging", metavar="DIR",
                     help="envdata staging root whose review/review_units.json drives this run")
    env.add_argument("--frame-root", metavar="DIR",
                     help="extracted frames root (default: root in metadata/index.json)")
    env.add_argument("--priority", metavar="LIST", default="P1,P2,P3,P4",
                     help="classes to process, comma separated (default P1,P2,P3,P4)")
    env.add_argument("--limit", type=int, default=0,
                     help="max envdata units processed in this invocation (0 = unlimited)")
    env.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                     default=True, help="skip envdata units with a successful proposal")
    env.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                     help="re-process units even if a successful proposal exists")

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
    cfg.add_argument("--evidence-max-side", type=int, default=0,
                     help="cap evidence longest side (px) before encoding "
                          "(0 = original resolution)")
    cfg.add_argument("--dry-run", action="store_true",
                     help="print selection + evidence plan; no API call, no writes")

    args = ap.parse_args()

    if args.env_staging:
        staging = Path(args.env_staging)
        if not (staging / "review" / "review_units.json").exists():
            ap.error(f"no envdata review file at {staging}/review/review_units.json")
        if args.unit or args.source or args.batch:
            ap.error("--env-staging cannot be combined with --unit/--source/--batch")
        return _run_envdata(args, staging)

    if not (args.unit or args.source or args.batch):
        ap.error("provide --unit, --source, --batch, or --env-staging")

    manifest = load_stage12_manifest()
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
                                    dry_run=True, max_side=args.evidence_max_side)
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
                           max_side=args.evidence_max_side,
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


def _run_envdata(args: argparse.Namespace, staging: Path) -> int:
    frames_root = Path(args.frame_root) if args.frame_root else None
    units = load_envdata_units(staging, frames_root=frames_root)
    wanted = {rank for rank, _ in enumerate(PRIORITY_RANK)}
    classes = {PRIORITY_RANK.get(p, 99) for p in
               [x.strip() for x in args.priority.split(",") if x.strip()]}
    if 99 in classes:
        print(f"ERROR: unknown priority class in {args.priority!r}", file=sys.stderr)
        return 2

    ordered = sorted(units.items(),
                     key=lambda kv: (kv[1]["priority"],
                                     0 if kv[1]["unit"].unit_type == "sequence" else 1,
                                     kv[0]))
    selected = [kv for kv in ordered if kv[1]["priority"] in classes]
    if args.limit > 0:
        selected = selected[:args.limit]

    out_path = staging / "review" / "vlm_proposals.json"
    skipped = 0
    store = {}
    if args.skip_existing:
        store = load_proposals(out_path)
        have_success = {p["review_unit_id"] for p in store.get("proposals", [])
                        if p.get("status") == "success"}
        pending = [kv for kv in selected if kv[0] not in have_success]
        skipped = len(selected) - len(pending)
        selected = pending

    if args.dry_run:
        print(f"dry-run envdata: {len(selected)} unit(s) selected "
              f"(skipped {skipped})")
        n_seq = sum(1 for _, v in selected if v["unit"].unit_type == "sequence")
        n_frm = sum(1 for _, v in selected if v["unit"].unit_type == "frame")
        print(f"  sequence units: {n_seq}  frame units: {n_frm}")
        for uid, v in selected:
            print(f"  {uid:<52} P{v['priority'] + 1} "
                  f"kind={v['evidence'].kind:<6} primary={v['evidence'].primary}")
        return 0

    print(f"processing {len(selected)} envdata unit(s) via {args.model} @ "
          f"{args.endpoint} (skipped {skipped})")
    updated = 0
    write_batch = 20
    try:
        for idx, (uid, v) in enumerate(selected, start=1):
            proposal = propose_unit(
                v["unit"], endpoint=args.endpoint, model=args.model,
                include_context=args.include_context, timeout=args.timeout,
                retries=args.retries, prompt=EXTENDED_REVIEW_PROMPT,
                failure_types=EXTENDED_FAILURE_TYPES, evidence=v["evidence"],
                max_side=args.evidence_max_side, extra=v["extra"])
            store = upsert_proposals(store, [proposal])
            updated += 1
            status = proposal.get("status")
            if status == "success":
                print(f"  [{idx}/{len(selected)}] {uid} -> "
                      f"{proposal['decision']} {proposal.get('failure_type')} "
                      f"{proposal['confidence']:.2f}")
            else:
                print(f"  [{idx}/{len(selected)}] {uid} -> ERROR "
                      f"[{proposal.get('error_type')}] {proposal.get('error')}",
                      file=sys.stderr)
            if idx % write_batch == 0 or idx == len(selected):
                write_proposals(store, out_path)
    except KeyboardInterrupt:
        print("interrupted; writing partial store", file=sys.stderr)
    store["generated_at"] = _now_iso()
    store["model"] = args.model
    store["endpoint"] = args.endpoint
    write_proposals(store, out_path)
    n_ok = sum(1 for p in store.get("proposals", []) if p.get("status") == "success")
    n_err = sum(1 for p in store.get("proposals", []) if p.get("status") == "error")
    print(f"written: {out_path}")
    print(f"  success: {n_ok}  error: {n_err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())