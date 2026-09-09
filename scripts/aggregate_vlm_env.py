#!/usr/bin/env python3
"""Aggregate VLM review proposals by environment for envdata staging.

Reads `<DIR>/review/review_units.json` (review units) and
`<DIR>/review/vlm_proposals.json` (VLM proposals, proposals-only) plus the
staged manifest, and writes per-environment aggregates that support the
release-data gate decision:

  <DIR>/review/env_evidence.json   per-environment aggregate statistics
  <DIR>/review/env_ranking.json    ranked failure-candidate environments
  <DIR>/review/human_candidates.json  representative units to human-verify

Proposals are treated strictly as VLM evidence: they never modify review units,
samples, or verified labels. Nothing produced here is human verification.

Usage:
  python3 scripts/aggregate_vlm_env.py --dir datasets/staging/trier
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

FAILURE_LABELS = {
    "spaghetti": "spaghetti",
    "stringing": "stringing",
    "under_extrusion": "under_extrusion",
    "extrusion_error": "extrusion_error",
    "layer_shift": "layer_shift",
    "bed_adhesion": "bed_adhesion",
    "warping": "warping",
}
MISSING_TYPE_SEARCH = {"spaghetti", "under_extrusion", "bed_adhesion", "warping"}


def load_json(path: Path, what: str) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {what}: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"malformed {what}: {path}")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(prog="aggregate_vlm_env.py")
    ap.add_argument("--dir", required=True, help="envdata staging root")
    ap.add_argument("--min-confidence", type=float, default=0.7,
                    help="failing proposal confidence floor for ranking (default 0.7)")
    args = ap.parse_args()

    root = Path(args.dir)
    units_meta = load_json(root / "review" / "review_units.json", "review units")
    store = load_json(root / "review" / "vlm_proposals.json", "VLM proposals")
    manifest = load_json(root / "manifests" / "env_manifest.json", "env manifest")
    proposals = store.get("proposals", [])

    env_by_id = {e["environment_id"]: e for e in manifest["environments"]}
    unit_by_rec: dict[str, list[dict]] = {}
    unit_by_env: dict[str, list[dict]] = {}
    for u in units_meta.get("units", []):
        unit_by_rec.setdefault(u["recording_id"], []).append(u)
        unit_by_env.setdefault(u["environment_id"], []).append(u)

    prop_by_unit = {p.get("review_unit_id"): p for p in proposals}
    rec_of_prop = {p.get("review_unit_id"): p.get("recording_id")
                   for p in proposals if p.get("recording_id")}

    env_groups: dict[str, dict] = {}
    for env_id, env in env_by_id.items():
        group = env.get("environment_group") or "unknown"
        intended = env.get("metadata", {}).get("failure_type", "none")
        is_failure_intent = bool(intended and intended != "none")
        env_units = unit_by_env.get(env_id, [])

        seq_units = [u for u in env_units if u.get("type") == "sequence"]
        totals = {"normal": 0, "failing": 0, "uncertain": 0}
        failing_props = []
        ft_hist: dict[str, int] = {}
        missing_hits: list[str] = []
        reviewed_ids = []
        seen_normal = []
        err_count = 0
        for u in seq_units:
            p = prop_by_unit.get(u["review_unit_id"])
            if p is None:
                continue
            reviewed_ids.append(u["review_unit_id"])
            if p.get("status") == "error":
                err_count += 1
                continue
            decision = p.get("decision")
            if decision == "failing":
                totals["failing"] += 1
                failing_props.append(p)
                ft = p.get("failure_type")
                if ft:
                    ft_hist[ft] = ft_hist.get(ft, 0) + 1
                    if ft in MISSING_TYPE_SEARCH:
                        missing_hits.append(ft)
            elif decision == "normal":
                totals["normal"] += 1
                if len(seen_normal) < 1:
                    seen_normal.append(p)
            elif decision == "uncertain":
                totals["uncertain"] += 1

        n_seq = len(seq_units)
        n_reviewed = len(reviewed_ids)
        failing_ratio = (totals["failing"] / n_reviewed) if n_reviewed else 0.0
        confs = [p.get("confidence", 0.0) for p in failing_props]
        top_failing = sorted(failing_props, key=lambda p: -p.get("confidence", 0.0))
        env_groups[env_id] = {
            "environment_id": env_id,
            "environment_key": env.get("environment_key"),
            "environment_group": group,
            "intended_failure_type": intended,
            "intended_failure": is_failure_intent,
            "split": env.get("split_assignment"),
            "n_units": len(env_units),
            "n_sequence_units": n_seq,
            "n_reviewed_sequence_units": n_reviewed,
            "n_error_proposals": err_count,
            "margin_of_unreviewed": n_seq - n_reviewed,
            "vlm_normal": totals["normal"],
            "vlm_failing": totals["failing"],
            "vlm_uncertain": totals["uncertain"],
            "vlm_failing_ratio": round(failing_ratio, 3),
            "failing_confidence": {
                "mean": round(statistics.mean(confs), 3) if confs else None,
                "min": round(min(confs), 3) if confs else None,
                "max": round(max(confs), 3) if confs else None,
                "count": len(confs),
            },
            "failure_type_histogram": ft_hist,
            "missing_type_hits": sorted(set(missing_hits)),
            "representative_failing": [
                {
                    "review_unit_id": p["review_unit_id"],
                    "decision": p["decision"],
                    "failure_type": p.get("failure_type"),
                    "confidence": p.get("confidence"),
                    "visual_evidence": p.get("visual_evidence"),
                    "sheet": str(root / "review" / "sheets"
                                 / f"{p['review_unit_id']}.jpg"),
                } for p in top_failing[:3]
            ],
            "representative_normal": [
                {
                    "review_unit_id": p["review_unit_id"],
                    "decision": p["decision"],
                    "confidence": p.get("confidence"),
                    "visual_evidence": p.get("visual_evidence"),
                    "sheet": str(root / "review" / "sheets"
                                 / f"{p['review_unit_id']}.jpg"),
                } for p in seen_normal[:1]
            ],
        }

    # Rank failure candidates: strong persistent visual evidence wins.
    rank = []
    for env_id, agg in env_groups.items():
        n_conf = sum(
            1 for r in agg["representative_failing"]
            if r["confidence"] >= args.min_confidence
        )
        rank.append({
            "environment_id": env_id,
            "environment_key": agg["environment_key"],
            "split": agg["split"],
            "score": round(
                agg["vlm_failing_ratio"] * 2.0
                + min(agg["vlm_failing"] / 18.0, 2.0)
                + (agg["failing_confidence"]["mean"] or 0.0)
                + (1.0 if agg["vlm_failing"] > 0 else 0.0)
                - (0.25 if agg["margin_of_unreviewed"] > 0 else 0.0),
                3,
            ),
            "vlm_failing_ratio": agg["vlm_failing_ratio"],
            "vlm_failing": agg["vlm_failing"],
            "vlm_normal": agg["vlm_normal"],
            "mean_confidence": agg["failing_confidence"]["mean"],
            "margin_of_unreviewed": agg["margin_of_unreviewed"],
            "failure_type_histogram": agg["failure_type_histogram"],
            "intended_failure_type": agg["intended_failure_type"],
            "intended_failure": agg["intended_failure"],
        })
    rank.sort(key=lambda r: (-r["score"], r["environment_key"]))

    # Intended-failure environments the VLM did NOT flag. Montage evidence can
    # hide thin defects (e.g. stringing), so these must still be surfaced for
    # human review rather than silently dropped.
    intended_not_flagged = [
        {
            "environment_id": r["environment_id"],
            "environment_key": r["environment_key"],
            "split": r["split"],
            "intended_failure_type": r["intended_failure_type"],
            "vlm_failing": r["vlm_failing"],
            "vlm_normal": r["vlm_normal"],
            "vlm_uncertain": env_groups[r["environment_id"]]["vlm_uncertain"],
            "representative_normal": env_groups[r["environment_id"]]["representative_normal"],
        }
        for r in rank
        if r["intended_failure"] and r["vlm_failing"] == 0
    ]

    candidates = []
    for r in rank:
        if r["vlm_failing"] == 0:
            continue
        agg = env_groups[r["environment_id"]]
        candidates.append({
            "environment_id": r["environment_id"],
            "environment_key": r["environment_key"],
            "split": r["split"],
            "score": r["score"],
            "vlm_failing": r["vlm_failing"],
            "vlm_normal": r["vlm_normal"],
            "failure_type_histogram": r["failure_type_histogram"],
            "units_to_verify": agg["representative_failing"]
                               + agg["representative_normal"],
            "instructions": (
                "Visually confirm each representative unit decision. If any "
                "unit is ambiguous, record 'uncertain' and do not count the "
                "environment. Use tv review-confirm to record the human "
                "decision, then tv review-merge to stamp the samples."
            ),
        })

    out_dir = root / "review"
    (out_dir / "env_evidence.json").write_text(
        json.dumps({"schema": "tinyvision-env-vlm-evidence",
                    "min_confidence_floor": args.min_confidence,
                    "n_environments": len(env_groups),
                    "n_proposals": len(proposals),
                    "environments": env_groups},
                   indent=2, ensure_ascii=False))
    (out_dir / "env_ranking.json").write_text(
        json.dumps({"schema": "tinyvision-env-vlm-ranking",
                    "ranking": rank,
                    "intended_failure_envs_not_flagged": intended_not_flagged},
                   indent=2, ensure_ascii=False))
    (out_dir / "human_candidates.json").write_text(
        json.dumps({"schema": "tinyvision-env-human-candidates",
                    "candidates": candidates,
                    "intended_failure_envs_not_flagged": [
                        {
                            "environment_id": e["environment_id"],
                            "environment_key": e["environment_key"],
                            "split": e["split"],
                            "intended_failure_type": e["intended_failure_type"],
                            "vlm_normal": e["vlm_normal"],
                            "units_to_verify": e["representative_normal"],
                            "instructions": (
                                "Montage evidence may hide thin defects (e.g. "
                                "stringing). Confirm at least one sequence unit "
                                "per intended-failure environment by direct "
                                "visual review before trusting the source "
                                "intent. tv review-confirm records the decision; "
                                "tv review-merge stamps the samples."
                            ),
                        } for e in intended_not_flagged
                    ],
                    "gate_verification_min": 5},
                   indent=2, ensure_ascii=False))

    print(f"environments: {len(env_groups)}  proposals: {len(proposals)}")
    print(f"env_evidence.json / env_ranking.json / human_candidates.json written "
          f"under {out_dir}")
    missing_type_envs = sorted({
        (a["environment_key"], m)
        for a in env_groups.values() for m in a["missing_type_hits"]
    })
    if missing_type_envs:
        print("\nmissing-type candidates found in Trier footprint:")
        for key, ft in missing_type_envs:
            print(f"  {ft:16s} {key}")
    else:
        print("\nno VLM-confirmed spaghetti/under_extrusion/bed_adhesion/warping "
              "units yet")
    print("\ntop-5 failure candidates by VLM evidence:")
    for i, r in enumerate(rank[:5], start=1):
        print(f"  {i}. {r['environment_key']:<62} split={r['split']:<16} "
              f"failing={r['vlm_failing']:>3} normal={r['vlm_normal']:>3} "
              f"intent={r['intended_failure_type']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())