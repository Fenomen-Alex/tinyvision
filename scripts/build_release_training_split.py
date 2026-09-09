"""Build the release-candidate (rc1) training split directory.

Composition (provenance-typed; no VLM-only labels):

  train.csv =
    frozen printing-errors train rows (source: printing-errors / Kaggle),
    Trier train-environment frames with SOURCE annotations
      (source_annotation = Trier FFF-QC v2 folder-level labels; only
       environments whose manifest split_assignment is ``train`` — base_data +
       additional_defect_data roots; test_data environments are excluded),
    human-verified failing frames from video-frames NORMAL001/SP001
      (provenance: human_reviewed via ``verified_labels.json``).
  val.csv / test.csv =
    byte-identical copies of the frozen printing-errors splits.

Excluded by policy:
  * Trier test_data environments (external_holdout) — never in training.
  * Trier VLM proposals (proposals are not labels).
  * Stage-13 VLM-screened normal frames and Stage-14 VLM-screened failing
    frames (VLM-only labels; excluded from the release training set).
  * UE003 / external normal eval frames (evaluation-only).

Environment-level isolation is enforced here: rows are produced only from
train-split environments, the manifest is hashed into the ledger, and every
frame file is existence-checked at build time.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

STAGING = REPO / "datasets/staging/trier"
MANIFEST = STAGING / "manifests/env_manifest.json"
SAMPLES_DIR = STAGING / "samples"
INDEX = STAGING / "metadata/index.json"
FROZEN = REPO / "datasets/processed/printing-errors"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_many(paths: list[Path]) -> dict[str, str]:
    out = {}
    for p in paths:
        out[str(p)] = sha256_of(p)
    return out


def load_samples(env_ids: set[str], root: Path) -> list[dict]:
    rows = []
    ids = set()
    for f in sorted((SAMPLES_DIR).glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r["environment_id"] not in env_ids:
                continue
            p = Path(r["metadata"]["file"])
            if not p.is_absolute():
                p = root / p
            if not p.exists():
                raise FileNotFoundError(f"missing frame: {p}")
            ids.add(r["environment_id"])
            rows.append({
                "path": str(p),
                "label": r["source_label"],  # failing | normal (source annotation)
                "failure_type": r["source_failure_type"],
                "environment_id": r["environment_id"],
                "recording_id": r["recording_id"],
            })
    missing = env_ids - ids
    if missing:
        raise ValueError(f"no samples for train environments: {sorted(missing)}")
    return rows


def read_csv_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label"])
        writer.writeheader()
        for r in rows:
            writer.writerow({"path": r["path"], "label": r["label"]})


def verified_failing_rows() -> list[dict]:
    """Human-verified failing augmentation frames (NORMAL001/SP001)."""
    from tv.exp2_external import verified_training_rows

    rows = verified_training_rows()
    out = []
    for r in rows:
        p = Path(r["image_path"])
        if not p.exists():
            raise FileNotFoundError(f"missing verified frame: {p}")
        out.append({"path": r["image_path"], "label": r["binary_label"],
                    "source": "human_verified(video-frames NORMAL001/SP001)",
                    "recording_id": r["source_id"]})
    return out


def build(out_dir: Path) -> dict:
    manifest = json.loads(MANIFEST.read_text())["environments"]
    env_by_id = {e["environment_id"]: e for e in manifest}

    train_envs = [e for e in manifest if e["split_assignment"] == "train"]
    train_ids = {e["environment_id"] for e in train_envs}
    holdout_ids = {e["environment_id"] for e in manifest
                   if e["split_assignment"] == "external_holdout"}

    root = Path(json.loads(INDEX.read_text())["root"])
    if not root.is_absolute():
        root = (REPO / root).resolve()

    trier_rows = load_samples(train_ids, root)

    printing_rows = read_csv_rows(FROZEN / "splits/train.csv")
    val_rows = read_csv_rows(FROZEN / "splits/val.csv")
    test_rows = read_csv_rows(FROZEN / "splits/test.csv")
    human_rows = verified_failing_rows()

    # provenance tags on the copies we emit
    for r in printing_rows:
        r["source"] = "printing-errors(frozen)"
        r["recording_id"] = Path(r["path"]).parts[-2]
    for r in trier_rows:
        r["source"] = "trier(source_annotation)"
    for r in human_rows:
        pass  # already tagged

    train_rows = printing_rows + trier_rows + human_rows

    # --- isolation asserts -------------------------------------------------
    bad = [r for r in trier_rows if r["environment_id"] in holdout_ids]
    if bad:
        raise ValueError(f"holdout environment leaked into train: {bad[0]}")
    if len(trier_rows) != len({r["path"] for r in trier_rows}):
        raise ValueError("duplicate Trier frame paths in train rows")
    train_paths = {r["path"] for r in train_rows}
    if len(train_paths) != len(train_rows):
        raise ValueError("duplicate image path across train rows")
    holdout_path_sample = None
    for eid, e in env_by_id.items():
        if e["split_assignment"] == "external_holdout":
            pass

    from collections import Counter
    comp = Counter((r["source"], r["label"]) for r in train_rows)

    # provenance section ----------------------------------------------------
    trier_envs = {eid: env_by_id[eid]
                  for eid in train_ids}
    src_samples = {eid: sum(1 for r in trier_rows
                            if r["environment_id"] == eid) for eid in train_ids}
    ft_by_key = {}
    for eid, e in trier_envs.items():
        ft_by_key[e["environment_key"]] = (e["metadata"].get("failure_type", "none"),
                                           src_samples[eid])

    derived = {
        "schema": "release-rc1-training-directory",
        "schema_version": 1,
        "mode": "release-candidate-1",
        "data_dir": str(FROZEN),
        "source_splits": str(FROZEN / "splits"),
        "stats": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "class_to_idx": {"failing": 0, "normal": 1},
        "label_authority": {
            "printing-errors": "source annotations (Kaggle printing-errors; frozen splits)",
            "trier": "source_annotation = Trier FFF-QC v2 folder-level labels (CC BY 4.0, Zenodo 18628777); NOT human-verified",
            "human_verified": "human_reviewed provenance in datasets/external/video-frames/labels/verified_labels.json (NORMAL001/SP001)",
            "excluded": [
                "Trier VLM proposals (never labels)",
                "Stage-13 VLM-screened normal frames",
                "Stage-14 VLM-screened failing frames",
                "Trier test_data environments (external_holdout)",
                "UE003 external normal eval frames",
            ],
        },
        "composition": {
            "printing-errors(frozen)": dict(Counter(r["label"] for r in printing_rows)),
            "trier(source_annotation)": dict(Counter(r["label"] for r in trier_rows)),
            "human_verified": dict(Counter(r["label"] for r in human_rows)),
        },
        "trier_environments": {
            e["environment_key"]: {
                "split": e["split_assignment"],
                "failure_type": e["metadata"].get("failure_type", "none"),
                "frames": src_samples[eid],
            }
            for eid, e in trier_envs.items()
        },
        "holdout_environments_excluded": {
            e["environment_key"]: {"failure_type": e["metadata"].get("failure_type", "none")}
            for e in manifest if e["split_assignment"] == "external_holdout"
        },
        "integrity": {
            "env_manifest_sha256": sha256_of(MANIFEST),
            "sample_files_sha256": sha256_many(sorted(SAMPLES_DIR.glob("*.jsonl"))),
            "frozen_train_sha256": sha256_of(FROZEN / "splits/train.csv"),
            "frozen_val_sha256": sha256_of(FROZEN / "splits/val.csv"),
            "frozen_test_sha256": sha256_of(FROZEN / "splits/test.csv"),
            "n_trier_train_envs": len(trier_envs),
            "n_holdout_envs_excluded": len(holdout_ids),
            "assert_env_level_isolation": True,
        },
        "caveats": [
            "Trier labels are source annotations (dataset-author folder-level), not human-verified.",
            "No VLM proposals, no VLM-screened labels in this training set.",
            "val/test are byte-identical to the frozen printing-errors splits.",
        ],
    }

    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    write_csv(splits_dir / "train.csv", train_rows)
    # byte-identical frozen copies
    (splits_dir / "val.csv").write_bytes((FROZEN / "splits/val.csv").read_bytes())
    (splits_dir / "test.csv").write_bytes((FROZEN / "splits/test.csv").read_bytes())

    splits_json = {
        "data_dir": str(FROZEN),
        "mode": "release-rc1",
        "stats": derived["stats"],
        "splits": {
            "train": dict(Counter(r["label"] for r in train_rows)),
            "val": dict(Counter(r["label"] for r in val_rows)),
            "test": dict(Counter(r["label"] for r in test_rows)),
        },
    }
    (splits_dir / "splits.json").write_text(json.dumps(splits_json, indent=2))
    (out_dir / "derived_manifest.json").write_text(json.dumps(derived, indent=2))

    for (src, label), n in sorted(comp.items()):
        print(f"{src:<45} {label:<8} {n}")
    print(f"TOTAL train rows: {len(train_rows)}  (val {len(val_rows)}, test {len(test_rows)})")
    print(f"wrote {out_dir}/splits/ and derived_manifest.json")
    return derived


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "datasets/processed/release-rc1"))
    args = ap.parse_args()
    build(Path(args.out))