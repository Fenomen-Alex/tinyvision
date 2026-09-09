"""Experiment 2: external video-frames augmentation (verified Stage 12 labels).

The frozen TinyVision baseline is binary (normal vs failing). This module
builds the Experiment-2 (``2A``) derived split and evaluation for augmenting
the printing-errors training set with human-reviewed external pilot footage.

Label authority (Strict, Experiment 2):
  * TRAINING augmentation  -> the 6 units MERGED into
    ``verified_labels.json`` (NORMAL001 stringing, SP001 spaghetti) -> ``failing``.
  * External NORMAL eval   -> the 8 UE003 units with human ``normal`` decisions
    recorded in ``review_manifest.json``. These are human-review decisions that
    were NEVER merged into ``verified_labels.json`` (Stage 12F left the merge
    undecided). They are used here ONLY as an external evaluation set (truly
    unseen by both baseline and 2A, which never train on UE003), never as
    training labels. Their labels are read verbatim; nothing is silent-merged.
  * EXCLUDED everywhere    -> ``uncertain`` decisions, undecided units, VLM
    proposals, provisional Stage-12B labels, and all other sources.

Source-group rule: every source belongs ENTIRELY to one role.
  * NORMAL001, SP001  -> training augmentation (failing).
  * UE003             -> external evaluation (normal) ONLY.
A source is therefore never split across training and evaluation.

Evaluation mirrors the existing deep-eval and SKZ conventions: failing-as-positive
is the primary view (decision threshold on P(class=failing)), both binary
conventions are reported, AUROC is reported for the failing class. External
normal frames are a single-source descriptive signal (24 frames), NOT a
benchmark; all external numbers are labeled limited/descriptive. Evaluating 2A
on NORMAL001/SP001 is a TRAINING-FIT (memorization) check because those frames
are in its training set; it is reported separately and clearly labeled.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "datasets" / "external" / "video-frames"
MANIFEST = EXT / "labels" / "review_manifest.json"
VERIFIED = EXT / "labels" / "verified_labels.json"
DECISION_VOCAB = {"normal", "failing", "uncertain"}

SCHEMA = "exp2-external"
SCHEMA_VERSION = 1

# Binary label -> index, identical to the frozen TinyVision splits
# (class_to_idx sorted alphabetically: failing=0, normal=1).
BINARY_LABELS = {"failing": 0, "normal": 1}
FAIL_IDX = 0

# Human decision / failure type -> binary label (faithful to the original
# binary task, never a redefinition).
DECISION_TO_BINARY = {
    "normal": "normal",
    "failing": "failing",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_unit_frames(unit: dict) -> list[int]:
    """Return ``unit``'s frame indices with a hard existence check."""
    frames = [int(f) for f in unit["frames"]]
    missing = [
        (f, frame_path(unit["source_id"], f).resolve())
        for f in frames if not frame_path(unit["source_id"], f).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{unit['review_unit_id']}: missing extracted frame(s): "
            + ", ".join(f"{idx} -> {path}" for idx, path in missing[:5])
        )
    return frames


def frame_path(source_id: str, seq_index: int) -> Path:
    """Path to the extracted 1-FPS sequence frame for this source/index."""
    return EXT / "extracted" / source_id / "sequence" / f"frame_{seq_index:04d}.jpg"


def expanded_rows(units: list[dict]) -> list[dict]:
    """Expand verified/human units to per-frame rows.

    Each row carries the absolute image path, the binary label derived from the
    human decision, and full provenance (source, unit, frame index, failure
    type). Nothing is silently relabeled.
    """
    rows = []
    for unit in units:
        decision = unit["decision"]
        if decision not in DECISION_VOCAB:
            raise ValueError(f"{unit['review_unit_id']}: unknown decision {decision!r}")
        binary = DECISION_TO_BINARY.get(decision)
        if binary is None:  # uncertain -> no usable label
            continue
        frames = verify_unit_frames(unit)
        for idx in frames:
            rows.append({
                "image_path": str(frame_path(unit["source_id"], idx).resolve()),
                "source_id": unit["source_id"],
                "review_unit_id": unit["review_unit_id"],
                "frame_index": idx,
                "decision": decision,
                "failure_type": unit.get("failure_type"),
                "binary_label": binary,
                "binary_idx": BINARY_LABELS[binary],
            })
    rows.sort(key=lambda r: (r["source_id"], r["review_unit_id"], r["frame_index"]))
    return rows


def verified_training_rows() -> list[dict]:
    """Rows for 2A TRAINING augmentation = the 6 MERGED verified failing units."""
    data = load_json(VERIFIED)
    if data.get("provenance") != "human_reviewed":
        raise ValueError(
            f"verified_labels.json provenance unexpected: {data.get('provenance')!r}"
        )
    units = [u for u in data["units"] if u["decision"] == "failing"]
    failing = [u for u in units if u["decision"] == "failing"]
    if len(failing) != 6:
        raise ValueError(f"expected 6 verified failing units, got {len(failing)}")
    return expanded_rows(failing)


def external_normal_rows() -> list[dict]:
    """Rows for external NORMAL evaluation = the 8 UE003 human ``normal`` units.

    These decisions live in ``review_manifest.json`` only (never merged into
    ``verified_labels.json``). They are evaluation-only here.
    """
    data = load_json(MANIFEST)
    normal = [
        u for u in data["units"]
        if u.get("decision") == "normal" and u["source_id"] == "UE003"
    ]
    if len(normal) != 8:
        raise ValueError(f"expected 8 UE003 normal units, got {len(normal)}")
    rows = expanded_rows(normal)
    if any(r["source_id"] != "UE003" for r in rows):
        raise ValueError("external normal rows must be UE003-only")
    return rows


def coverage(rows: list[dict]) -> dict:
    return {
        "frames": len(rows),
        "sources": sorted({r["source_id"] for r in rows}),
        "units": sorted({r["review_unit_id"] for r in rows}),
        "per_binary": dict(sorted(Counter(r["binary_label"] for r in rows).items())),
        "per_source": dict(
            sorted(
                (s, {"frames": n, "units": len({r['review_unit_id'] for r in grp})})
                for s, grp in ((k, [r for r in rows if r["source_id"] == k])
                              for k in sorted({r["source_id"] for r in rows}))
                for n in (len([r for r in grp]),)
            )
        ),
    }


def prepare_exp2_splits(
    printing_errors_dir: str | Path,
    out_dir: str | Path,
) -> dict:
    """Build the Experiment-2A derived split directory.

    train.csv = printing-errors train rows (unchanged) + the verified external
                failing rows (absolute paths; NORMAL001 + SP001).
    val.csv   = printing-errors val rows (unchanged byte-identical copy).
    test.csv  = printing-errors test rows (unchanged byte-identical copy).

    The existing printing-errors splits, review_manifest.json and
    verified_labels.json are never modified.
    """
    printing_errors_dir = Path(printing_errors_dir).resolve()
    out_dir = Path(out_dir)
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    def read_csv(path: Path) -> list[dict]:
        with open(path) as f:
            return list(csv.DictReader(f))

    src = printing_errors_dir / "splits"
    train_rows = read_csv(src / "train.csv")
    val_rows = read_csv(src / "val.csv")
    test_rows = read_csv(src / "test.csv")

    aug_rows = verified_training_rows()
    for r in aug_rows:
        p = Path(r["image_path"])
        if not p.exists():
            raise FileNotFoundError(f"external image missing: {p}")
        train_rows.append({"path": r["image_path"], "label": "failing"})

    def write_csv(path: Path, rows: list[dict]) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    write_csv(splits_dir / "train.csv", train_rows)
    write_csv(splits_dir / "val.csv", val_rows)
    write_csv(splits_dir / "test.csv", test_rows)

    normal_rows = external_normal_rows()
    stats = {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)}

    derived = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "mode": "exp2-external-video-aug",
        "data_dir": str(printing_errors_dir),
        "source_splits": str(src),
        "baseline_experiment_id": "shufflenet_v2_x1_0_printing-errors_cdc9e2cad4",
        "stats": stats,
        "class_to_idx": BINARY_LABELS,
        "training_augmentation": {
            "label": "failing",
            "provenance": "human_reviewed (merged in verified_labels.json)",
            **coverage(aug_rows),
        },
        "external_normal_eval": {
            "label": "normal",
            "provenance": (
                "human-review decisions in review_manifest.json (UE003); "
                "NOT merged into verified_labels.json; evaluation-only, "
                "never used for training"
            ),
            **coverage(normal_rows),
        },
        "role_by_source": {
            "NORMAL001": "training_augmentation (failing)",
            "SP001": "training_augmentation (failing)",
            "UE003": "external_normal_eval (normal)",
        },
        "excluded": {
            "uncertain_units": len([
                u for u in load_json(MANIFEST)["units"]
                if u.get("decision") == "uncertain"
            ]),
            "claim": "uncertain decisions, undecided units, VLM proposals, "
                     "provisional Stage-12B labels, and non-UE003 sources are "
                     "never used as labels here",
        },
        "integrity": {
            "review_manifest_sha256": sha256_of(MANIFEST),
            "verified_labels_sha256": sha256_of(VERIFIED),
        },
        "caveats": [
            "External frame counts are tiny (44 failing / 24 normal) — descriptive only, "
            "not a benchmark.",
            "UE003 normal decisions are human-review manifest decisions that Stage 12F "
            "left unmerged; they are evaluation-only, never training labels.",
            "No verified under_extrusion external frames exist.",
            "2A evaluated on NORMAL001/SP001 is a TRAINING-FIT (memorization) check, "
            "not a generalization result.",
            "External UE003 evaluation is a single-source descriptive signal.",
        ],
    }
    (out_dir / "derived_manifest.json").write_text(json.dumps(derived, indent=2))

    splits_json = {
        "data_dir": str(printing_errors_dir),
        "mode": "exp2-external-video-aug",
        "stats": stats,
        "splits": {
            "train": count_by_label(train_rows),
            "val": count_by_label(val_rows),
            "test": count_by_label(test_rows),
        },
    }
    (splits_dir / "splits.json").write_text(json.dumps(splits_json, indent=2))
    return derived


def count_by_label(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["label"]] += 1
    return dict(counts)


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, pos_integer: int) -> dict:
    """Precision/recall/F1/accuracy/FNR/FPR with ``pos_integer`` as positive.

    Confusion matrix orientation (rows=true, cols=pred, order [0,1]) and the
    per-convention TP/FP/FN/TN mapping mirror deep-eval exactly.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    c00, c01, c10, c11 = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
    if pos_integer == 1:
        tp, fn, fp, tn = c11, c10, c01, c00
    else:
        tp, fn, fp, tn = c00, c01, c10, c11
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "positive_class": int(pos_integer),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0, 4),
        "fnr": round(fn / (fn + tp) if (fn + tp) else 0.0, 4),
        "fpr": round(fp / (fp + tn) if (fp + tn) else 0.0, 4),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def _auroc_failing(y_true: np.ndarray, fail_scores: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    if len(set(int(x) for x in y_true)) < 2:
        return None
    try:
        # Higher score must predict the POSITIVE (failing=0) class; sklearn's
        # implicit positive is label 1, so score with P(normal)=1-P(failing),
        # matching the deep-eval failing-positive AUROC convention.
        return round(float(roc_auc_score(y_true, 1.0 - fail_scores)), 4)
    except ValueError:
        return None


def infer(checkpoint_path: Path, rows: list[dict], image_size: int = 224,
          batch_size: int = 32, threshold: float = 0.5) -> dict:
    """Score every row with a frozen binary classifier (deterministic, no aug)."""
    import torch
    from PIL import Image

    from .data.loaders import _build_transforms
    from .models.factory import build_classifier

    device = torch.device("cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"))

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = int(ckpt.get("num_classes", 2))
    model_name = ckpt.get("config_model", "shufflenet_v2_x1_0")

    model = build_classifier(model_name, num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    tfm = _build_transforms(image_size, augmentation={}, is_train=False)

    fail_scores, preds = [], []
    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i: i + batch_size]
        tensors = torch.stack([
            tfm(Image.open(r["image_path"]).convert("RGB")) for r in batch_rows
        ])
        with torch.no_grad():
            probs = torch.softmax(model(tensors.to(device)), dim=1).cpu().numpy()
        for row, prob in zip(batch_rows, probs):
            fail_score = float(prob[FAIL_IDX])
            fail_scores.append(fail_score)
            preds.append(FAIL_IDX if fail_score >= threshold else 1 - FAIL_IDX)

    return {
        "true_idx": np.array([r["binary_idx"] for r in rows]),
        "fail_scores": np.array(fail_scores),
        "preds": np.array(preds),
    }


def evaluate_external(
    checkpoint_path: str | Path,
    subset: str = "normal",
    image_size: int = 224,
    batch_size: int = 32,
    threshold: float = 0.5,
    kind: str = "generalization",
) -> dict:
    """Evaluate a frozen binary checkpoint over selected external video frames.

    ``subset`` selects which external frames to score:
      * ``normal``  -> UE003 human-normal frames (external eval; unseen by training)
      * ``failing`` -> NORMAL001+SP001 verified failing frames (baseline: external
                       generalization; 2A: TRAINING-FIT, see ``kind``)
      * ``all``     -> union of both

    ``kind`` labels the result ``generalization`` (unseen by the model) or
    ``training_fit`` (frames used in the model's own training set).
    """
    rows = {
        "normal": external_normal_rows,
        "failing": verified_training_rows,
    }[subset]() if subset != "all" else (
        external_normal_rows() + verified_training_rows()
    )
    inference = infer(Path(checkpoint_path), rows, image_size, batch_size, threshold)
    y_true, y_pred, fail_scores = inference["true_idx"], inference["preds"], inference["fail_scores"]

    per_source = {}
    for sid in sorted({r["source_id"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["source_id"] == sid]
        per_source[sid] = {
            "frames": len(idx),
            "n": len(idx),
            **{k: v for k, v in _binary_metrics(y_true[idx], y_pred[idx], FAIL_IDX).items()
               if k not in ("positive_class",)},
        }

    per_unit_blocks = defaultdict(list)
    for i, r in enumerate(rows):
        per_unit_blocks[r["review_unit_id"]].append(i)
    per_unit = {}
    for uid in sorted(per_unit_blocks):
        idx = np.array(per_unit_blocks[uid])
        yp = y_pred[idx]
        yt = y_true[idx]
        per_unit[uid] = {
            "source": rows[idx[0]]["source_id"],
            "n": int(len(idx)),
            "predicted_failing": int(np.sum(yp == FAIL_IDX)),
            "ground_truth": rows[idx[0]]["binary_label"],
        }

    normal_idx = [i for i, r in enumerate(rows) if r["binary_label"] == "normal"]
    result = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "subset": subset,
        "kind": kind,
        "threshold": threshold,
        "image_size": image_size,
        "evaluated_frames": len(rows),
        "sources": sorted({r["source_id"] for r in rows}),
        "metrics_failing_positive": _binary_metrics(y_true, y_pred, FAIL_IDX),
        "metrics_normal_positive": _binary_metrics(y_true, y_pred, 1),
        "auroc_failing": _auroc_failing(y_true, fail_scores),
        "per_source": per_source,
        "per_unit": per_unit,
        "normal_frames": {
            "frames": len(normal_idx),
            "predicted_failing": int(np.sum(y_pred[normal_idx] == FAIL_IDX)),
            "false_positive_rate": round(
                float(np.mean(y_pred[normal_idx] == FAIL_IDX)), 4
            ) if normal_idx else 0.0,
        },
        "full_coverage": coverage(rows),
        "label_class": sorted(set(r["binary_label"] for r in rows)),
    }
    return result


def coverage_stats(printing_errors_dir: str | Path, out_dir: str | Path) -> dict:
    """Stats helper used by tests: rows in the derived CSVs."""
    out_dir = Path(out_dir)
    splits_dir = out_dir / "splits"

    def read_csv(path: Path) -> list[dict]:
        with open(path) as f:
            return list(csv.DictReader(f))

    return {
        "train": len(read_csv(splits_dir / "train.csv")),
        "val": len(read_csv(splits_dir / "val.csv")),
        "test": len(read_csv(splits_dir / "test.csv")),
        "data_dir": load_json(splits_dir / "splits.json")["data_dir"],
    }