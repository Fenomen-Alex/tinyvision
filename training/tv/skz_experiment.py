"""SKZ Nozzle-Cam probe experiments (A: external generalization, B: auxiliary training).

The TinyVision model is BINARY (normal vs failing). The SKZ probe records four
image-level classes (normal/stringing/spaghetti/under_extrusion) derived from
test-split YOLO boxes in ``datasets/external/skz-probe/manifest.json``.

Binary mapping used here (faithful to the original binary task, never a
redefinition):

* stringing, spaghetti, under_extrusion  -> failing
* normal (a CANDIDATE label, provider annotation: "no defect-class box") -> normal

The probe is read-only. Experiment B consumes the probe through a NEW derived
split directory (``skz-prep``) so nothing in the probe or in the existing
TinyVision splits is modified.

Group-unit rule: a ``grouping_key`` (Profile config token) is treated as a single
unit of independence across ALL classes, because the audit found tokens spanning
more than one derived class (transition prints). Splitting is token-based via
``split_by_group``; every frame of a token lands in the same split.

Evaluation mirrors the existing deep-eval conventions: failing-as-positive is
the primary view (failing score = P(class=failing), decision threshold on the
failing score), both binary conventions are reported, AUROC is reported for the
failing class, and failure-type detail is preserved from the manifest.

Licensing is recorded without interpretation (HF card CC BY-SA 4.0; Zenodo
umbrella record CC BY 4.0); this module never resolves the discrepancy.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from .data.loaders import _build_transforms
from .models.factory import build_classifier
from .skz_probe import CLS, PROBE_CLASSES, split_by_group

SCHEMA = "skz-experiment"
SCHEMA_VERSION = 1
DEFAULT_SPLIT_TRAIN_FRACTION = 0.8
DEFAULT_SEED = 42

# Derived probe class -> binary label understood by the TinyVision pipeline.
# The 0.5 decision rule always references the failing score.
DERIVED_TO_BINARY = {
    "normal": "normal",
    "stringing": "failing",
    "spaghetti": "failing",
    "under_extrusion": "failing",
}
# Binary label -> index, identical to the frozen TinyVision splits
# (class_to_idx sorted alphabetically: failing=0, normal=1).
BINARY_LABELS = {"failing": 0, "normal": 1}
FAIL_IDX = 0

# Failure type names recorded in a manifest entry's all_failure_types.
FAILURE_TYPE_NAMES = ("Spaghetti", "Stringing", "Unterextrusion")


def binary_label(derived_class: str) -> str:
    """Map a probe derived class to the TinyVision binary label."""
    return DERIVED_TO_BINARY[derived_class]


def load_manifest(manifest_path: str | Path) -> dict:
    """Load a probe manifest and validate its schema."""
    manifest_path = Path(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("schema") != "skz-probe":
        raise ValueError(
            f"expected schema 'skz-probe' in {manifest_path}, got {manifest.get('schema')!r}"
        )
    return manifest


def entry_rows(entries: list[dict]) -> list[dict]:
    """Enrich manifest entries with binary label + absolute image path."""
    rows = []
    for e in entries:
        derived = e["derived_class"]
        image_path = Path(e["probe_image_path"]).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"probe image missing: {image_path}")
        rows.append(
            {
                "probe_id": e["probe_id"],
                "image_path": str(image_path),
                "derived_class": derived,
                "binary_label": binary_label(derived),
                "binary_idx": BINARY_LABELS[binary_label(derived)],
                "grouping_key": e["grouping_key"],
                "grouping_verified": e["grouping_verified"],
                "all_failure_types": list(e["all_failure_types"]),
                "multi_defect": e["multi_defect"],
                "image_sha256": e.get("image_sha256"),
            }
        )
    return rows


def token_split(
    entries: list[dict], train_fraction: float | None = None, seed: int = DEFAULT_SEED
) -> dict[str, str]:
    """Assign every probe GROUPING TOKEN to ``train`` (skz-aux) or ``eval``.

    Token counts are taken over the UNION of all classes so a token spanning
    several derived classes (transition print) is never split across aux/eval.
    """
    if train_fraction is None:
        train_fraction = DEFAULT_SPLIT_TRAIN_FRACTION
    counts = Counter(e["grouping_key"] for e in entries)
    return split_by_group(dict(counts), train_fraction=train_fraction, seed=seed)


def subset_entries(
    entries: list[dict], split: dict[str, str], subset: str = "all"
) -> list[dict]:
    """Filter entries by token split: 'all', 'skz_eval', or 'skz_aux'."""
    if subset == "all":
        return entries
    split_value = {"skz_aux": "train", "skz_eval": "eval"}.get(subset)
    if split_value is None:
        raise ValueError(f"subset must be all|skz_eval|skz_aux, got {subset!r}")
    return [e for e in entries if split[e["grouping_key"]] == split_value]


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, pos_integer: int) -> dict:
    """Precision/recall/F1/accuracy/FNR/FPR with ``pos_integer`` as positive."""
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
        "accuracy": round((tp + tn) / (tp + tn + fp + fn), 4),
        "fnr": round(fn / (fn + tp) if (fn + tp) else 0.0, 4),
        "fpr": round(fp / (fp + tn) if (fp + tn) else 0.0, 4),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def _auroc_failing(y_true: np.ndarray, fail_scores: np.ndarray) -> float | None:
    if len(set(int(x) for x in y_true)) < 2:
        return None
    try:
        # Higher score must predict the POSITIVE class (failing=0). sklearn's
        # implicit positive is label 1, so score with P(normal)=1-P(failing),
        # matching the deepeval convention for the failing-positive AUROC.
        return float(round(roc_auc_score(y_true, 1.0 - fail_scores), 4))
    except ValueError:
        return None


def _run_inference(
    checkpoint_path: Path,
    rows: list[dict],
    image_size: int,
    batch_size: int,
    num_workers: int,
    threshold: float,
) -> dict:
    """Score every probe image with a frozen binary classifier."""
    import torch

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
        batch_rows = rows[i : i + batch_size]
        from PIL import Image

        tensors = torch.stack([tfm(Image.open(r["image_path"]).convert("RGB")) for r in batch_rows])
        with torch.no_grad():
            probs = torch.softmax(model(tensors.to(device)), dim=1).cpu().numpy()
        for row, prob in zip(batch_rows, probs):
            fail_score = float(prob[FAIL_IDX])
            fail_scores.append(fail_score)
            preds.append(FAIL_IDX if fail_score >= threshold else 1 - FAIL_IDX)

    true_idx = np.array([r["binary_idx"] for r in rows])
    return {
        "true_idx": true_idx,
        "fail_scores": np.array(fail_scores),
        "preds": np.array(preds),
    }


def _failure_type_block(rows: list[dict], preds: np.ndarray) -> dict:
    """Per-failure-type subset statistics (recall that a subset is predicted failing)."""
    preds = np.asarray(preds)
    out = {}
    for name in FAILURE_TYPE_NAMES:
        idx = [i for i, r in enumerate(rows) if name in r["all_failure_types"]]
        if not idx:
            continue
        pred_fail = int(np.sum(preds[idx] == FAIL_IDX))
        groups = sorted({rows[i]["grouping_key"] for i in idx})
        out[name] = {
            "images": len(idx),
            "groups": len(groups),
            "predicted_failing": pred_fail,
            "recall_failing": round(pred_fail / len(idx), 4),
        }
    return out


def evaluate_probe(
    checkpoint_path: str | Path,
    manifest_path: str | Path | None = None,
    subset: str = "all",
    train_fraction: float | None = None,
    seed: int = DEFAULT_SEED,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    threshold: float = 0.5,
    split: dict[str, str] | None = None,
) -> dict:
    """Evaluate a frozen binary checkpoint against the SKZ probe.

    ``split`` optionally provides a precomputed token split (skips recomputing);
    otherwise ``token_split`` is computed from the manifest with the seed given.
    """
    from .skz_probe import PROBE as PROBE_DIR

    manifest_path = Path(manifest_path) if manifest_path else PROBE_DIR / "manifest.json"
    manifest = load_manifest(manifest_path)
    entries = manifest["entries"]
    if split is None:
        split = token_split(entries, train_fraction=train_fraction, seed=seed)
    selected = subset_entries(entries, split, subset)
    rows = entry_rows(selected)

    inference = _run_inference(
        Path(checkpoint_path), rows, image_size, batch_size, num_workers, threshold
    )
    y_true = inference["true_idx"]
    y_pred = inference["preds"]
    fail_scores = inference["fail_scores"]

    classes_block = []
    for c in PROBE_CLASSES:
        idx = [i for i, r in enumerate(rows) if r["derived_class"] == c]
        classes_block.append(
            {
                "class": c,
                "binary_label": binary_label(c),
                "images": len(idx),
                "groups": len({rows[i]["grouping_key"] for i in idx}),
                "predicted_failing_rate": round(
                    float(np.mean(y_pred[idx] == FAIL_IDX)), 4
                ) if idx else 0.0,
            }
        )

    result = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "manifest": str(manifest_path.resolve()),
        "subset": subset,
        "threshold": threshold,
        "image_size": image_size,
        "evaluated_images": len(rows),
        "independent_groups": len({r["grouping_key"] for r in rows}),
        "classes": classes_block,
        "per_failure_type": _failure_type_block(rows, y_pred),
        "multi_defect": {
            "images": sum(1 for r in rows if r["multi_defect"]),
            "predicted_failing": int(np.sum(y_pred[[i for i, r in enumerate(rows) if r["multi_defect"]]] == FAIL_IDX))
            if any(r["multi_defect"] for r in rows) else 0,
        },
        "metrics_failing_positive": _binary_metrics(y_true, y_pred, pos_integer=FAIL_IDX),
        "metrics_normal_positive": _binary_metrics(y_true, y_pred, pos_integer=1),
        "auroc_failing": _auroc_failing(y_true, fail_scores),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "candidate_normal": {
            "images": sum(1 for r in rows if r["binary_label"] == "normal"),
            "groups": len({r["grouping_key"] for r in rows if r["binary_label"] == "normal"}),
            "false_positive_rate": round(
                float(np.mean(y_pred[[i for i, r in enumerate(rows) if r["binary_label"] == "normal"]] == FAIL_IDX)), 4
            ) if any(r["binary_label"] == "normal" for r in rows) else 0.0,
        },
        "evaluated_probe_ids": [r["probe_id"] for r in rows],
    }
    return result


def prepare_expB_splits(
    printing_errors_dir: str | Path,
    out_dir: str | Path,
    manifest_path: str | Path | None = None,
    train_fraction: float | None = None,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Build the Experiment B derived split directory.

    train.csv  = printing-errors train rows (unchanged) + SKZ-aux rows.
    val.csv    = printing-errors val rows (unchanged copy).
    test.csv   = printing-errors test rows (unchanged copy).

    SKZ image paths are written absolute; printing-errors rows stay relative and
    resolve against ``splits.json`` data_dir (the printing-errors root). The
    existing printing-errors splits and the SKZ probe are never modified.
    """
    from .skz_probe import PROBE as PROBE_DIR

    printing_errors_dir = Path(printing_errors_dir).resolve()
    manifest_path = Path(manifest_path) if manifest_path else PROBE_DIR / "manifest.json"
    manifest = load_manifest(manifest_path)

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

    entries = manifest["entries"]
    split_map = token_split(entries, train_fraction=train_fraction, seed=seed)
    aux = [e for e in entries if split_map[e["grouping_key"]] == "train"]
    eval_ = [e for e in entries if split_map[e["grouping_key"]] == "eval"]

    for e in aux:
        image = Path(e["probe_image_path"]).resolve()
        if not image.exists():
            raise FileNotFoundError(f"probe image missing: {image}")
        train_rows.append(
            {
                "path": str(image),
                "label": binary_label(e["derived_class"]),
            }
        )

    def write_csv(path: Path, rows: list[dict]) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    write_csv(splits_dir / "train.csv", train_rows)
    write_csv(splits_dir / "val.csv", val_rows)
    write_csv(splits_dir / "test.csv", test_rows)

    stats = {
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
    }
    manifest_out = {
        "data_dir": str(printing_errors_dir),
        "source_splits": str(src),
        "ratios": [0.7, 0.15, 0.15],
        "seed": seed,
        "train_fraction": DEFAULT_SPLIT_TRAIN_FRACTION if train_fraction is None else train_fraction,
        "skz_probe_manifest": str(manifest_path.resolve()),
        "skz_split": {"train_tokens": sorted(k for k, v in split_map.items() if v == "train"),
                      "eval_tokens": sorted(k for k, v in split_map.items() if v == "eval")},
        "skz_aux": {
            "images": len(aux),
            "groups": len({e["grouping_key"] for e in aux}),
            "per_class": dict(Counter(e["derived_class"] for e in aux)),
        },
        "skz_eval": {
            "images": len(eval_),
            "groups": len({e["grouping_key"] for e in eval_}),
            "per_class": dict(Counter(e["derived_class"] for e in eval_)),
        },
        "caveats": [
            "SKZ normal labels are CANDIDATE labels (no defect-class box in the annotation), "
            "not human-verified ground truth.",
            "SKZ annotations are provider annotations (exploratory); correctness is not claimed.",
            "Tokens spanning multiple derived classes are split as one unit (no per-class split).",
            "Full train-side probe leakage cannot be verified (HF tree API pagination); "
            "eval uses only held-out probe tokens.",
        ],
        "stats": stats,
        "class_to_idx": BINARY_LABELS,
    }
    with open(out_dir / "derived_manifest.json", "w") as f:
        json.dump(manifest_out, f, indent=2)

    splits_json = {
        "data_dir": str(printing_errors_dir),
        "mode": "expB-aux",
        "seed": seed,
        "stats": stats,
        "splits": {"train": count_by_label(train_rows), "val": count_by_label(val_rows), "test": count_by_label(test_rows)},
    }
    with open(splits_dir / "splits.json", "w") as f:
        json.dump(splits_json, f, indent=2)

    return manifest_out


def count_by_label(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["label"]] += 1
    return dict(counts)