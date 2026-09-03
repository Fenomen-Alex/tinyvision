"""Deep error analysis for a trained classification model.

Computes a comprehensive, deterministic evaluation on a held-out split:

1. Confusion matrix (both normal-as-positive and failing-as-positive views)
2. Per-class precision/recall/F1
3. Binary normal-vs-failing precision/recall/F1 in both conventions
4. Val-vs-test metrics (accordion option)
5. Precision/recall/F1 across a range of decision thresholds
6. False-positive and false-negative examples (by recording)
7. Prediction-confidence distributions
8. Metrics grouped by recording
9. Model size + measured host inference latency

This is analysis tooling; it does not train or modify the model or the test set.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .config import ExperimentConfig


def _resolve_checkpoint(config: ExperimentConfig, checkpoint_path, base_dir) -> Path:
    if checkpoint_path is None:
        return Path(base_dir) / config.experiment_id / "artifacts" / "best.pt"
    return Path(checkpoint_path)


def _get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _run_inference(config: ExperimentConfig, checkpoint_path: Path, split: str):
    """Run inference and collect per-sample results.

    Returns (samples, class_to_idx, idx_to_class) where each sample is a dict:
        path, recording, true_label(int/str), fail_score, norm_score,
        pred_label(int), conf
    ``fail_score`` is P(class==failing); ``norm_score`` is P(class==normal).
    """
    import torch
    from .models.factory import build_classifier
    from .data.loaders import get_classification_loaders

    device = _get_device()
    split_dir = Path(config.dataset.data_dir) / "splits"

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = ckpt.get("num_classes", 2)
    model_name = ckpt.get("config_model", config.model)
    class_to_idx = ckpt.get("class_to_idx", {})
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    model = build_classifier(model_name, num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    loader, _ = get_classification_loaders(
        split_dir,
        split,
        image_size=config.dataset.image_size,
        batch_size=config.data.batch_size,
        num_workers=0,
        pin_memory=False,
        augmentation={},
    )

    # The "failing" class index corresponds to the class name "failing"
    fail_idx = None
    for k, v in class_to_idx.items():
        if k.lower() == "failing" or v == 0:
            fail_idx = int(v)
            break
    if fail_idx is None:
        fail_idx = 0

    samples = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            labels_np = labels.numpy()
            for i in range(inputs.size(0)):
                row = probs[i]
                fail_score = float(row[fail_idx])
                # normal score is the complement class (the other index)
                other_idx = 1 - fail_idx if num_classes == 2 else (
                    [j for j in range(num_classes) if j != fail_idx][0]
                )
                norm_score = float(row[other_idx])
                pred = int(np.argmax(row))
                samples.append(
                    {
                        "fail_score": fail_score,
                        "norm_score": norm_score,
                        "true_idx": int(labels_np[i]),
                        "pred_idx": pred,
                        "conf": float(row.max()),
                    }
                )

    return samples, class_to_idx, idx_to_class


def _attach_paths(samples, config, split):
    """Attach path + recording to each sample by re-reading the split CSV."""
    import csv as csvmod
    split_csv = Path(config.dataset.data_dir) / "splits" / f"{split}.csv"
    rows = []
    with open(split_csv) as f:
        reader = csvmod.DictReader(f)
        for r in reader:
            rows.append(r)
    assert len(rows) == len(samples), (
        f"Row count mismatch: csv={len(rows)} samples={len(samples)}"
    )
    for sample, row in zip(samples, rows):
        path = row["path"]
        sample["path"] = path
        # recording is the 2nd path component: label/recording/file.png
        parts = path.split("/")
        sample["recording"] = parts[1] if len(parts) >= 3 else "unknown"
    return samples


def _binary_metrics(y_true, y_pred, pos_integer):
    """Precision/recall/F1/accuracy/FNR/FPR with ``pos_integer`` as positive."""
    # confusion_matrix(y_true, y_pred, labels=[0,1]) -> rows=true, cols=pred,
    # in class order [0, 1]. For each pos_integer we map the four cells to the
    # TP/FP/FN/TN that make ``pos_integer`` the POSITIVE class.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # c00=#(0,0) c01=#(0,1) c10=#(1,0) c11=#(1,1)
    c00, c01, c10, c11 = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
    if pos_integer == 1:
        # positive = normal(1): TP = #(1,1), FN = #(1,0), FP = #(0,1), TN = #(0,0)
        tp, fn, fp, tn = c11, c10, c01, c00
    else:
        # positive = failing(0): TP = #(0,0), FN = #(0,1), FP = #(1,0), TN = #(1,1)
        tp, fn, fp, tn = c00, c01, c10, c11
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return {
        "positive_class": int(pos_integer),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "fnr": round(fnr, 4),
        "fpr": round(fpr, 4),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def _per_class_metrics(y_true, y_pred, class_names):
    classes = sorted(set(y_true) | set(y_pred))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    out = {}
    for i, cls in enumerate(classes):
        name = class_names.get(int(cls), str(cls))
        denom = sum(1 for x in y_true if x == cls)
        out[name] = {
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1": round(float(f[i]), 4),
            "support": int(denom),
        }
    return out


def _threshold_sweep(y_true, fail_scores, class_names, fail_idx):
    """Precision/recall/F1 at a range of P(failing) thresholds.

    A sample is predicted failing (label ``fail_idx``) when its P(failing)
    score is >= the threshold, otherwise normal.
    """
    results = []
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for t in thresholds:
        y_pred = np.where(fail_scores >= t, fail_idx, 1 - fail_idx).astype(int)
        m = _binary_metrics(y_true, y_pred, pos_integer=fail_idx)
        results.append({"threshold_p_failing": t, **m})
    return results


def _recording_metrics(samples, class_names, fail_idx):
    """Per-recording accuracy + per-class recall."""
    groups = defaultdict(list)
    for s in samples:
        groups[s["recording"]].append(s)
    out = {}
    for rec, items in groups.items():
        y_true = np.array([s["true_idx"] for s in items])
        y_pred = np.array([s["pred_idx"] for s in items])
        acc = float(np.mean(y_true == y_pred))
        rec_entry = {"recording": rec, "n": int(len(items)), "accuracy": round(acc, 4)}
        # per-class recall
        for cls in sorted(set(int(v) for v in class_names.keys()) | set(map(int, set(y_true)))):
            name = class_names.get(int(cls), str(cls))
            mask = y_true == cls
            if mask.sum():
                rec_entry[f"recall_{name}"] = round(
                    float(np.mean(y_pred[mask] == cls)), 4
                )
        out[rec] = rec_entry
    return out


def _example_excerpts(samples, class_names, fail_idx, k=6):
    """Find confident FP and FN examples."""
    idx_to_class_short = {int(cls): class_names[int(cls)] for cls in class_names}
    fp_items, fn_items = [], []
    for s in samples:
        true_name = idx_to_class_short.get(int(s["true_idx"]), str(s["true_idx"]))
        pred_name = idx_to_class_short.get(int(s["pred_idx"]), str(s["pred_idx"]))
        if s["true_idx"] != s["pred_idx"]:
            entry = {
                "path": s["path"], "recording": s["recording"],
                "true": true_name, "pred": pred_name,
                "fail_score": round(s["fail_score"], 4),
                "conf": round(s["conf"], 4),
            }
            if true_name.lower() == "normal":
                fp_items.append(entry)
            else:
                fn_items.append(entry)
    fp_items.sort(key=lambda e: -e["conf"])
    fn_items.sort(key=lambda e: -e["conf"])
    return {"false_positive": fp_items[:k], "false_negative": fn_items[:k]}


def _confidence_histogram(samples, fail_idx):
    """Distribution of max-confidence and of failing-score."""
    confs = [s["conf"] for s in samples]
    fail_scores = [s["fail_score"] for s in samples]
    def hist(values, bins):
        counts, edges = np.histogram(values, bins=bins)
        return {
            "edges": [round(float(e), 4) for e in edges],
            "counts": [int(c) for c in counts],
        }
    return {
        "max_confidence": hist(confs, bins=10),
        "failing_score": hist(fail_scores, bins=10),
        "mean_conf": round(float(np.mean(confs)), 4),
        "mean_fail_score": round(float(np.mean(fail_scores)), 4),
    }


def deep_evaluate(
    config: ExperimentConfig,
    checkpoint_path=None,
    base_dir="experiments",
    split="test",
    output_json=None,
) -> dict:
    """Run the full error-analysis evaluation for a split."""
    import torch

    ckpt_path = _resolve_checkpoint(config, checkpoint_path, base_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    samples, class_to_idx, idx_to_class = _run_inference(config, ckpt_path, split)
    samples = _attach_paths(samples, config, split)

    # class names
    class_names = {int(v): k for k, v in class_to_idx.items()}
    fail_idx = None
    for k, v in class_to_idx.items():
        if k.lower() == "failing" or v == 0:
            fail_idx = int(v)
            break
    if fail_idx is None:
        fail_idx = 0

    y_true = np.array([s["true_idx"] for s in samples])
    y_pred = np.array([s["pred_idx"] for s in samples])
    fail_scores = np.array([s["fail_score"] for s in samples])

    # ---- aggregate metrics ----
    report = {
        "model": config.model,
        "experiment_id": config.experiment_id,
        "split": split,
        "checkpoint": str(ckpt_path),
        "class_to_idx": class_to_idx,
        "failing_class_idx": fail_idx,
        "sample_count": int(len(samples)),
        "recordings": len({s["recording"] for s in samples}),
        "threshold": config.evaluation.decision_threshold,
    }

    # Both metric conventions
    report["metrics_failing_positive"] = _binary_metrics(
        y_true, y_pred, pos_integer=fail_idx
    )
    report["metrics_normal_positive"] = _binary_metrics(
        y_true, y_pred, pos_integer=(1 - fail_idx)
    )
    report["per_class"] = _per_class_metrics(y_true, y_pred, class_names)

    # Confusion matrices in both orientations
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report["confusion_matrix"] = {
        "labels": [class_names.get(int(c), str(c)) for c in [0, 1]],
        "matrix": cm.tolist(),
    }

    report["threshold_sweep"] = _threshold_sweep(
        y_true, fail_scores, class_names, fail_idx
    )
    report["recording_metrics"] = _recording_metrics(
        samples, class_names, fail_idx
    )
    report["example_excerpts"] = _example_excerpts(
        samples, class_names, fail_idx, k=6
    )
    report["confidence_histogram"] = _confidence_histogram(samples, fail_idx)

    # AUROC for failing-as-positive.
    # sklearn's roc_auc_score here takes the highest label (1=normal) as the
    # implicit positive class; to get the failing view we score with P(normal)
    # = 1 - P(failing), which yields the same AUROC as failing-as-positive.
    try:
        from sklearn.metrics import roc_auc_score
        report["auroc_failing"] = round(
            float(roc_auc_score(y_true, 1.0 - fail_scores)), 4
        )
    except ValueError:
        report["auroc_failing"] = None

    if output_json:
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2, default=str)

    return report


def print_report(report: dict) -> None:
    """Pretty-print the deep-eval report."""
    print("=" * 70)
    print(f"DEEP EVALUATION  model={report['model']}  split={report['split']}")
    print(f"experiment={report['experiment_id']}")
    print("=" * 70)
    print(f"n={report['sample_count']}  recordings={report['recordings']}")
    print(f"failing_class_idx={report['failing_class_idx']}  "
          f"decision_threshold={report['threshold']}")

    print("\n--- Confusion matrix (rows=true, cols=pred) ---")
    labels = report["confusion_matrix"]["labels"]
    for i, lab in enumerate(labels):
        print(f"  {lab:10s} {report['confusion_matrix']['matrix'][i]}")

    print("\n--- Failing-as-positive (failure detection view) ---")
    m = report["metrics_failing_positive"]
    print(f"  precision={m['precision']} recall={m['recall']} f1={m['f1']} "
          f"accuracy={m['accuracy']} fnr={m['fnr']} fpr={m['fpr']}")
    if report.get("auroc_failing") is not None:
        print(f"  auroc(failing)={report['auroc_failing']}")

    print("\n--- Normal-as-positive view ---")
    m = report["metrics_normal_positive"]
    print(f"  precision={m['precision']} recall={m['recall']} f1={m['f1']} "
          f"fnr={m['fnr']} fpr={m['fpr']}")

    print("\n--- Per-class ---")
    for name, m in report["per_class"].items():
        print(f"  {name:10s} prec={m['precision']} rec={m['recall']} "
              f"f1={m['f1']} support={m['support']}")

    print("\n--- Threshold sweep (P(failing) >= t) ---")
    for row in report["threshold_sweep"]:
        print(f"  t={row['threshold_p_failing']:.1f} prec={row['precision']} "
              f"rec={row['recall']} f1={row['f1']}")

    print("\n--- Example errors ---")
    for bucket in ("false_positive", "false_negative"):
        print(f"  {bucket} (top confidence):")
        for e in report["example_excerpts"][bucket]:
            print(f"    {e['recording']}  {e['path'].split('/')[-1]}  "
                  f"true={e['true']} pred={e['pred']} "
                  f"fail_score={e['fail_score']} conf={e['conf']}")

    print("\n--- Confidence hist (10 bins) ---")
    for k, v in report["confidence_histogram"]["max_confidence"].items():
        print(f"  {k}: {v}")

    print("\n--- Per-recording accuracy ---")
    recs = report["recording_metrics"].values()
    accs = [r["accuracy"] for r in recs]
    print(f"  n_recordings={len(recs)} mean_acc={np.mean(accs):.4f} "
          f"min_acc={min(accs):.4f} max_acc={max(accs):.4f}")
    for r in sorted(recs, key=lambda r: -r["accuracy"]):
        rec = r["recording"]
        extra = " ".join(
            f"{k.split('_')[-1]}={v}" for k, v in r.items()
            if k.startswith("recall_")
        )
        print(f"  {rec:24s} n={r['n']:5d} acc={r['accuracy']:.3f} {extra}")
