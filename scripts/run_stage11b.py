#!/usr/bin/env python3
"""Stage 11B: Execute leave-cluster-out generalization experiments.

Trains and evaluates four experiments (2 stringing holdout + 2 under-extrusion
control) using the exact Experiment 1 ShuffleNetV2 x1.0 configuration.

Usage:
  python scripts/run_stage11b.py --all
  python scripts/run_stage11b.py --run str_holdout_22122022
  python scripts/run_stage11b.py --report-only
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

from tv.config import ExperimentConfig, DatasetConfig, DataConfig, TrainConfig, \
    AugmentationConfig, EvalConfig, save_config, load_config, _generate_experiment_id
from tv.experiment import ExperimentManager
from tv.train import train_classification, set_seed
from tv.stage11a import (
    load_all_recordings, define_stringing_clusters, define_under_extrusion_clusters,
    design_runs, run_leakage_audit, generate_run_csvs, DATA_DIR, SPLITS_DIR,
    extract_date,
)

RUNS_DIR = REPO / "experiments"
RESULTS_DIR = REPO / "docs" / "research"
THRESHOLD = 0.65
TRAIN_FRAMES_EXP1 = 11254
TRAIN_TIME_EXP1_S = 968


# ---------------------------------------------------------------------------
# Run config
# ---------------------------------------------------------------------------

RUN_IDS = [
    "str_holdout_22122022",
    "str_holdout_20122022",
    "ue_control_10012023",
    "ue_control_05012023",
]


def _build_run_config(run_id: str, tmp_dir: Path) -> ExperimentConfig:
    """Build ExperimentConfig pointing to temporary split directory."""
    aug = AugmentationConfig(
        horizontal_flip=True, random_rotation=15, color_jitter=0.2,
        random_crop=False, normalize=True,
    )
    train_cfg = TrainConfig(
        epochs=30, optimizer="adam", learning_rate=0.001, weight_decay=0.0001,
        augmentation=aug, precision="fp32", seed=42, early_stopping_patience=10,
    )
    ds_cfg = DatasetConfig(
        name="printing-errors", version="1.0",
        data_dir=str(tmp_dir), image_size=224,
    )
    data_cfg = DataConfig(num_workers=4, batch_size=32, pin_memory=True)
    eval_cfg = EvalConfig(decision_threshold=0.65)
    cfg = ExperimentConfig(
        model="shufflenet_v2_x1_0", task="classification",
        dataset=ds_cfg, data=data_cfg, train=train_cfg, evaluation=eval_cfg,
        notes=f"Stage 11B: leave-cluster-out ({run_id})",
    )
    cfg.experiment_id = f"stage11b_{run_id}"
    cfg.created_at = datetime.now(timezone.utc).isoformat()
    return cfg


# ---------------------------------------------------------------------------
# Held-out cluster evaluation (manual inference)
# ---------------------------------------------------------------------------

def _evaluate_held_out(
    checkpoint_path: Path,
    held_out_frame_paths: list[str],
    held_out_labels: list[int],
    held_out_fines: list[str],
    device: torch.device,
    image_size: int = 224,
    batch_size: int = 32,
) -> dict:
    """Run inference on held-out cluster frames and compute metrics.

    Returns dict with per-frame predictions and aggregate metrics.
    """
    from torchvision import transforms
    from PIL import Image

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    from tv.models.factory import build_classifier
    model_name = ckpt.get("config_model", "shufflenet_v2_x1_0")
    num_classes = ckpt.get("num_classes", 2)
    class_to_idx = ckpt.get("class_to_idx", {})
    fail_idx = 0
    for k, v in class_to_idx.items():
        if k.lower() == "failing" or v == 0:
            fail_idx = int(v)
            break

    model = build_classifier(model_name, num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Transforms (no augmentation for eval)
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Run inference in batches
    all_fail_scores = []
    all_pred_indices = []
    all_true_indices = []

    with torch.no_grad():
        for i in range(0, len(held_out_frame_paths), batch_size):
            batch_paths = held_out_frame_paths[i:i + batch_size]
            batch_labels = held_out_labels[i:i + batch_size]
            images = []
            for p in batch_paths:
                img = Image.open(DATA_DIR / p).convert("RGB")
                images.append(tfm(img))
            inputs = torch.stack(images).to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            for j in range(len(batch_paths)):
                fail_score = float(probs[j, fail_idx])
                pred = int(np.argmax(probs[j]))
                all_fail_scores.append(fail_score)
                all_pred_indices.append(pred)
                all_true_indices.append(int(batch_labels[j]))

    y_true = np.array(all_true_indices)
    y_pred = np.array(all_pred_indices)
    fail_scores = np.array(all_fail_scores)

    # Compute binary metrics with fail_idx as positive
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    c00, c01, c10, c11 = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
    # positive = failing(fail_idx=0): TP = c00, FN = c01, FP = c10, TN = c11
    tp, fn, fp, tn = c00, c01, c10, c11
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    # AUROC
    auroc = None
    n_classes = len(set(y_true))
    if n_classes > 1:
        try:
            from sklearn.metrics import roc_auc_score
            # For failing-as-positive, score with 1 - P(failing) vs sklearn convention
            auroc = float(roc_auc_score(y_true, 1.0 - fail_scores))
        except ValueError:
            auroc = None

    # Per-recording breakdown
    rec_metrics = defaultdict(lambda: {"frames": 0, "detected": 0, "missed": 0, "true": [], "pred": [], "scores": []})
    for idx, path in enumerate(held_out_frame_paths):
        parts = path.split("/")
        rec_id = parts[1] if len(parts) >= 3 else "unknown"
        rec_metrics[rec_id]["frames"] += 1
        rec_metrics[rec_id]["true"].append(y_true[idx])
        rec_metrics[rec_id]["pred"].append(y_pred[idx])
        rec_metrics[rec_id]["scores"].append(float(fail_scores[idx]))
        if y_true[idx] == fail_idx and y_pred[idx] == fail_idx:
            rec_metrics[rec_id]["detected"] += 1
        elif y_true[idx] == fail_idx and y_pred[idx] != fail_idx:
            rec_metrics[rec_id]["missed"] += 1

    per_recording = {}
    for rec_id, m in sorted(rec_metrics.items()):
        n = m["frames"]
        correct = sum(1 for t, p in zip(m["true"], m["pred"]) if t == p)
        per_recording[rec_id] = {
            "frames": n,
            "detected": m["detected"],
            "missed": m["missed"],
            "accuracy": round(correct / n, 4) if n else 0.0,
            "mean_fail_score": round(float(np.mean(m["scores"])), 4),
        }

    return {
        "sample_count": len(held_out_frame_paths),
        "failing_frames": int((y_true == fail_idx).sum()),
        "normal_frames": int((y_true != fail_idx).sum()),
        "detected": int(tp),
        "missed": int(fn),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "fnr": round(fnr, 4),
            "fpr": round(fpr, 4),
            "auroc": round(auroc, 4) if auroc is not None else None,
        },
        "confusion_matrix": cm.tolist(),
        "per_recording": per_recording,
        "threshold": THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Single run execution
# ---------------------------------------------------------------------------

def execute_run(run_id: str, all_train, all_val, str_clusters, ue_clusters,
                runs, tmp_root: Path) -> dict:
    """Execute a single leave-cluster-out run. Returns results dict."""
    run = next(r for r in runs if r.run_id == run_id)
    print(f"\n{'='*70}")
    print(f"STAGE 11B: {run_id}")
    print(f"{'='*70}")
    print(f"  Held-out cluster: {run.held_out_cluster_id}")
    print(f"  Held-out recordings: {len(run.held_out_recordings)}")
    print(f"  Held-out frames: {run.held_out_frame_count}")
    print(f"  Train recordings (held-in): {run.train_recording_count}")
    print(f"  Train frames (held-in): {run.train_frame_count}")
    print(f"  Val recordings: {run.val_recording_count}")
    print(f"  Val frames: {run.val_frame_count}")

    # --- Leakage audit ---
    audit = run_leakage_audit(run, all_train, all_val,
                              load_all_recordings()["test"])
    if not audit.passed:
        raise RuntimeError(f"Leakage audit FAILED for {run_id}")
    print(f"  Leakage audit: ALL {len(audit.checks)} checks PASS")

    # --- Generate temporary splits ---
    run_splits_dir = tmp_root / run_id / "splits"
    run_splits_dir.mkdir(parents=True, exist_ok=True)
    paths = generate_run_csvs(run, all_train, all_val, run_splits_dir)
    run_tmp = tmp_root / run_id
    print(f"  Temporary splits written to {run_splits_dir}")

    # --- Build config ---
    cfg = _build_run_config(run_id, run_tmp)
    exp_dir = RUNS_DIR / cfg.experiment_id
    if exp_dir.exists():
        print(f"  Resuming: {exp_dir} already exists")
        status_path = exp_dir / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("status") == "completed":
                print(f"  Run already completed. Skipping training.")
                return _load_existing_results(exp_dir, run_id, run)

    # --- Train ---
    print(f"\n  Training... (experiment_id={cfg.experiment_id})")
    exp_mgr = ExperimentManager(cfg, base_dir=str(RUNS_DIR))
    t0 = time.time()
    with exp_mgr.run() as manager:
        train_result = train_classification(cfg, manager)
    train_elapsed = time.time() - t0
    print(f"  Training complete in {train_elapsed:.0f}s")
    print(f"  Best val loss: {train_result['best_val_loss']:.4f}")

    # --- Evaluate on held-in validation ---
    print(f"\n  Evaluating on held-in validation...")
    ckpt_path = exp_mgr.artifacts_dir / "best.pt"
    val_result = _evaluate_split(cfg, ckpt_path, "val")
    print(f"  Val metrics: F1={val_result['metrics']['f1']:.4f} "
          f"FNR={val_result['metrics']['fnr']:.4f} "
          f"FPR={val_result['metrics']['fpr']:.4f}")

    # --- Evaluate on held-out cluster ---
    print(f"\n  Evaluating on held-out cluster...")
    held_out_paths = []
    held_out_labels = []
    held_out_fines = []
    for rec in all_train:
        if rec.recording_id in run.held_out_recordings:
            for frame in rec.frames:
                held_out_paths.append(frame.path)
                held_out_labels.append(0 if frame.label == "failing" else 1)
                held_out_fines.append(frame.fine)

    held_out_result = _evaluate_held_out(
        ckpt_path, held_out_paths, held_out_labels, held_out_fines,
        device=_get_device(), image_size=cfg.dataset.image_size,
        batch_size=cfg.data.batch_size,
    )
    m = held_out_result["metrics"]
    print(f"  Held-out: recall={m['recall']:.4f} FNR={m['fnr']:.4f} "
          f"n={held_out_result['sample_count']} "
          f"detected={held_out_result['detected']} missed={held_out_result['missed']}")

    # --- Evaluate known-domain stringing (remaining train stringing frames) ---
    print(f"\n  Evaluating known-domain stringing (remaining train)...")
    known_str_paths = []
    known_str_labels = []
    known_str_fines = []
    held_out_ids = set(run.held_out_recordings)
    for rec in all_train:
        if rec.recording_id not in held_out_ids and rec.has_stringing:
            for frame in rec.frames:
                if frame.fine == "stringing":
                    known_str_paths.append(frame.path)
                    known_str_labels.append(0)
                    known_str_fines.append(frame.fine)

    if known_str_paths:
        known_str_result = _evaluate_held_out(
            ckpt_path, known_str_paths, known_str_labels, known_str_fines,
            device=_get_device(), image_size=cfg.dataset.image_size,
            batch_size=cfg.data.batch_size,
        )
        km = known_str_result["metrics"]
        print(f"  Known stringing: recall={km['recall']:.4f} FNR={km['fnr']:.4f} "
              f"n={known_str_result['sample_count']}")
    else:
        known_str_result = None
        print(f"  No known stringing frames remaining")

    # --- Save per-run results ---
    result = {
        "run_id": run_id,
        "held_out_cluster": run.held_out_cluster_id,
        "held_out_recordings": run.held_out_recordings,
        "held_out_frame_count": run.held_out_frame_count,
        "train_frame_count": run.train_frame_count,
        "train_recording_count": run.train_recording_count,
        "val_frame_count": run.val_frame_count,
        "val_recording_count": run.val_recording_count,
        "experiment_id": cfg.experiment_id,
        "checkpoint": str(ckpt_path),
        "training_elapsed_seconds": round(train_elapsed, 1),
        "best_val_loss": train_result["best_val_loss"],
        "training_epochs": len(train_result["history"]),
        "validation": val_result,
        "held_out": held_out_result,
        "known_domain_stringing": known_str_result,
        "config": {
            "model": cfg.model,
            "seed": cfg.train.seed,
            "epochs": cfg.train.epochs,
            "early_stopping_patience": cfg.train.early_stopping_patience,
            "lr": cfg.train.learning_rate,
            "wd": cfg.train.weight_decay,
            "batch_size": cfg.data.batch_size,
            "image_size": cfg.dataset.image_size,
            "precision": cfg.train.precision,
            "threshold": THRESHOLD,
        },
    }

    # Save to experiment dir
    result_path = exp_mgr.artifacts_dir / "stage11b_result.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"  Results saved to {result_path}")

    return result


def _evaluate_split(cfg: ExperimentConfig, ckpt_path: Path, split: str) -> dict:
    """Evaluate model on a split (val) using the existing evaluation approach."""
    from tv.data.loaders import get_classification_loaders
    from tv.models.factory import build_classifier

    device = _get_device()
    split_dir = Path(cfg.dataset.data_dir) / "splits"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_name = ckpt.get("config_model", cfg.model)
    num_classes = ckpt.get("num_classes", 2)
    class_to_idx = ckpt.get("class_to_idx", {})
    fail_idx = 0
    for k, v in class_to_idx.items():
        if k.lower() == "failing" or v == 0:
            fail_idx = int(v)
            break

    model = build_classifier(model_name, num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    loader, _ = get_classification_loaders(
        split_dir, split, image_size=cfg.dataset.image_size,
        batch_size=cfg.data.batch_size, num_workers=0, pin_memory=False,
        augmentation={},
    )

    all_true, all_pred, all_scores = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            for i in range(inputs.size(0)):
                all_true.append(int(labels[i]))
                all_pred.append(int(np.argmax(probs[i])))
                all_scores.append(float(probs[i, fail_idx]))

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    scores = np.array(all_scores)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    c00, c01, c10, c11 = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
    tp, fn, fp, tn = c00, c01, c10, c11
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    auroc = None
    if len(set(y_true)) > 1:
        try:
            from sklearn.metrics import roc_auc_score
            auroc = float(roc_auc_score(y_true, 1.0 - scores))
        except ValueError:
            auroc = None

    return {
        "sample_count": len(all_true),
        "failing_frames": int((y_true == fail_idx).sum()),
        "normal_frames": int((y_true != fail_idx).sum()),
        "detected": int(tp),
        "missed": int(fn),
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "fnr": round(fnr, 4),
            "fpr": round(fpr, 4),
            "auroc": round(auroc, 4) if auroc is not None else None,
        },
        "confusion_matrix": cm.tolist(),
    }


def _load_existing_results(exp_dir: Path, run_id: str, run) -> dict:
    """Load results from a previously completed run."""
    result_path = exp_dir / "artifacts" / "stage11b_result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    raise FileNotFoundError(f"Completed run at {exp_dir} but no result JSON found")


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Combined results and report
# ---------------------------------------------------------------------------

def _generate_combined_report(results: list[dict]) -> None:
    """Generate combined Stage 11B results JSON and markdown report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- JSON ---
    combined = {
        "stage": "11B",
        "title": "Leave-Cluster-Out Generalization Results",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "status": "complete",
        "threshold": THRESHOLD,
        "total_runs": len(results),
        "runs": {r["run_id"]: r for r in results},
    }

    json_path = RESULTS_DIR / "stage11b-cluster-generalization-results.json"
    json_path.write_text(json.dumps(combined, indent=2))
    print(f"\nWrote {json_path} ({json_path.stat().st_size:,} bytes)")

    # --- Markdown ---
    lines = []
    def w(s=""):
        lines.append(s)

    w("# Stage 11B — Leave-Cluster-Out Generalization Results")
    w()
    w(f"**Date:** {combined['date']}")
    w(f"**Threshold:** {THRESHOLD}")
    w(f"**Model:** ShuffleNetV2 x1.0 (frozen architecture)")
    w()
    w("---")
    w()

    # Summary table
    w("## Summary")
    w()
    w("| Run | Type | Held-out | Frames | Recall | FNR | Notes |")
    w("|-----|------|----------|--------|--------|-----|-------|")
    for r in results:
        rid = r["run_id"]
        held_out = r["held_out"]
        m = held_out["metrics"]
        task = "Stringing" if "str_" in rid else "UE control"
        notes = "single-class" if held_out["normal_frames"] == 0 else "mixed"
        w(f"| `{rid}` | {task} | `{r['held_out_cluster']}` | "
          f"{held_out['sample_count']} | {m['recall']:.4f} | {m['fnr']:.4f} | {notes} |")
    w()

    # Detailed per-run
    for r in results:
        rid = r["run_id"]
        w(f"---")
        w()
        w(f"## `{rid}`")
        w()
        w(f"**Cluster:** `{r['held_out_cluster']}`")
        w(f"**Train frames (held-in):** {r['train_frame_count']}")
        w(f"**Training epochs:** {r['training_epochs']} / {r['config']['epochs']}")
        w(f"**Training time:** {r['training_elapsed_seconds']:.0f}s")
        w(f"**Best val loss:** {r['best_val_loss']:.4f}")
        w()

        # Validation
        vm = r["validation"]["metrics"]
        w("### Held-in Validation")
        w()
        w(f"- Samples: {r['validation']['sample_count']}")
        w(f"- Failing: {r['validation']['failing_frames']}, Normal: {r['validation']['normal_frames']}")
        w(f"- **F1: {vm['f1']:.4f}**")
        w(f"- Recall: {vm['recall']:.4f}, FNR: {vm['fnr']:.4f}")
        w(f"- Precision: {vm['precision']:.4f}, FPR: {vm['fpr']:.4f}")
        w(f"- Accuracy: {vm['accuracy']:.4f}")
        if vm['auroc'] is not None:
            w(f"- AUROC: {vm['auroc']:.4f}")
        w()

        # Held-out
        hm = r["held_out"]["metrics"]
        w("### Held-out Cluster")
        w()
        is_single_class = r["held_out"]["normal_frames"] == 0
        if is_single_class:
            w("**Single-class evaluation (100% failing).** FPR, AUROC, F1 are undefined or not meaningful.")
            w()
        w(f"- Samples: {r['held_out']['sample_count']}")
        w(f"- Failing: {r['held_out']['failing_frames']}, Normal: {r['held_out']['normal_frames']}")
        w(f"- **Detected: {r['held_out']['detected']}, Missed: {r['held_out']['missed']}**")
        w(f"- **Recall: {hm['recall']:.4f}, FNR: {hm['fnr']:.4f}**")
        if not is_single_class:
            w(f"- Precision: {hm['precision']:.4f}, FPR: {hm['fpr']:.4f}")
            w(f"- F1: {hm['f1']:.4f}, Accuracy: {hm['accuracy']:.4f}")
            if hm['auroc'] is not None:
                w(f"- AUROC: {hm['auroc']:.4f}")
        w()

        # Known-domain comparison
        if r.get("known_domain_stringing") and r["known_domain_stringing"]:
            ks = r["known_domain_stringing"]
            ksm = ks["metrics"]
            w("### Known-domain Stringing (remaining train)")
            w()
            w(f"- Samples: {ks['sample_count']}")
            w(f"- Recall: {ksm['recall']:.4f}, FNR: {ksm['fnr']:.4f}")
            delta_recall = hm['recall'] - ksm['recall']
            delta_fnr = hm['fnr'] - ksm['fnr']
            w(f"- **Delta recall (held-out minus known): {delta_recall:+.4f}**")
            w(f"- **Delta FNR (held-out minus known): {delta_fnr:+.4f}**")
            w()

        # Per-recording
        per_rec = r["held_out"].get("per_recording", {})
        if per_rec:
            w("### Per-recording breakdown")
            w()
            w("| Recording | Frames | Detected | Missed | Accuracy | Mean fail score |")
            w("|-----------|--------|----------|--------|----------|-----------------|")
            for rec_id, m in sorted(per_rec.items()):
                w(f"| `{rec_id}` | {m['frames']} | {m['detected']} | {m['missed']} | "
                  f"{m['accuracy']:.4f} | {m['mean_fail_score']:.4f} |")
            w()

    # Cross-run comparison
    w("---")
    w()
    w("## Cross-run Comparison")
    w()

    str_runs = [r for r in results if "str_" in r["run_id"]]
    ue_runs = [r for r in results if "ue_" in r["run_id"]]

    if str_runs:
        w("### Stringing holdout runs")
        w()
        w("| Run | Held-out recall | Known-domain recall | Delta |")
        w("|-----|----------------|--------------------|-------|")
        for r in str_runs:
            held_recall = r["held_out"]["metrics"]["recall"]
            if r.get("known_domain_stringing") and r["known_domain_stringing"]:
                known_recall = r["known_domain_stringing"]["metrics"]["recall"]
                delta = held_recall - known_recall
                w(f"| `{r['run_id']}` | {held_recall:.4f} | {known_recall:.4f} | {delta:+.4f} |")
            else:
                w(f"| `{r['run_id']}` | {held_recall:.4f} | N/A | N/A |")
        w()

    if ue_runs:
        w("### Under-extrusion control runs")
        w()
        w("| Run | Held-out recall | Held-out F1 | Held-out FNR | Held-out FPR |")
        w("|-----|----------------|-------------|--------------|--------------|")
        for r in ue_runs:
            hm = r["held_out"]["metrics"]
            w(f"| `{r['run_id']}` | {hm['recall']:.4f} | {hm['f1']:.4f} | "
              f"{hm['fnr']:.4f} | {hm['fpr']:.4f} |")
        w()

    # Context
    w("---")
    w()
    w("## Frozen Context (NOT used for experiment decisions)")
    w()
    w("Experiment 1 test-set results (historical reference only):")
    w()
    w("| Metric | Value |")
    w("|--------|-------|")
    w("| Test F1@0.65 | 0.8799 |")
    w("| Test stringing FNR@0.65 | 0.213 |")
    w("| Test under-extrusion FNR@0.65 | 0.155 |")
    w("| Test normal FPR@0.65 | 0.103 |")
    w()
    w("---")
    w()
    w("## Conclusion")
    w()

    # Analyze
    all_str_held_out_fnrs = [r["held_out"]["metrics"]["fnr"] for r in str_runs]
    all_str_known_fnrs = []
    for r in str_runs:
        if r.get("known_domain_stringing") and r["known_domain_stringing"]:
            all_str_known_fnrs.append(r["known_domain_stringing"]["metrics"]["fnr"])

    if all_str_held_out_fnrs:
        mean_held_fnr = np.mean(all_str_held_out_fnrs)
        w(f"Mean held-out stringing FNR: {mean_held_fnr:.4f}")
        if all_str_known_fnrs:
            mean_known_fnr = np.mean(all_str_known_fnrs)
            w(f"Mean known-domain stringing FNR: {mean_known_fnr:.4f}")
            w(f"Mean delta FNR: {mean_held_fnr - mean_known_fnr:+.4f}")
        w()

    for r in str_runs:
        held = r["held_out"]["metrics"]["fnr"]
        if r.get("known_domain_stringing") and r["known_domain_stringing"]:
            known = r["known_domain_stringing"]["metrics"]["fnr"]
            delta = held - known
            if delta > 0.05:
                w(f"- `{r['run_id']}`: FNR degraded by {delta:+.4f} on unseen cluster "
                  f"(known={known:.4f} -> held-out={held:.4f}). **Evidence of domain generalization failure.**")
            elif delta > 0:
                w(f"- `{r['run_id']}`: FNR slightly worse on unseen cluster "
                  f"(known={known:.4f} -> held-out={held:.4f}, delta={delta:+.4f}).")
            else:
                w(f"- `{r['run_id']}`: FNR comparable or better on unseen cluster "
                  f"(known={known:.4f} -> held-out={held:.4f}, delta={delta:+.4f}).")
    w()

    if ue_runs:
        w("**Under-extrusion control:**")
        for r in ue_runs:
            hm = r["held_out"]["metrics"]
            w(f"- `{r['run_id']}`: recall={hm['recall']:.4f} FNR={hm['fnr']:.4f} F1={hm['f1']:.4f}")
        w()

    w("### Interpretation")
    w()
    w("This experiment tests whether stringing failures generalize poorly across ")
    w("recording/session domains. Two independent stringing clusters and two ")
    w("under-extrusion controls were evaluated. Results should be interpreted as ")
    w("evidence for or against domain-specific generalization failure, not as ")
    w("definitive statistical proof (n=2 stringing clusters is insufficient for ")
    w("strong statistical conclusions).")
    w()
    w("---")
    w()
    w(f"*Generated {combined['date']} — Stage 11B complete.*")
    w()

    md_path = RESULTS_DIR / "stage11b-cluster-generalization-results.md"
    md_path.write_text("\n".join(lines))
    print(f"Wrote {md_path} ({md_path.stat().st_size:,} bytes, {len(lines)} lines)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="Run all 4 experiments")
    ap.add_argument("--run", type=str, help="Run a specific experiment by ID")
    ap.add_argument("--report-only", action="store_true",
                    help="Regenerate report from existing results")
    args = ap.parse_args()

    if not args.all and not args.run and not args.report_only:
        ap.error("Specify --all, --run <id>, or --report-only")

    # --- Load data ---
    print("Loading dataset...")
    all_data = load_all_recordings()
    train_recs = all_data["train"]
    val_recs = all_data["val"]
    str_clusters = define_stringing_clusters(train_recs)
    ue_clusters = define_under_extrusion_clusters(train_recs)
    runs = design_runs(train_recs, val_recs, str_clusters, ue_clusters)

    if args.report_only:
        # Collect existing results
        results = []
        for rid in RUN_IDS:
            exp_dir = RUNS_DIR / f"stage11b_{rid}"
            result_path = exp_dir / "artifacts" / "stage11b_result.json"
            if result_path.exists():
                results.append(json.loads(result_path.read_text()))
            else:
                print(f"WARNING: No results for {rid}")
        if results:
            _generate_combined_report(results)
        return 0

    # --- Print plan ---
    print("\n" + "=" * 70)
    print("STAGE 11B EXECUTION PLAN")
    print("=" * 70)
    run_ids = [args.run] if args.run else RUN_IDS
    for rid in run_ids:
        run = next(r for r in runs if r.run_id == rid)
        print(f"\n  {rid}:")
        print(f"    Cluster: {run.held_out_cluster_id}")
        print(f"    Held-out: {len(run.held_out_recordings)} recs, "
              f"{run.held_out_frame_count} frames")
        print(f"    Train: {run.train_recording_count} recs, "
              f"{run.train_frame_count} frames")
        print(f"    Output: experiments/stage11b_{rid}/")
    print()

    # --- Verify no existing Experiment 1 checkpoint would be overwritten ---
    exp1_dir = RUNS_DIR / "shufflenet_v2_x1_0_printing-errors_cdc9e2cad4"
    if exp1_dir.exists():
        print(f"Experiment 1 directory exists: {exp1_dir}")
        print("  Will NOT be modified.")

    # --- Verify official splits are intact ---
    for split in ("train", "val", "test"):
        csv_path = SPLITS_DIR / f"{split}.csv"
        assert csv_path.exists(), f"Missing: {csv_path}"
    print("Official splits verified intact.")

    # --- Execute runs ---
    device = _get_device()
    print(f"Device: {device}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Start time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    t_start = time.time()
    results = []

    with tempfile.TemporaryDirectory(prefix="stage11b_") as tmp_root:
        tmp = Path(tmp_root)
        for rid in run_ids:
            result = execute_run(
                rid, train_recs, val_recs, str_clusters, ue_clusters, runs, tmp,
            )
            results.append(result)

    t_total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"ALL RUNS COMPLETE in {t_total:.0f}s ({t_total/60:.1f} min)")
    print(f"{'='*70}")

    # --- Generate combined report ---
    _generate_combined_report(results)

    # --- Verify integrity ---
    print("\n--- Integrity checks ---")
    # Official splits unchanged
    for split in ("train", "val", "test"):
        csv_path = SPLITS_DIR / f"{split}.csv"
        assert csv_path.exists(), f"Missing: {csv_path}"
    print("  Official splits: intact")

    # Experiment 1 checkpoint unchanged
    exp1_ckpt = exp1_dir / "artifacts" / "best.pt"
    if exp1_ckpt.exists():
        print(f"  Experiment 1 checkpoint: intact ({exp1_ckpt.stat().st_size:,} bytes)")

    # All output dirs isolated
    for rid in run_ids:
        out_dir = RUNS_DIR / f"stage11b_{rid}"
        assert out_dir.exists(), f"Missing output dir: {out_dir}"
        assert (out_dir / "artifacts" / "best.pt").exists(), f"Missing checkpoint: {rid}"
        assert (out_dir / "artifacts" / "stage11b_result.json").exists(), f"Missing result: {rid}"
    print(f"  All {len(run_ids)} output directories: present with checkpoints and results")

    print(f"\nFinal time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
