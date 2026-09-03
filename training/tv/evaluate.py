"""Model evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .data.loaders import get_classification_loaders
from .metrics import compute_classification_metrics


def evaluate_classification(
    checkpoint_path: Path,
    config: ExperimentConfig,
) -> dict:
    """Evaluate a classification model on the test split.

    Loads the checkpoint, runs inference on the test set, computes all
    configured metrics, and returns the results dict.
    """
    split_dir = Path(config.dataset.data_dir) / "splits"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = ckpt.get("num_classes", 2)
    model_config_name = ckpt.get("config_model", config.model)

    from .models.factory import build_classifier

    model = build_classifier(model_config_name, num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Test loader
    test_loader, class_to_idx = get_classification_loaders(
        split_dir,
        "test",
        image_size=config.dataset.image_size,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=False,
        augmentation={},
    )

    # Inference
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_scores: list[float] = []
    threshold = config.evaluation.decision_threshold

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            if num_classes == 2:
                scores = probs[:, 1].cpu().numpy()
                preds = (scores >= threshold).astype(int)
                all_scores.extend(scores.tolist())
                all_preds.extend(preds.tolist())
            else:
                preds = probs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_scores.extend(probs.max(dim=1).values.cpu().numpy().tolist())

            all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_scores = np.array(all_scores) if all_scores else None

    metrics = compute_classification_metrics(y_true, y_pred, y_scores)
    result = metrics.to_dict()
    result["class_to_idx"] = class_to_idx
    result["checkpoint"] = str(checkpoint_path)

    return result


def evaluate_detection(
    model_path: Path,
    config: ExperimentConfig,
) -> dict:
    """Evaluate a YOLO detection model."""
    from ultralytics import YOLO

    data_yaml = Path(config.dataset.data_dir) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"YOLO data config not found: {data_yaml}")

    model = YOLO(str(model_path))
    results = model.val(data=str(data_yaml), imgsz=config.dataset.image_size)

    return {
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
        "precision": float(results.box.p),
        "recall": float(results.box.r),
        "f1": float(results.box.f1),
        "support": int(results.box.mp_count) if hasattr(results.box, "mp_count") else 0,
        "checkpoint": str(model_path),
    }


def evaluate(
    config: ExperimentConfig,
    checkpoint_path: str | Path | None = None,
    base_dir: str | Path = "experiments",
) -> dict:
    """Main evaluation entry point."""
    if checkpoint_path is None:
        exp_dir = Path(base_dir) / config.experiment_id
        checkpoint_path = exp_dir / "artifacts" / "best.pt"
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if config.task == "classification":
        return evaluate_classification(checkpoint_path, config)
    elif config.task == "detection":
        return evaluate_detection(checkpoint_path, config)
    else:
        raise ValueError(f"Unknown task: {config.task}")
