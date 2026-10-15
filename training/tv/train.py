"""Classification and detection training."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .experiment import ExperimentManager
from .models.factory import build_classifier, get_model_spec


def set_seed(seed: int) -> torch.Generator:
    """Set all relevant RNG seeds for reproducible training.

    Returns a ``torch.Generator`` to pass to DataLoaders so worker shuffling
    is deterministic as well.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.Generator().manual_seed(seed)


class _WorkerInit:
    """Picklable worker_init_fn that seeds each DataLoader worker deterministically."""

    def __init__(self, base_seed: int):
        self.base_seed = base_seed

    def __call__(self, worker_id: int) -> None:
        import random
        np.random.seed(self.base_seed + worker_id)
        random.seed(self.base_seed + worker_id)
        torch.manual_seed(self.base_seed + worker_id)


def make_worker_init(base_seed: int):
    """Return a worker_init_fn that seeds each DataLoader worker deterministically."""
    return _WorkerInit(base_seed)


def train_classification(
    config: ExperimentConfig,
    exp_manager: ExperimentManager,
) -> dict:
    """Train a torchvision classification model.

    Returns a dict with training history and final metrics.
    """
    from .data.loaders import get_classification_loaders

    spec = get_model_spec(config.model)
    split_dir = Path(config.dataset.data_dir) / "splits"
    image_size = config.dataset.image_size
    aug = config.train.augmentation.__dict__

    generator = set_seed(config.train.seed)
    worker_seed_base = (config.train.seed * 100003) % (2**31)

    # Data
    train_loader, class_to_idx = get_classification_loaders(
        split_dir,
        "train",
        image_size=image_size,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        augmentation=aug,
        generator=generator,
        worker_init_fn=make_worker_init(worker_seed_base),
    )
    val_loader, _ = get_classification_loaders(
        split_dir,
        "val",
        image_size=image_size,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        augmentation=aug,
        generator=generator,
        worker_init_fn=make_worker_init(worker_seed_base),
    )

    num_classes = len(class_to_idx)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Model
    model = build_classifier(config.model, num_classes, pretrained=True)
    model = model.to(device)

    # Optimizer
    if config.train.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
    elif config.train.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.train.learning_rate,
            momentum=0.9,
            weight_decay=config.train.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.train.optimizer}")

    criterion = nn.CrossEntropyLoss()
    use_amp = config.train.precision in ("fp16", "bf16")
    amp_dtype = torch.bfloat16 if config.train.precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(enabled=use_amp) if use_amp else None

    # Training loop
    history: list[dict] = []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.train.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(record)
        print(
            f"Epoch {epoch + 1:3d}/{config.train.epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            exp_manager.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "class_to_idx": class_to_idx,
                    "config_model": config.model,
                    "num_classes": num_classes,
                    # Deterministic preprocessing: inference must reuse the exact
                    # normalisation the model was trained with.
                    "normalize_mean": config.train.augmentation.normalize_mean,
                    "normalize_std": config.train.augmentation.normalize_std,
                    "image_size": image_size,
                },
                filename="best.pt",
            )
        else:
            patience_counter += 1

        if patience_counter >= config.train.early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return {"history": history, "best_val_loss": best_val_loss, "class_to_idx": class_to_idx}


def train_detection(config: ExperimentConfig, exp_manager: ExperimentManager) -> dict:
    """Train a YOLO detection model via ultralytics."""
    from ultralytics import YOLO

    data_yaml = Path(config.dataset.data_dir) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"YOLO data config not found: {data_yaml}")

    model = YOLO(f"{config.model}.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=config.train.epochs,
        imgsz=config.dataset.image_size,
        batch=config.data.batch_size,
        lr0=config.train.learning_rate,
        seed=config.train.seed,
        project=str(exp_manager.artifacts_dir),
        name="detection",
        exist_ok=True,
    )
    return {"results": str(results)}


def train(config: ExperimentConfig, base_dir: str | Path = "experiments") -> dict:
    """Main training entry point."""
    exp = ExperimentManager(config, base_dir=base_dir)
    with exp.run() as manager:
        if config.task == "classification":
            return train_classification(config, manager)
        elif config.task == "detection":
            return train_detection(config, manager)
        else:
            raise ValueError(f"Unknown task: {config.task}")
