"""Data loaders for classification and detection tasks."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def _build_transforms(
    image_size: int,
    augmentation: dict | None = None,
    is_train: bool = True,
) -> transforms.Compose:
    """Build image transforms from config."""
    aug = augmentation or {}
    ops: list[transforms.Transform] = []

    if is_train:
        if aug.get("horizontal_flip", True):
            ops.append(transforms.RandomHorizontalFlip())
        rot = aug.get("random_rotation", 15)
        if rot:
            ops.append(transforms.RandomRotation(rot))
        cj = aug.get("color_jitter", 0.2)
        if cj:
            ops.append(transforms.ColorJitter(brightness=cj, contrast=cj, saturation=cj))
        if aug.get("random_crop", False):
            ops.append(transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)))

    ops.append(transforms.Resize((image_size, image_size)))
    ops.append(transforms.ToTensor())

    if aug.get("normalize", True):
        ops.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    return transforms.Compose(ops)


class CSVImageDataset(Dataset):
    """Dataset that reads image paths and labels from a CSV file."""

    def __init__(self, csv_path: Path, transform: transforms.Compose | None = None):
        self.samples: list[tuple[str, int]] = []
        self.class_to_idx: dict[str, int] = {}
        self.transform = transform
        self._load(csv_path)

    def _load(self, csv_path: Path) -> None:
        import csv as csvmod

        with open(csv_path) as f:
            reader = csvmod.DictReader(f)
            labels = sorted({row["label"] for row in reader})
            self.class_to_idx = {label: i for i, label in enumerate(labels)}

        with open(csv_path) as f:
            reader = csvmod.DictReader(f)
            for row in reader:
                self.samples.append((row["path"], self.class_to_idx[row["label"]]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        from PIL import Image

        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_classification_loaders(
    split_dir: Path,
    split: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    augmentation: dict | None = None,
    generator: torch.Generator | None = None,
    worker_init_fn=None,
) -> tuple[DataLoader, dict[str, int]]:
    """Get a DataLoader for a given split.

    Args:
        split_dir: Directory containing ``splits.json`` and per-split CSVs.
        split: One of ``train``, ``val``, ``test``.
        image_size: Resize target.
        batch_size: Batch size.
        num_workers: Data loading workers.
        pin_memory: Pin CUDA memory.
        augmentation: Augmentation config dict.
        generator: Optional ``torch.Generator`` for reproducible shuffling.
        worker_init_fn: Optional callable to seed each worker deterministically.

    Returns:
        (DataLoader, class_to_idx mapping)
    """
    csv_path = split_dir / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    is_train = split == "train"
    tfm = _build_transforms(image_size, augmentation or {}, is_train=is_train)
    dataset = CSVImageDataset(csv_path, transform=tfm)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    return loader, dataset.class_to_idx
