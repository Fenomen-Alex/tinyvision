"""Data loaders for classification and detection tasks."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class _AddGaussianNoise:
    """Additive Gaussian noise on a ``[0, 1]`` float tensor (train-only)."""

    def __init__(self, std: float):
        self.std = float(std)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return tensor
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0.0, 1.0)


def normalize_stats(augmentation: dict | None = None) -> tuple[list[float], list[float]]:
    """Return the (mean, std) normalisation used by :func:`_build_transforms`.

    Custom statistics (``normalize_mean``/``normalize_std``) take precedence;
    otherwise ImageNet statistics are used. A single source of truth keeps the
    training and inference transforms identical.
    """
    aug = augmentation or {}
    mean = aug.get("normalize_mean")
    std = aug.get("normalize_std")
    if mean and std:
        return list(mean), list(std)
    return list(IMAGENET_MEAN), list(IMAGENET_STD)


def _build_transforms(
    image_size: int,
    augmentation: dict | None = None,
    is_train: bool = True,
) -> transforms.Compose:
    """Build image transforms from config.

    Custom ``normalize_mean``/``normalize_std`` (fit on training data) replace
    the ImageNet defaults for *both* training and evaluation, so deterministic
    preprocessing is identical across train/val/inference. Appearance
    augmentations (``appear_*``, ``blur_*``, ``noise_std``) apply only when
    ``is_train``.
    """
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

        ab = aug.get("appear_brightness", 0.0)
        ac = aug.get("appear_contrast", 0.0)
        asat = aug.get("appear_saturation", 0.0)
        ah = aug.get("appear_hue", 0.0)
        if any(v for v in (ab, ac, asat, ah)):
            ops.append(
                transforms.ColorJitter(
                    brightness=ab, contrast=ac, saturation=asat, hue=ah
                )
            )

        blur_prob = aug.get("blur_prob", 0.0)
        blur_sigma = aug.get("blur_sigma", 0.0)
        if blur_prob > 0 and blur_sigma > 0:
            ops.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, blur_sigma))],
                    p=blur_prob,
                )
            )

        if aug.get("random_crop", False):
            ops.append(transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)))

    ops.append(transforms.Resize((image_size, image_size)))
    ops.append(transforms.ToTensor())

    noise_std = aug.get("noise_std", 0.0)
    if is_train and noise_std > 0:
        ops.append(_AddGaussianNoise(noise_std))

    if aug.get("normalize", True):
        mean, std = normalize_stats(aug)
        ops.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(ops)


class CSVImageDataset(Dataset):
    """Dataset that reads image paths and labels from a CSV file."""

    def __init__(self, csv_path: Path, data_dir: Path | None = None,
                 transform: transforms.Compose | None = None):
        self.samples: list[tuple[str, int]] = []
        self.class_to_idx: dict[str, int] = {}
        self.transform = transform
        self.data_dir = data_dir
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
                path = row["path"]
                # Resolve relative paths against data_dir
                if self.data_dir and not Path(path).is_absolute():
                    path = str(self.data_dir / path)
                self.samples.append((path, self.class_to_idx[row["label"]]))

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

    # Read data_dir from splits.json to resolve relative paths
    import json as jsonmod
    splits_json = split_dir / "splits.json"
    data_dir = None
    if splits_json.exists():
        with open(splits_json) as f:
            manifest = jsonmod.load(f)
        data_dir = Path(manifest["data_dir"])

    is_train = split == "train"
    tfm = _build_transforms(image_size, augmentation or {}, is_train=is_train)
    dataset = CSVImageDataset(csv_path, data_dir=data_dir, transform=tfm)

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
