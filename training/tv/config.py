"""Experiment configuration schema and loading."""
from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import yaml


@dataclass
class DatasetConfig:
    name: str = ""
    version: str = "1.0"
    data_dir: str = "datasets/processed/default"
    image_size: int = 224
    source_mode: str = "directory"  # "directory" or "filename"
    split_ratios: list[float] = field(default_factory=lambda: [0.7, 0.15, 0.15])


@dataclass
class DataConfig:
    num_workers: int = 4
    batch_size: int = 32
    pin_memory: bool = True


@dataclass
class AugmentationConfig:
    horizontal_flip: bool = True
    random_rotation: int = 15
    color_jitter: float = 0.2
    random_crop: bool = False
    normalize: bool = True
    # Custom per-channel normalisation (fit on training data). ``None`` keeps
    # the ImageNet defaults, so leaving these unset reproduces old behaviour.
    normalize_mean: list[float] | None = None
    normalize_std: list[float] | None = None
    # Train-only appearance augmentations (applied on top of the baseline).
    appear_brightness: float = 0.0
    appear_contrast: float = 0.0
    appear_saturation: float = 0.0
    appear_hue: float = 0.0
    blur_prob: float = 0.0
    blur_sigma: float = 0.0
    noise_std: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 50
    optimizer: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    precision: str = "fp32"
    seed: int = 42
    early_stopping_patience: int = 10


@dataclass
class EvalConfig:
    metrics: list[str] = field(
        default_factory=lambda: [
            "precision",
            "recall",
            "f1",
            "fnr",
            "fpr",
            "auroc",
        ]
    )
    decision_threshold: float = 0.5


@dataclass
class ExportConfig:
    format: str = "onnx"
    quantization: str = "int8"
    dynamic_axes: bool = False


@dataclass
class HardwareConfig:
    machine: str = ""
    gpu: str = ""


@dataclass
class SoftwareConfig:
    python: str = ""
    torch: str = ""
    torchvision: str = ""
    ultralytics: str = ""
    opencv: str = ""
    os: str = ""


@dataclass
class ExperimentConfig:
    experiment_id: str = ""
    model: str = "efficientnet_b0"
    task: str = "classification"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    software: SoftwareConfig = field(default_factory=SoftwareConfig)
    created_at: str = ""
    notes: str = ""


def record_software() -> SoftwareConfig:
    import cv2
    import torchvision

    # ultralytics is an optional extra (AGPL-3.0); the validated torchvision
    # path does not need it, so record an empty string when it is absent.
    try:
        import ultralytics

        ultralytics_version = str(ultralytics.__version__)
    except ImportError:
        ultralytics_version = ""

    # ``str(...)`` is required: some versions (e.g. ``torch.__version__``, a
    # ``torch.torch_version.TorchVersion``) are ``str`` subclasses that PyYAML
    # would otherwise serialize with a ``!!python/object/new:`` tag, producing a
    # config that ``yaml.safe_load`` (used by :func:`load_config`) cannot read.
    return SoftwareConfig(
        python=str(sys.version.split()[0]),
        torch=str(__import__("torch").__version__),
        torchvision=str(torchvision.__version__),
        ultralytics=ultralytics_version,
        opencv=str(cv2.__version__),
        os=str(f"{platform.system()} {platform.release()}"),
    )


def record_hardware() -> HardwareConfig:
    import torch

    machine = f"{platform.machine()} ({platform.processor()})"
    gpu = "None"
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return HardwareConfig(machine=machine, gpu=gpu)


def _experiment_identity(config: "ExperimentConfig") -> str:
    """Return a canonical string capturing every knob that determines the result.

    Environmental metadata (hardware/software/created_at) is intentionally
    excluded so the ID is stable across machines and lets ``evaluate``/``export``
    locate the checkpoint produced by ``train``.
    """
    a = config.train.augmentation
    train_blob = "|".join([
        str(config.train.epochs), str(config.train.optimizer),
        str(config.train.learning_rate), str(config.train.weight_decay),
        str(config.train.precision), str(config.train.seed),
        str(config.train.early_stopping_patience),
        str(a.horizontal_flip), str(a.random_rotation), str(a.color_jitter),
        str(a.random_crop), str(a.normalize),
        str(a.normalize_mean), str(a.normalize_std),
        str(a.appear_brightness), str(a.appear_contrast),
        str(a.appear_saturation), str(a.appear_hue),
        str(a.blur_prob), str(a.blur_sigma), str(a.noise_std),
    ])
    data_blob = "|".join([
        str(config.data.batch_size), str(config.data.num_workers),
        str(config.data.pin_memory),
    ])
    ds_blob = "|".join([
        str(config.dataset.name), str(config.dataset.version),
        str(config.dataset.data_dir), str(config.dataset.image_size),
    ])
    eval_blob = str(config.evaluation.decision_threshold)
    export_blob = "|".join([
        str(config.export.format), str(config.export.quantization),
        str(config.export.dynamic_axes),
    ])
    return "|".join([
        config.model, config.task, ds_blob, data_blob, train_blob,
        eval_blob, export_blob,
    ])


def _generate_experiment_id(config: "ExperimentConfig") -> str:
    """Derive a deterministic, content-unique experiment ID.

    Hashing the full experiment-determining config (not just model+dataset)
    guarantees that experiments differing in any hyperparameter get distinct
    IDs, so they cannot overwrite each other. Re-loading the same config yields
    the same ID, letting ``evaluate`` and ``export`` locate the checkpoint.
    """
    import hashlib

    content = _experiment_identity(config)
    digest = hashlib.sha1(content.encode()).hexdigest()[:10]
    ds = config.dataset.name.replace("/", "_")
    return f"{config.model}_{ds}_{digest}"


def _dict_to_dataclass(cls, d: dict):
    """Recursively convert a dict to a nested dataclass instance."""
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        return d
    fieldtypes = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for name, ftype in fieldtypes.items():
        val = d.get(name)
        if val is None:
            continue
        # Resolve forward references
        import typing

        actual = typing.get_type_hints(cls).get(name, ftype)
        if dataclasses.is_dataclass(actual) and isinstance(val, dict):
            kwargs[name] = _dict_to_dataclass(actual, val)
        else:
            kwargs[name] = val
    return cls(**kwargs)


class _ConfigLoader(yaml.SafeLoader):
    """SafeLoader that also understands the legacy ``TorchVersion`` tag.

    Configurations written before software versions were coerced to ``str``
    serialized ``torch.__version__`` (a ``str`` subclass) as
    ``!!python/object/new:torch.torch_version.TorchVersion``. This maps exactly
    that one tag back to the wrapped version string; nothing is imported or
    executed, and every other unsafe tag still raises.
    """


def _construct_legacy_torch_version(loader: yaml.SafeLoader, node: yaml.Node) -> str:
    values = loader.construct_sequence(node, deep=True)
    return str(values[0]) if values else ""


_ConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/object/new:torch.torch_version.TorchVersion",
    _construct_legacy_torch_version,
)


def load_config(path: Path) -> ExperimentConfig:
    with open(path) as f:
        raw = yaml.load(f, Loader=_ConfigLoader) or {}
    config = _dict_to_dataclass(ExperimentConfig, raw)
    if not config.software.python:
        config.software = record_software()
    if not config.hardware.machine:
        config.hardware = record_hardware()
    if not config.experiment_id:
        config.experiment_id = _generate_experiment_id(config)
    config.created_at = datetime.now().isoformat()
    return config


def save_config(config: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        # ``safe_dump`` guarantees the file is readable by ``load_config`` and
        # fails loudly if a non-plain type slips in, instead of emitting a
        # ``!!python/object`` tag that ``safe_load`` would reject.
        yaml.safe_dump(asdict(config), f, default_flow_style=False, sort_keys=False)
