"""Architecture-agnostic model registry."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelSpec:
    name: str
    task: str  # "classification" or "detection"
    family: str  # "torchvision" or "ultralytics"
    license: str
    default_image_size: int
    pretrained: bool = True


MODEL_REGISTRY: dict[str, ModelSpec] = {
    # --- Classification (torchvision) ---
    # NOTE: this is standard EfficientNet-B0 from torchvision, NOT the TF
    # EfficientNet-Lite0 family. torchvision ships no "lite" variant. If Lite0
    # is required, a separate builder must be added (e.g. via timm).
    "efficientnet_b0": ModelSpec(
        name="efficientnet_b0",
        task="classification",
        family="torchvision",
        license="BSD-3-Clause",
        default_image_size=224,
        # Rationale: EfficientNet-B0 was a research candidate (best ImageNet
        # acc@1 among the Apache/Apache-family candidates). Used as a strong
        # accuracy baseline; a true Lite0 builder is TBD.
    ),
    "resnet18": ModelSpec(
        name="resnet18",
        task="classification",
        family="torchvision",
        license="BSD-3-Clause",
        default_image_size=224,
        # Rationale: ubiquitous small-Baseline reference architecture included
        # to bound the accuracy/latency trade-off curve.
    ),
    "mobilenet_v3_small": ModelSpec(
        name="mobilenet_v3_small",
        task="classification",
        family="torchvision",
        license="BSD-3-Clause",
        default_image_size=224,
        # Rationale: strong edge-friendly candidate (high accuracy-to-cost ratio)
        # and good small-model representative for the edge requirement.
    ),
    "mobilenet_v3_large": ModelSpec(
        name="mobilenet_v3_large",
        task="classification",
        family="torchvision",
        license="BSD-3-Clause",
        default_image_size=224,
        # Rationale: higher-accuracy sibling of mobilenet_v3_small; included to
        # measure the accuracy/latency trade-off within one family.
    ),
    "shufflenet_v2_x1_0": ModelSpec(
        name="shufflenet_v2_x1_0",
        task="classification",
        family="torchvision",
        license="BSD-3-Clause",
        default_image_size=224,
        # Rationale: PrintGuard, the strongest open-source competitor, uses
        # ShuffleNetV2 as its backbone. Included for an apples-to-apples measure.
    ),
    # --- Classification (ultralytics) ---
    "yolov8n-cls": ModelSpec(
        name="yolov8n-cls",
        task="classification",
        family="ultralytics",
        license="AGPL-3.0",
        default_image_size=224,
        # Rationale: fastest classification candidate (research RPi5 ~10ms);
        # AGPL-3.0 constrains distribution, so treated as an upper speed bound.
    ),
    "yolov11n-cls": ModelSpec(
        name="yolov11n-cls",
        task="classification",
        family="ultralytics",
        license="AGPL-3.0",
        default_image_size=224,
        # Rationale: newest tiny YOLO classifier; compare against v8n-cls.
    ),
    # --- Detection (ultralytics) ---
    "yolov8n": ModelSpec(
        name="yolov8n",
        task="detection",
        family="ultralytics",
        license="AGPL-3.0",
        default_image_size=640,
        # Rationale: detection experiment path (localize the defect box).
    ),
    "yolov11n": ModelSpec(
        name="yolov11n",
        task="detection",
        family="ultralytics",
        license="AGPL-3.0",
        default_image_size=640,
        # Rationale: newest tiny detector for the detection experiment path.
    ),
}


def list_models(task: str | None = None) -> list[str]:
    """List registered model names, optionally filtered by task."""
    if task:
        return [name for name, spec in MODEL_REGISTRY.items() if spec.task == task]
    return list(MODEL_REGISTRY.keys())


def get_model_spec(name: str) -> ModelSpec:
    """Get model specification by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]


def build_classifier(name: str, num_classes: int, pretrained: bool = True):
    """Build a torchvision classification model.

    Returns an ``nn.Module`` with the final layer replaced for ``num_classes``.
    """
    import torch.nn as nn
    import torchvision.models as models

    spec = get_model_spec(name)
    if spec.family != "torchvision":
        raise ValueError(
            f"Model {name} is not a torchvision classifier. "
            f"Use build_ultralytics() for {spec.family} models."
        )

    model_fn = getattr(models, name)
    weights = "DEFAULT" if pretrained else None
    model = model_fn(weights=weights)

    # Replace the final classifier layer
    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Cannot find classifier layer for {name}")

    return model
