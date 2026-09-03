"""ONNX export and INT8 quantization."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig


def export_onnx(
    checkpoint_path: Path,
    output_path: Path,
    num_classes: int,
    image_size: int = 224,
) -> dict:
    """Export a torchvision classifier to ONNX."""
    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt.get("config_model", "efficientnet_b0")

    from .models.factory import build_classifier

    model = build_classifier(model_name, num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=17,
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )

    size_kb = output_path.stat().st_size / 1024

    # Benchmark latency on CPU
    import onnxruntime as ort

    sess = ort.InferenceSession(str(output_path))
    dummy_np = dummy.numpy()
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        sess.run(None, {"input": dummy_np})
        latencies.append(time.perf_counter() - t0)
    avg_ms = np.mean(latencies[10:]) * 1000  # skip warmup

    return {
        "format": "onnx",
        "path": str(output_path),
        "size_kb": round(size_kb, 1),
        "avg_latency_ms": round(avg_ms, 2),
        "image_size": image_size,
        "num_classes": num_classes,
    }


def quantize_int8(
    onnx_path: Path,
    output_path: Path,
) -> dict:
    """Quantize an ONNX model to INT8 using ONNX Runtime dynamic quantization.

    Uses ``quantize_dynamic`` (weights-only / dynamic activation ranges), which
    needs no calibration dataset. Because no data (training, validation, or
    test) is fed into quantization, there is no opportunity for test-set leakage
    here. Calibration-based static INT8 quantization is intentionally not used.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

    size_kb = output_path.stat().st_size / 1024

    # Benchmark
    import onnxruntime as ort

    sess = ort.InferenceSession(str(output_path))
    input_meta = sess.get_inputs()[0]
    h = input_meta.shape[2] if len(input_meta.shape) > 2 else 224
    w = input_meta.shape[3] if len(input_meta.shape) > 3 else 224
    dummy = np.random.randn(1, 3, h, w).astype(np.float32)

    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        sess.run(None, {input_meta.name: dummy})
        latencies.append(time.perf_counter() - t0)
    avg_ms = np.mean(latencies[10:]) * 1000

    return {
        "format": "onnx_int8",
        "path": str(output_path),
        "size_kb": round(size_kb, 1),
        "avg_latency_ms": round(avg_ms, 2),
    }


def export(
    config: ExperimentConfig,
    checkpoint_path: str | Path | None = None,
    base_dir: str | Path = "experiments",
) -> dict:
    """Main export entry point. Exports to ONNX, then optionally quantizes."""
    if checkpoint_path is None:
        checkpoint_path = (
            Path(base_dir) / config.experiment_id / "artifacts" / "best.pt"
        )
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    num_classes = ckpt.get("num_classes", 2)
    out_dir = checkpoint_path.parent
    onnx_path = out_dir / "model.onnx"

    result = export_onnx(checkpoint_path, onnx_path, num_classes, config.dataset.image_size)

    if config.export.quantization == "int8":
        int8_path = out_dir / "model_int8.onnx"
        int8_result = quantize_int8(onnx_path, int8_path)
        result["quantized"] = int8_result

    return result
