"""ONNX export correctness: torch <-> ONNX numerical parity and INT8.

These tests use random weights (pretrained=False) to avoid downloads, and verify
that the exported model produces the same predictions as the source framework.
"""
from __future__ import annotations

import pytest
import torch

from tv.models.factory import build_classifier

torch.manual_seed(0)

MODELS = ["resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0"]


@pytest.mark.parametrize("model_name", MODELS)
def test_onnx_matches_torch(model_name: str, tmp_path) -> None:
    model = build_classifier(model_name, 3, pretrained=False)
    model.eval()
    onnx_path = tmp_path / f"{model_name}.onnx"
    x = torch.randn(1, 3, 64, 64)  # fixed batch 1 (dynamic_axes=None)

    torch.onnx.export(
        model, x, str(onnx_path), opset_version=17, dynamo=False,
        input_names=["input"], output_names=["output"], dynamic_axes=None,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path))
    with torch.no_grad():
        torch_out = model(x).numpy()
    onnx_out = sess.run(None, {"input": x.numpy()})[0]

    assert torch_out.shape == onnx_out.shape
    # Predictions (argmax) must agree.
    import numpy as np

    assert (torch_out.argmax(1) == onnx_out.argmax(1)).all()
    # Logits must be numerically close.
    assert np.allclose(torch_out, onnx_out, atol=1e-4)


@pytest.mark.parametrize("model_name", MODELS)
def test_int8_keeps_predictions(model_name: str, tmp_path) -> None:
    """INT8 dynamic quantization must not change argmax predictions."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model = build_classifier(model_name, 3, pretrained=False)
    model.eval()
    onnx_path = tmp_path / f"{model_name}.onnx"
    int8_path = tmp_path / f"{model_name}_int8.onnx"
    x = torch.randn(1, 3, 64, 64)

    torch.onnx.export(
        model, x, str(onnx_path), opset_version=17, dynamo=False,
        input_names=["input"], output_names=["output"], dynamic_axes=None,
    )
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(int8_path))
    with torch.no_grad():
        torch_pred = model(x).argmax(1).numpy()
    int8_out = sess.run(None, {"input": x.numpy()})[0]
    assert (torch_pred == int8_out.argmax(1)).all()
    assert int8_path.exists()