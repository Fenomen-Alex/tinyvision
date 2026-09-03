# Experiment 1: Binary classification across registry candidates

Status: **Complete**

## Objective

Train and evaluate all torchvision classification candidates in the model registry
on the binary classification task (normal vs failing), using the 3d-print-errors
dataset (NilsHagenBeyer, Apache 2.0), then export each to ONNX + INT8.

The winner is selected from measured results, not intuition.

## Method

- **Task**: Binary classification (2 classes: `failing`, `normal`)
- **Dataset**: `printing-errors` v1.0 (16,287 images, 151 recordings)
- **Class balance**: failing 56.9% (9,261), normal 43.1% (7,026)
- **Split**: source-level (by `recording`), 70/15/15, seed 42
  - train 11,254 (69.1%), val 2,174 (13.3%), test 2,859 (17.6%)
  - Verified zero recording overlap across splits (leakage-free)
- **Image size**: 224×224 (center-cropped from 256×256 source)
- **Augmentation**: horizontal flip, random rotation 15°, color jitter 0.2, ImageNet normalization
- **Training**: 30 epochs max, Adam, lr 1e-3, weight decay 1e-4, batch 32, fp32
- **Early stopping**: patience 10 (val loss)
- **Pretrained weights**: ImageNet (torchvision, BSD-3-Clause)
- **Hardware**: Apple Silicon MPS (Mac); latencies measured on same host (not edge)
- **Seed**: 42 (deterministic; verified reproducible)

## Results (held-out test set)

F1/precision/recall are reported for the positive (`failing`) class at threshold 0.5.

| Model | Accuracy | F1 | Precision | Recall | FNR | FPR | AUROC | fp32 size | fp32 lat | int8 size | int8 lat |
|---|---|---|---|---|---|---|---|---|---|---|---|
| efficientnet_b0 | 0.789 | 0.695 | 0.625 | 0.782 | 0.218 | 0.208 | 0.897 | 16.0 MB | 8.85 ms | 4.36 MB | 25.4 ms |
| resnet18 | 0.740 | 0.685 | 0.545 | 0.923 | 0.077 | 0.341 | 0.871 | 44.7 MB | 13.4 ms | 11.2 MB | 9.03 ms |
| **mobilenet_v3_small** | **0.823** | **0.761** | 0.650 | **0.917** | 0.083 | 0.219 | **0.925** | 6.09 MB | **1.84 ms** | 1.69 MB | 6.72 ms |
| mobilenet_v3_large | 0.821 | 0.726 | 0.686 | 0.770 | 0.230 | 0.156 | 0.906 | 16.8 MB | 4.87 ms | 4.42 MB | 14.2 ms |
| shufflenet_v2_x1_0 | **0.836** | 0.757 | **0.695** | 0.830 | 0.170 | 0.162 | 0.900 | 5.05 MB | 1.99 ms | 1.45 MB | 4.95 ms |

### Experiment IDs

| Model | Experiment ID | Train time |
|---|---|---|
| efficientnet_b0 | `efficientnet_b0_printing-errors_c9a8cd7f6e` | ~66 min |
| resnet18 | `resnet18_printing-errors_3d95c29af5` | ~23 min |
| mobilenet_v3_small | `mobilenet_v3_small_printing-errors_b7ac0cdc2e` | ~18 min |
| mobilenet_v3_large | `mobilenet_v3_large_printing-errors_fe02d643ef` | ~60 min |
| shufflenet_v2_x1_0 | `shufflenet_v2_x1_0_printing-errors_cdc9e2cad4` | ~16 min |

## Analysis

### Top candidates

**mobilenet_v3_small** achieves the best F1 (0.761) and AUROC (0.925), with strong
recall (0.917), a small footprint (6.09 MB fp32 / 1.69 MB int8), and the fastest
fp32 latency (1.84 ms). It is the standout edge candidate: high accuracy, small,
fast.

**shufflenet_v2_x1_0** is nearly tied (F1 0.757, accuracy 0.836) and even smaller
(5.05 MB fp32 / 1.45 MB int8). It matches the architecture used by PrintGuard.

### Precision / recall trade-off

For failure detection, false negatives are the most harmful (a missed failure can
ruin a print). The mobile-nets deliver low FNR (8–17%) while the larger models vary
more. Note that a low FNR comes with a higher FPR (more false alarms); temporal
smoothing (Experiment 4) is designed to suppress transient false positives.

### Latency note (important)

Latencies above were measured on this Apple Silicon host, **not** on an edge device.
INT8-on-this-host is often slower than fp32 because the INT8 graph runs on CPU while
fp32 runs on GPU here. These are not the production latency numbers; edge latency
must be benchmarked per `EDGE.md` on Raspberry Pi. Model-size and accuracy rankings
transfer, but latency rankings may not.

## Recommendation

**Provisional winner: mobilenet_v3_small** (best F1 + AUROC, small, fast).
**Runner-up: shufflenet_v2_x1_0** (comparable F1 with smallest footprint).

Final selection should be confirmed after:
1. Edge latency / memory benchmarking on Raspberry Pi (rankings may shift).
2. Temporal smoothing experiment to confirm FPR suppression.
3. Quantization-accuracy check (int8 vs fp32 accuracy degradation).

## Files / artifacts

- Configs: `training/configs/experiment1_<model>.yaml`
- Experiment dirs: `experiments/<model>_printing-errors_<hash>/`
  - `artifacts/best.pt` — training checkpoint
  - `artifacts/model.onnx` — ONNX fp32
  - `artifacts/model_int8.onnx` — ONNX int8 (dynamic quantization)
  - `config.yaml`, `status.json` — recorded configuration and status
- Prepared dataset: `datasets/processed/printing-errors/` (gitignored)
- Split manifest: `datasets/processed/printing-errors/splits/splits.json`

## Limitations

- All latencies measured on Apple Silicon, not edge hardware.
- Dynamic (post-training) int8 quantization; static/quantization-aware not used.
- Single dataset / fixed seed; per-class confusion not broken down by failure mode
  (underextrusion vs stringing vs spaghetti).
- Normalization stats hardcoded in loaders and not serialized for the C++ runtime
  (existing known issue).
