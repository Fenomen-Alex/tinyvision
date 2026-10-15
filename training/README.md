# TinyVision Training Pipeline

This directory contains the **model-agnostic, reproducible training and evaluation
pipeline**. It is intentionally **not** coupled to any single model architecture:
experiments select the model from a registry, so the best model is decided by
measured results, not preconception.

Per the project principles (see `AGENTS.md`):

- Every experiment records a full configuration and measurable results.
- Data splitting avoids leakage by grouping at the *source* (print / capture session) level.
- Training infrastructure stays separate from the C++ production runtime (`runtime/`).
- Deterministic CLI tools are preferred over notebook-only workflows.

## Commands

Run everything through the `tv` CLI (installed in this venv):

```
uv run tv <command> [options]
```

| Command | Description |
|---|---|
| `tv split --data-dir DIR` | Generate leakage-free train/val/test splits |
| `tv train --config <yaml>` | Train a model from an experiment config |
| `tv evaluate --config <yaml>` | Evaluate a trained model on the held-out test set |
| `tv export --config <yaml>` | Export to ONNX and optionally INT8 |
| `tv capture --output-dir DIR` | Capture webcam frames for data collection |
| `tv list-models` | List registered model architectures |
| `tv demo` | Launch the local web UI for inspecting a video recording |
| `tv poc --input CLIP --checkpoint CKPT` | Headless per-frame inference / temporal decision |
| `tv calibrate --clips SPEC.yaml` | Per-printer threshold calibration |
| `tv sessions validate\|build` | Validate a footage session sheet / build a calibration spec |
| `tv reliability evaluate\|validate-suite` | Apply the release acceptance policy (PASS/FAIL/INCONCLUSIVE) |

## Local demo (`tv demo` / `tinyvision-demo`)

`tv demo` starts a small local web UI (stdlib HTTP server, no extra framework)
that runs the **same** inference path as `tv poc` on a video file and streams
per-frame `P(failing)` progress to the browser. It is **experimental and not
validated for production use**, and it needs a checkpoint (`.pt`).

```
tv demo [--host 127.0.0.1] [--port 8000] [--data-dir DIR] [--open-browser] [--check]
```

`--check` only verifies the inference dependencies and exits. The installable
console script `tinyvision-demo` (declared in `pyproject.toml`) is the
recommended entry point for non-developers because it reports missing
dependencies with actionable install instructions:

```
uv pip install -e .          # or: python -m pip install -e .
tinyvision-demo --check      # "All inference dependencies are present (...)"
tinyvision-demo --open-browser
```

- **Data dir** (uploads + exported reports) defaults to a per-user directory
  (`~/Library/Application Support/TinyVision/demo` on macOS,
  `~/.local/share/tinyvision/demo` on Linux,
  `%LOCALAPPDATA%\TinyVision\demo` on Windows). Override with `--data-dir` or
  `TINYVISION_DEMO_DATA`.
- **Checkpoints** are discovered under `experiments/*/artifacts/best.pt` and any
  directories in `TINYVISION_MODELS_DIR`; the UI also accepts a typed path.
- Binds **loopback** by default; a non-loopback `--host` prints a warning. A busy
  port raises an actionable error (use `--port 0` for a free port).
- The UI asset ships with the package
  (`[tool.setuptools.package-data] tv = ["demo_ui.html"]`), so a wheel install
  serves it without the source tree.

## Workflow

The typical flow is:

1. **Collect data** with `tv capture` (one capture session = one *source*).
2. **Split** the dataset with `tv split`. Frames are grouped by source and the
   train/val/test split happens at the group level, so frames from the same
   print/session never span splits (prevents leakage).
3. **Train** with `tv train --config experiment.yaml`.
4. **Evaluate** with `tv evaluate --config experiment.yaml` (same config, so the
   checkpoint is located automatically).
5. **Export** with `tv export --config experiment.yaml` to produce ONNX / INT8
   artifacts for the C++ runtime.

## Experiment config

Each experiment is described by one YAML file (see `configs/classification_base.yaml`).
The config records every field required by the project's experiment policy:
experiment ID, model, task, dataset name/version/dir, image size, batch size,
epochs, optimizer, learning rate, weight decay, augmentation, precision, seed,
evaluation metrics/threshold, export settings, hardware, and software versions.

Key points:

- **`experiment_id`** is auto-generated and is **deterministic**: re-loading the
  same config yields the same ID, so `evaluate`/`export` find the checkpoint
  produced by `train` without manual bookkeeping.
- **`model`** selects any architecture from the registry. The registry is the
  single source of truth for what can be trained.
- **`hardware`** and **`software`** are auto-filled from the runtime at load time
  unless explicitly provided.
- The full resolved config (including the runtime-recorded versions) is written
  to `<experiments-dir>/<experiment_id>/config.yaml`.

## Directory output

Each experiment gets a directory:

```
experiments/<experiment_id>/
├── config.yaml         # fully-resolved config (with ID, versions, hardware)
├── status.json         # started / completed / failed + elapsed time
├── traceback.txt       # written on failure
└── artifacts/
    ├── best.pt         # best checkpoint (model + optimizer + metadata)
    ├── model.onnx      # ONNX export
    ├── model_int8.onnx # INT8-quantized ONNX
    └── metrics.json    # evaluation metrics (written by a future step)
```

## Dataset layout & splitting

Classification data is expected under a root directory in one of two layouts:

```
Layout A (recommended — source subdirectories):   Layout B (flat):
  data/<label>/<source_id>/img.jpg                  data/<label>/img.jpg
  data/<label>/<source_id>/img.jpg                  data/<label>/img.jpg
```

With **Layout A**, each `source_id` (e.g. one print, one capture session) is an
indivisible group for splitting, which is what prevents data leakage. With
**Layout B**, each image is treated as its own source (no leakage protection).

`tv split` writes:

- `splits/splits.json` — full manifest with per-label file lists and stats
- `splits/train.csv`, `splits/val.csv`, `splits/test.csv` — `path,label` rows
  consumed by the training loaders.

## Supported models

The registry (`tv/models/factory.py`) currently exposes these architectures:

| Model | Task | License |
|---|---|---|
| `efficientnet_b0` | classification | Apache-2.0 |
| `resnet18` | classification | BSD-3-Clause |
| `mobilenet_v3_small` | classification | Apache-2.0 |
| `mobilenet_v3_large` | classification | Apache-2.0 |
| `shufflenet_v2_x1_0` | classification | BSD-3-Clause |
| `yolov8n-cls` / `yolov11n-cls` | classification | AGPL-3.0 |
| `yolov8n` / `yolov11n` | detection | AGPL-3.0 |

Add new architectures by extending `MODEL_REGISTRY` and the corresponding
builder. Nothing in the pipeline assumes a particular model is "best".

> **Licensing note:** Under AGPL-3.0 and GPL-2.0 (UL principle 6), if any experiment's
> winning model is a copyleft-licensed architecture, that shapes how the project may
> be distributed/relicensed. Prefer permissively-licensed architectures where possible.

## Evaluation metrics

Classification evaluation reports (see `tv/metrics.py`):

- accuracy, precision, recall, F1
- **false-negative rate (FNR)** and **false-positive rate (FPR)** — FNR is
  especially important for failure detection (missing a failure is worst)
- AUROC (when probability scores are available)
- full 2×2 confusion matrix and sample count

Detection evaluation (via Ultralytics) reports mAP50, mAP50-95, precision,
recall, and F1.

## Notes

- `onnxscript` is required by modern `torch.onnx.export`; ONNX exports use the
  legacy (`dynamo=False`) exporter at opset 17 for maximum onnxruntime
  quantization compatibility.
- INT8 quantization uses ONNX Runtime dynamic quantization.
- Pillow is used for image loading and is a declared dependency (`pillow`).
