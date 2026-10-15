# Third-party software and licensing

TinyVision source code is licensed under the **Apache License 2.0** (see
[LICENSE](LICENSE)). This file summarizes the licenses of the third-party
software the project depends on, and the licensing constraints that matter for
redistribution.

> Licenses below are as reported by the upstream projects at the time of
> writing and are listed for convenience only. They are not legal advice.
> Verify the current license of any dependency before redistribution.

## Python dependencies (`training/pyproject.toml`)

| Package | License (upstream) |
|---|---|
| torch, torchvision | BSD-3-Clause |
| numpy | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| pandas | BSD-3-Clause |
| matplotlib | PSF-based (BSD-compatible) |
| opencv-python | Apache-2.0 (OpenCV); MIT (wrapper packaging) |
| onnx | Apache-2.0 |
| onnxruntime | MIT |
| onnxscript | MIT |
| pillow | MIT-CMU (HPND) |
| pyyaml | MIT |
| setuptools (build) | MIT |

## Optional extras

| Extra | Package | License | Notes |
|---|---|---|---|
| `yolo` | ultralytics | **AGPL-3.0** | **Copyleft.** Only needed for the `yolov8*`/`yolov11*` model families, which are **not** used by the validated torchvision (ShuffleNetV2) path, the export pipeline, or the C++ inference runtime. It is excluded from the default install so the released software stays permissively licensed. If you enable this extra and distribute a combined/derivative work or offer it as a network service, the AGPL-3.0 obligations apply. |

The model registry (`training/tv/models/factory.py`) tags each architecture with
its license. The registered `yolov8*`/`yolov11*` entries are AGPL-3.0 and are
declared for comparison only; they are not required to build, export, or run the
released classifier.

## Runtime dependencies (`runtime/`)

The C++ inference runtime links against:

| Library | License (upstream) |
|---|---|
| ONNX Runtime | MIT |
| OpenCV | Apache-2.0 |
| CMake / Ninja (build only) | BSD-3-Clause / Apache-2.0 |

## Datasets

Dataset licensing, provenance, and allowed usage are tracked separately per the
project dataset policy — see [DATASETS.md](DATASETS.md) and
[docs/research/datasets/](docs/research/datasets/). Datasets are **not**
redistributed in this repository; raw and derived media are excluded by
`.gitignore`.

## Models

Trained checkpoints are **not** committed (see `.gitignore`). Any model released
separately is subject to its own provenance and license record.
