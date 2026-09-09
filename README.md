# TinyVision

Open-source edge computer-vision project focused on small, specialized visual
sensors. First target domain: **3D-print failure detection** from a fixed
webcam, running local inference on ordinary hardware.

**Status: research prototype — v1.0.0-rc1 is NOT approved for production or
release deployment.**

## What exists

- **Runtime** (C++, ONNX Runtime, CPU): validated on development hardware —
  [EDGE.md](EDGE.md), [cpp-runtime-parity.md](docs/research/cpp-runtime-parity.md).
- **Training/evaluation pipeline** (Python, PyTorch): ShuffleNetV2 x1.0 binary
  classifier (failing vs normal) — [ML.md](ML.md).
- **RC1 trained model + exports** (local artifacts; exact paths in
  [ML.md](ML.md) and [rc1-release-candidate.md](docs/research/rc1-release-candidate.md)):
  PyTorch `best.pt`, ONNX `model.onnx`, INT8 ONNX `model_int8.onnx`.
- **Dataset + provenance**: Trier FFF-QC v2 (CC BY 4.0), frozen printing-errors,
  human-verified and `source_annotation` labels with environment-level isolation —
  [release-dataset-build.md](docs/research/release-dataset-build.md),
  [dataset-staging-pipeline.md](docs/research/dataset-staging-pipeline.md).
- **Research record**: problem selection, data strategy, staging pipeline,
  experiments, and evaluations under [docs/research/](docs/research/) — most
  notably [tinyvision-data-strategy.md](docs/research/tinyvision-data-strategy.md)
  and [rc1-release-candidate.md](docs/research/rc1-release-candidate.md).

## Known limitations (do not skip)

- **Normal-source false alarms / poor cross-environment generalization** is the
  dominant unresolved issue: RC1 flags the independent normal holdout at
  FPR 1.000 and alarms on every real-world normal clip in seconds; Trier
  holdout predictions are strongly environment-correlated (camera-3
  environments missed). Full evidence in
  [rc1-release-candidate.md](docs/research/rc1-release-candidate.md).
- Target hardware (Raspberry Pi) and physical cameras have **not** been
  validated; runtime benchmarks are development-hardware baselines.

## Layout

```
runtime/   C++ inference runtime (OpenCV + ONNX Runtime, CMake/Ninja)
training/  Python training & evaluation pipeline
cli/       command-line tools
web/       browser demonstration (not yet present)
datasets/  raw and processed data (large items are git-ignored, see .gitignore)
scripts/   automation and build tools
docs/      research notes, data strategy, and release records
experiments/  trained model artifacts (git-ignored, local)
```