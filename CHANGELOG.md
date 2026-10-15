# Changelog

All notable changes to TinyVision are documented here. Versions refer to the
**software** (`tinyvision-training`, the `tv` CLI, the demo, and the runtime).
Model/release-candidate names such as "RC1" are separate from software versions.

## [0.1.0] — 2026-10-15 — research preview

First public release. Published as a **reproducible research artifact**, not a
validated product. The included classifier is **not** approved for production
use: the reliability gate returns **FAIL**.

### Software

- **Training / evaluation pipeline** (Python, PyTorch): model-agnostic registry,
  leakage-free source-level splitting, training, evaluation, and ONNX/INT8
  export, driven by the `tv` CLI.
- **Local inference demo** (`tv demo` / `tinyvision-demo`): a local-only web UI
  to inspect an existing recording with the frozen classifier, view the per-frame
  `P(failing)` stream and temporal decision, and export a JSON report. No
  printer, camera, account, or cloud required.
- **Headless point-of-care inference** (`tv poc`) and per-printer threshold
  **calibration** (`tv calibrate`).
- **Reliability gate** (`tv reliability`): applies an acceptance policy and
  returns `PASS` / `FAIL` / `INCONCLUSIVE`; missing required evidence always
  blocks release. Exit codes are stable (`2` = fail, `3` = inconclusive).
- **Environment-isolated dataset tooling** (`tv env-*`): manifest schema,
  structural + license/consent + leakage validation, splitting, and readiness.
- **Footage intake workflow** (`tv sessions`): validate a session sheet, build a
  calibration spec, and run an end-to-end intake that keeps a locked holdout
  unopened until the decision threshold is fixed — see
  [docs/research/footage-intake-workflow.md](docs/research/footage-intake-workflow.md).
- **C++ inference runtime** (`runtime/`): OpenCV + ONNX Runtime (CPU), CMake /
  Ninja; builds and passes parity tests on development hardware.
- **CI** (`.github/workflows/ci.yml`): Python unit tests and a synthetic
  release-gate assertion that a missing dataset can never produce a passing
  real-world evaluation; a C++ build job.

### Research record (not benchmark results)

- **RC1 classifier + exports** (torchvision ShuffleNetV2 x1.0, binary
  failing-vs-normal) trained on the Trier frozen printing-errors dataset.
  Checkpoints and exports are local artifacts and are **not** committed.
- **RC1 reliability gate: FAIL.** `dev_validation` recall 0.8603 (< 0.9 floor,
  FPR 0.1744, n = 2174); locked `test` recall 0.8430 (n = 2859); independent
  normal-holdout evidence missing (INCONCLUSIVE). Onnx/INT8 parity and
  deployment-size checks pass.
- **Domain-shift finding:** thresholds do not transfer across printers/cameras.
  A held-out printer improved AUROC from 0.708 (baseline) to 0.809–0.933 with
  photometric/appearance perturbations, but **no single operating point
  transferred** — the record documents a limitation, not a fix.
- **RC2 real-domain diagnostic:** an automated acquire → autolabel (VLM) →
  streams → train pipeline was exercised end-to-end. Its labels are weak/VLM and
  **not human-verified**; the run does **not** satisfy the release policy and its
  numbers must not be quoted as benchmarks.

### Known limitations

- Normal-source false alarms and poor cross-environment generalization are the
  dominant unresolved problem (RC1 flags the independent normal holdout at
  FPR 1.000).
- Target hardware (Raspberry Pi) and physical cameras are unvalidated; runtime
  numbers are development-hardware baselines.

### Licensing

- Released under the **Apache License 2.0**.
- `ultralytics` (AGPL-3.0) moved to the optional `yolo` extra; the validated
  torchvision path and the inference runtime do not require it. See
  [THIRD_PARTY.md](THIRD_PARTY.md).
- Datasets and checkpoints are not redistributed; licensing is tracked per the
  project dataset policy ([DATASETS.md](DATASETS.md)).

[0.1.0]: https://github.com/Fenomen-Alex/tinyvision/releases/tag/v0.1.0
