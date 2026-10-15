# TinyVision

Open-source edge computer-vision project focused on small, specialized visual
sensors. First target domain: **3D-print failure detection** from a fixed
webcam, running local inference on ordinary hardware.

**Status: experimental research preview (v0.1.0).** This is a first public
release of the *software* — the training/evaluation pipeline, the `tv` CLI, the
local inference demo, and the reliability gate — as a reproducible research
artifact. It is **not** a validated product.

What is and is not established:

- **Established (software):** the pipeline, CLI, local demo, tests, and the
  reliability gate work end-to-end and are covered by automated tests; the C++
  runtime builds and its parity tests pass on development hardware.
- **Established (accuracy):** *nothing about real-world detection accuracy.*
  The included classifier (RC1) is **not** approved for production or release
  deployment — the reliability gate currently returns **FAIL** (recall below
  the required floor; independent normal-holdout evidence missing). See
  [Known limitations](#known-limitations-do-not-skip).
- **Not established:** cross-printer/camera generalization, target-hardware
  (Raspberry Pi) behavior, and real-camera performance in the field.

Do not deploy this model to protect real prints. The honest next step is
acquiring licensed, printer-disjoint real footage; see
[footage-intake-workflow.md](docs/research/footage-intake-workflow.md).

## What exists

- **Runtime** (C++, ONNX Runtime, CPU): builds and passes parity tests on
  development hardware —
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
- **Reliability gate**: `tv reliability evaluate` applies an acceptance policy
  and returns PASS / FAIL / INCONCLUSIVE; missing evidence always blocks release
  — [docs/reliability-evaluation.md](docs/reliability-evaluation.md),
  [docs/research/deployment-parity.md](docs/research/deployment-parity.md).
  The current RC1 gate result is **FAIL** (recall below floor; normal-holdout
  evidence missing) — [rc2-audit-final-report.md](docs/research/rc2-audit-final-report.md),
  [reliability-assessment.md](docs/research/reliability-assessment.md).

Start here: [ARCHITECTURE.md](ARCHITECTURE.md), [DATASETS.md](DATASETS.md),
[EXPERIMENTS.md](EXPERIMENTS.md), [BENCHMARKS.md](BENCHMARKS.md),
[CONTRIBUTING.md](CONTRIBUTING.md).

## Try it: local inference demo (no printer/camera/cloud)

Inspect an existing 3D-print recording with the frozen classifier, watch the
per-frame `P(failing)` stream and the temporal failure decision, and export the
JSON report. This runs entirely locally on the Python backend — no printer,
camera, account, hosted service, or paid API. **It is experimental and not
validated for production use.**

You do **not** need to be a Python developer: install once, launch with one
command, and use it in your browser. You do need a **model checkpoint** (a
`.pt` file) — this repository keeps checkpoints out of Git, so a fresh clone has
none. If you have one, the demo's model picker (or its checkpoint-path field)
will use it.

Install (one time), then launch:

```sh
# --- install the demo (creates an isolated venv; nothing is installed globally)
# macOS / Linux (with uv: https://docs.astral.sh/uv/)
uv venv training/.venv && uv pip install --python training/.venv -e training
# ...or with a plain Python 3.11+ (pip):
python3 -m venv training/.venv && training/.venv/bin/python -m pip install -e training

# --- check that the inference dependencies are present
training/.venv/bin/tinyvision-demo --check

# --- launch the local web UI (serves http://127.0.0.1:8000 and opens a browser)
training/.venv/bin/tinyvision-demo --open-browser
```

On **Windows** (PowerShell) the equivalent is
`py -3 -m venv training\.venv; training\.venv\Scripts\python -m pip install -e training`
then run `training\.venv\Scripts\tinyvision-demo.exe --open-browser`.
(`tv demo` is an equivalent entry point on every platform.)

Get a deterministic synthetic test clip if you have no real recording:

```sh
training/.venv/bin/python scripts/make_synthetic_demo_video.py
```

Then in the browser: pick a checkpoint (or paste a path), choose the video
(upload it, or paste a local path), adjust sample FPS / threshold / warm-up if
you like, and press **Analyze video**.

- **Where data goes.** Uploaded videos and exported reports are written to a
  per-user data directory (`~/Library/Application Support/TinyVision/demo` on
  macOS, `~/.local/share/tinyvision/demo` on Linux,
  `%LOCALAPPDATA%\TinyVision\demo` on Windows). Override it with
  `--data-dir PATH` or the `TINYVISION_DEMO_DATA` environment variable. Nothing
  is uploaded off your machine; the data stays until you delete it.
- **Checkpoints elsewhere.** Point the launcher at any directory of `*.pt`
  checkpoints with the `TINYVISION_MODELS_DIR` environment variable
  (`:`-separated on macOS/Linux, `;` on Windows), or type a path into the UI.
- **Networking.** The server binds `127.0.0.1` (loopback) by default. Binding a
  non-loopback `--host` exposes uploads and the endpoint to your network and
  prints a warning.
- **Port busy?** Use `--port 0` to pick a free port, or `--port 8001`.

The same pipeline is available headless via
`tv poc --input CLIP.mp4 --checkpoint CKPT`. Details and the exact commands:
[docs/research/poc-inference-architecture.md](docs/research/poc-inference-architecture.md).

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
training/  Python training & evaluation pipeline + `tv` CLI + reliability gate
cli/       reserved for standalone tooling (currently empty; `tv` is Python)
web/       TypeScript/Vite scaffold (unused; the working demo is the Python
           `tv demo` local web UI in training/tv/webdemo.py)
datasets/  raw and processed data (large items are git-ignored, see .gitignore)
scripts/   automation, provenance builders, audits
docs/      research notes, data strategy, and release records
experiments/  trained model artifacts (git-ignored, local)
models/    reserved for model metadata/release config (currently empty)
benchmarks/  reserved for benchmark records (currently empty)
```

## License

Source code is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE).
Third-party dependencies and their licenses (including the optional, AGPL-3.0
`ultralytics` extra) are summarized in [THIRD_PARTY.md](THIRD_PARTY.md).
Datasets and trained model checkpoints are **not** redistributed in this
repository; their licensing is recorded per the project dataset policy in
[DATASETS.md](DATASETS.md). Contributions are accepted under the terms in
[CONTRIBUTING.md](CONTRIBUTING.md).