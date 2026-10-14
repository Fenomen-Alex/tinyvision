# Contributing

TinyVision is a research prototype. Correctness, provenance and reproducibility
matter more than speed. Read [`AGENTS.md`](AGENTS.md) (project principles) and
[`ARCHITECTURE.md`](ARCHITECTURE.md) before making changes.

## Setup

```bash
python -m venv training/.venv
training/.venv/bin/pip install -e "training[test]"
```

C++ runtime (macOS Homebrew / Raspberry Pi apt):

```bash
cmake -S runtime -B runtime/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build runtime/build
```

## Running tests

```bash
# Python (from training/)
.venv/bin/python -m pytest tests -q

# C++ (from runtime/build/)
ctest --output-on-failure
```

Data-dependent Python tests **skip** (never fail) when large datasets/models are
absent, so the suite runs on a clean checkout. The C++ tests require the
git-ignored `model.onnx` artifact.

## CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml): a Python job (unit tests
+ assertions that the release gate exits non-zero on missing evidence) and a C++
job (compiles against ONNX Runtime + OpenCV; runs CTest only when the model
artifact is present).

## The release gate

The release decision is `tv reliability evaluate`, which applies
`training/configs/reliability/policy.yaml` to a suite and returns PASS / FAIL /
INCONCLUSIVE. **Missing evidence is INCONCLUSIVE and always blocks release.**
Never loosen the policy, tune thresholds on the locked test, or fabricate
results to make a gate pass. See
[`docs/reliability-evaluation.md`](docs/reliability-evaluation.md).

## Data

* Every external dataset needs a full provenance record before use; unknown
  licensing means it cannot be included. See [`DATASETS.md`](DATASETS.md).
* Never commit large data, footage, or model artifacts (they are git-ignored).
* Never split at frame/recording level — split at **environment** level.
* Never mix datasets without recording provenance.

## Pull requests

* Keep the change minimal and focused; don't rewrite working components without
  a clear reason.
* Add or update tests for behaviour changes.
* Run the Python and (where possible) C++ tests before opening a PR.
* Do not introduce a dependency for convenience.
* Record any change to model architecture, dataset definitions, evaluation
  criteria or benchmark methodology explicitly, with rationale.
