# Architecture

TinyVision is split into four independent layers. The hard boundary is
**training (Python) ≠ production inference (C++)**: they share only the ONNX
model contract and the score/decision conventions, never code or environments.

```
            datasets/                         experiments/ (git-ignored artifacts)
                │                                       ▲
                ▼                                       │
   ┌───────────────────────────┐   PyTorch ckpt   ┌──────────────────────┐
   │ training/  (Python)       │ ───────────────► │ ONNX / INT8 export   │
   │  config → train → eval    │                  └──────────┬───────────┘
   │  reliability gate  ◄──────┼──── suite+policy            │ model.onnx
   └───────────────────────────┘                             ▼
                                        ┌──────────────────────────────────┐
                                        │ runtime/  (C++ / ONNX Runtime)    │
                                        │  preprocess → model → temporal    │
                                        │  CLI + JSON report                │
                                        └──────────────────────────────────┘
                                                             │
                                          web/ (TS demo, scaffold only) ──┘
```

## Layers

### 1. Data (`datasets/`, `scripts/`)
Environment-centric manifests (`tinyvision-env-manifest`), staged under
`datasets/staging/`. Splits are made at **environment** level, never frame
level, to prevent leakage. Intake is validated by `tv env-validate`; readiness
is gated by `tv env-readiness`. Only manifests/labels are committed. See
[`DATASETS.md`](../DATASETS.md).

### 2. Training & evaluation (`training/`, Python)
`tv` CLI: `config.py` (content-hashed experiment IDs) → `train.py`
(deterministic seeding) → `experiment.py` (artifact manager) →
`evaluate.py`/`export.py`. The **reliability gate** (`tv reliability ...`,
`tv/reliability.py`) is the release decision surface: it scores a model over an
evaluation suite, applies an acceptance policy, and returns
PASS/FAIL/INCONCLUSIVE. See [`reliability-evaluation.md`](../reliability-evaluation.md).

### 3. Production inference (`runtime/`, C++17, CMake/Ninja)
`tinyvision_core` (`preprocess.cpp`, `model.cpp`, `temporal.cpp`, `cli.cpp`)
built against OpenCV + ONNX Runtime CPU EP; binaries `tinyvision` (CLI) and
`tinyvision_tests` (CTest). The Python `tv poc` path is the behavioral
reference; parity is asserted by golden-logit tests. See
[`deployment-parity.md`](deployment-parity.md) and
[`edge-runtime-benchmark.md`](edge-runtime-benchmark.md).

### 4. Web (`web/`, TypeScript/Vite/React)
Scaffold only (stock Vite template); no inference demo yet. Intended as an
accessible demonstration and, later, optional in-browser inference.

## Cross-cutting contracts

* **Score convention:** `failing` is the positive class (index 0);
  `score = P(failing)`; predicted failing iff `score >= threshold`.
* **Temporal decision:** per-frame scores → EWMA → baseline-relative signal →
  fire-once FSM (`OBSERVING→NORMAL→SUSPECT→FAILING→PAUSED`). See
  [`temporal-decision-logic.md`](temporal-decision-logic.md).
* **Release state:** v1.0.0-rc1 is **not** approved. The gate must remain
  INCONCLUSIVE while independent evidence is missing.

## Repository layout

| Path | Contents |
|---|---|
| `runtime/` | C++ inference runtime (OpenCV + ONNX Runtime, CMake/Ninja) |
| `training/` | Python training/eval pipeline + `tv` CLI + reliability gate |
| `cli/` | reserved for standalone C++/shell tooling (currently empty; `tv` is Python) |
| `web/` | TypeScript scaffold (no demo yet) |
| `datasets/` | data + manifests (large data git-ignored) |
| `scripts/` | automation, provenance builders, audits |
| `experiments/` | trained artifacts (git-ignored) |
| `models/` | reserved for model metadata/release config (currently empty) |
| `benchmarks/` | reserved for benchmark records (currently empty) |
| `docs/` | research, data strategy, release records |
