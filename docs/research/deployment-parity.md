# Deployment parity: PyTorch → ONNX → INT8 → C++

Goal: the production C++ runtime must compute the same decision as the training
framework, through the same export chain, within stated tolerances. This page
records the measured parity at each hop for RC1.

Score convention: `failing` is the positive class (index 0); `score =
P(failing)`; predicted failing iff `score >= threshold`. Artifacts under
`experiments/rc1_release_candidate_shufflenet_v2_x1_0/artifacts/`:
`best.pt`, `model.onnx`, `model_int8.onnx`.

Reproduce:

```
training/.venv/bin/python -m tv.cli reliability parity \
    --suite training/configs/reliability/suite.yaml --surface dev_val \
    --framework experiments/rc1_release_candidate_shufflenet_v2_x1_0/artifacts/best.pt \
    --other     experiments/rc1_release_candidate_shufflenet_v2_x1_0/artifacts/model.onnx \
    --output-json docs/research/parity-rc1-onnx.json
# ... --other model_int8.onnx ... --output-json docs/research/parity-rc1-int8.json
```

Surface: `dev_val` (n = 2174; printing-errors val fold; single ELP venue —
see [`evaluation-independence.md`](evaluation-independence.md)).

## Hop 1 — PyTorch → ONNX (fp32)

Source: [`parity-rc1-onnx.json`](parity-rc1-onnx.json).

| metric | value |
|---|---|
| prediction disagreement | **0.0000** |
| mean score abs drift | 0.000000 |
| max score abs drift | **0.000004** |
| metric delta (fpr/fnr/recall) | 0.0 / 0.0 / 0.0 |

Verdict: the fp32 export is **behaviorally identical** to PyTorch at 1e-6; the
policy parity floor (`max_onnx_score_abs_drift: 0.05`,
`max_onnx_prediction_disagreement: 0.01`) passes with large margin.

## Hop 2 — ONNX → INT8 (dynamic quantization)

Source: [`parity-rc1-int8.json`](parity-rc1-int8.json).

| metric | value |
|---|---|
| prediction disagreement | 0.0322 (3.2%) |
| mean score abs drift | 0.0351 |
| max score abs drift | 0.7168 |
| recall delta | −0.0007 |
| **FPR delta** | **+0.0410** |
| FNR delta | −0.0007 |

Verdict: INT8 preserves recall essentially exactly but **raises the false-alarm
rate by ~4 points** on this surface (confusion matrix fp32 `[[658,99],[231,1186]]`
→ int8 `[[627,130],[230,1187]]`). For a failure detector where normal-source
false alarms are the documented failure mode, INT8 must be treated as a
**different model** and gated by the normal-holdout FPR floor, not assumed
equivalent to fp32. Model size 5.05 MB (fp32) → 1.45 MB (int8).

## Hop 3 — ONNX → C++ runtime

The C++ runtime replays a fixed input through ONNX Runtime and must reproduce
the Python logits bit-for-bit within tolerance; this is asserted by the runtime
test suite (`runtime/tests/test_model.cpp`, golden `golden_model_input.npy` /
`golden_model_logits.npy`).

```
cd runtime/build && ctest --output-on-failure
  Test #1: tinyvision_tests      ... Passed
  Test #2: integration_smoke     ... Passed
```

`tinyvision_tests` includes: model load + input-shape check, golden-logit
parity, `PredictFailing`, missing-model handling, and wrong-input-shape
handling. `integration_smoke` runs the end-to-end CLI over a fixture video and
validates the JSON report fingerprint.

**Note (CI):** the C++ tests depend on the git-ignored `model.onnx` artifact, so
CI compiles but does not run them when the artifact is absent
(`.github/workflows/ci.yml`). They run wherever the artifact is present.

## Summary table

| Hop | Disagreement | Recall Δ | FPR Δ | Status |
|---|---|---|---|---|
| PyTorch → ONNX fp32 | 0.0000 | 0.0 | 0.0 | exact |
| ONNX fp32 → INT8 | 0.0322 | −0.0007 | +0.0410 | usable with FPR caveat |
| ONNX → C++ | 0 (golden) | — | — | verified (ctest) |

## Caveats

* All parity numbers are on a **single-venue** surface; they show export
  fidelity, not generalization. INT8's FPR impact on other venues is unknown
  until an independent surface is available.
* Edge latency/size/FPS numbers live in
  [`edge-runtime-benchmark.md`](edge-runtime-benchmark.md) and
  `runtime/EDGE.md`.
