# Reliability assessment — claim-by-claim

Date: 2026-10-13
Baseline under assessment: `rc1_release_candidate_shufflenet_v2_x1_0`
(see `rc1-baseline.json` for hashes).

This document states, for each reliability claim one might make about TinyVision,
what evidence actually exists, at what tier, and the honest verdict. **A failed
claim is not a failure of the project — an unsupported claim is.** Evidence tiers:

* **artifact** — corroborated by a committed artifact/JSON.
* **local-test** — exercised by the test suite on this machine.
* **real-data** — produced by running a model on real footage
  (`real-footage-analysis.json`, `stage14-real-frames-eval.json`).
* **target-hardware** — measured on the deployment device.
* **not-yet-verified** — no supporting evidence.

| # | Claim | Evidence | Tier | Verdict |
|---|---|---|---|---|
| 1 | RC1 detects failures on the frozen in-distribution test | `rc1-core-eval.json` reproduced exactly by `scripts/reliability_repro_rc1.py` (TN=798, FP=80, FN=372, TP=1609 @0.5) | artifact + local-test | **TRUE but insufficient** (P=0.95, R=0.81) |
| 2 | RC1 meets the acceptance policy | `release-gate-rc1/summary.json` | artifact + local-test | **FALSE** — dev R 0.86 < 0.9, locked R 0.84 < 0.9 |
| 3 | RC1 generalises to real external footage | `real-footage-analysis.json`: 5/5 VLM-normal videos fire; median P(failing) 0.74–0.99 on normal | real-data | **FALSE** |
| 4 | RC1 improved real-domain reliability over the model it replaced | real-data: normal false-alarm ≥0.9 rises (38gnsb 0.35→0.77, nbkf372 0.83→0.93) | real-data | **FALSE / contradicted** |
| 5 | The classifier's P(failing) is meaningful (calibrated) on real frames | `stage14-real-frames-eval.json`: idle/empty frames all >0.5 (median 0.92) | real-data | **FALSE** |
| 6 | The temporal layer removes false alarms on normal footage | real-data: video-level FAR = 1.0 under the shipped run config; synthetic `temporal-synthetic-benchmark.json` | real-data | **FALSE on real data** |
| 7 | Low-FPR temporal logic misses gradual drift | `temporal-synthetic-benchmark.json` (default delta-only misses drift; abs-floor catches at frame 92) | local-test | **TRUE** (documented; not silently changed) |
| 8 | ONNX export is numerically faithful to PyTorch | `parity-rc1-onnx.json`: disagreement 0.0, max drift 4e-6 | artifact | **TRUE** |
| 9 | INT8 export is deployable without meaningful degradation | `parity-rc1-int8.json`: disagreement 0.032, max drift 0.72, FPR +0.041 | artifact | **MARGINAL** — passes declared thresholds, real effect not negligible |
| 10 | The C++ runtime matches the Python reference | `runtime/build` ctest 2/2; `cpp-runtime-parity.md` | local-test | **TRUE (behavioral smoke)** — not on target HW |
| 11 | It runs on ordinary edge hardware | none | target-hardware | **NOT VERIFIED** (`raspberry-pi-benchmark-procedure.md` is a procedure only) |
| 12 | Experiment configs are reproducible/loadable | was broken (all 10 configs unreadable) | — | **FIXED this session** (see below) |
| 13 | Evaluation surfaces are environment-independent | `evaluation-independence.json`: dev 22 / test 24 recordings, 0 overlap, **single venue** | artifact | **FALSE** (source-disjoint, not environment-disjoint) |
| 14 | All used datasets are licence-cleared | `dataset-inventory.md`: SKZ unresolved | artifact | **FALSE** — SKZ blocks release |

## Defect fixed this session

**Configs were un-loadable (reproducibility break).** `record_software()`
returned `torch.__version__` (a `torch.torch_version.TorchVersion`, a `str`
subclass). `save_config` used `yaml.dump`, which emitted
`!!python/object/new:torch.torch_version.TorchVersion`, while `load_config` used
`yaml.safe_load`, which refuses that tag. **Every config written by `train`
(all committed experiments) therefore failed to load**, breaking
`tv evaluate` / `tv export` on any historical run. Fixed in
`training/tv/config.py`: versions coerced to `str`, `save_config` switched to
`yaml.safe_dump`, and `load_config` gained a tolerant loader that recovers the
legacy tag as a plain string without importing or executing anything (4
regression tests in `tests/test_config.py`). All 15 configs now load.

## The single most important result

On real footage, **P(failing) is high on essentially any frame** — including
normal prints and frames with no print visible. This is a representation /
domain-shift failure, not a threshold or smoothing problem. It is the same
mechanism behind the documented FPR = 1.0 normal-holdout failure, now shown on
seven independent videos and 100 real failure frames. No threshold on the
existing checkpoint fixes it; RC2 must change the training distribution (the
prepared `training/configs/planned/rc2_domain_robust.yaml`, **not run**).

## What would change these verdicts

* A **human-verified**, **environment-diverse** holdout of normal and failing
  real footage (per `evaluation-independence.md`, the current surfaces share one
  venue). This is the gating prerequisite; it is data/labelling work, not code.
* Resolving the SKZ licence.
* A target-hardware benchmark (procedure ready).
* An RC2 training run on a domain-robust mixture, evaluated on the holdout
  above — which must be *locked before* any tuning.
