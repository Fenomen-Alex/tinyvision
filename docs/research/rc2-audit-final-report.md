# TinyVision RC2 audit — final engineering report

Date: 2026-10-13
Branch: `main` @ `1c9e8f46` ("release: finalize TinyVision RC1"); nothing committed
or pushed during this audit.
Baseline audited: `rc1_release_candidate_shufflenet_v2_x1_0`
(`rc1-baseline.json` has the hashes).

Related: `reliability-assessment.md` (claim-by-claim),
`reliability-audit.md` (FPR=1.0 trace), `release-gate-rc1/` (live gate run),
`real-footage-analysis.md` / `.json`, `stage14-real-frames-eval.json`,
`evaluation-independence.md`, `dataset-inventory.md`,
`raspberry-pi-benchmark-procedure.md`.

---

## 1. Scope and constraints honoured

Audit of the current tree + the smallest scientifically defensible RC2 steps.
Honoured: no commit/push; working tree and all user data preserved (the 183
`datasets/external/**` files were **read** via `git show HEAD:<path>`, never
restored/cleaned); no expensive training; no large downloads; no fabricated
results; policy/gates not loosened; locked test not tuned on; no printer-control
integration added; unresolved-licence media not downloaded or redistributed.

## 2. Working-tree state

Pre-existing (not created here): 25 untracked, 183 deleted, 11 modified —
deliberate RC1 finalisation. This session **added** research artifacts and
tooling only (`docs/research/*`, `scripts/*`), plus two small source fixes
(§3). Nothing was deleted or reverted. Full `git status` diff reviewed;
changes are intentional.

## 3. Defects found and fixed

1. **Config serialization (reproducibility break).** `record_software()`
   returned `torch.__version__`, a `torch.torch_version.TorchVersion` (a `str`
   subclass). `save_config` used `yaml.dump`, emitting a
   `!!python/object/new:...` tag that `load_config`'s `yaml.safe_load` refuses —
   so **every config written by `train` failed to load**, breaking
   `evaluate`/`export` on all historical experiments. Fixed in
   `training/tv/config.py`: versions coerced with `str()`; `save_config` →
   `yaml.safe_dump`; tolerant `_ConfigLoader` recovers the legacy tag as a plain
   string (no object construction). All 15 configs load. 4 regression tests.
2. **Parity scope not shown in the gate summary (observability).** The gate
   computes ONNX/INT8 parity on the *largest* surface, bounded to
   `PARITY_MAX_SAMPLES = 512` (here frozen_test, n=512), whereas the standalone
   `parity` command uses the whole surface (here dev_val, n=2174) — so
   `parity[int8]` legitimately differs between the gate report (0.013672 /
   0.583879 on frozen_test) and the committed `parity-rc1-int8.json`
   (0.032199 / 0.716802 on dev_val). `summary.txt` printed neither the surface
   nor `n`, so the two looked contradictory. Fixed: summary parity lines now
   show `on <surface> n=<n>`; 2 regression tests. (Not verdict-affecting: the
   policy constrains only onnx 0.01/0.05, and fp32 onnx is exact either way.)

No other defect was found in the audited paths.

## 4. Developer workflow verification

* Python suite: **407 passed, 17 skipped, 0 failed** (`training/tests`).
  Skips are all justified missing-data skips (11 SKZ probe, 6 experiment-2
  external), never silent passes.
* C++: `cd runtime/build && ctest --output-on-failure` → **2/2**.
* CLI contracts: `evaluate` without `--policy` → exit 2; `validate-suite`
  non-strict → 0, `--strict` on missing required surface → 1; `env-readiness`
  → 1 (0/5 verified failure environments). `ci.yml` parses; both jobs (`python`,
  `cpp-build`) present.
* New scripts compile; `env-validate --dir datasets/staging/trier` → 0.

## 5. RC1 baseline

`best.pt` `780d20ad…a279` (15,359,069 B) · `model.onnx` `c46537bd…4548`
(5,053,216 B) · `model_int8.onnx` `d8ab7b98…01ef` (1,454,746 B). Split CSVs
`val.csv` `0c552f89…`, `test.csv` `18155da2…`; `release-rc1/splits/{val,test}.csv`
are byte-identical to `printing-errors/splits/{val,test}.csv` (`splits.json`
lives under `datasets/processed/printing-errors/splits/splits.json`).

## 6. Reproducibility of RC1's headline result

Re-scoring the committed checkpoint reproduces the frozen original test exactly:
`TN=798, FP=80, FN=372, TP=1609` (P=0.9526, R=0.8122, FPR=0.0911 @0.5),
matching `rc1-core-eval.json`. The scoring path, checkpoint, and `FAIL_IDX=0`
convention are consistent. The standalone int8 parity file also reproduces
byte-for-byte (0.032199 / 0.716802). **However**, the FPR = 1.0 normal-holdout
number is **not re-executable offline**: its frames are not committed and the
source is non-redistributable — confirmed-by-artifact and corroborated, not
re-run. The gate reports that surface INCONCLUSIVE, never pass.

## 7. Real-data findings (the decisive result)

* **Real-footage run artifacts** (committed C++ ONNX Runtime outputs on 7
  videos; `real-footage-analysis.json`): under the shipped run config
  (`use_delta:false, fail_threshold:0.3, fail_threshold_abs:0.65`), **all five
  VLM-normal videos fire** (video-level FAR = 1.0); baseline/RC1 median
  P(failing) is 0.74–0.99 on normal footage. RC1 is **worse** than the model it
  replaced on two normal videos (38gnsb frac≥0.9 0.35→0.77; nbkf372 0.83→0.93).
  exp3 suppresses everything including the true failure (median 0.000).
* **Stage-14 real failure frames** (`stage14-real-frames-eval.json`; labels are
  VLM-verified, `pending_human_review=true`, so diagnostic not benchmark): RC1
  scores 67/89 fail windows above 0.5 (75.3%), **and all 11 idle/empty frames
  above 0.5** (median 0.92).
* **Conclusion:** P(failing) is high on essentially any real frame — a
  representation/domain-shift failure, not a threshold or smoothing problem.
  Same mechanism as the FPR = 1.0 holdout failure. No threshold on this
  checkpoint fixes it.

## 8. Export / quantization parity

* **ONNX fp32: faithful** — prediction disagreement 0.0, max score drift 3e-06
  (exact).
* **INT8: marginal** — dev_val (n=2174, standalone): disagreement 0.0322, max
  drift 0.7168, mean 0.0351, FPR +0.0410, recall −0.0007; frozen_test (n=512,
  gate): disagreement 0.0137, max drift 0.5839. Not gated by policy (no int8
  thresholds), but the FPR inflation is a real effect, not negligible.

## 9. C++ runtime validation

`ctest` 2/2 (behavioral smoke against the framework reference). Not benchmarked
on target hardware; C++ tests skip when the gitignored `model.onnx` is absent,
so a fresh clone runs fewer C++ tests.

## 10. Temporal validation

Synthetic (`temporal-synthetic-benchmark.json`): the default delta-only logic
misses gradual drift; an absolute floor (`fail_threshold_abs=0.6`) catches it.
Real footage: the temporal layer does **not** suppress the false alarms — FAR
stays 1.0 with the shipped config. The run config is more aggressive than
`temporal.py` defaults; both are documented, neither is silently changed.

## 11. Dataset and licensing readiness

`dataset-inventory.md` + `DATASETS.md` record provenance/licence per set.
Blocker: **SKZ licence unresolved** (CC BY-SA vs CC BY) → cannot ship. Staging
tooling (`env-validate`, `env-split`, `review-prep/confirm/merge`) and capture
tooling (`tv capture`) are present and pass their checks; capture needs a camera
and was not exercised here.

## 12. Release-gate outcome (honest)

`docs/research/release-gate-rc1/summary.{json,txt}`, exit code **2**.
`RELEASE GATE: FAIL (blocks_release=True)`.

| gate | status | reason |
|---|---|---|
| dev_validation | FAIL | recall 0.8603 < min 0.9 |
| locked_test | FAIL | recall 0.8430 < min 0.9 |
| normal_holdout | INCONCLUSIVE | required surface missing |
| failure_holdout | WARN | optional, not present |
| cross_environment | WARN | optional, not present |
| parity | PASS | fp32 onnx exact |
| deployment | PASS | size 5.05 MB, latency within bound |

Threshold selection PASS: threshold 0.3 (max recall with FPR ≤ 0.2 on dev_val,
n=2174: P=0.9023, R=0.8603, F1=0.8808, AUROC=0.9310; frozen_test n=2859:
P=0.9361, R=0.8430, F1=0.8871, AUROC=0.8922). No leakage detected.

## 13. Evidence tiers — what is and is not verified

* **Verified (artifact/local-test):** frozen-test reproduction; fp32 onnx
  parity; C++ smoke; config loadability (after fix); gate mechanics/exit codes;
  threshold selection uses dev_val only.
* **Verified (real-data):** P(failing) uncalibrated on real frames; temporal
  layer does not suppress normal false alarms; RC1 no better (worse on some)
  than its predecessor on real footage.
* **Not yet verified:** target-hardware (Raspberry Pi) performance — procedure
  only; environment-independent evaluation (surfaces share one venue);
  INT8 sign-off; SKZ licensing.
* **Confirmed-by-artifact, not re-executed:** FPR = 1.0 on the stage-13 normal
  holdout.

## 14. Recommendation and RC2 prerequisites

**Do not release RC1.** The gate is right to block it: it misses the recall
floor and the evidence that exposed the false-alarm failure cannot even be
re-run. RC2 is a **data/problem** step, not a code tweak.

Ordered prerequisites:

1. **Build one human-verified, environment-diverse holdout** (normal + failing,
   ≥2 venues/sources). This is the gating item; lock it **before** any tuning
   (`evaluation-independence.md` gives the criteria). Everything else is
   unverifiable without it.
2. **Resolve the SKZ licence.**
3. **Train RC2 on a domain-robust mixture** — the prepared, unrun
   `training/configs/planned/rc2_domain_robust.yaml`, evaluated on (1).
4. **Run the Raspberry Pi benchmark** using
   `raspberry-pi-benchmark-procedure.md`.
5. **Re-run the release gate**; a FAIL is acceptable, an unsupported PASS is
   not.

The immediate codebase is healthy: tests, C++, CLI contracts, parity, and the
gate all behave as specified, and the two source defects found are fixed with
regression coverage.
