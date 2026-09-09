# TinyVision: Final Experimental Synthesis

**Date:** 2026-09-07
**Scope:** Consolidates Stages 12C–14 plus all prior measured work (Experiment 1,
SKZ Experiments A and B, Experiment 2, Exp3/Exp4, POC, C++ parity, real-world
evaluation).
**Status:** STOP AND SYNTHESIZE. No new experiment, no new data, no model or
threshold or architecture change was made by this task. Only one factual/indexing
correction was applied to a Stage-14 report (SKZ numbers standardized to the
canonical `skz_eval` subset — see §3.4 correction note).
**Machine-readable mirror:** `docs/research/tinyvision-final-synthesis.json`

---

## 1. Complete experiment timeline

Historical reports are the authority; summaries here reference them rather than
repeat tables.

### Stage 12C — External pilot dataset (2026-09-05)
`docs/research/stage12c-external-pilot-dataset.md`
Converted the six Stage-12B videos (SP001, SP002, NORMAL001, UE001, UE003,
STR001) into a deterministic 1-FPS external pilot dataset: 1,636 extracted →
1,546 kept after exact-md5 dedup. Only **36 labeled frames** (21 normal from
NORMAL001, 15 failing from SP001), all **provisional and unverified**. The
validation split was degenerate (zero labeled frames). Diversity analysis showed
the external pilot is measurably distinct from Kaggle (darker, higher phash
distance) — evidence of genuine domain shift. Conclusion: far too little data for
Experiment 2; labels are the blocker.

### Stage 12D — Human review workflow (2026-09-05)
`docs/research/stage12d-human-review-workflow.md`
Built the manifest + contact-sheet workflow (1,006 units = 86 sequence + 920
frame; 715 sequence-reviewable seconds; 86 sheets). Key evidence-based finding:
the acquisition criterion must be **"continuous footage first, failure type
second"** — tutorial/montage videos are low-value training data; stable printer-cam
feeds are the valuable kind. SP001/NORMAL001/UE003 were the only high-value
sources; SP002 and STR001 essentially contribute nothing.

### VLM review pipe (Stage 12D-extra + Stage 12E, 2026-09-05)
`docs/research/stage12d-extra-vlm-review-pipe.md`, `stage12e-vlm-batch-audit.md`
A local `qwen3-vl-8b` (LM Studio) reviewer produces **opinions only — proposals,
never ground truth, never merged into labels**. Neutral prompts; strict JSON
validation; 120 high-value units reviewed (120/120 success, ~1h52m). SP001
revealed a VLM `stringing` vs structural `spaghetti` disagreement, unresolved at
that point.

### Stage 12F — First human review tranche (2026-09-05)
`docs/research/stage12f-first-human-review.md`
Six human-reviewed units merged into `verified_labels.json` (NORMAL001×3 =
failing/stringing; SP001×3 = failing/spaghetti) = **44 verified failing frames,
2 sources**. VLM agreement on the overlapped units was **0/6** (NORMAL001 VLM
calls normal; SP001 VLM calls stringing) — an audit observation, not an accuracy
metric. Manual human review was **stopped at 24 units** (982 undecided); UE003
was judged largely setup/non-print footage.

### SKZ probe — Experiments A and B (2026-09-06)
`docs/research/skz-probe-experiments.md`
166-image nozzle-cam probe, 76 independent groups, license unresolved
(CC BY-SA vs CC BY).
* **A (no training):** failure detection transfers almost perfectly (recall 0.966,
  FNR 0.034); **normal recognition does not** (candidate-normal FPR 0.92,
  AUROC 0.61).
* **B (+135 SKZ-aux, group-split):** frozen test improves on every metric
  (acc 0.836→0.878, FPR 0.170→0.060); the held-out SKZ-normal FPR problem
  persists (~1.0). 31-image slice too small to be robust.

### Experiment 2 — Verified video-frames augmentation (2026-09-06)
`docs/research/exp2-external-video-augmentation.md`
Trained frozen train + **44 human-verified failing frames** (2 sources).
Frozen test regressed (failing recall 0.839→0.628; acc 0.836→0.738); FPR
improvement was a side effect of the normal-leaning shift, not a corrected
mechanism. UE003 normal FPR stayed ~1.0. Verified set too small; **adding failing
frames does not fix normal false alarms.**

### POC inference (2026-09-06)
`docs/research/poc-inference-architecture.md`
End-to-end loop verified: capture → eval-transform → ShuffleNet → EWMA → FSM →
alert, no train/deploy drift. All known-failing sources alert quickly; UE003
false-alarms as expected. Sustained 1 fps cadence with large headroom. **Verdict:
architecture viable; data is the blocker.**

### C++ runtime and parity (2026-09-06)
`docs/research/cpp-runtime-parity.md`, `edge-runtime-benchmark.md`
C++/ONNX Runtime reference runtime with a bit-exact Pillow-bilinear resize port,
aligned sampling, bit-exact temporal port. **Parity: state sequences and
first-alert frames exact on all four frozen videos; raw-P within the measured
decoder-build floor (0.15).** Inference ~2.2 ms; cadence 19–53 fps on the dev
machine. Edge target (Raspberry Pi) not yet benchmarked.

### Independent real-world evaluation (2026-09-06)
`docs/research/real-world-video-evaluation.md`
Frozen C++ pipeline against **7 independent YouTube videos** (3 normal, 4 failure).
Result: FAILING for **96–99% of every video**, including two VLM-confirmed
continuous real-time normal prints (`nbkf372s_cA` raw mean 0.924, `UzULqvGoi1Y`
0.750). Two spaghetti clips detected plausibly. **"Normal-print false alarming is
the dominant failure mode on external footage."** No software problem found:
Python/C++ identical.

### Stage 13 — Target-domain normal augmentation, Experiment 3 (2026-09-07)
`docs/research/stage13-experiment.md`
Trained frozen train + **333 VLM-screened normal frames from 7 independent
sources** (pending human review). Held-out normal FPR **1.000 → 0.258**; 7-video
normal clips ~7–8× down. Cost at fixed threshold 0.5: frozen failing recall
0.839→0.751; the confirmed spaghetti `87gsL-Luvis` is now **missed entirely**
(raw-failing 0.000); `mGH3p3AF1YY` weakens (1.000→0.562); SKZ failure recall
collapses (1.0→0.10). Hypothesis (normal diversity drives the FP tail)
**confirmed**; order of magnitude of the claim is ~26% FPR on held-out normals.

### Stage 14 — Failure balancing, Experiment 4 (2026-09-07)
`docs/research/stage14-experiment.md`, `stage14-target-failure-dataset.md`
Trained frozen train + 333 normals + **100 VLM-verified failing frames from 2
independent sources** (70 spaghetti `ZYfXW0JN2k4`, 30 blob `3zXQrw0E-Ag`;
pending human review). Frozen failing recall achieved **0.950** (best), FNR 0.050;
the missed spaghetti `87gsL-Luvis` is re-detected (0.000→0.212, first event 9);
SKZ failure recall 0.10→0.40. **But** held-out normal FPR returns to **1.000** and
7-video normal clips return to ~39–54% raw-failing. The balancing hypothesis is
**refuted as stated**: count-based rebalancing does not yield a stable operating
point; the two desired properties never co-exist.

---

## 2. What is proven

Three confidence tiers are used: **PROVEN** (measured, reproducible,
cross-checked), **SUPPORTED** (consistent with multiple surfaces but involves an
interpretive step), **NOT PROVEN** (§3).

### Runtime

| Claim | Tier | Evidence |
|---|---|---|
| C++ implementation is behaviorally equivalent to the Python POC | **PROVEN** | State sequences, sampled-frame counts, first-alert frames exact on 4 frozen videos (cpp-runtime-parity.md) and 2 independent videos (real-world §4); raw-P within the 0.15 decoder-build floor |
| Preprocessing parity is established | **PROVEN** | Pillow-bilinear port is bit-identical to PIL/torchvision resize; normalized input matches |
| Temporal logic parity is established | **PROVEN** | EWMA + FSM port bit-exact on synthetic golden traces (cpp-runtime-parity §Temporal) |
| Inference pipeline works | **PROVEN** | End-to-end loop, correct fire-once semantics, alarm generation; 344 training tests + C++ ctest pass |
| Failure handling works | **PROVEN** | Missing/malformed model and session errors with clear failures (C++ tests); bounded VLM/HTTP error handling in review pipe |
| Performance sufficient for the intended 1 fps cadence on the dev machine | **PROVEN** | C++ 19–53 processed fps (infer ~2.2 ms); POC 13–27 fps; huge headroom at 1 fps (edge-runtime-benchmark.md). Note: dev machine = Apple M1 Pro, **not** the edge target |

### Model

| Claim | Tier | Evidence |
|---|---|---|
| Frozen model detects obvious failures | **PROVEN** | Frozen test failing recall 0.839; both confirmed spaghetti clips get high raw P; VLM independently confirms gross failure |
| Failure detection transfers to independent footage | **PROVEN** | SKZ probe recall 0.966 (FNR 0.034); `87gsL-Luvis`, `mGH3p3AF1YY` correctly alarmed |
| Model has severe out-of-domain normal false-positive behavior | **PROVEN** | 100% FPR on held-out normals; 96–99% FAILING on confirmed normal real prints; SKZ candidate-normal FPR 0.92 |
| Target-domain normal augmentation substantially reduces that problem | **PROVEN** | Held-out normal FPR 1.000→0.258; 7-video normal clips 7–8× down (all at fixed threshold 0.5) |
| Adding a small failure tail reverses the normal improvement | **PROVEN** | Exp4: held-out normal FPR back to 1.000; normal clips back to ~39–54% |
| Simple count-based class balancing does not produce a stable operating point | **SUPPORTED** | Exp3 (normal-biased, misses spaghetti) vs Exp4 (failure-biased, FPR 1.0). Both single-axis; counts here ranged 100/333 vs an 11.3k frozen base. Bounded claim: true for these distributions/augmentation sets |

### Data

| Claim | Tier | Evidence |
|---|---|---|
| Original public dataset is insufficiently representative of target-domain normal footage | **PROVEN** | Baseline normal FPR 1.000 on every held-out normal source; 96–99% FAILING on confirmed normal real-time prints |
| Continuous fixed-camera footage is substantially more informative than arbitrary stills | **SUPPORTED** | Acquisition outcome evidence (12B–12D: tutorials/montages/slicer screens rejected, continuous feeds yield usable data), sequence-based human+VLM review rationale. Not itself a head-to-head measured effect |
| Source diversity matters more than simply increasing frame count | **SUPPORTED** | 44 frames/2 sources (Exp2) → no effect; 333/7 (Exp3) → large effect; 100/2 (Exp4) → boundary flip. Outcomes track sources+distribution, not count alone |
| Two-camera failure augmentation was insufficient and caused style/domain overfitting | **SUPPORTED** | 100 frames/2 sources → held-out normal FPR 1.000 (measured). The overfit-to-two-cameras interpretation is the consistent explanation but is inference |
| Target-domain normal data is the current dominant missing ingredient | **SUPPORTED** | Largest single measured effect (1.000→0.258 from 333 frames/7 sources); baseline FP dominates all external surfaces |
| Target-domain failure diversity is also insufficient | **SUPPORTED** | 2-camera failure set insufficient; in-domain failure robustness is fragile at threshold 0.5 (SKZ failure recall collapses under normal augmentation) |

---

## 3. What is NOT proven (must not be claimed)

1. **No quantitative real-world accuracy estimate.** The 7-video set is small and
   4/7 category labels were unreliable; no precision/recall on real footage was
   asserted (real-world-video-evaluation §6).
2. **No physical printer validation.** The pipeline was never run against a
   live, real print to verdict.
3. **No hardware deployment validation.** Latency/RSS measured on an Apple M1 Pro
   workstation only; Raspberry Pi 4/5 not benchmarked (explicitly open).
4. **No broad generalization guarantee.** Augmentation sources are 7 normal + 2
   failure environments; independent failure evidence is 2 clips + the SKZ probe.
   Per-source spread is large (held-out normal FPR 0.10 vs 0.42).
5. **No demonstrated under-extrusion robustness on truly independent
   target-domain footage.** No verified target-domain under-extrusion frames
   exist; the VLM grid is too coarse to adjudicate thin extrusion lines, and the
   two under/over-extrusion videos were VLM-classified normal.
6. **No evidence the threshold 0.5 is globally optimal.** Threshold is fixed by
   policy; the Stage-5 val sweep peaked F1 near 0.7.
7. **No release-ready checkpoint** (see §5).
8. **Not proven additionally:** that interpolation between Exp3/Exp4 counts would
   find a stable point (only adverse movement measured); that synthetic failure
   data works (unexplored); that INT8 quantization preserves behavior on the edge
   target (exported but unvalidated there); correct failure-type identification
   (only the binary decision is validated; VLM/human disagreement on failure types
   remains unresolved).

---

## 4. Architecture assessment

> **Is the current TinyVision architecture fundamentally viable?**

### Architecture layer (component evidence above, §2 Runtime)

`camera/video → Pillow-bilinear 224 resize → ShuffleNetV2 → ONNX Runtime → P(failing) → EWMA + FSM → event`

Every stage is **PROVEN**: preprocessing parity (bit-exact), inference
correctness, temporal logic parity (bit-exact), event generation, robust failure
handling, and sufficient dev-machine performance at the intended cadence. The
POC verdict (A — proceed) and the C++ parity results stand; tests are green
(344 training + 2 CTest pass).

### Data/model-limitation layer (not architecture)

All observed failures sit here:

* **Domain shift** — baseline cannot recognize arbitrary-camera normal prints.
* **Normal-print diversity** — the single largest measured cause of the FP tail.
* **Failure diversity** — 2-failure-camera assumption breaks down (style
  overfitting, boundary flip).
* **Source/style overfitting** — Exp4's held-out normal FPR 1.000.
* **Calibration/boundary instability** — both augmentation rounds flip the whole
  decision boundary at a fixed threshold.

### Conclusion

**No evidence justifies replacing the architecture. The architecture is sound
enough to preserve; the training data is the primary unresolved problem.** A
different model would not fix a dataset that cannot represent target-domain
normal footage, and the decision-boundary instability is a data-diversity effect,
not a pipeline defect. Do not redesign.

---

## 5. Checkpoint status

| Checkpoint | Status | Reason |
|---|---|---|
| Frozen baseline (`shufflenet_v2_x1_0_printing-errors_cdc9e2cad4`) | **Experimental / benchmark reference — NOT release** | FPR 1.000 on all held-out normal sources and 96–99% FAILING on confirmed normal real prints |
| Exp3 normal augmentation (`exp3_target_normal_aug_shufflenet_v2_x1_0`) | **Experimental — NOT release** | Fixes normal FPR (→0.258) but misses confirmed spaghetti `87gsL-Luvis` (0.000), SKZ failure recall 0.10, frozen failing recall −0.09 |
| Exp4 normal + failure augmentation (`exp4_target_balanced_shufflenet_v2_x1_0`) | **Experimental — NOT release** | Best failure side (frozen recall 0.950; spaghetti re-detected) but held-out normal FPR 1.000 and normal clips 39–54% |

No checkpoint is called "release candidate"; there is no evidence any satisfies
both failure sensitivity and normal false-alarm behavior.

---

## 6. Dataset status

| Dataset | Frames / unit | Role | Status |
|---|---|---|---|
| printing-errors (NilsHagenBeyer, Apache 2.0) — train | 11,254 | **Training** (all experiments) | Used; frozen |
| printing-errors — val | 2,174 | **Validation** (frozen, unchanged) | Byte-identical in every derived split (sha verified) |
| printing-errors — test | 2,859 | **Evaluation** (frozen) | Byte-identical in every derived split (sha verified) |
| SKZ probe (CC BY-SA/CC BY unresolved) | 166 | **Experimental only** | 135 in ExpB train-side; 31 held out (`skz_eval`); candidate-normal labels not verified |
| External pilot video-frames | 1,546 kept | **Experimental only / eval-only / unsuitable** | 44 verified failing frames used in Exp2 train; 24 UE003 normal = eval-only; rest uncertain/unlabeled |
| Stage 13 normal (Standard YT license) | 333 + 120 | **Experimental only / evaluation** | 333 train-side (Exp3/Exp4), 120 held-out eval (FPR surface); **pending human review** |
| Stage 14 failure (Standard YT license) | 100 | **Experimental only** | Exp4 train augmentation only; **pending human review** |
| 7-video external holdout | 7 clips | **External holdout** (eval-only, never trained on) | Runs baseline/exp3/exp4 in `real-world-eval/` |
| POC / parity videos (NORMAL001, SP001, UE003, STR001) | — | **Evaluation / parity** | Never trained on |

### Leakage boundaries (verified this task)

* Frozen `val`/`test` byte-identical across all derived splits (sha256 compared).
* No within-file duplicate rows in any train split.
* No 7-video, UE003, Stage-13 held-out-eval, or SKZ-eval frame appears in any
  training split (programmatic check over the union of all train splits).
* Source-atomic grouping throughout (a source is never split between roles).
* VLM proposals, provider labels, and SKZ candidate-normals are held out of
  training with two exceptions by design: Exp2's 44 verified frames and ExpB's
  135 SKZ-aux images (train-side, explicitly so; eval used only held-out tokens).
* Exp3/Exp4 trained on VLM-**screened** (not human-verified) labels — a declared
  caveat, not a silent promotion.

---

## 7. Human review status

* **Human-verified to date:** 44 failing frames (NORMAL001 stringing + SP001
  spaghetti) in `verified_labels.json` (sha `27038df1…`, unchanged). Used by
  Exp2 only.
* **Pending human review:** Stage 13 **333 train + 120 eval** normal frames
  (VLM-screened `decision==normal AND part_visible`); Stage 14 **100 failure**
  frames (VLM-verified contiguous windows). Both recorded as
  `pending_human_review` in their manifests.
* **The VLM is not ground truth.** The Stage-12F audit overlapped six units:
  **0/6 agreement** (NORMAL001: VLM=normal vs human=stringing; SP001:
  VLM=stringing vs human=spaghetti). VLM decisions (and especially failure-type
  and "normal" decisions) must never be silently promoted to verified.
* **Because of this, Exp3/Exp4 results carry a label-quality caveat**: they were
  trained on VLM-screened labels. Re-signing or retraining after a changed human
  pass is a required step before release use.

### What must happen before any dataset is release-quality training data

1. Complete a **human visual pass** over the 333 + 120 Stage-13 frames and 100
   Stage-14 frames using the Stage-12D workflow (sequence contact sheets; per-unit
   `normal`/`failing`/`uncertain`; optional failure type and transition bounds).
2. **Merge** accepted decisions to a verified labels artifact with
   `provenance: human_reviewed`.
3. Require **visible-printing-evidence justification** for `normal` labels
   (absence of failure ≠ verified normal), per the Exp-3 design lesson.
4. Re-check agreement vs VLM proposals as a calibration signal only.
5. If the human pass changes material labels, **retrain or explicitly re-sign**
   the Exp3/Exp4 conclusions with the corrected labels.
6. Resolve **licensing** before any redistributable use; current YouTube-sourced
   data is Standard-YouTube-License (non-redistributable).

---

## 8. Final engineering verdict

1. **Is TinyVision's current software/runtime architecture worth keeping?** **Yes.**
   Preprocessing, inference, temporal logic, event generation, and failure
   handling are proven, parity-checked, and fast enough on the dev machine at the
   target cadence. Nothing in the evidence implicates the architecture.
2. **Is the current model release-ready?** **No.** Baseline FP-ridden; Exp3
   normal-biased (misses confirmed spaghetti); Exp4 failure-biased (normal FPR
   back to 1.000). No checkpoint has both properties.
3. **Is the current training data sufficient for deployment?** **No.** Domain
   shift on normal footage is the dominant deficit; failure diversity is also
   insufficient; labels for everything outside the 44 verified frames are
   VLM-screened and pending human review.
4. **What is the primary technical blocker?** **Target-domain training data
   diversity** — a diverse, source-grouped, human-verified corpus of continuous
   normal printing (and a controlled, diverse failure set), not model/architecture
   or threshold tuning.
5. **Highest expected-information-value experiment if development resumes?**
   A **data-collection program** per §9 followed by an identical retraining and
   re-evaluation on the frozen surfaces (four surfaces used in Stage 13/14). This
   is the only intervention the evidence says moves the needle. A close second
   place: validating synthetic/nightly failure data (flagged in the original
   research, still unproven).
6. **What should NOT be attempted again?**
   * More count-based rebalancing tweaks on the current 7-normal/2-failure-source
     YouTube corpus (both directions measured; no stable point).
   * Threshold tuning against the frozen external sets (would convert evaluation
     into tuning).
   * Training on SKZ candidate-normal labels as ground truth.
   * Trusting titles/channel claims ("failure"-labelled videos that are montages
     or tutorials were repeatedly shown to be unusable; "normal"-labelled video
     contained stringing).
   * Another arbitrary "Stage 15" experiment without the data first.
7. **What would a genuinely production-quality dataset look like?** See §9.
   Minimum defensible bar: enough independent printer-camera environments that a
   source-out retrain leaves failure recall and normal FPR simultaneously
   acceptable at a fixed threshold — currently not achievable at any measured
   count/distribution.

---

## 9. Future data requirements (collection specification)

The unit of independence is the **printer-camera-environment session (a
   "source")**, not the frame. Raw frame counts are secondary; the evidence shows
   333 normal frames from 7 sources changed the FP behavior while 100 failure
   frames from 2 sources reversed it — so **source/environment diversity is what
   counts.**

### Normal data (the binding constraint)

* **≥ ~10 independent normal sources** (distinct printer + camera + environment;
  prefer distinct creators/locations); **≥2–4 reserved entirely** as held-out
  evaluation.
* Each source: **continuous, fixed-camera, real-time footage of a full print
  (start to finish)** — live-cam/printer-cam style, not tutorials, montages,
  timelapses, slicer screens, or control-panel footage (the repeated acquisition
  failures all trace to these).
* Deliberately broad coverage: multiple FDM printers (Ender-class, Prusa, Bambu,
  Voron, …), camera placement (bed-level, side, top), lighting (room light,
  enclosure LED), backgrounds, colors/materials.
* Per-frame binary labels **human-verified** with visible-printing-evidence
  justification; `normal` must show successful printing, not merely absence of
  failure.

### Failure data

* **≥ 3–6 independent failure sources**, human-confirmed, spanning at least
  spaghetti and under-extrusion (and stringing/blob if available) plus their
  **onset/transition windows** (sequences beat isolated stills).
* Prefer live-cam feeds of real failures over "failure" tutorials; title claims
  are never ground truth.
* The 2-source evidence explicitly showed insufficiency; do not ship on fewer
  than 3 evaluated failure environments.

### Structure & process

* **Source-grouped splits** (train/val/eval segmented by source, never by frame).
* Human review via the existing Stage-12D workflow; VLM as a proposal/screener
  only; agreement statistics optional.
* Full metadata + license records per the dataset policy (name, source, URL,
  license, counts, resolution, collection method, limitations, retrieval date).
* Prefer **own/controlled collection** (own printers) for a redistribution-clean
  license; otherwise resolve licensing before any public release.
* Freeze the evaluation surfaces (frozen val/test, held-out normals, 7-video set,
  SKZ slice) and re-measure unchanged after any future retrain.

Do not invent a required frame count: the evidence supports a **source/environment
count and continuous-footage requirement**, not an arbitrary total.

---

## 10. Product-readiness statement

```text
Runtime: validated (dev hardware only)
Model: research prototype
Data: insufficient for release
Hardware: unvalidated
Real-world generalization: inadequate
Release candidate: none
```

Every clause is evidence-backed: "Runtime: validated" refers to behavior/parity on
the Apple M1 Pro dev machine; edge hardware (Raspberry Pi) is explicitly
unvalidated. "Real-world generalization: inadequate" reflects that all three
checkpoints fail at least one of the two required behaviors on independent
footage.

---

## 11. Repository integrity (verified this task)

* **Test suite:** `uv run --project training python -m pytest training/tests/` →
  **344 passed**; `ctest --test-dir runtime/build` → **2/2 passed**.
* **Frozen splits unchanged:** printing-errors train/val/test sha256 stable;
  all four derived train sets copy the frozen train verbatim and their val/test
  are byte-identical to frozen.
* **Historical evaluation data unchanged:** `verified_labels.json`
  (`27038df1…`) and `vlm_review_proposals.json` (`6457afdd…`) match recorded
  hashes; SKZ/real-world/stage-13/stage-14 JSON artifacts are consistent with the
  reports (one Stage-14 SKZ indexing correction applied, §3.4).
* **Experiment artifacts reproducible:** every experiment has `config.yaml` +
  `status.json` + checkpoints/ONNX; Exp4 config note corrected to the actual
  100-frame label count.
* **No accidental leakage:** programmatic check — no eval source in any training
  split; no within-file duplicate rows.
* One **factual/indexing correction** made during this review (allowed by the
  task): Stage-14 SKZ row standardized to the canonical `skz_eval` subset for all
  three models (was mixing the 166-image subset for baseline/Exp3 with the
  31-image subset for Exp4). Historical reports were otherwise not modified.

---

## 12. Resuming later

To resume development cleanly:

1. **Do the human review pass** (§7) — unblocks the current augmentation labels.
2. **Run the collection program** (§9) until ≥ ~10 normal + ≥3–6 failure
   independent environments are verified.
3. **Retrain identically** (same config/hyperparameters/seed) and **re-measure the
   four frozen surfaces unchanged**.
4. Then, and only then, consider secondary avenues (synthetic failure data,
   anomaly detection, deployment benchmarking on the Pi).

The project is left in a scientifically defensible state: a proven runtime, a
research-prototype model, a fully documented dataset/experiment trail, honest
negative results, and an explicit, evidence-grounded path to the missing
ingredient (target-domain data diversity).