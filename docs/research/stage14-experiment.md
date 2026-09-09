# Stage 14 — Target-Domain Failure Balancing (Experiment 4)

**Status: complete — experiment STOPPED, no release candidate identified.**

## 1. Question

After Stage 13 showed that adding 333 target-domain **normal** frames fixed the
persistent normal-print false-positive tail (held-out FPR 1.000 → 0.258) but made
the model too **normal-biased** (external confirmed spaghetti `87gsL-Luvis` missed
entirely, SKZ failure recall collapsed), Stage 14 asked:

> Did adding a small (100-300 frame) diverse target-domain **failure** set on top
> of the Stage-13 normal augmentation successfully balance the model — restoring
> failure detection while retaining the normal-FP improvement?

## 2. Method

Identical architecture / preprocessing / temporal logic to the frozen baseline
(`shufflenet_v2_x1_0`, seed 42, epochs 30, Adam 1e-3). Exp4 train = frozen
printing-errors train + 333 Stage-13 normal frames + **100 VLM-verified
target-domain failing frames** from 2 independent fixed-cam sources. Val/test
are byte-identical copies of frozen printing-errors splits. Threshold 0.5
everywhere; no threshold tuning.

Failing set: `ZYfXW0JN2k4` (70 spaghetti/blob, unknown blue-filament printer) +
`3zXQrw0E-Ag` (30 blob-of-death, Bambu A1). All frames in VLM-verified contiguous
failure windows; boundary healthy/idle frames excluded; `pending_human_review`.

## 3. Results — four surfaces

### 3.1 Frozen original test (failing = positive, threshold 0.5)

| metric | baseline | Exp3 | Exp4 |
|--------|---------|------|------|
| precision | 0.695 | 0.606 | **0.701** |
| recall (failure) | 0.830 | 0.866 | **0.950** |
| F1 | 0.757 | 0.713 | **0.807** |
| FNR | 0.170 | 0.134 | **0.050** |
| AUROC | 0.900 | 0.877 | **0.925** |
| FPR | 0.162 | 0.249 | 0.180 |

Exp4 has the **best frozen-test failure detection of all three** checkpoints.

### 3.2 Held-out normal sources (LY3jlcO34f4, zkJgLanCSUw; 120 frames)

| metric | baseline | Exp3 | Exp4 |
|--------|---------|------|------|
| normal FPR | 1.000 | **0.258** | **1.000** |

**Decisive regression**: Exp4 labels *every* held-out normal frame as failing —
identical to the original baseline defect that Stage 13 fixed.

### 3.3 7-video external holdout (C++ runtime, validated config)

Raw-failing fraction (`raw_decision == failing` at threshold 0.5):

| clip | intended | VLM visual | baseline | Exp3 | Exp4 |
|------|----------|------------|---------|------|------|
| `nbkf372s_cA` | normal | normal | 0.961 | **0.110** | 0.514 |
| `UzULqvGoi1Y` | normal | normal | 0.870 | **0.141** | 0.543 |
| `0TY6LVvMpUo` | normal | normal | 0.926 | **0.114** | 0.393 |
| `38gnsbGqcx4` | failure | normal print | 0.787 | 0.036 | 0.360 |
| `G56LQfQCUnA` | failure | normal print | 0.847 | 0.411 | 0.448 |
| `87gsL-Luvis` | failure | **spaghetti** | 0.818, evt 8 | **0.000, none** | **0.212, evt 9** |
| `mGH3p3AF1YY` | failure | **spaghetti** | 1.000, evt 8 | 0.562, evt 9 | 0.500, evt 9 |

Exp4 **re-detects the confirmed external spaghetti `87gsL-Luvis`** that Exp3
missed (0% → 21%, first event at frame 9), but **regresses normal clips** from
~11-14% back to ~39-54% false alarms.

### 3.4 SKZ held-out slice (continuity reference, not optimized for)

All three models evaluated on the same canonical `skz_eval` subset (31 images,
16 groups):

| metric | baseline | Exp3 | Exp4 |
|--------|---------|------|------|
| failure recall | 1.000 | 0.100 | 0.400 |
| normal FPR | 0.909 | 0.273 | 0.364 |
| AUROC (failing) | 0.723 | 0.332 | 0.518 |

Exp4 partially recovers SKZ failure recall (0.10 → 0.40) at some normal-FP cost
(0.273 → 0.364). Continuity only; not a tuning target.

## 4. Interpretation

Exp4 **partially confirms and partially refutes** the balancing hypothesis:

- The failure side is fixed. External spaghetti missed by Exp3 (`87gsL-Luvis`) is
  detected again; frozen failure recall reaches 0.950 (best); SKZ failure recall
  nearly doubles. The 100-frame failure augmentation does restore failure
  sensitivity.
- The balance point was **overshot**. The normal side regressed *catastrophically*
  on the held-out sources (FPR back to 1.000) and *substantially* on real footage
  (normal clips ~11-14% → ~39-54%). Exp4 is almost as failure-biased as the
  baseline was, only with much better genuine-failure discrimination.

### Net assessment vs Stage-14 success criteria

- *Materially better failure detection than Stage 13 while retaining most normal
  improvement?* **No.** Failure detection is better, but the normal-FP
  improvement is *completely* lost on the held-out sources. The two required
  properties did not co-exist.
- *Is Exp4 clearly superior → freeze release candidate?* **No. Do not freeze.**
- *If none acceptable: diagnose root cause, stop.* **Diagnosis below; STOP.**

## 5. Root-cause diagnosis (no blind Stage 15)

1. **Count-based balance is not a valid proxy.** The 100 failure frames are
   sub-second near-duplicates from 2 cameras; the 333 normals are 30-35 s apart
   from 7 cameras. Effective diversity, not frame count, moves the boundary.
2. **Extreme domain scarcity vs 11.3k frozen frames.** Both 100-failure and
   333-then-100 adjustments flip the entire decision boundary. There is no
   stable operating point at this data scale; naive augmentation is dominated by
   whichever tail was added last.
3. **The two tails are anti-correlated at threshold 0.5.** No single checkpoint
   satisfied both surfaces. A per-surface threshold or per-source weighting is a
   different intervention (out of scope, would require threshold tuning) — not a
   data-balancing fix.
4. **Catastrophic held-out FPR (1.000) indicates strong overfitting to the two
   failure cameras' visual style**, not learned generalizable failure evidence.

## 6. Conclusion and recommendation

**The simple "add a small failure set to rebalance" hypothesis is not supported.**
Exp4 traded the entire Stage-13 normal-FP win for failure detection, producing a
third checkpoint with the *opposite* single-axis bias. Small-count target-domain
balancing at threshold 0.5 cannot satisfy both failure recall and normal FPR in
this setting.

**Stop experimentation.** Neither Exp3 nor Exp4 is a release candidate. The
evidence points to a representation/diversity gap rather than a data-count
problem:

- Recommended (future, not started): a *collection-based* approach — gather a
  diverse set of real target-domain footage (both normal and failure, many
  printers/cameras) at scale, or validate synthetic/simulated failure data
  (which the initial research report flagged as a high-potential avenue), rather
  than continued count tweaks on 2-camera YouTube sourcing.
- Do not ship either checkpoint for failure detection.
- Keep all three checkpoints archived; baseline remains the reference.

## 7. Required human review before any release use

- All 333 Stage-13 normal frames (VLM-screened only).
- All 100 Stage-14 failing frames (VLM-verified only, `pending_human_review`).
No human review was available in this session; it remains mandatory and
unstarted.

## Artifacts

- Derived split: `datasets/processed/external-aug-exp4-balanced/{splits,derived_manifest.json}`
- Dataset: `datasets/external/stage14-failure/{raw,screening,frames,manifest.json}`
- Config: `training/configs/exp4_target_balanced.yaml`
- Experiment: `experiments/exp4_target_balanced_shufflenet_v2_x1_0/`
- Comparison JSON: `docs/research/stage14-exp4-comparison.json`
- Runtime runs: `datasets/external/real-world-eval/exp4_runs/*.json`
- SKZ: `docs/research/skz-exp4.json`