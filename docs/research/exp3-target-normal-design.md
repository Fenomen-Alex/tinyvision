# Experiment 3 — Target-Domain Normal Data: Audit and Design (Planning Only)

**Date:** 2026-09-06
**Status:** Feasibility/design investigation only. **No training, no dataset changes, no label
changes, no experiment artifacts, no VLM review, no downloads.**
**Scope:** determine whether enough genuinely-normal target-domain printing footage exists to
make "normal augmentation improves out-of-domain normal recognition" a scientifically
worthwhile Experiment 3.

---

## 1. Bottom line

**The correct recommendation is option 2: not enough data — acquire a small targeted normal
set before implementing Experiment 3.**

- **Verified normal frames: 0.** `verified_labels.json` contains only the 6 human-verified
  *failing* units (NORMAL001×3 stringing, SP001×3 spaghetti; 44 frames). No unit carries a
  verified `normal` decision.
- **Human-review normal decisions in the manifest (unmerged): 8 units, 24 frames, all from one
  source (UE003), all within a single 27-second opening window.** Even counting these,
  independent normal sources = **1**, independent normal groups = **1**, camera/view diversity
  = **1**, temporal diversity = **one** continuous span.
- A controlled Experiment 3 needs at minimum **one independent normal training source AND at
  least one independent held-out normal evaluation source**. With a single candidate source
  (and zero verified ones), the minimal viable experiment **cannot be constructed** from
  existing data.

---

## 2. Current normal data (exact counts)

| Quantity | Count |
|---|---|
| Verified (`verified_labels.json`) normal frames | **0** |
| Verified normal units | **0** |
| Human-review manifest normal decisions | **8 units / 24 frames** |
| Independent normal sources (manifest-based) | **1** (UE003) |
| Independent normal groups | **1** (UE003 is a single continuous session) |
| Camera/view diversity | **1** (UE003, 1920×1080 @ 23.98 fps, fixed camera) |
| Temporal diversity | **1** (t = 0–26 s of one 367 s video) |
| Verified normal frames suitable for training | **0** |
| Manifest-normal frames whose labels are justified by visible printing evidence | **0 confirmed** |
| Independent sources suitable for a held-out normal eval set | **0** |

**Not counted** (per policy): provisional Stage 12B labels, VLM proposals (Qwen3-VL),
`uncertain` decisions, candidate normal labels from the SKZ probe, and unreviewed footage.

### The 8 manifest `normal` units (all UE003)

| Unit | Type | Time window (s) | Frames | Notes |
|---|---|---|---|---|
| UE003__frm0000 | frame | 0.0 | 1 | – |
| UE003__frm0001 | frame | 1.001 | 1 | – |
| UE003__frm0002 | frame | 2.002 | 1 | – |
| UE003__frm0003 | frame | 3.003 | 1 | – |
| UE003__frm0004 | frame | 4.004 | 1 | – |
| UE003__seq000 | sequence | 5–11 | 7 | – |
| UE003__seq001 | sequence | 15–24 | 10 | – |
| UE003__seq002 | sequence | 25–26 | 2 | – |

All 24 frames exist on disk (`extracted/UE003/sequence/frame_0XXX.jpg`, indexes match the
integer-second grid; 24/24 present). These are genuine human review decisions — they are
**not** reinterpreted here. But the following facts matter for *suitability*, distinct from
label validity:

- The window t=0–26 s is the **opening segment** of a "how to fix under-extrusion"
  demonstration video (UE003, "Creality Ender 3 V2 — Solving Under-extrusion").
- The immediately following units (t≥29 s, `UE003__seq003…`) were reviewed `uncertain`, with
  notes such as *"Printer is being setting up, no print progress or printed object visible"*.
- The normal units carry **no notes** and no per-frame justification; the label alone does not
  document visible printing evidence.
- Stage 12F stopped exhaustive UE003 review because the source has **low information value**
  (substantial setup/non-print footage).

Per the Experiment-3 definition of *useful normal footage* (actual printing visibly occurring,
printed object visible, print progressing, deployment-like view, temporal context, label
justifiable from visual evidence), the UE003 normal units are **weak candidates**: single
source, single short temporal span, adjacent to documented setup footage, no visual-evidence
notes. They are evaluation-references at best — and they are already committed to that role
(Exp 2 used exactly these 24 frames as the external-normal eval set: baseline FPR 1.0, 2A FPR
0.958). Consuming them for training would remove the only existing target-domain normal
reference signal.

---

## 3. Per-source availability (review state)

Source inventory from `review_manifest.json` (1,006 units) and Stage 12C metadata:

| Source | Video / channel | Decided | Undecided | Content (documented) |
|---|---|---|---|---|
| NORMAL001 | Timelapse / TheeHustleHouse | 3 failing (stringing) | 0 | “normal”-named but all 3 reviewed as stringing; vertical B-roll |
| SP001 | Live-cam fail / GXS 3D | 3 failing (spaghetti) | 3 frame (t=23–25) | Strongest failure feed; not normal |
| SP002 | Timelapse montage / BetterLifeCreations | 0 | 55 | 41 cuts in 55 s; unusable as continuous footage |
| STR001 | Slicer tutorial / 3D Printsmith | 0 | 520 (19 seq + 501 frame) | PrusaSlicer screens (`STRINGING_SOURCE_FAILED`); not printing footage |
| UE001 | Tutorial / Callum Coles | 0 | 311 (28 seq + 283 frame) | On-screen troubleshooting checklist; largely explanatory framing |
| UE003 | Under-extrusion how-to / NeedItMakeIt | 8 normal + 15 uncertain | 88 | Substantial setup/non-print footage; low information value (Stage 12F) |

Verified failing: 6 units / 44 frames (NORMAL001 + SP001). Everything else either is
`uncertain`, is undecided, or comes from sources documented as unusable for normal footage.
The 977 undecided units are dominated by STR001/UE001 (tutorial/slicer) and SP002 (montage);
none is established as continuous normal-print footage, and none is counted.

---

## 4. The actual data gap

- **0 verified normal frames**; even optimistic counting yields **1 candidate source / 24
  frames / 1 camera / 1 temporal window**.
- Stage 12C established that a meaningful external validation signal requires **≥ 5
  independent labeled sources**; minimum any experiment needs is several independent sources.
- Experiment 3 additionally requires that some of those sources be **held out** (never trained,
  never tuned). With 1 candidate source, train and eval cannot both exist.
- Public sources do not close the gap today (see §8): SKZ normal is explicitly excluded
  (candidate label, per policy); the Kaggle normal class is already the in-domain training
  class; other candidates are detection datasets whose `GOOD` labels are not verified
  TinyVision normals or are sensor (non-video) data.

**If the answer were counted as zero, the honest answer is zero — and it is.**

---

## 5. Minimum viable Experiment 3 (design, pending data)

Minimal requirement before implementation: **≥ 3 independent, human-verified normal sources** —
at least one for training augmentation and at least two (separate) held out for evaluation.
Target scale for scientific value, not frame-count inflation:

| Role | Minimum | Rationale |
|---|---|---|
| Normal training augmentation | ≥ 2 sources | avoid memorizing a single camera/scene |
| Normal held-out evaluation | ≥ 2 sources, ≥ 30–50 frames each | group-level FPR needs >1 independent source to mean anything |
| Failing augmentation (if any) | unchanged (none in Exp 3) | keep the failing question fixed |

### Baseline vs Experiment 3A

- **Baseline:** frozen `shufflenet_v2_x1_0_printing-errors_cdc9e2cad4` (stored results).
- **3A:** existing printing-errors train (11,254) **+ verified target-domain normal frames** from
  the training-side sources. Hyperparameters, seed, augmentation, and eval identical to the
  established method (Adam lr 1e-3, batch 32, epochs 30 / patience 10, seed 42, hflip / rot15 /
  colorjitter 0.2 / ImageNet-normalize, threshold 0.5 on P(failing)).
- **Primary question:** does target-domain normal augmentation reduce FPR on held-out
  target-domain normal footage?
- **Secondary question:** does failing recall on the frozen TinyVision test remain stable?

### Split construction (once data exists)

Reuse the Exp 2 derived-split mechanism (`prepare_exp2_splits`): copy frozen `train.csv`
byte-identically, append external rows as absolute paths, keep `val`/`test` byte-identical,
record a derived manifest with role-by-source, integrity sha256s, and caveats. Source-atomic:
a source is entirely train-side or entirely eval-side, never split.

---

## 6. Evaluation (proposed success criteria)

Use the project's existing metric conventions (failing-positive primary, threshold 0.5 on
P(failing); recall on the frozen test is the safety gate; FPR on external normal frames;
AUROC convention `roc_auc_score(y_true, 1 − fail_scores)`).

| Criterion | Metric | Baseline reference | Proposed success bar |
|---|---|---|---|
| **Primary** | FPR on held-out target-domain normal frames | 1.0 (UE003), 0.92 (SKZ candidate) | Material reduction with ≥2 independent eval sources; do not tune on this set |
| **Safety gate** | Failing-positive recall on frozen test | 0.8385 | No material degradation (recall loss ≲ 0.02) |
| **Secondary** | F1 / AUROC / accuracy on frozen test | F1 0.876, AUROC 0.900, acc 0.836 | No material degradation (F1 / AUROC loss ≲ 0.01) |
| Integrity | Val/test remain byte-identical to frozen | – | Mandatory |

The ~0.02/0.01 bars are conservative and justified by the magnitude of differences the project
already treats as meaningful (Exp 2's recall drop of 0.21 and AUROC drop of 0.03 were
material regressions; the report must state any chosen bar explicitly rather than tune it
after results exist).

Because the held-out normal set has few sources, external numbers remain **descriptive, not a
benchmark** — consistent with Exp 2 and SKZ reporting.

---

## 7. Acquisition specification (if we decide to collect)

Do **not** download hundreds of videos. The bottleneck is *diversity of genuinely normal
continuous printing*, verified by humans — not frame count.

**Target (justified, not arbitrary):**
- **5–10 independent normal-print sources** (distinct printer + camera + session; prefer
  distinct creators/locations where possible).
- **Several dozen useful, verified frames per source** (continuous, not sampled montage).
- **At least 2 sources reserved for held-out evaluation**; the rest for training augmentation.
- Net target: ~300–800 human-verified normal frames across 5–10 sources, with
  ≥2 independent eval sources. Fewer sources → report the experiment as under-powered, do not
  over-claim.

**Preferred footage type (from Stage 12D §6):**
- Continuous printer-cam / live-cam / remote-monitor feeds of a successful print running start
  to finish (live-cam channels over tutorial/montage channels).
- Full-length print runs (minutes to hours) captured by a fixed camera; the longer and more
  continuous, the better.

**Acceptable variation (deliberately broad):**
- Multiple FDM printers (Ender-class, Prusa, Bambu, Voron, etc.); different colors/materials;
- Interior/side/bed-level camera angles; different lighting (room light, enclosure LED);
- Some motion/noise is fine — this is a binary per-frame classifier, not a SLAM system.

**Reject:**
- Tutorials, montages, timelapses with micro-cuts, slicer/OCR screens, control-panel /
  touchscreen footage, setup scenes, hands obstructing the print, end-cards/credits, and any
  footage whose "normal" class is provider-inferred (`GOOD` label ≠ verified normal).

**Cost of wrong approach (already evidenced):** NORMAL001 (named "normal", turned out stringing),
STR001 (stringing tutorial = slicer screens), UE001 (tutorial checklist), SP002 (montage) — each
source must be **human-reviewed** via the Stage 12D/12F workflow before it becomes a label.
Expect to screen more sources than you keep.

---

## 8. Public sources — only if necessary

Existing relevant public material, honestly assessed:

| Candidate | Why it does **not** close the gap now |
|---|---|
| SKZ Nozzle-Cam probe (already fetched) | Candidate `normal` = “no defect box”, provider-inferred, **explicitly excluded** by Experiment-3 policy; license CC BY-SA 4.0 vs CC BY 4.0 unresolved |
| Kaggle “3d printing errors” (already training) | Its normal class is the **in-domain** training data; does not test out-of-domain normal |
| Roboflow seprinder / syLucauc, wengmhu (detection) | `GOOD`/normal boxes are not human-verified TinyVision normals; detection framing differs |
| joanna-/3D-Printing-Data (CC BY 4.0) | Sensor (IMU/interface) readings from one printer; not video camera footage |

**No public source currently provides human-verified, continuous, target-domain normal printing
footage.** A public source would still require the full human-verification and license pipeline
and is not a shortcut. Own acquisition of continuous printer-cam feeds + the existing review
workflow is the defensible path.

---

## 9. Risks

1. **Insufficient independent sources** — with 1 candidate source, any result is camera-specific
   noise; external numbers must stay descriptive.
2. **Label ambiguity** — “normal” is the least justifiable label (absence of failure ≠ visible
   successful printing). All labels must document visible printing evidence.
3. **Temporal correlation** — adjacent frames from one continuous run are not independent
   samples; do not inflate counts by treating frames as statistically independent.
4. **Camera/domain mismatch** — target-domain normal must resemble the intended TinyVision
   deployment (fixed printer camera), not B-roll/timelapse aesthetics (NORMAL001 lesson).
5. **Evaluation leakage** — no train/eval overlap, no source split across train and eval, no
   tuning on the held-out normal set, val/test kept byte-identical to frozen.
6. **Over-optimism from a training-only augmentation** — without a held-out target-domain eval
   set, a “normal augmentation improves frozen test” result does **not** prove improved
   generalization on target-domain normal footage. State this explicitly in any future report.

---

## 10. Recommendation

**Option 2 — Not enough data. Acquire a small targeted normal set before implementing
Experiment 3.**

- The research question is correct and aligned with all findings (failing data is not the
  constraint; out-of-domain *normal* recognition is).
- Existing footage cannot build even the minimum viable train/eval split: zero verified normal
  frames, one candidate source, one camera, one temporal window.
- Do not implement Experiment 3 training yet. First: (a) acquire 5–10 continuous normal-print
  sources per §7, (b) human-review them through the existing Stage 12D/12F workflow to produce
  `verified_labels.json` normal entries, (c) reserve ≥2 independent sources for held-out
  evaluation, then (d) implement the §5/§6 design with the same derived-split/eval method as
  Experiment 2.

**Validation performed for this planning task:** read-only audit only. No training, no dataset
change, no label change, no experiment-artifact change, no VLM call, no downloads. Git working
tree unchanged by this task (pre-existing uncommitted changes intact: `.gitignore`, `ML.md`,
`training/tv/cli.py`, `review_manifest.json` with its 5 pre-existing UE003 `uncertain`
decisions, plus prior Experiment-2/SKZ artifacts and untracked source files). Nothing committed.