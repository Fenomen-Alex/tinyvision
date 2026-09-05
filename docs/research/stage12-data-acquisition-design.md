# Stage 12 — Targeted Data Acquisition Design

**Date:** 2026-09-04
**Status:** Design only — no training, no downloads, no dataset modification

---

## Context

Stage 11B confirmed that leave-cluster-out degradation affects ALL failure types, not just stringing:

| Cluster | Type | Held-out FNR | Known-domain FNR | Delta |
|---------|------|-------------|-------------------|-------|
| 22122022 | stringing | 0.062 | 0.000 | +0.062 |
| 20122022 | stringing | 0.162 | 0.001 | +0.161 |
| 10012023 | under-extrusion | 0.016 | 0.005 | +0.011 |
| 05012023 | under-extrusion | 0.337 | 0.000 | +0.337 |

The 05012023 under-extrusion cluster (FNR=0.337) performed worse than either stringing holdout. Two recordings in that cluster were nearly undetectable (18_14_03: 1/65 detected, 14_32_45: 7/37 detected).

**Conclusion:** The bottleneck is not failure-type-specific. It is cross-domain/session generalization. The existing 105 training recordings come overwhelmingly from a single creator dataset (NilsHagenBeyer). Adding recordings from diverse sources, printers, materials, and cameras is the highest-value intervention.

---

## 1. What Data We Need

### Priority 1: Stringing

**Why:** 23 train recordings exist, but Stage 11B showed 20122022 cluster had catastrophic failures (3/5 recordings near-zero detection). The model memorizes session-specific stringing appearance rather than learning failure features.

| Property | Requirement |
|----------|-------------|
| Independent sessions | 6–10 new videos from ≥4 different creators/sources |
| Frames per session | 30–200 (extracted from video) |
| Useful conditions | Different printers (FDM brands), different filament colors, different camera angles, different lighting |
| Acceptable sources | YouTube timelapse of FDM print showing visible stringing artifacts between parts or on surface |
| Unusable sources | Discussion-only (no printing footage), heavily occluded print area, resolution <360p, CG renders, resin printing |

**Specific stringing variants to target:**
- Light stringing (thin wisps between columns)
- Heavy stringing (dense web-like artifacts)
- Stringing on curved surfaces
- Stringing with different filament colors (white shows more, black shows less)

### Priority 2: Under-extrusion

**Why:** 37 train recordings but 05012023 cluster showed FNR=0.337. Two recordings with visible under-extrusion patterns were nearly undetected, suggesting the model has not learned robust under-extrusion features.

| Property | Requirement |
|----------|-------------|
| Independent sessions | 5–8 new videos from ≥3 different creators/sources |
| Frames per session | 30–200 |
| Useful conditions | Different under-extrusion severities (partial gaps vs. complete layer failure), different materials, different cameras |
| Acceptable sources | FDM print with visible gaps, thin walls, or missing material in specific layers |
| Unusable sources | Over-extrusion (different failure), no visible print defects, resin printing |

**Specific under-extrusion variants to target:**
- Partial under-extrusion (gaps in 1–3 layers)
- Severe under-extrusion (multiple consecutive failed layers)
- Under-extrusion on top surfaces vs. vertical walls
- Under-extrusion with different filament colors

### Priority 3: Spaghetti / Major Failures

**Why:** Only 5 train recordings, 191 frames. Severely underrepresented. These are high-value failures (print has completely failed and must be stopped).

| Property | Requirement |
|----------|-------------|
| Independent sessions | 3–5 new videos from ≥3 different creators |
| Frames per session | 20–100 |
| Useful conditions | Different spaghetti formations (clumped, filament trail, full bed coverage), different stages of spaghetti progression |
| Acceptable sources | FDM print where the model has detached from the bed or lost layer adhesion, producing extruded filament loops |
| Unusable sources | Print failure discussion without footage, resin printing failures |

### Priority 4: Normal Printing

**Why:** 61 train recordings, 5391 frames. Adequate in quantity but all from the same dataset. Diverse normal footage helps reduce false positives on unseen setups.

| Property | Requirement |
|----------|-------------|
| Independent sessions | 5–8 new videos from ≥3 different creators/sources |
| Frames per session | 30–200 |
| Useful conditions | Different printers, materials, lighting, camera angles, print stages (first layer, mid-print, completion) |
| Acceptable sources | Smooth FDM printing with no visible defects |
| Unusable sources | Print failure footage, post-processing/cleanup, resin printing |

### Priority 5: Domain Diversity

**Why:** The existing dataset is from one creator with presumably similar printers/cameras. Stage 11B showed that even within this dataset, different dates/sessions cause substantial performance variation.

Domain diversity is achieved by satisfying the above priorities with sources from:
- Different YouTube channels / creators
- Different FDM printer brands (Prusa, Creality, Bambu Lab, Voron, etc.)
- Different camera setups (fixed webcam, phone, Raspberry Pi camera)
- Different lighting conditions (natural, artificial, mixed)
- Different filament types (PLA, PETG, ABS) and colors

---

## 2. Definition of "Independent Session"

An **independent session** is a set of frames that satisfies at minimum ONE of these conditions:

1. **Different physical print:** A distinct print job, even if recorded by the same camera on the same printer. Different G-code, different model, different day.

2. **Different recording session:** A distinct continuous recording period, even if the same physical print is being recorded. Different start/stop time, different camera angle, different lighting.

3. **Different source video:** A distinct video from a distinct creator/uploader. This is the strongest independence signal available for public video data.

**Operational rules:**

- If a source video contains cuts or multiple angles of the SAME physical print, all segments from that print form ONE session. The video should not be split across sessions.
- If a source video shows MULTIPLE DISTINCT prints (e.g., a compilation), each print should be extracted as a separate session if clearly identifiable.
- If two source videos clearly show the same physical print (e.g., same creator, same model, same filament color), they should be treated as one session.
- If independence cannot be determined, err on the side of treating them as the same session.

**Minimum independence for inclusion:** Each session must come from a source video that is visually distinguishable from all existing training recordings. A video from the same creator showing the same printer with the same filament does NOT add domain diversity, even if it is a different print.

---

## 3. Labeling Methodology

### Label schema

The primary task remains **binary per-frame classification**:

| Label | Definition |
|-------|-----------|
| `normal` | Frame shows the printer operating without visible failure artifacts |
| `failing` | Frame shows visible print failure (stringing, under-extrusion, spaghetti, or other) |
| `uncertain` | Frame where failure status is ambiguous (e.g., unclear whether a mark is a defect or shadow) |

**Do NOT use fine-class labels during acquisition.** The binary task is the priority. Fine-class labels can be added later if multi-class experiments are pursued.

### Labeling procedure

For each source video:

1. **Extract frames** at 1 FPS (or original FPS if < 1 FPS). Record the timestamp of each frame.

2. **Watch the full video** before labeling to understand the print, the failure type (if any), and the timeline.

3. **Label segments**, not individual frames. Identify contiguous spans of normal printing and contiguous spans of failure. Apply the label to all frames within each span.

4. **Mark transitions** as a separate span if the transition itself is ambiguous. Do NOT assume the first frame of a failure segment is the physical onset of failure.

5. **Mark uncertain frames** with `uncertain`. These will be excluded from training but retained for reference.

6. **Record provenance metadata** for each frame (see §3.1).

### 3.1 Minimum metadata schema

Every labeled frame must have:

```
path:           relative path from dataset root (e.g., normal/VideoCreator_001/frame_0042.jpg)
label:          "normal" | "failing" | "uncertain"
source_video:   unique identifier for the source video
source_url:     URL of the source video (if public)
source_creator: creator/uploader name or identifier
source_date:    upload date or recording date if known (ISO 8601 or "unknown")
failure_type:   "stringing" | "under_extrusion" | "spaghetti" | "none" | "unknown"
printer:        printer model if identifiable, else "unknown"
filament:       filament type/color if identifiable, else "unknown"
camera:         camera description if identifiable, else "unknown"
lighting:       "natural" | "artificial" | "mixed" | "unknown"
extraction_fps: frame extraction rate used
segment_start:  timestamp in source video where this labeled segment begins
segment_end:    timestamp in source video where this labeled segment ends
notes:          free-text observations (e.g., "stringing visible between two towers")
```

### Labeling quality rules

- **Visible failure only:** A frame is `failing` ONLY if failure artifacts are visible in that frame. A video titled "3D print failure" where the first 5 minutes show normal printing must be labeled `normal` for those frames.

- **Discussion ≠ failure:** A creator discussing a past failure while showing a successful print must be labeled `normal`.

- **No inference from titles:** Video titles and descriptions are context, not labels. "My 3D printer failed!" does not tell you WHICH frames show failure.

- **Transitions are explicit:** The boundary between `normal` and `failing` segments must be marked with a timestamp. Do not label the first potentially-failing frame as failing unless the artifact is clearly visible.

- **Uncertain is acceptable:** If you cannot determine whether a visual artifact is a real failure or a camera/compression artifact, label it `uncertain`. This is better than a wrong label.

---

## 4. Source-Video Acceptance Criteria

### Checklist

Rate each criterion: ✅ (met), ⚠️ (partial), ❌ (not met).

| # | Criterion | Weight | Notes |
|---|-----------|--------|-------|
| 1 | Actual FDM printing footage visible | REQUIRED | Must see the print bed/nozzle area during printing |
| 2 | Print area sufficiently visible | REQUIRED | ≥50% of the print area visible in most frames |
| 3 | Failure is actually visible (if failure video) | REQUIRED | Must be able to see the specific failure type being claimed |
| 4 | Stable or predictable viewpoint | HIGH | Fixed camera preferred; gentle movement acceptable |
| 5 | Adequate resolution | HIGH | ≥360p; 720p+ preferred |
| 6 | Can label both normal and failing segments | HIGH | Video must contain enough of both for meaningful labeling |
| 7 | Compression artifacts not dominant | MEDIUM | Heavy blocking/banding that obscures print details is disqualifying |
| 8 | No permanent overlay obscuring print area | MEDIUM | Watermarks in corner acceptable; full-screen overlays are not |
| 9 | Cuts do not make labeling impossible | MEDIUM | Frequent cuts are acceptable if each segment is labelable |
| 10 | Timelapse or real-time distinguishable | LOW | Timelapse is acceptable; just note extraction_fps accordingly |
| 11 | Camera movement minimal | LOW | Gentle movement acceptable if print area remains visible |
| 12 | Filament/printer identifiable | LOW | Helpful for metadata but not required |

### Decision rules

- **Acceptable:** Criteria 1–3 met, at least 4 of criteria 4–8 are ✅ or ⚠️.
- **Borderline:** Criteria 1–3 met but only 2–3 of criteria 4–8 are ✅. Accept with documented caveats.
- **Reject:** Any of criteria 1–3 not met, OR fewer than 2 of criteria 4–8 are ✅.

### What makes a source UNUSABLE

- No actual printing footage (discussion, timelapse of finished prints, unboxing)
- Print area completely occluded by printer housing
- Resolution below 360p
- Failure not visible in any frame (e.g., "my print failed" but the camera shows only the creator's face)
- CG/animated rendering of 3D printing (not real footage)
- Resin/SLA printing (different failure modes, different visual appearance)
- Video is a compilation of hundreds of clips with no way to identify independent sessions

---

## 5. Leakage Prevention

### Principles

1. **Source-level atomicity:** All frames from one source video stay in one split. Never split a video across train/val/test.

2. **Session-level atomicity:** If multiple videos show the same physical print, they must all be in the same split.

3. **No near-duplicate crossing:** Frames from different videos of the same printer/model should be placed in the same split if their visual similarity exceeds a threshold (to be determined during acquisition).

4. **Official test frozen:** The existing `datasets/processed/printing-errors/splits/test.csv` is never modified. External data never enters the official test set.

5. **External data is additive:** External data is added to training only, or to a separate external validation set. It never replaces or混入 the existing official splits.

### Split architecture for external data

```
datasets/
  processed/
    printing-errors/          # EXISTING — UNTOUCHED
      splits/
        train.csv             # Frozen
        val.csv               # Frozen
        test.csv              # Frozen
  external/
    video-frames/
      sourcing/               # Raw extracted frames, not yet labeled
      labeled/                # Labeled frames ready for training
        normal/
        failing/
      metadata.csv            # Frame-level metadata
      sources.csv             # Source video registry
      splits/                 # External-only splits
        train.csv
        val.csv               # External validation (NOT official val)
```

### Training data composition

When external data is used for training:

```
training_data = official_train (existing) + external_labeled_train (new)
```

The combined training CSV is generated at experiment time, never stored as a permanent file. This preserves reproducibility: the original `train.csv` is always available.

### Validation strategy

- **Official val** remains frozen and is used for compatibility checks only.
- **External val** is a held-out subset of external data (source-level held out).
- **Primary evaluation** remains the frozen official test set.

---

## 6. Initial Acquisition Target

### Recommended allocation

Based on Stage 11B findings (both stringing AND under-extrusion showed degradation; spaghetti is severely underrepresented):

| Failure type | New sessions | New source videos | Target frames | Priority |
|-------------|-------------|-------------------|---------------|----------|
| Stringing | 6–10 | ≥4 creators | 300–800 | Highest |
| Under-extrusion | 5–8 | ≥3 creators | 250–600 | High |
| Spaghetti | 3–5 | ≥3 creators | 100–250 | High |
| Normal | 5–8 | ≥3 creators | 250–600 | Medium |
| **Total** | **19–31** | **≥10 creators** | **900–2250** | |

### Why these numbers

- **Stringing 6–10:** Existing 23 recordings come from 6 dates (1 creator). Stage 11B showed 20122022 cluster (5 recs) was catastrophic. We need ≥4 independent sources to ensure the model is not memorizing one creator's stringing appearance.

- **Under-extrusion 5–8:** Existing 37 recordings come from 13 dates (1 creator). Stage 11B showed 05012023 (7 recs, FNR=0.337) was as bad as stringing. We need diverse under-extrusion appearances.

- **Spaghetti 3–5:** Existing 5 recordings, 191 frames. Severely underrepresented. Even 3 new independent sessions would more than double the representation.

- **Normal 5–8:** Adequate quantity (61 recs) but low diversity. 5–8 new sources with different printers/cameras will help reduce FPR on unseen setups.

### Frame extraction strategy

From each source video:
- Extract at **1 FPS** for real-time footage
- Extract at the video's native rate for timelapse (typically 0.5–2 FPS effective)
- Skip frames where the print area is not visible (e.g., creator talking to camera)
- Target 30–200 labeled frames per session (not thousands)

**Do not inflate frame count with high-FPS extraction.** 100 frames from one video are NOT 100 independent examples. They are ~5–10 seconds of one printing session. The value is in session diversity, not frame count.

---

## 7. Dataset Integration

### Directory structure

```
datasets/
  processed/
    printing-errors/                    # EXISTING — NEVER MODIFIED
      metadata.csv
      splits/
        train.csv                       # 11,254 rows (frozen)
        val.csv                         # 2,174 rows (frozen)
        test.csv                        # 2,859 rows (frozen)
      failing/
        Recording_.../*.png
      normal/
        Recording_.../*.png

  external/                             # NEW — SEPARATE NAMESPACE
    video-frames/
      sources.csv                       # Registry of all source videos
      metadata.csv                      # Frame-level metadata (per-frame labeling)
      labeled/
        normal/
          CreatorA_Video001/
            frame_0001.jpg
            frame_0002.jpg
            ...
        failing/
          CreatorB_Video003/
            frame_0001.jpg
            ...
      splits/
        train.csv                       # External training frames
        val.csv                         # External validation frames (source-level held out)
        test.csv                        # Optional: external test for ablation studies
```

### Metadata files

**`sources.csv`** — one row per source video:

```
source_id,source_url,source_creator,source_date,upload_date,failure_type_visible,printer,filament,camera,lighting,accepted,rejection_reason,notes
CreatorA_Vid001,https://youtube.com/watch?v=...,CreatorA,2024-03-15,2024-03-16,stringing,Prusai3MK3S,PLA white,Logitech C920,artificial,yes,,"Heavy stringing between towers"
```

**`metadata.csv`** — one row per frame:

```
path,label,source_id,source_url,source_creator,source_date,failure_type,printer,filament,camera,lighting,extraction_fps,segment_start,segment_end,notes
failing/CreatorB_Vid003/frame_0042.jpg,failing,CreatorB_Vid003,https://...,CreatorB,unknown,stringing,Creality Ender3,PLA red,RPi camera,natural,1.0,00:02:15,00:03:40,Stringing visible between two pillars
```

### Integration with existing training

A script (not yet implemented) will generate combined training CSVs at experiment time:

```python
# Pseudocode for combined training CSV generation
existing_train = read_csv("datasets/processed/printing-errors/splits/train.csv")
external_train = read_csv("datasets/external/video-frames/splits/train.csv")

combined = pd.concat([existing_train, external_train])
combined.to_csv(tmp_dir / "train.csv", index=False)
```

The existing official splits are never read from or written to during this process.

### Reproducibility guarantee

- Existing Experiment 1 results are reproducible from the frozen official splits alone.
- Stage 11B results are reproducible from the frozen official splits alone.
- Any future experiment using external data will document the exact external training set composition in its config.

---

## 8. Next Experiment After Acquisition

### High-level design

**Experiment 2: External data augmentation study**

1. **Baseline:** Current frozen Experiment 1 model (ShuffleNetV2 x1.0, threshold 0.65).

2. **Experimental condition:** Same model, same config, same frozen official test set. Training data = original train + external labeled data.

3. **Evaluation:** Frozen official test set (primary). External validation set (secondary, for ablation).

4. **Success criteria:**
   - FNR on the frozen test set improves (especially stringing FNR from 0.213).
   - FPR does not regress substantially (currently 0.103).
   - Leave-cluster-out performance (Stage 11B protocol) improves.

5. **What NOT to do:**
   - Do not replace the official test set.
   - Do not change the model architecture.
   - Do not perform hyperparameter search on the test set.
   - Do not change the threshold based on test results.

6. **Duration estimate:** ~2–3 hours (training + evaluation on combined dataset).

### Ablation dimensions (future)

If Experiment 2 shows improvement, subsequent ablations could explore:

- Effect of adding only stringing data vs. all failure types
- Effect of domain diversity (same-type data from many sources vs. few)
- Effect of external data proportion (10%, 25%, 50% of training data)
- Whether external data helps under-extrusion generalization specifically

These are NOT part of the immediate next step.

---

## Summary

### What data we are missing

The dataset has 105 training recordings from essentially one creator. It lacks:
- Cross-creator diversity (different printers, materials, cameras)
- Cross-session diversity for the same failure type
- Sufficient spaghetti representation (only 5 recordings)

### Exactly what data to acquire first

19–31 new independent sessions from ≥10 different YouTube sources:
- 6–10 stringing sessions (≥4 creators)
- 5–8 under-extrusion sessions (≥3 creators)
- 3–5 spaghetti sessions (≥3 creators)
- 5–8 normal sessions (≥3 creators)

Target: 900–2,250 labeled frames total.

### How to label it

Binary per-frame labeling (`normal` / `failing` / `uncertain`) with segment-based annotation, full provenance metadata, and explicit transition marking. No inference from video titles.

### How to preserve independence/leakage safety

Source-level atomicity (all frames from one video in one split). External data in separate namespace (`datasets/external/`). Official splits frozen. Combined training CSVs generated at experiment time only.

### Proposed directory/metadata structure

`datasets/external/video-frames/` with `sources.csv`, `metadata.csv`, `labeled/`, and `splits/`.

### Proposed next experiment

Experiment 2: Same ShuffleNetV2 x1.0, same frozen test set, training data augmented with external labeled data. Compare against frozen Experiment 1 baseline. ~2–3 hours.

---

*Stage 12 design complete. No training, no downloads, no dataset modification.*
