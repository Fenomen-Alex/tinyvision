# Stage 12B — External Video Acquisition Pilot

**Date:** 2026-09-05
**Status:** Pilot acquisition + inspection of 6 public videos (research only). No training, no modification of `datasets/processed/printing-errors/`, no modification of the frozen official test set, no construction of the final external dataset.

---

## 1. Objective

Validate the external-video pipeline on a small acquired pilot batch before committing to the full Stage 12 acquisition (target 19–31 independent sessions from ≥10 creators, Stage 12A). For each of the 6 acquired videos we:

1. Preserve the original downloaded file + provenance (`.info.json`).
2. Extract deterministic sample frames.
3. Assess whether the footage is frame-labelable for the intended failure class.
4. Create pilot labels only where the evidence supports them.
5. Produce a decision report including an honest `STRINGING_SOURCE_FAILED` verdict if the stringing candidate is not labelable.

**Hard constraints honored (no violation):**
- No training run.
- No modification of `datasets/processed/printing-errors/`.
- No modification of the frozen official test set.
- No construction of the final external dataset.
- No downloads beyond the 6 acquired videos.
- No fabricated visual claims.

**Critical capability limitation (must be stated prominently):** this agent has **no image-input (vision) capability**. It cannot visually confirm a failure type in any frame. All labelability conclusions below are therefore based exclusively on **deterministic programmatic structure analysis** (frame-difference, camera-`motion`-cut detection, foreground-growth trend, and OCR text overlays). Visual confirmation is **NOT established** for any video and remains pending human review of the contact-sheet montages.

---

## 2. Pilot batch

Selected in Stage 12A (SP001, SP002, NORMAL001, UE001, UE003, STR001; UE002 excluded as a 15 s vertical short).

| ID | Source video (YouTube) | Codec | Res | FPS | Dur (s) | Size (MB) | Reviewer note |
|---|---|---|---|---|---|---|---|
| SP001 | `3Ci-3ZYHBkM` | h264 | 720x1280 | 30 | 25.5 | 4.1 | vertical Short, upload 2023-09-18 |
| SP002 | `qcVFnRcJcP4` | h264 | 1920x1080 | 30 | 54.97 | 29.2 | channel BetterLifeCreations, 2026-04-12 |
| NORMAL001 | `dX4eGEFrthE` | h264 | 1080x1920 | 30 | 20.07 | 5.0 | channel TheeHustleHouse, vertical, 2026-01-02 |
| UE001 | `6V319bseO3U` | h264 | 1280x720 | 25 | 519.64 | 67.4 | channel Callum Coles, 2021-03-29 |
| UE003 | `k-inUpjRVLA` | h264 | 1920x1080 | 23.98 | 367.62 | 87.7 | channel NeedItMakeIt, 2020-09-24 |
| STR001 | `kuAMtoUJGgE` | av1 | 2560x1440 | 23.98 | 645.06 | 194.7 | channel 3D Printsmith, 2024-11-28 |

All originals preserved under `datasets/external/video-frames/raw/<ID>/` with `*.info.json` provenance.

---

## 3. Method (reproducible)

- **Download:** `yt-dlp` v2026.08.19 (`--write-info-json`) for each video id.
- **Extraction:** `scripts/extract_pilot_samples.py` (deterministic). Videos ≤ 60 s → 1 frame/s; longer videos → evenly spaced to a target count. **207 sample frames** produced into `extracted/<ID>/sample/`; parameters recorded in `metadata/pilot_extraction_params.json`.
- **Analysis:** `scripts/analyze_pilot_footage.py` (OpenCV 5.0.0) computes at 1 Hz: gray mean/std, RMS frame-difference vs previous sample, fraction of changed pixels, and the bounding-box coverage of changed pixels. A sample is flagged `cut_flag` when change is large and spatially broad (camera swap / scene change). For spaghetti candidates, a foreground-growth trend is measured in the central crop vs an early reference. Full per-sample table in `metadata/pilot_footage_analysis.json`; per-video summary in `metadata/pilot_footage_summary.json`.
- **OCR:** `tesseract` (psm 6/11) on sample frames to detect on-screen overlay text (identifies tutorial/slicer content).
- **Montages:** `datasets/external/video-frames/montage_*.jpg` contact sheets retained for mandatory human visual review.

---

## 4. Findings per video

The "stable run" concept is central: a sample is **unstable** when flagged as a cut or when >30% of pixels changed across >50% of the frame (camera pan / scene change). Only the longest continuous stable run is a candidate for frame labeling.

### 4.1 SP001 (spaghetti/live-cam-fail) — **PARTIALLY_USABLE**

One continuous 22 s stable run (t=0–22), then a hard scene change at t=23. Foreground-growth trend (central crop vs t=0): area **0.008 → 0.137** from t=0→22 (**17× growth, slope ≈ 0.0067/s**) with the **largest-blob fraction climbing to ~0.08**, then a discontinuous jump to **area 0.726** at t=23 (camera delivery).

This is the strongest **structural** failure signal in the batch: a steadily growing object in the print area over a stable, well-framed feed, then an abrupt scene change congruent with post-failure handling. However — and this must be explicit — **the growing object is not visually confirmed to be a spaghetti failure**; it could be a normally-growing print. Failure-type confirmation requires human review or a future vision model.

### 4.2 SP002 (spaghetti/timelapse) — **UNUSABLE as acquired**

**41 scene cuts in 55 s (~44.8 cuts/min), 74.5% of samples unstable, and no stable run ≥ 3 s.** `gray_std` reaches 0.0 and `gray_mean` 0.0 (black/full-frame transitions). This is a heavily-edited montage (many micro-cuts) rather than continuous capture. As acquired it is not frame-labelable for a specific failure without dense per-segment human work, and even then the failure class cannot be anchored. **Does not contribute usable frames as-is.**

### 4.3 NORMAL001 (normal/timelapse) — **USABLE (framing) / content UNCONFIRMED**

Fully stable 20 s run, **0 cuts, 0% unstable**. However the high `gray_std` (≈66–71) and OCR watermark ("2|®") are consistent with a **vertical B-roll / aesthetic demo** rather than a continuous real-time production capture. As a *normal-class* candidate it is provisionally usable in framing terms, but **whether it depicts a genuine successful print (vs. pure aesthetics) is NOT confirmed**. Should not be relied on as failure-free ground truth without human confirmation.

### 4.4 UE001 (under-extrusion) — **PARTIALLY_USABLE**

520 s with **169 cuts** and **34% unstable**; the two longest stable windows are t=318–345 (27 s) and t=431–519 (88 s). **Deep OCR over the 431–519 window shows a persistent on-screen troubleshooting checklist** ("STRAIGHT EXTRUSION", "CLEAN EXTRUSION", "UNIFORM PTFE?", "RETRACT") — i.e. this is a **tutorial with on-screen diagnostic text**, not raw stable printer footage across the whole window. Labelable under-extrusion frames may exist within dense sub-segments, but the video is dominated by explanatory framing. Labelability requires sub-second human segmentation.

### 4.5 UE003 (under-extrusion) — **PARTIALLY_USABLE**

368 s with **51 cuts, 16% unstable**, and a long stable run t=148–256 (108 s). OCR across that window is low-contrast and ambiguous (no reliable text), so **content is not visually confirmable**. The long stable window is promising in *framing* terms, but the **presence of a visible, labelable under-extrusion failure is UNCONFIRMED**. Requires human review before any label is trusted.

### 4.6 STR001 (stringing) — **UNUSABLE → STRINGING_SOURCE_FAILED**

645 s with **321 scene cuts (~29.9/min), 52% of samples unstable, and the longest stable run only 17 s.** Deep OCR at the stable windows shows **PrusaSlicer settings screens** ("Retract on layer change", "Retract amount before wipe", "Dynamic overhang speed ... Enable dynamic overhang speeds") — i.e. the video is a **stringing tutorial dominated by slicer-configuration screen-captures**, switching rapidly between screens and any brief printer footage. There is **no long, stable run of continuous print-area footage**, and the stringing material is fragmented and likely narrated. **Result: the Stage 12A stringing candidate does not provide labelable frames. The stringing source is FAILED for this acquisition pass.** Per design, this is reported explicitly rather than silently swapping in another class. `STR_002` and `STR_003` remain the next stringing candidates to evaluate (not downloaded).

---

## 5. Pilot labels

Written to `datasets/external/video-frames/labels/pilot_labels.json`. These are **provisional, structure-derived, NOT ground truth**, and clearly marked `uncertain`/`MEDIUM`/`LOW` certainty throughout.

- **SP001:** 26 frame labels — t=8–23 `suspect_growth` (MEDIUM, "foreground object growing pre-cut"), t≥23 `scene_transition` (LOW), remainder `uncertain`. **No frame asserts "spaghetti" as fact.**
- **NORMAL001:** 21 frame labels — `normal_provisional` (MEDIUM), pending visual confirmation of a real successful print.
- **SP002, STR001:** labeled `UNUSABLE` (no usable frames); STR001 carries `STRINGING_SOURCE_FAILED`.
- **UE001, UE003:** labeled with a note only (`PARTIALLY_USABLE`, no frame labeled confidently) because content could not be visually confirmed.

**Labeling status:** 47 structure-derived frames (SP001+NORMAL001) + explicit unusable/uncertain markers for the rest. No visual model and no human reviewer ran during this stage, so **all of these are unverified for actual failure type.**

---

## 6. Deliverables

- `datasets/external/video-frames/raw/<ID>/*.mp4` + `*.info.json` — original files + provenance (source of truth).
- `datasets/external/video-frames/extracted/<ID>/sample/` — 207 sample frames.
- `datasets/external/video-frames/metadata/pilot_extraction_params.json` — extraction parameters.
- `datasets/external/video-frames/metadata/pilot_footage_analysis.json` — full per-sample metrics.
- `datasets/external/video-frames/metadata/pilot_footage_summary.json` — per-video structural summary.
- `datasets/external/video-frames/labels/pilot_labels.json` — provisional labels.
- `datasets/external/video-frames/montage_*.jpg` — contact sheets for human review.
- `scripts/extract_pilot_samples.py`, `scripts/analyze_pilot_footage.py` — reproducible tooling.
- `docs/research/stage12a-video-source-candidates.csv` — updated with acquisition/inspection status (original candidate info preserved).

---

## 7. Stage 12A CSV update

The `stage12a-video-source-candidates.csv` has been updated to add acquisition/inspection status columns to the six acquired IDs (SP_001, SP_002, NORMAL_001, UE_001, UE_003, STR_001) while **preserving all original candidate information**. Status summary:

| ID | Acquired | Inspected | Labelability (this pass) | % unstable | Longest stable run |
|---|---|---|---|---|---|
| SP_001 | yes | yes | PARTIALLY_USABLE | 7.7% | 22 s |
| SP_002 | yes | yes | UNUSABLE | 74.5% | 1 s |
| NORMAL_001 | yes | yes | USABLE (content unconfirmed) | 0% | 20 s |
| UE_001 | yes | yes | PARTIALLY_USABLE | 34.2% | 88 s |
| UE_003 | yes | yes | PARTIALLY_USABLE | 16.3% | 108 s |
| STR_001 | yes | yes | UNUSABLE (→ STRINGING_SOURCE_FAILED) | 51.6% | 17 s |

---

## 8. Conclusions

1. **Pipeline works end-to-end** (download → deterministic extraction → structure analysis → provisional labels → report) with full provenance and no dataset/test modification.
2. **Continuous real-time feeds are the realistic label source.** The single best structural candidate is **SP001** (one stable 22 s feed with strong object-growth evidence). The two long-duration candidates (UE001, UE003) are dominated by tutorial/screen content or are visually ambiguous and need dense human work. The two high-cut montages (SP002, STR001) are effectively unusable as acquired.
3. **The stringing source failed this pass.** STR001 is a slicer-screen tutorial with no long stable print-area run. This is reported as `STRINGING_SOURCE_FAILED`; STR_002/STR_003 should be evaluated before any class replacement is considered.
4. **No failure type is visually confirmed.** All conclusions are structural; the montages are retained for the mandatory human review. A future stage with a vision-capable reviewer or model is required to turn "object growth" into "confirmed spaghetti/under-extrusion failure."
5. **Dataset-creation risk is materially higher than the Stage 12A assumptions suggested.** Most "failure" videos are either heavily edited (unusable) or tutorial-dominated (only sub-segments labelable). This materially raises the per-video screening effort for achieving the 19–31 independent-session target.

---

## 9. Recommendation / next step (not executed)

- **Do not extend downloads yet.** First complete a **human visual review** of the six montages and the SP001/UE001/UE003 stable windows to (a) confirm failure types and (b) decide whether SP001's growth is truly a spaghetti event.
- **Replace the STRINGING source:** evaluate `STR_002`/`STR_003` (or other stringing sources) before committing.
- **Revise the acquisition plan** to favor (i) longer continuous real-time feeds, (ii) dedicated live-camera/printer-cam channels, over tutorial/montage channels that dominate search results.
- **Only then** build the Stage 12 candidate shortlist for the full acquisition, keeping the frozen official test set and `datasets/processed/printing-errors/` untouched.

**Stage stops here. No training initiated.**
