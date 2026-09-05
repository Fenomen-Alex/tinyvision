# Stage 12A — Candidate Source Videos

**Date:** 2026-09-04
**Status:** Candidate identification only — no training, no downloads, no dataset modification

---

## Objective

Identify 10–15 strong candidate public videos (primarily YouTube) that could later be converted into frame-level 3D-print failure datasets. The goal is **quality and independence**, not volume.

This stage catalogs candidates with structured metadata and scores them. **No videos have been downloaded.** Labelability fields marked `UNVERIFIED` must be confirmed by manual review before acquisition.

---

## Method

Searched public web sources (Google + YouTube) for real 3D-printing footage across the four priority categories:

- Stringing
- Under-extrusion
- Spaghetti / major failures
- Normal printing

For each candidate found, recorded the title, URL, creator, failure type, and any metadata available from search results and video descriptions. Camera stability, print visibility, resolution, and duration could not be visually confirmed without downloading, so these are flagged `UNKNOWN` pending manual review.

**Critical rule applied throughout:** A video being titled "stringing" or "under extrusion" is NOT treated as evidence that visible failure footage exists. Labelability is marked `UNVERIFIED` unless the search description or channel format strongly indicates continuous print-area footage.

---

## Candidate Details

### Stringing (priority 1)

#### STR_001 — "Understanding and fixing stringing in 3d prints"
- **URL:** `https://www.youtube.com/watch?v=kuAMtoUJGgE`
- **Creator:** unknown (needs identification during manual review)
- **Source date:** 2024-11-28
- **Video class:** tutorial / troubleshooting
- **Notes:** Description explicitly targets stringing and "the MOST important setting in your slicer." The visual centerpiece is likely a stringing test tower or a printed part showing webs between towers. Requires manual confirmation that a stringing print is actually shown and clearly framed.

#### STR_002 — "6 Ways to Fix Stringing on 3D Prints"
- **URL:** `https://www.youtube.com/watch?v=if1yztM4K9I`
- **Creator:** unknown
- **Source date:** 2025-06-09
- **Video class:** tutorial / troubleshooting
- **Notes:** Fix-format tutorial. Likely shows a stringing print before and after adjustment. Must verify the "before" print is clearly visible and close enough to label. Tutorials sometimes show only the after/success state, which would be unsuitable.

#### STR_003 — "3D Printer Settings Test: Geometry, Overhang, Stringing, ..."
- **URL:** `https://www.youtube.com/watch?v=GaAHA2Uv4DA`
- **Creator:** unknown
- **Source date:** 2025-04-25
- **Video class:** test print
- **Notes:** A comprehensive printer test covering geometry, overhang, stringing, bridging, and dimensional accuracy. The stringing section is only one part of the print; the usable segment would be the portion where stringing is visible. Good candidate for its diversity but requires segmenting out the stringing portion.

#### STR_004 — "3D Print Basics: MASTERING The Temperature Tower!"
- **URL:** `https://www.youtube.com/watch?v=RXjJaM0-V2s`
- **Creator:** unknown
- **Source date:** 2025-04-02
- **Video class:** tutorial / test print
- **Notes:** Temperature towers show varying stringing across their temperature zones. If the printed tower is shown in good detail, this could be a useful stringing source. Verify the tower is actually displayed with visible web formation.

### Under-extrusion (priority 2)

#### UE_001 — "How to Diagnose & Fix Serious Under Extrusion"
- **URL:** `https://www.youtube.com/watch?v=6V319bseO3U`
- **Creator:** unknown
- **Source date:** 2021-03-29
- **Video class:** failure-diagnosis
- **Notes:** "Almost always caused by hardware issue." Walking through the diagnosis of severe under-extrusion. Likely shows close-ups of gaps and thin walls. Strong candidate given the explicit focus on SEVERE under-extrusion.

#### UE_002 — "Ender 3 Under Extrusion Fix"
- **URL:** `https://www.youtube.com/watch?v=VF4kA0P73B8`
- **Creator:** unknown
- **Source date:** 2022-05-22
- **Video class:** failure-diagnosis
- **Notes:** Direct title match for under-extrusion fix on Ender 3. Print showing under-extrusion (gaps) likely visible. Verify.

#### UE_003 — "Creality Ender 3 V2 — Solving Under-extrusion To Save You Hours"
- **URL:** `https://www.youtube.com/watch?v=k-inUpjRVLA`
- **Creator:** unknown
- **Video class:** failure-diagnosis
- **Notes:** Ender 3 V2-specific under-extrusion fix. Likely shows the print surface with gaps. Manufacturer-specific, useful for a distinct printer in the diversity mix.

#### UE_004 — "Fixing Under/Over Extrusion — Adjust #ESteps"
- **URL:** `https://www.youtube.com/watch?v=G56LQfQCUnA`
- **Creator:** unknown
- **Video class:** failure-diagnosis
- **Notes:** Ender 3 Pro ESteps calibration. May show both under- and over-extrusion prints as part of calibration demonstration. Verify.

#### UE_005 — "Failed Filament Extrusion — 3D Print Vlog #56"
- **URL:** `https://www.youtube.com/watch?v=aoRxl5DYtpA`
- **Creator:** unknown
- **Source date:** 2025-07-25
- **Video class:** failure-footage (vlog)
- **Notes:** Vlog format. Likely contains real failure footage but interspersed with narration and other content. Disjoint cuts may complicate labeling independence; must verify which segments show the actual failed extrusion.

### Spaghetti / major failures (priority 3)

#### SP_001 — "3d printer makes spaghetti — 3d printing fails"
- **URL:** `https://www.youtube.com/watch?v=3Ci-3ZYHBkM`
- **Creator:** channel offers live 3D printer cam / nozzle cam / bed cam
- **Video class:** live-cam-fail
- **Notes:** This channel's format is continuous live 3D printer camera footage with time-lapse and failure clips. A "makes spaghetti" video from such a channel is highly likely to show clear, continuous footage of spaghetti developing. **This is the strongest spaghetti candidate** for labelability (real-time, stable cam, full print-area view). Must verify framing and resolution during manual review.

#### SP_002 — "From Spaghetti Fails to Perfect Prints (Timelapse)"
- **URL:** `https://www.youtube.com/watch?v=qcVFnRcJcP4`
- **Creator:** unknown (Bambu Lab printer)
- **Source date:** 2026-04-12
- **Video class:** timelapse
- **Notes:** Bambu Lab timelapse showing a failed (spaghetti) print transitioning to a clean print. The spaghetti sequence is visually obvious but is compressed timelapse, which reduces effective frame count and may skip the failure-onset. Recording extraction_fps will need to be handled.

### Normal (priority 4)

#### NORMAL_001 — "3D Printing Timelapse | Satisfying Print From Start to Finish"
- **URL:** `https://www.youtube.com/watch?v=dX4eGEFrthE`
- **Creator:** unknown
- **Source date:** 2026-01-01
- **Video class:** timelapse
- **Notes:** Full successful print documented from start to finish. Potentially strong normal-class source with complete print visibility. Must verify no failure frames or cuts to another subject.

#### NORMAL_002 — "3D Printing TimeLapses" playlist (WildRoseBuilds)
- **URL:** `https://www.youtube.com/playlist?list=PLBW1h8azPIa4wboMV82TH-FZhNMl8vBuw`
- **Creator:** WildRoseBuilds
- **Video class:** timelapse compilation
- **Notes:** Prusa i3 Mk3 Octolapse compilation, 3.3M views. Satisfying-print style. Compilation format shows multiple prints; independence requires identifying each distinct print. Usable but lower independence confidence.

---

## Scoring system

Scores are assigned on 7 dimensions (each 0–2), summed to a 0–14 total:

| Dimension | Description |
|-----------|-------------|
| Labelability | Can frame-level `normal`/`failing` labels be established from footage? |
| Footage quality | Resolution, stability, compression |
| Duration/usable footage | Sufficient usable segment length |
| Viewpoint usefulness | Does the camera show the print area well? |
| Domain diversity | Distinct printer/camera/material/environment contribution |
| Metadata quality | How much of printer/material/camera is known |
| Independence confidence | Confidence that this is a distinct session/print |

**Classification bands:**
- **ACCEPTABLE_CANDIDATE:** score ≥ 8, no disqualifying weakness, likely labelable
- **MARGINAL_CANDIDATE:** score 5–7, or a serious uncertainty in one dimension (esp. labelability)
- **REJECT:** score < 5, or a disqualifying flaw (no visible failure, discussion-only, occluded, CG, resin, etc.)

> Note: Because no videos have been downloaded/watched, most candidates carry labelability = `UNVERIFIED` and scores reflect the *known metadata*, not confirmed footage. The classification is a **pre-screening** to prioritize manual review, not a guarantee.

| Candidate | Label | Quality | Duration | Viewpoint | Diversity | Metadata | Independence | **Score** | Class |
|-----------|-------|---------|----------|-----------|-----------|----------|--------------|-----------|-------|
| STR_001 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | **7** | MARGINAL |
| STR_002 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | **7** | MARGINAL |
| STR_003 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | **7** | MARGINAL |
| STR_004 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | **7** | MARGINAL |
| UE_001 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | **8** | MARGINAL |
| UE_002 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | **7** | MARGINAL |
| UE_003 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | **7** | MARGINAL |
| UE_004 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | **7** | MARGINAL |
| UE_005 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | **7** | MARGINAL |
| SP_001 | 2 | 2 | 2 | 2 | 1 | 1 | 1 | **9** | ACCEPTABLE |
| SP_002 | 2 | 2 | 1 | 2 | 1 | 1 | 1 | **9** | ACCEPTABLE |
| NORMAL_001 | 2 | 2 | 2 | 2 | 1 | 0 | 1 | **10** | ACCEPTABLE |
| NORMAL_002 | 1 | 2 | 1 | 2 | 1 | 1 | 0 | **6** | MARGINAL |

---

## Independence assessment

- All candidates are from distinct video URLs.
- Creator identity is largely **unknown** pending manual review, so creator independence across candidates is not yet confirmed.
- Multiple videos from the same unsourced creator (STR_*, UE_*) may turn out to be the same channel. **Creator identity must be recorded during manual review** before any candidate is used, per Stage 12 §2.
- SP_001 is from a channel whose format is continuous camera footage; SP_002 is a separate Bambu Lab timelapse. These are almost certainly independent sessions.
- NORMAL_002 is a compilation; independence must be established per individual print inside it.

---

## Sourcing problems discovered

1. **Most search results are blog articles/tutorial pages, not direct footage.** A large fraction of the web content about stringing/under-extrusion is text guides embedding a few seconds of video. These are not usable as frame-level datasets.

2. **Tutorial videos frequently show "the fix" (the good result) rather than the failure in progress.** Many "how to fix stringing" videos might show only the successful print after adjustment, with little or no footage of the actual stringing failure. This is the single biggest risk to labelability.

3. **Resolution and duration metadata are rarely available from search results alone.** Approx. duration/resolution/camera-stability could not be confirmed for any candidate without opening the video. All marked `UNKNOWN`.

4. **Vlog/compilation formats complicate session independence.** Videos containing multiple prints (vlogs, compilations, test prints) require segmenting and session identification, which increases labeling effort and leakage risk.

5. **No candidate yet guarantees a recording date.** Most YouTube URLs don't expose upload date without fetching the page's metadata. Recording date (the Stage 10/11 cluster variable) is largely unknown.

6. **Stringing candidate labelability is the weakest link.** Stringing is a subtle, thin artifact; whether it is large/clear enough to label in tutorial footage is uncertain until manual review. The highest-confidence stringing source would be a dedicated stringing test-tower print shown live, which none of the current candidates guarantees.

---

## Recommendation for first acquisition batch

Given the need for **domain diversity** and **reliable visible labels**, the first batch should prioritize:

1. **SP_001** (live-cam spaghetti) — Real-time, continuous, high labelability. Highest value for the most underrepresented class.
2. **NORMAL_001** (full successful print) — Complete normal print, immediately labelable as `normal`.
3. **UE_001** (serious under-extrusion diagnosis) — Description strongly implies close-up views of severe under-extrusion, a top-priority class.
4. **SP_002** (Bambu spaghetti timelapse) — Second spaghetti source; independent from SP_001.
5. **UE_002 / UE_003** — More under-extrusion sources to reach the 5–8 target.

This focus maximizes reliable labels for the classes with the most evidence of generalization failure (under-extrusion FNR 0.337, stringing FNR 0.162) and the most underrepresented class (spaghetti, only 5 sessions).

Manual review order should confirm:
- Is real print-area footage shown continuously for a labelable segment?
- What is the actual resolution/duration/camera stability?
- Which creator/channel, and what recording date (if any) can be established?
- Can normal AND failing segments both be labeled?

---

## Feasibility of the 19–31-session target

**Assessment: The target (19–31 independent sessions from ≥10 creators) appears **realistically achievable**, but with important caveats.**

Evidence for achievability:
- The four target categories are all well-covered by public YouTube content. Spaghetti and under-extrusion (the two hardest classes) already have strong live-cam and diagnosis-format candidates.
- The failure-detection niche is popular enough that dozens of relevant videos exist; the current search surfaced ~13 candidates in a few queries.
- Timelapse and live-cam channels (e.g., the SP_001 channel) are a rich source of continuous failure footage with high labelability.

Evidence of difficulty:
- **Stringing is genuinely the hardest to source.** Tutorials dominate and may not show the failure being labeled. Reaching 6–10 *independent* stringing sessions from ≥4 creators will require more targeted searching (stringing test towers, "spider web" prints, PETG stringing fail clips).
- Resolution is often low in older failure clips; user footage on TikTok/Reddit/Instagram is mostly short but can contribute sessions.
- The 10-creator diversity requirement means we cannot rely on a handful of channels; we must spread the search.

**Realistic plan: 13 candidate videos pre-screened here provide a solid start; after manual review confirms labelability, a second targeting pass (especially for stringing) should close the gap to the 19–31 target.** I therefore judge the full Stage 12 target achievable provided we (a) confirm labelability of the current pool and (b) run one more targeted search for stringing test towers and additional distinct creators.

---

## Rejected / deferred candidates and reasons

| Source | Reason |
|--------|--------|
| Numerous blog articles (All3DP, Creality, 3dprinterstuff, etc.) | Text guides with embedded still images or brief clips; not continuous footage usable for frame labeling |
| `lazerhorse.org` "When 3D Printing Goes Wrong" | Still photographs only, not video |
| TikTok/Instagram/Snapchat short clips (spaghetti fails) | Short-form (usually <60s), vertical framing, often with overlays/filters and unreliable provenance; deferred pending pool status |
| Stock video sites (Pexels) "3D printing timelapse" | Stock footage may meet licensing but lacks failure content and provenance |
| AppropeDia 3D Printing Failure Database | Image database (photos), not video; could be a separate future stills source |
| Reddit thread "help with horrible stringing" | Still images/forum discussion, not video footage |

---

## Deliverables

- This document: `docs/research/stage12a-video-source-candidates.md`
- Structured metadata: `docs/research/stage12a-video-source-candidates.csv`

No code was changed, so the existing test suite was not re-run (per instructions: "Run the existing test suite only if repository code was changed").

---

## Addendum (2026-09-05, Stage 12B)

This document was updated with acquisition/inspection status for the six videos piloted in **Stage 12B** (`stage12b-video-acquisition-pilot.md`). Original candidate information is fully preserved; four new columns (`stage12b_acquired`, `stage12b_inspected`, `stage12b_labelability`, `stage12b_evidence`) were appended. Key pilot outcomes: **SP_002** and **STR_001** are unusable as acquired (STR_001 → `STRINGING_SOURCE_FAILED`); **SP_001** shows the strongest structural failure signal; **UE_001/UE_003** are partially usable but content-unconfirmed; **NORMAL_001** is framing-usable but its "genuine successful print" status is unconfirmed. No failure type is visually confirmed by this agent; human montage review is required before further acquisition.

*Stage 12B complete. No training, no modification of `datasets/processed/printing-errors/` or the frozen official test set.*

---

*Stage 12A complete. No training, no downloads, no dataset modification.*
