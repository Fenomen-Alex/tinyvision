# YouTube & Public Video Feasibility for 3D-Print Failure Training Data

**Date:** 2026-07-05  
**Status:** Research-only  
**Purpose:** Determine whether YouTube and public video platforms can realistically provide enough genuinely independent, diverse training/evaluation footage for 3D-print failure detection.

---

## Executive Summary

**Verdict: YouTube is unlikely to be a primary training data source.** While there are some continuous live-stream channels, the volume of genuinely independent, raw, fixed-camera footage is small (estimated 3–7 independent normal-print environments and 1–3 independent failure environments discoverable). The Standard YouTube License prohibits downloading and redistribution, making legal use for an open-source dataset impractical without per-channel licensing agreements. The best path forward is self-collected data with a small number of volunteer contributors, supplemented by a few existing research datasets.

---

## 1. Candidate Sources Table

| # | Channel / Source | URL | Type | Approx. Duration | Camera Setup | License | Continuous Raw? | Independent Env? |
|---|-----------------|-----|------|-------------------|--------------|---------|-----------------|-------------------|
| 1 | **GXS 3D** (The GrafX Shop) | https://www.youtube.com/@GXS3d | Live/Normal/Failure | Multi-hour live streams | Fixed bed cam (SV08), nozzle cam, overhead cam; Ender 3 Max & SV08 | Standard YouTube | **Yes** – streams run for hours during prints | **Yes** – distinct printers, multiple cameras, home workshop |
| 2 | **What is Jared 3D printing today?** | https://www.youtube.com/watch?v=8GYUAVQoiOc | Normal/Live | ~2–3 hours per stream | Fixed camera watching printer; single angle | Standard YouTube | **Yes** – continuous print streams with full print start-to-finish | **Yes** – distinct environment, solo hobbyist setup |
| 3 | **Satisfying Prints 3D** | https://www.youtube.com/watch?v=UMI7K3HwdjM | Normal/Live | Multi-hour streams | Fixed camera, likely bed-level view | Standard YouTube | **Yes** – continuous layer-by-layer streams | **Yes** – distinct environment |
| 4 | **Nero3D / Canuck Creator** | https://www.youtube.com/watch?v=2LAK9g76mpo | Normal/Live | Multi-hour streams (e.g., "FASTEST WOODY") | Overhead/fixed camera | Standard YouTube | **Yes** – live streams during prints | **Yes** – distinct Canadian maker environment |
| 5 | **I Do Know Somethings (IDKST)** | https://www.youtube.com/watch?v=iWlK-RltfNI | Normal/Live | "Longest 3D Print Ever" – multi-hour | Fixed camera, likely OctoPrint-connected | Standard YouTube | **Yes** – continuous live stream until print finishes | **Yes** – distinct environment |
| 6 | **Canuck Creator** (community stream) | https://www.youtube.com/watch?v=DDs9XxbTiUU | Normal/Live | ~3 hours | Overhead camera shown intermittently; primarily discussion stream | Standard YouTube | **Partial** – camera shown at times, but stream is primarily talk | **Same as #4** – same person |
| 7 | **GXS 3D – Failure clips** | https://www.youtube.com/watch?v=iqrvQDhdOfw | Failure | Short clips (seconds–minutes) | Fixed bed cam, nozzle cam on SV08 | Standard YouTube | **No** – short edited clips of failures | **Same as #1** – same environment |
| 8 | **GXS 3D – Bed adhesion failure** | https://www.youtube.com/watch?v=3QzW5hbps5c | Failure | ~43 seconds | Fixed camera, bed cam | Standard YouTube | **No** – short clip | **Same as #1** |
| 9 | **MihaiDesigns** | https://www.youtube.com/watch?v=Ox4qKvh_pzE | Failure | ~8 seconds | Bambu Lab X1C built-in chamber cam | Standard YouTube | **No** – very short clip | **Yes** – distinct Bambu Lab environment |
| 10 | **Synthetic Electron 3D** | https://www.youtube.com/watch?v=Rsa77O5QBKk | Failure/Normal | ~10 min (edited) | Gantry-mounted nozzle camera, custom mount on Elegoo Centauri Carbon | Standard YouTube | **No** – edited video, not raw footage | **Yes** – distinct custom camera setup |
| 11 | **Makers Mashup** (tutorial) | https://www.youtube.com/watch?v=-J5qAfRMwYU | Tutorial | ~9 min | Multi-cam setup demonstration | Standard YouTube | **No** – tutorial, not print footage | N/A – tutorial only |
| 12 | **Reticulated** (tutorial) | https://reticulated.net/posts/making-the-ideal-3d-printer-montioring-live-stream/ | Tutorial/Guide | N/A (blog post) | Yi 4k+, Kasa EC60, RTMP streaming guide | N/A (blog) | N/A | N/A – documentation only |
| 13 | **Newquay Weather Station** | https://newquayweather.com/wx3dprinter.php | Normal/Live | Continuous (web-based) | Axis M1011 webcam on Ender 3 | N/A (personal site) | **Yes** – always-on printer cam (web) | **Yes** – distinct UK environment |
| 14 | **kd7aat (Duff)** | https://kd7aat.net/3d_printing.html | Normal/Live | Continuous (web-based) | Multiple webcams on Prusa printers | N/A (personal site) | **Yes** – always-on webcams | **Yes** – distinct environment |
| 15 | **Project Biped** | https://www.projectbiped.com/blog/streaming-3d-printer-feed | Normal/Live | Continuous (UStream) | Webcam inside printer enclosure | N/A (blog) | **Yes** – UStream feed | **Yes** – distinct environment |

**Notes on the table:**
- "Continuous Raw" means the footage is real-time, unedited, fixed-camera, showing an actual print in progress (not timelapse, not compilation).
- "Independent Environment" means a distinct physical printer + camera + location setup.
- Many failure "videos" on YouTube are short edited clips (seconds to minutes), not continuous footage with visible failure onset.
- Channels #4 and #6 are the same person; #1 and #7/#8 are the same person.
- The web-based printer cams (#13–15) are not on YouTube but represent publicly accessible footage.

---

## 2. Estimated Independent Environments Discoverable

### Normal Print Environments (healthy, continuous footage)
**Estimated: 5–8 independent environments**

- **GXS 3D** (Sovol SV08, Creality Ender 3 Max, possibly Anycubic Kobra S1): multiple printers in one workshop = 1 environment (can be subdivided by printer, but same lighting/background)
- **What is Jared 3D printing today?**: 1 environment (single printer, fixed cam)
- **Satisfying Prints 3D**: 1 environment (single printer, fixed cam)
- **Nero3D / Canuck Creator**: 1 environment (Voron/CoreXY style, Prusa Core 1)
- **IDKST**: 1 environment (single printer)
- **Newquay Weather Station**: 1 environment (Ender 3, Axis webcam, UK)
- **kd7aat**: 1 environment (Prusa, multiple webcams)
- **Project Biped**: 1 environment (enclosed printer)

**Verdict:** We can identify roughly 5–8 genuinely distinct camera+printer+environment setups with continuous or near-continuous footage. Lighting conditions, background clutter, printer model, camera angle, and color balance differ across these environments. However, only 3–5 of these appear to produce content regularly enough to be a reliable source.

### Failure Environments (raw footage showing failure onset)
**Estimated: 1–3 independent environments**

- **GXS 3D** is the only channel that clearly posts both live streams AND failure clips from the same printer environment. The failure clips are short and edited, not continuous raw footage.
- **Synthetic Electron 3D** shows a failure onset in an edited video but not in raw continuous footage.
- **MihaiDesigns** shows a spaghetti result but only as a very short clip.
- **Bambu Lab X1C** spaghetti is shown as a brief clip, not continuous.

**Critical gap:** There is a severe shortage of **continuous, real-time, raw footage** where a failure **onset** is visible and the camera is stationary. Nearly all failure content on YouTube is:
- Short clips (seconds) showing the aftermath
- Timelapses where the failure onset is invisible
- Edited compilations
- Tutorials explaining failures, not showing them in real-time

---

## 3. Qualitative Verdict

### Can public footage scale past ~7–10 independent normal sources?

**Barely, with significant caveats.** There are approximately 5–8 discoverable independent normal-print environments with continuous footage. Adding platforms beyond YouTube (Twitch, personal websites) might add 2–3 more. However:
- Many streams are intermittent or seasonal
- Some channels are small and may go inactive
- Camera setups change over time (same person upgrades equipment)
- Content is not reliably archived or searchable

**Estimated ceiling: 8–12 independent normal environments** if we exhaust all discoverable sources across platforms.

### Can public footage scale past ~5 independent failure environments?

**No.** The bottleneck is severe. I could identify only **1–3 independent environments** that produce any failure footage at all, and none of them provide the specific type of footage we need (continuous raw footage with visible failure onset from a stationary camera). The 3D printing community shares failures primarily as:
- Short clips (before/after)
- Timelapses
- Photos of aftermath

This is fundamentally a **sampling problem**: failures are rare events, and when they happen, most people grab a quick photo or short clip rather than running a continuous stream. Even channels that do continuous live streams rarely capture failures because (a) the streamer is watching and usually intervenes, and (b) failures tend to happen during unattended prints.

---

## 4. Licensing Findings

### Standard YouTube License

**What it is:** The default license for all YouTube uploads. It is NOT a Creative Commons license. ([YouTube License Types](https://support.google.com/youtube/answer/2797468))

**Key restrictions:**
- Viewers may watch content but **may not download, reproduce, distribute, adapt, or create derivative works** without permission from the uploader. ([YouTube ToS](https://www.youtube.com/t/terms))
- The license grants YouTube the right to host and display, and grants users the right to view through the YouTube player only. It does **not** grant any rights for use independent of the platform. ([YouTube ToS, Section 6.B](https://www.youtube.com/t/terms))
- YouTube's ToS explicitly prohibits accessing content by "automated means" except as permitted by the service. ([YouTube ToS](https://www.youtube.com/t/terms))

**Implications for training a model:**
- **(a) Using footage to train a model:** This is **not explicitly permitted** under the Standard YouTube License. Downloading video to extract frames for training requires downloading the video, which violates the ToS. Even frame-by-frame viewing through the YouTube player does not give you rights to the underlying data. **This is legally risky without explicit permission from each uploader.**
- **(b) Redistributing extracted frames in an open-source dataset:** This is **clearly prohibited**. The Standard YouTube License does not grant any redistribution rights. Uploading extracted frames to a public dataset repository (e.g., on GitHub, Hugging Face, or Zenodo) would constitute unauthorized redistribution of copyrighted content.

### Third-Party AI Training Setting (December 2024)

YouTube introduced a new creator setting allowing creators to opt in to third-party AI training. ([YouTube Help: Content & Third-Party Training](https://support.google.com/youtube/answer/15509945))

- By default, **third-party training is OFF** for all creators.
- Creators can select from a list of 18 approved companies (Adobe, Amazon, Anthropic, Apple, Meta, Microsoft, OpenAI, etc.) or "All third-party companies."
- This setting controls whether YouTube may **share** videos with authorized companies for training purposes.
- The videos' training permission status is queryable via the YouTube Data API (`videoTrainability` endpoint). ([YouTube Trainability API](https://developers.google.com/youtube/v3/video-trainability))

**Critical note:** This system is designed for large AI companies with formal agreements, not for open-source projects. Even if a creator opts in to "all third-party companies," the mechanism requires YouTube to facilitate the sharing, and there is no public mechanism for downloading training data. An open-source project would not qualify under this framework.

### Google v. Singular Computing

**Clarification:** The case *Singular Computing LLC v. Google LLC* (Case No. 1:19-cv-12551, D. Mass.) was a **patent infringement** case about TPU hardware design, not about data scraping or copyright. Google settled in January 2024. ([Law360](https://www.law360.com/cases/5dfccc53eafeb50264163d9d/articles), [TopClassActions](https://topclassactions.com/lawsuit-settlements/lawsuit-news/google-class-action-lawsuit-and-settlement-news/google-reaches-settlement-in-1-6b-patent-infringement-case/))

**This case has no direct bearing on the question of using YouTube footage for ML training.** It was about hardware patents, not content scraping or copyright.

### Relevant Copyright Cases (for context)

The ongoing *Andersen v. Google* case (N.D. Cal.) involves allegations that Google trained generative AI models on copyrighted works, including YouTube content. The court's partial dismissal (April 2025) did not resolve the core question of whether training on copyrighted data constitutes fair use. This remains an open legal question. ([Court filing](https://www.pacermonitor.com/public/filings/GONDP45Y/USCOURTS-mad-1_24-cv-11611-4.pdf))

### Practical Licensing Path

The only safe path for using YouTube footage in an open-source dataset is:
1. **Contact each creator individually** and obtain explicit written permission to download, extract frames, and redistribute under an open license (e.g., CC BY 4.0).
2. **Filter for CC BY videos** using YouTube's search filters (Features > Creative Commons). However, very few 3D printing videos use CC BY licensing.
3. **Use the YouTube Trainability API** to check if videos are opted in for third-party training, but this only confirms permission for the 18 approved companies, not for open-source projects.

---

## 5. What Could NOT Be Verified

1. **Whether the identified YouTube channels are currently active** – channel activity status was inferred from recent upload dates but not confirmed as of the research date.
2. **Exact camera model, resolution, and field of view** for most channels – camera specifications are only documented for GXS 3D (which lists specific models in descriptions).
3. **Whether any identified channels use Creative Commons licensing** – I did not check each video's license field individually; the Standard YouTube License assumption is based on it being the default and CC BY being rare.
4. **Whether the web-based printer cams (#13–15) are still online** – personal websites may go offline.
5. **Twitch, Kick, or other platform live streams** – the research focused on YouTube; other platforms may have additional sources not identified here.
6. **Non-English-language 3D printing YouTube communities** – there may be significant communities in Chinese, German, or Polish that were not searched.
7. **Whether YouTube's Trainability API is currently functional and returns meaningful data** for 3D printing videos – the API exists but was not tested.
8. **The exact number of videos per channel** that meet the "continuous raw footage" criterion – a full audit was not performed.
9. **Whether the GXS 3D live streams capture failure onset** in real-time – the channel posts failure clips separately, suggesting failures may be clipped from streams, but this was not verified by watching streams.
10. **Fair use applicability** – I am not a lawyer and cannot determine whether extracting frames for ML training constitutes fair use under U.S. copyright law. This is an unsettled legal question.

---

## 6. Recommendations

1. **Do not rely on YouTube as a primary training data source.** The licensing restrictions are too severe for an open-source project, and the volume of suitable footage is too low.

2. **Prioritize self-collected data** with a small number of volunteer contributors (5–10 people) who set up standardized fixed cameras on their printers and agree to CC BY 4.0 licensing.

3. **Supplement with existing research datasets** where available:
   - [Minds-R-Lab/3D-Printing-Dataset](https://github.com/Minds-R-Lab/3D-Printing-Dataset) – 2,638 images, 9 classes, Prusa MK4, but photographed (not in-print webcam)
   - [YOLO-Strings](https://zenodo.org/records/14711320) – stringing detection, camera + microscope images
   - [Zenodo multi-material dataset](https://zenodo.org/records/19200472) – 11,593 images, PLA/ABS, warping/stringing/cracking, Raspberry Pi HQ Camera
   - Roboflow Universe datasets for spaghetti/stringing (small, ~700 images each)

4. **Consider contacting GXS 3D and What is Jared** as the most promising potential collaborators – they have continuous raw footage setups and active channels, and may be willing to contribute data under a CC BY license.

5. **Document the licensing risk explicitly** in any dataset release, noting that YouTube footage cannot be included without per-channel written permission.
