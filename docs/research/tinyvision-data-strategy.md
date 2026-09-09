# TinyVision: Release Dataset Strategy

**Date:** 2026-09-07
**Status:** Research-only. No training, no thresholds, no model/runtime changes,
no synthetic generation, no dataset download beyond metadata-level investigation.
**Scope:** Decide the most credible and practical path to the target-domain
dataset required for a release candidate, per the Stage 12–14 synthesis.
**Supporting studies:** `docs/research/youtube-footage-feasibility.md`,
`docs/research/synthetic-data-feasibility.md`, `docs/research/tinyvision-final-synthesis.md`.
**Machine-readable mirror:** `docs/research/tinyvision-data-strategy.json`

---

## 1. The actual data problem

### 1.1 The independence hierarchy

The evidence from Stages 12–14 forces a distinction between five levels of
data that are routinely conflated:

| Unit | Definition | Independence |
|------|------------|--------------|
| **frame** | one image the detector scores | not an information unit |
| **video** | contiguous footage of one camera session | may contain one or more runs; **not** independent of its camera environment |
| **printer run / build** | one print job, start to finish, on one printer | within a fixed camera environment runs are correlated (same view, sensor, lighting, background) |
| **printer** | physical machine | a component of the environment; model variability is observed only through the camera's view of it |
| **camera environment** | a stationary camera + printer + site/background/lighting combination | **the true independent unit** |

The camera environment is the operative independent unit because it bundles the
**causal domain factors** the frozen model failed on: sensor chip, optics/resolution,
view geometry, field of view, background clutter, lighting, bed/plate appearance,
and the printer itself as it appears to the camera. A run, video, or frame inside one
environment is autocorrelated; an environment collected on another day, in another
room, with another printer and camera is not.

### 1.2 What our own data says about the unit of independence

Measured on the frozen split (`tv`-tracked CSV): the entire Kaggle
`printing-errors` corpus is **151 recordings across one creator site, one
camera model (`ELP`), one naming/site pattern** — see §3 of
`stage12-data-acquisition-design.md` ("the existing 105 training recordings come
overwhelmingly from a single creator dataset"). In other words: **temporal
diversity (151 runs), zero environmental diversity.** Stage 14 then showed the
mirror failure: **100 frames from 2 cameras** (ZYfXW0JN2k4 + 3zXQrw0E-Ag) taught
Exp4 to prefer those two environments' appearance over real footage, while
**333 normal frames from 7 sources** produced the single largest measured win
(held-out normal FPR 1.000 → 0.258). Both effects are *environment-count* effects.

### 1.3 Requirements derived from the experiments

A useful target-domain dataset must therefore guarantee, at minimum:

- **Independent printers:** many distinct machines, ideally several models/brands,
  because printed-object appearance, motion, and frame around the part change.
- **Independent cameras:** different sensors/optics/resolutions; camera models
  must not cluster (stage-4 forensics-style dedup of identical sources).
- **Camera positions:** side/front/overhead/bed-mounted views already observed to
  change what the detector sees (Pillow-bilinear 224 is view-sensitive).
- **Lighting:** day/night, window light, enclosure LEDs, RGB vs. webcam white balance.
- **Backgrounds / beds / plates:** textured PEI, glass, garages vs. desks.
- **Filament colors/materials:** reflectance and string appearance differ.
- **Healthy continuous printing:** full unattended runs with no defects (the normal class).
- **Failure footage with onset and development:** continuous, fixed-camera, from
  first-layer haze to obvious-failure aftermath — not just aftermath stills.
- **Source-level independence in training/eval/val split** — never frame-random splits.

### 1.4 The counting rule

Do not specify a huge frame count. Frame counts measure throughput; **environment
counts measure generalization**. Every source contributes a per-environment frame
cap (sampled evenly across its duration so temporal diversity is preserved). Two
hundred diverse environments × 500 frames each are more informative than 100,000
frames from 2 cameras. The Stage-14 lesson stands: **100 frames from 2 cameras
are not equivalent to 100 diverse frames.**

---

## 2. Public real-world footage (YouTube and other platforms)

Full study: `docs/research/youtube-footage-feasibility.md`. It catalogued 15
candidate sources (13 channels/videos, 2 always-on webcams, 1 blog) and reached
these conclusions:

| Question | Finding |
|----------|---------|
| Independent **normal** environments, continuous footage | **~5–8 discoverable**, only ~3–5 active enough to rely on; ceiling ~8–12 exhausting all platforms |
| Independent **failure** environments, continuous raw onset footage | **~1–3**, and none reliably provide onset-from-stationary-camera footage |
| Different printers/cameras identifiable? | Yes for the well-documented channels (e.g. GXS 3D runs multiple printers/cams), mostly not for others |
| Useful frames extractable? | Yes for normal (full multi-hour live streams), mostly **no** for failures (short edited clips, timelapses, compilations, tutorials) |
| License | Default **Standard YouTube License**; downloading violates ToS and redistribution is clearly prohibited |
| Legal to train on? | Not clearly permitted; unsettled area (per-Channel); the Dec-2024 third-party-training mechanism targets large AI companies, not open-source projects |
| Redistribution possible? | **No**, without per-channel written permission granting an open license |
| Can it scale past Stage 13/14? | As-is, **no**: Stage 13 already used ~7 normal + 2 failure sources; YouTube adds marginal *new environments* and zero reliable failure onset |

Representative candidates worth contacting (short list, not a corpus):

| Source | Type | Continuous? | Independent env? | License note |
|--------|------|-------------|------------------|--------------|
| GXS 3D (`@GXS3d`) | live + failure clips | yes (streams); clips are edited | yes; multiple printers, one workshop | Standard YT — needs permission |
| "What is Jared 3D printing today?" | live | yes | yes | Standard YT — needs permission |
| "Satisfying Prints 3D" | live | yes | yes | Standard YT |
| Nero3D / Canuck Creator | live | yes | yes | Standard YT |
| Newquay printer cam (web) | always-on | yes | yes (UK, Ender 3 + Axis webcam) | personal site — needs permission |
| kd7aat printer cams (web) | always-on | yes | yes (Prusa) | personal site — needs permission |

**Verdict:** YouTube/public footage is a **supplemental, permission-gated** source
for *normal environments only*. It cannot serve as the failure backbone: genuine
continuous raw failure-onset footage is extraordinarily scarce — failures are rare,
streamers intervene, and the community archives aftermath clips, not onset.

---

## 3. Existing public / research datasets

The four already-investigated corpora (Kaggle `printing-errors`, Cambridge Caxton,
SKZ Nozzle-Cam, SKZ Plattform-Cam) are not re-reported. Genuinely additional
candidates found in this study:

| # | Dataset | URL / DOI | License | Envs (printers/cameras) | Format | Normal data | Failure types | Usable size | Domain match | Notes |
|---|---------|-----------|---------|------------------------|--------|-------------|---------------|-------------|--------------|-------|
| 1 | **Trier FFF QC Dataset** (v2) | zenodo.org/records/18628777 (v1: 14712897) | **CC BY 4.0** (verified on Zenodo rights) | UNVERIFIED single site, **3 fixed camera angles** during production | high-res images (+crops), train/test zips | **yes** (defect-free prints) | extrusion errors, layer shifts, stringing | ~48 GB; exact frame count **TO VERIFY** | **HIGH** — in-production fixed cameras | Strongest bulk candidate; same-site multi-angle ⇒ counts as limited environments |
| 2 | **Bambu Timelapse Dataset** | huggingface.co/datasets/v2thegreat/bambu-timelapse-dataset | **CC BY 4.0** (verified) | **~3 contributors** (envs), Bambu P1/X1 built-in cam | 627 timelapses + **81 full-length videos** + metadata (~36 GB) | **yes**, unlabeled | none labeled yet | 627+81 videos | **VERY HIGH** for normal | Contribution-based growth model to replicate; timelapses are skimmed (cadence mismatch) |
| 3 | **Bed-Adhesion Failure-Onset** | github.com/EmptyDeck/3DPrint-Failure-Video-Dataset | **CC BY-NC 4.0** (annotations/masks); videos hosted **externally (YouTube)** | **2 envs** (P1 1024×768, P2 1920×1080, bed-mounted 1fps) | 30 sequences, 261,583 frames, onset GT + masks | **yes** (normal baseline pre-onset) | bed-adhesion/fallover only | 30 sequences, 300 masks | **VERY HIGH** — exact target setup | Onset labels ideal for our task; NC license + YT-hosted videos ⇒ containment flags |
| 4 | **UTFPR / Wartronick** | github.com/Wartronick/3D-Printer-Failure-Detection | **MIT** | 1 env (RPi 5 + ActionCam) | 505 images, 2-class | yes (290) | binary only | tiny | **HIGH** (right hardware) | reference/repro only |
| 5 | **Multi-material PLA/ABS layerwise (Zenodo)** | zenodo.org/records/19200472 | license UNVERIFIED; **files restricted (login-gated)** | 1 env (Delta printer + RPi HQ cam) | 11,593 images, DoE-varied defects | yes (balanced) | warping/stringing/cracking | 11,593 | HIGH | access-gated ⇒ cannot use yet |
| 6 | **FT3D FFF defects** | zenodo.org/records/20379284 | **MIT** (per Zenodo license picker, details to verify) | UNVERIFIED | defect images | UNVERIFIED | defect images | UNVERIFIED | UNVERIFIED | verify before use |
| 7 | **3DVis (layer-wise faults)** | Kaggle `adiparamartha/3d-vis` / IEEE DataPort | **CC BY-SA 4.0** (Kaggle; IEEE pages login) | UNVERIFIED | timelapse layer images, 4 fault classes | UNVERIFIED | 4 faults | 2,297 raw / 4,261 aug | MED-HIGH | small, class names unverified |
| 8 | **Minds-R-Lab 3D-Printing-Dataset** | github.com/Minds-R-Lab/3D-Printing-Dataset | permissive (README) | 1 env (Prusa MK4) | 2,638 images, 9 classes | UNVERIFIED | 9 classes | 2,638 | MED — post-print photos, not in-print webcam | metadata check needed |
| 9 | **YOLO-Strings** | zenodo.org/records/14711320 | UNVERIFIED | UNVERIFIED | camera + microscope images | no | stringing | UNVERIFIED | MED-LOW | stringing taxonomy only |
| 10 | **3D-ADAM** | HF `wuzhao73/3D-ADAM` / arXiv 2507.07838 | **CC BY-NC-SA 4.0** | 217 parts, 4 industrial depth sensors (post-print cell) | RGB+3D scans + pixel masks | yes (unsupervised protocol) | 12 classes incl. over/under-extrusion | 14,120 scans | MOD-LOW — post-print, different cameras | highest annotation quality, NC |
| 11 | **already-recorded small sets** | `docs/research/datasets/*.md` (Hu FDM MIT 1,912; Roboflow seprinder/syLucauc CC BY 4.0; Kaggle 3D-printer-defected unverified; Javiai unknown) | mixed | mostly UNVERIFIED, community | det/multi-class | mixed | spaghetti/stringing/etc. | ~500–5k each | MED | **provenance risk** — many re-uploads |

### Key observations

- **The single most valuable find is Trier (CC BY 4.0, in-production, defect-free +
  extrusion/layer-shift/stringing, 3 fixed angles, ~48 GB).** It is the only large,
  permissively-licensed, in-production corpus. Exact frame/printer counts must be
  verified at acquisition (Zenodo JS-walled during this study).
- **Bambu (CC BY 4.0) is the domain-perfect normal source and a working template
  for community contribution.** But its cardinality lesson applies to us too:
  **627 timelapses from ~3 contributors ≈ 3 environments** (the card explicitly
  lists 3 contributors; the visible sample is one, "Borillion").
- **Bed-Adhesion is the only onset-annotated failure corpus in our domain** and is
  exactly our camera setup — but bed-adhesion-only and CC BY-NC.
- Academic papers in this space recycle the same tiny Kaggle/Roboflow sets (e.g.
  the 2024 *J. Intelligent Manufacturing* YOLO study cites a "3D Printing Failure
  Detection Dataset" on Roboflow, June 2024), so **"more papers" ≠ "more independent
  environments."** No dataset fills the continuous under-extrusion + multi-env
  failure gap.
- **Obico/Spaghetti Detective monitored ~90M printing hours but has no public
  dataset** — the field's largest bioresource is closed.

---

## 4. Synthetic data as a serious alternative

Full study: `docs/research/synthetic-data-feasibility.md` (33 verified sources).
Summary of what the evidence supports:

| Claim | Evidence | Verdict |
|-------|----------|---------|
| Rendering FDM defects (spaghetti/stringing/under-extrusion) in Blender/simulators | **No published/verified pipeline exists.** Existing Blender FDM sims model the process, not defects; industrial defect sims target cosmetic surface defects | Building credible FDM defect rendering is novel *research*, not a tool |
| Photometric augmentation closes the camera/environment gap | **Refuted, quantified**: Xiao et al. 2024 — optimized illumination+ColorJitter leaves a persistent **4–14% accuracy gap** vs. real illumination diversity | Necessary, never sufficient for environment diversity |
| **Cut/paste defect compositing onto real backgrounds** (defect-diversity augmentation) | **Best evidence, from steel NEU-DET ablation**: simple cut-paste peaks at **n=50 synthetic/class** (mAP@50 0.7165→0.7782, **+6.2pp**), Poisson at n=100, then **monotonic degradation** (−2.8–3.0pp at n=300; features shift toward the augmented distribution). ControlNet-on-SD1.5 without domain fine-tuning **underperforms cut-paste** | The one synthetic avenue validated; use as **bounded augmentation**, never the backbone |
| Diffusion/GAN defect synthesis | Promising on MVTec-AD/NEU-DET; **zero verified evidence for 3D-print spaghetti/stringing**; SynSur 2026: "synthetic-only training does not replace real data; augmenting the **full real set** yields consistent gains" | Not justified yet for this project |

**The decisive framing:** the question is not "can an image look like spaghetti?" It
is **"would synthetic variation improve generalization across real camera
environments without teaching synthetic artifacts?"** The evidence says:

1. It **cannot fix environmental diversity** — that requires real footage from more
   real environments (the Stage-13 result was real, not synthetic).
2. It **can modestly help defect diversity** — pasting segments of *real* failure
   footage onto *real* normal backgrounds of other environments is
   environment-agnostic augmentation with a strict optimum.
3. **Synthetic-only data is never a replacement**; full-scene rendering of this
   failure morphology is unproven and high-risk engineering.

**Decision:** pursue synthetic **only** as a later, volume-capped, cut-paste
augmentation of real defect pixels onto real normal frames — after the real
environment collection reaches its target. Do not build rendering or generative
pipelines now.

---

## 5. Better use of existing real footage

Within the current corpus the **environment ceiling is hard**, and reuse must
respect every evaluation boundary:

| Existing asset | Environment count | Can it grow the *environment* stock? |
|----------------|-------------------|---------------------------------------|
| `printing-errors` frozen train (11,254) | **1 env** (ELP, one site) | no — frozen, single-env |
| Stage 13 train normals (333) | 7 envs | already in train side |
| Stage 14 failures (100) | 2 envs | already in train side |
| external pilot video-frames (1,546) | ~5–6 sources, mostly tutorial/low-value | marginal; only after human review |
| SKZ probe (166, aux 135) | ~1–2 envs, candidate labels | experimental; labels unresolved |
| 7-video holdout | 7 envs | **never** (frozen external holdout) |
| Stage 13 eval normals (120) | 2 envs | **never** (evaluation-only) |
| SKZ eval (31) | held out | **never** |

Genuinely useful *within-corpus* levers (design rules, not new training):

1. **Source/run-level stratification** — a per-environment frame cap so one
   environment cannot dominate class balance (the Stage-14 failure came from
   exactly this imbalance: 100/2 sources).
2. **Temporal spread sampling** — evenly stride each run's full duration instead of
   clustering frames around a start window; preserves onset information while
   reducing run-local bias.
3. **Hard-normal mining** — the highest-value *existing* resource is the frozen
   train's normal class, plus SKZ candidate normals: frames the frozen model
   scores as failing are precisely what forced the 0.920→1.000 FP tail; mining and
   weighting them (after human verification) is the cheapest real win.
4. **Failure-onset preservation** — Stage 14 already contains onset windows
   (ZYfXW t≈98–156 main onset + t≈221–231 final; 3zXQ blob t≈142–171); future
   curation must keep onset/development, not just aftermath.

**Verdict:** better use of existing footage is a **real but bounded** gain —
it rebalances and hardens, but cannot add a single new environment, and the
environment shortage is the blocker. It belongs inside the eventual collection
design, not as a standalone strategy.

---

## 6. Comparison and decision matrix

Qualitative scores: ● high · ◐ medium · ○ low.

| Strategy | Diversity | Scale | Licensing | Cost | Effort | Expected usefulness | Risk |
|----------|-----------|-------|-----------|------|--------|--------------------|------|
| Public real footage | ◐ (5–8 normal envs, 1–3 failure) | ◐ (~8–12 envs ceiling) | ○ (SYL; no redistribution) | ● low | ◐ | ◐ supplemental normals | ● high (legal + scarcity) |
| Existing datasets | ◐ (Trier 3 angles + Bambu ~3 + Bed 2 + small) | ● (Trier ~48 GB, Bambu ~36 GB) | ● mostly CC BY/MIT; two NC | ● low | ● low | ● normals + some defects | ◐ (metadata noise, multi-angle mismatch, provenance) |
| Synthetic (compositing) | ● in defect *appearance*; ○ for real environments | ● (arbitrary volume) | ● none | ● low | ◐–● | ○–◐ (bounded defect augmentation only) | ◐ artifact overfitting (strict optimum) |
| **Combination** | ● (envs from datasets + community) | ◐–● | ◐ (manage mix; NC held eval-only) | ◐ | ● | ● best | ◐ integration/data-hygiene |

**Reasoning:** no single source meets the release bar. Public footage is
capacity-limited and legally constraining; existing datasets give bulk *volume*
but few *environments* and fail the onset/under-extrusion gap; synthetic gives
neither environments nor (yet) credible failures. Only a combination — bulk
permissive datasets **plus** new environments from a contributor/self-collection
program, with real-defect compositing as a capped augmentation — addresses the
measured blocker (environment count) at attainable cost.

---

## 7. Minimum credible release dataset design

Unit of everything below: **camera environment** (see §1). No frame-random splits.

### 7.1 Normal class

- **Training environments: ≥ 16 independent; target ~24.** Composition estimate:
  - Trier defect-free block: 1 station but **3 fixed camera angles** (count ≈1–3 env units; verify at acquisition).
  - Bambu: ≤3 contributor environments from full-length/timelapse footage.
  - **Contributor/self-collected: ≥ 8–12 genuinely new environments** (this is the load-bearing number — seen nowhere downloadable today).
  - 1–2 permissioned channels (e.g. GXS 3D, "What is Jared") only if CC BY granted.
- **Printer/camera diversity:** at least 3–4 printer models; cameras must not repeat a single model; require ≥2 of {bed-mounted, front, side, overhead} view types overall.
- **Continuous requirement:** per environment, ≥1 complete healthy run or ≥2 h of continuous fixed-camera footage at ~1 fps (≈ ≥7,200 raw frames), then **sampled to a per-env cap** (~1,000–1,600 frames stride-sampled across the run). Normal training pool ≈ 25–40k frames across ≥16 envs.
- **Filament/bed/lighting spread:** record and diversify these deliberately (bed types, colors, lights) — they are environment covariates.

### 7.2 Failure class

- **Training environments: ≥ 9 independent; target ~12.** Minimum per failure type (independent sources, not frames):
  - Spaghetti: **≥ 3** environments (have: ZYfXWJN2k4; need ≥2 new).
  - Under-extrusion: **≥ 3** environments (**the largest gap**; continuous footage scarce everywhere).
  - Stringing: **≥ 2** (SP001/STR001 partial; Trier per-image available).
  - Bed-adhesion / warping: **≥ 2** (Bed-Adhesion gives 2, onset-labeled; contained by its license).
- **Onset vs. obvious coverage:** every failure environment must supply continuous onset + development (window from first anomaly to aftermath), not aftermath stills.
- **Per-env frame cap** (~500–1,200 frames, stride-sampled). Failure training pool ≈ 6–12k frames across ≥9 envs.

### 7.3 Evaluation

| Set | Sources | Status |
|-----|---------|--------|
| Frozen test (Kaggle) | 1 env, 24 recordings | unchanged |
| 7-video real-world holdout | 7 envs | **frozen, external, never trained** |
| Stage 13 held-out normals | 2 envs (120 frames) | **locked evaluation-only** |
| SKZ eval | 1 env, 16 held-out groups | **locked** |
| **New external holdouts** | 2–3 new normal envs + 2–3 new failure envs from collection | held out before any training |

Splitting is strictly environment-level. A "division" horizon rule: label only
*after* assigning sources to train/val/holdout.

### 7.4 Why this is credible

This is the first design whose *independent-unit count* (≈25–35 environments for
train+holdout) is an explicit target tied to the measured failure mode, rather
than a frame budget. It is achievable because ~40% of the normal environments and
~5–6 failure environments are *already in hand* (Trier, Bambu, Bed-Adhesion,
Stage 13/14), leaving ~8–12 new normal and ~4–6 new failure environments to be
raised from contribution/collection — a realistic community program, not an
institutional one.

---

## 8. Licensing

Three separate buckets. Legal claims here are limited to what each source's
license *says*; anything nuanced is flagged for manual verification.

### 8.1 Training data (legitimately usable)

- `printing-errors` (Apache 2.0, recorded) — used; frozen.
- Trier FFF QC v2 — **CC BY 4.0** → training + redistribution with attribution.
- Bambu Timelapse — **CC BY 4.0** → training + redistribution (credit).
- Hu FDM-3D-Printing-Defect — **MIT**.
- UTFPR/Wartronick — **MIT**.
- 3DVis — **CC BY-SA 4.0** → training + redistribution (share-alike).
- Roboflow seprinder / syLucauc — **CC BY 4.0** (page-stated) but **provenance risk**: these are community re-uploads; verify original capture provenance before including in the *redistribution* trench.
- FT3D — MIT (verify content before use).
- Bed-Adhesion — **CC BY-NC 4.0** → training only for non-commercial scope; if TinyVision stays open-source/research this is usable; if commercial release is later intended, exclude.
- 3D-ADAM — **CC BY-NC-SA 4.0** → NC training only if included at all.
- Stage 13/14 + SKZ frames — **Standard YouTube License** (already in use); training record stands, **flagged: at least label sources as non-redistributable**, and ideally re-verify each uploader's current license state.
- Own/community-collected — fully ours; **we license the release CC BY 4.0** with an explicit per-contributor permission record.

### 8.2 Evaluation-only (inspect/run, not redistribute)

- 7-video holdout — kept as-is (Standard YT license; internal evaluation).
- Stage 13 eval normals — kept as-is.
- SKZ eval groups — kept as-is.
- Bed-Adhesion full sequences if we prefer to keep its NC/licensing out of the released artifact → evaluate-only.
- Permissioned YouTube channels used for *evaluation* only — even with permission, treat as eval unless the grant covers redistribution.

### 8.3 Redistribution-compatible (may sit in the released repo/dataset)

Apache-2.0 `printing-errors` subset, **Trier** (CC BY), **Bambu** (CC BY), Hu (MIT),
UTFPR (MIT), 3DVis (CC BY-SA — forces share-alike on any derived set that mixes it),
plus **our own collected footage under CC BY 4.0**.

**Flagged for manual license verification before any release:**
1. Trier exact count/printer metadata and any intra-dataset license statement.
2. Roboflow sets' original capture provenance.
3. FT3D content/details.
4. Kaggle "3D-Printer Defected" (record already notes license unverified).
5. Multi-material Delta Zenodo access/license (files restricted).
6. Minds-R-Lab README license wording.
7. YouTube `videoTrainability` state for any channel we contact, and each
   uploader's explicit written grant (CC BY preferred) — **no redistribution
   without it**.

---

## 9. Is the project salvageable with available resources?

Blunt assessment.

1. **Can we realistically build the required dataset without owning a printer?**
   Yes, but it changes the plan. We cannot fabricate failures at will; we must
   (a) absorb already-existing permissive corpora (Trier, Bambu, Bed-Adhesion), and
   (b) run a light contributor program where owners donate continuous full-per-footage.
   Slower and socially driven, not cheaper-but-faster.
2. **Can public footage provide enough independent normal sources?**
   Only with permission, and only ~8–12 environments at the absolute ceiling.
   Enough to *supplement* a ~24-target; not enough alone.
3. **Can public footage provide enough independent failure sources?**
   **No.** ~1–3 environments, and essentially no continuous onset footage. This is
   the binding constraint on the entire release timeline.
4. **Are existing research datasets sufficient to fill the gaps?**
   Partially. Trier fills a large supervised chunk (normals + extrusion/layer
   shift/stringing); Bed-Adhesion fills 2 onset-labeled adhesion environments;
   Bambu feeds healthy normals. **Nothing fills continuous under-extrusion, and
   nothing other than us/stage-14 gives multi-env spaghetti onset.** Not sufficient alone.
5. **Is synthetic data worth pursuing?**
   Only the **cut-paste composite** variant, and only as bounded augmentation
   after real env targets are met. Rendering/generative FDM failure synthesis is
   not evidence-backed yet — **not worth engineering now**.
6. **Is a hybrid dataset the most realistic route?**
   Yes — the only route that satisfies the environment-count requirement at this
   cost. Bulk permissive datasets provide base coverage; contribution provides the
   genuinely new environments; compositing adds defect-sample diversity.
7. **What is the minimum acquisition effort that would justify another training run?**
   The next training run is justified **only when**: (a) the train-side environment
   count roughly doubles over today's usable pool (≈10 usable envs → ≈20, counting
   Trier/Bambu/Bed-Adhesion/Stage-13+14 and ≥5–6 brand-new environments), AND
   (b) ≥5 genuinely new failure environments (≥2 spaghetti, ≥3 under-extrusion)
   have human-verified onset labels, AND (c) 2 new normal + 2 new failure
   environments are frozen as holdout before training. **Until those three
   conditions hold, another training run is not justified.**

---

## 10. Recommendation

```text
RECOMMENDED: Hybrid — permissive research datasets (Trier, Bambu, Bed-Adhesion-contained,
small MIT sets) as the bulk backbone, PLUS a contributor/self-collection program to add
genuinely new camera environments, PLUS permissioned YouTube channels as a supplemental
normal source, PLUS (after real targets are met) volume-capped real-defect cut-paste
compositing as defect-diversity augmentation.
```

**Why preferable:** it is the only option that attacks the measured blocker —
*camera-environment count* — at attainable cost and licensing.
Public-footage-only fails licensing and failure scarcity; datasets-only fails the
environment/onset/under-extrusion gap; synthetic-only is refuted for environment
diversity.

**What data must be collected (new):** ≥6–10 brand-new normal environments and
≥4–6 new failure environments (≥2 spaghetti, ≥3 under-extrusion) with continuous
onset+development footage, human-verified, under CC BY 4.0 (contributed or
self-collected).

**What existing data can be reused:** Trier v2 (CC BY, bulk normals + extrusion/
shift/stringing), Bambu (CC BY normals), Bed-Adhesion (2 onset-labeled adhesion
envs, NC-contained), Stage 13/14 and SKZ frames (already in train/eval splits as
recorded), `printing-errors` (frozen), and the existing frozen holdouts unchanged.

**What remains missing:** continuous under-extrusion footage (by far the hardest),
multi-environment spaghetti onset, and enough genuinely independent normals —
all of which must come from the contributor program, not from any downloadable
corpus identified.

**Estimated scale:** ~25–35 independent environments, ~30–50k annotated frames at
~1 fps (normals ≈25–40k, failures ≈6–12k) — modest by frame count, meaningful by
environment count.

**Licensing considerations:** assemble a provenance ledger per dataset policy
(name, source, URL, license, counts, resolution, collection method, limitations,
retrieval date); keep NC-restricted data in training/eval-only trenches; only
CC BY/MIT/Apache data enters the redistribution artifact; our own data released
CC BY 4.0. Flag list from §8.3 must be cleared before any public release.

**Next implementation task:** build the **dataset-staging + contributor-intake
pipeline** (no training): (1) download and fully characterize **Trier v2** (record
exact frame/printer/environment counts, class balance, angles, license file,
checksums); (2) verify **Bambu** contributor/environment count and label a metadata
subset; (3) extend the Stage-12D human-review manifest format to new sources and
write the environment-manifest schema (source → env, printer, camera, view,
lighting, bed, filament, license, verification state); (4) ship a contributor
intake kit (fixed-camera guidance, CC BY permission recording) and open the
collection program to ≥3 normal + ≥1 failure early adopters. Freeze the new
holdouts before and independently of any future training.

**Explicitly do NOT pursue:**
- Scraping YouTube (or any platform) footage for training without per-uploader permission.
- Including Standard-YouTube-License frames in the redistributable release.
- Generative/diffusion or rendered synthetic FDM data as a primary data source.
- Full-scene FDM defect rendering (unproven, high-cost research).
- Frame-random train/eval splits at any level.
- Re-using the 7-video holdout, Stage-13 eval, or SKZ eval for training.
- Another training run before the §9-U7 acquisition conditions are met.
- Arbitrary frame-count inflation from any single environment.

---

## 11. Deliverables

- `docs/research/tinyvision-data-strategy.md` (this file)
- `docs/research/tinyvision-data-strategy.json` (machine-readable mirror)
- Supporting studies written during research: `docs/research/youtube-footage-feasibility.md`,
  `docs/research/synthetic-data-feasibility.md`
- `ML.md` updated with a short link to this strategy.

Historical experiment results, `verified_labels.json`, VLM proposals, frozen
splits, the 7-video holdout, Stage-13 held-out normals, and SKZ evaluation
artifacts are unmodified.

---

## 12. HARD STOP compliance

This task performed **investigation only**. It did not: train, fine-tune, tune
thresholds, change the model, change preprocessing, change temporal logic,
generate synthetic training data, download a massive corpus, create another
experiment stage, create a checkpoint, or declare a release candidate. The
deliverable is the credible path to the dataset required for a release candidate.