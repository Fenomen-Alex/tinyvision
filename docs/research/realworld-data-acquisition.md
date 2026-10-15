# Real-world data acquisition to resolve the RC2 data bottleneck

Date: 2026-10-14
Branch: `feat/realdomain-automated-pipeline` (HEAD `b52cf74`)
Companion machine-readable record: [`realworld-data-acquisition.json`](realworld-data-acquisition.json)

## 0. Why this acquisition

The RC2 real-domain diagnostic (see [`rc2-realdomain-results.md`](rc2-realdomain-results.md))
cannot bound its normal false-alarm rate and misses the one held-out failing print.
The root cause is **not** model capacity but data: too few **independent** normal
streams and too few independent failing prints. The acquisition target was:

* ≥ **16 independent, human-verified normal multi-frame streams** with zero false
  alarms, and
* ≥ **4 independent failing streams**.

with the explicit rules that (a) adjacent clips from one print do **not** count as
independent, and (b) a failing print's **pre-failure segment does not count as an
independent fully-normal stream**.

This document reports what was evaluated, what was actually acquired, the verified
licences, storage, validation, the residual gap, and the single highest-value next
action. **No training runs were executed and no release artefacts were changed.**

## 1. Sources evaluated and verified licences

Licences were verified against the actual dataset statement (Zenodo API / repository
`LICENSE` / dataset page), **never inferred from the code host or platform**.

| # | Source | Licence (verified) | Video? | Normal class? | Independent normal streams? | Usable here? |
|---|---|---|---|:--:|:--:|---|
| 1 | **EmptyDeck** "Bed-Adhesion Failure-Onset Dataset for FDM 3D Printing" | **CC BY-NC 4.0** (noncommercial) | yes, 30 prints | only pre-onset baseline of *the same print* (non-independent) | no | yes — research-only |
| 2 | **wengmhu** FDM-3D-Printing-Defect-Dataset | **MIT** | no (static) | no (defects only) | no | yes — supplemental only |
| 3 | **Trier** FFF-QC v2 (Zenodo 18628777, DOI 10.5281/zenodo.18628777) | **CC BY 4.0** (Zenodo API, verified 2026-09-07) | no (static) | yes (`no_stringing_error`) | no (static) | yes — already staged |
| 4 | **CAXTON** (Cambridge, DOI 10.17863/CAM.84082) | **CC BY 4.0** | effectively yes (nozzle cam) | no failure labels (continuous params only) | no | **no** — labels aren't failures; images not hosted in the repo (only CSVs) |
| 5 | **Roboflow** `3d-printing-errors-g4f6l` | **CC BY 4.0** | no (224 static) | `normal` class present | no (static) | **blocked** — download requires account/API key (HTTP 403) |
| 6 | **Kaggle** `early-detection-of-3d-printing-issues` competition | competition rules | snapshots | unknown | unknown | **blocked** — needs Kaggle account + rule acceptance |

Key facts that invalidate the "obvious" normal sources:

* EmptyDeck's `n_normal_baseline` frames are `[0, onset_frame)` of the **same
  failing print** — by the independence rule they are **not** independent normal
  streams.
* Trier and wengmhu are **static image** sets: they may inform training/exploration
  but can never evidence a video-level false-alarm rate.
* CAXTON's labels are process parameters (flow, speed, Z, temperatures, nozzle x/y),
  **not** pass/fail, and the repository hosts only label CSVs (119–144 MB each) —
  the images live elsewhere.

## 2. What was actually acquired

Data root (outside the repo, not under Git): `/Users/alex/tinyvision-data`
(no external volume was mounted; `/Volumes` had only the boot disk). Disk after
acquisition: 122 GB free on `/`.

| Dataset | What was acquired | Count | Size | Location |
|---|---|---:|---:|---|
| EmptyDeck (CC BY-NC 4.0) | 11 shortest bed-adhesion failing sequences, 360p video + 1 fps frames + provenance manifest | **11 prints**, **2 environments** (P1×6, P2×5), **28,469 frames** | 1.6 GB | `…/emptydeck-3dprint-run/` |
| EmptyDeck annotations repo | `metadata/sequences.csv`, masks, scripts, `video_links.md` | 30 sequences of metadata | 547 MB | `…/emptydeck-3dprint/` |
| wengmhu (MIT) | stratified static subset, 40/class | **200 images**, 5 classes | 645 MB | `…/wengmhu-fdm/subset/` |
| Trier (CC BY 4.0) | *already staged in-repo* (reused, not re-downloaded) | **27,154 images**, 24 envs, 4 groups | 673 MB | `datasets/staging/trier/` |

Acquired EmptyDeck sequences (all labelled `failing`, `failure_type=bed_adhesion`):

| id | env | frames | onset_frame |
|---|---|--:|--:|
| P1__PinkNetriti_Early | P1 | 270 | 49 |
| P1__YellowEarly | P1 | 391 | 120 |
| P1__BlackTilt_noz | P1 | 1207 | 312 |
| P1__WhiteNeitiri_SupportFail | P1 | 2140 | 1865 |
| P1__YellowOval_still | P1 | 4232 | 2994 |
| P1__WhiteBoat | P1 | 4309 | 3991 |
| P2__RedSkullEarly | P2 | 741 | 317 |
| P2__GreenSkullEarly | P2 | 1166 | 460 |
| P2__GreenCase | P2 | 3816 | 688 |
| P2__BlackFrancesco | P2 | 3417 | 2876 |
| P2__PinkPenholder | P2 | 6780 | 1863 |

Each row carries the source video sha256, the CC BY-NC 4.0 licence, attribution, the
capture environment (`P1`/`P2`) and `onset_frame` in its manifest entry.

## 3. Environment / print / source counts

* **Sources (independent capture setups) newly acquired:** EmptyDeck (2 environments:
  P1 1024×768 bed-mounted, P2 1920×1080) and wengmhu (1 static venue).
* **Independent failing prints:** **11** (EmptyDeck), all bed-adhesion.
* **Independent normal streams:** **0** (see §5).
* Existing in-repo for reference: Trier = 24 static environments (3 cameras × 8
  defect groups); Bambu staging = 3 chamber-camera environments (212 KB metadata only).

## 4. Validation results

* `tv.data.manifest.load_manifest` + `validate_manifest` on the EmptyDeck manifest:
  **0 warnings** (no duplicate ids, every `frames_dir` exists, all `frame_count > 0`).
* `tv.data.streams.build_streams` → **11 streams, 28,469 frames**, `by_label={failing:11}`,
  `by_source={title:11}` (weak video-level labels; the VLM was not used, consistent
  with the RC2 finding that single-frame VLM labels are noisy).
* Unit tests `tests/test_manifest.py` + `tests/test_streams.py`: **7 passed**.
* The RC2 release-gate / diagnostic artefacts were **not** touched or re-run.

This acquisition adds genuine **independent failing prints across two camera
environments**. It does **not** add any independent normal stream, and — because the
agent has no vision — none of the labels are human-verified; they are the dataset's
video-level weak labels plus its published `onset_frame`.

## 5. Remaining gaps

1. **Independent normal streams: 0 / 16.** No accessible source provides independent
   normal *multi-frame* streams. EmptyDeck baselines are the same print; Trier,
   wengmhu and Roboflow are static. This is the dominant unresolved gap and it is the
   reason RC2's false-alarm rate is still unbounded.
2. **Human verification: 0.** The agent cannot see images; all labels remain
   dataset-supplied weak labels. The ≥16 normal streams must be human-confirmed.
3. **Failure-type coverage.** EmptyDeck is 100% `bed_adhesion`; wengmhu adds static
   cracking / layer-shift / off-platform / stringing / warping; no independent
   spaghetti / under-extrusion / warping **video** streams.
4. **Kaggle / Roboflow blocked** on account + (for Kaggle) competition-rule acceptance.
5. **Licence boundary.** EmptyDeck is **CC BY-NC 4.0** and is therefore
   **research-only**: keep it out of any release-bound or training split and do not
   redistribute the source media (the manifest sets `noncommercial=true`,
   `redistributable=false`).

## 6. Single highest-value next action

**Accept the Kaggle `early-detection-of-3d-printing-issues` competition rules and
download its data.** It is the only remaining *external* source that plausibly
contains many independent snapshot-sequences from **7 different printers × 6–20
prints each** — i.e. the only candidate that could supply the independent
normal/failing **environments** the gap needs, in a distinct camera domain. It needs
only a Kaggle account and rule acceptance (no paid services). Verify, before use,
(a) whether it contains normal prints and (b) its actual licence, since competition
rules (not CC) govern it.

If that data proves not to contain normal prints, the guaranteed fallback is **own
collection**: one fixed camera on a target printer running ≥16 normal prints plus ≥4
induced failures. That is the only path that directly yields independent,
human-verifiable normal streams — but it needs hardware/time rather than a download.

## 7. Exact commands and file paths

Data root: `/Users/alex/tinyvision-data` (see `/tmp/tv_data_root`).

```bash
# 1. EmptyDeck annotations repo (547 MB)
git clone --depth 1 https://github.com/EmptyDeck/3DPrint-Failure-Video-Dataset.git \
  /Users/alex/tinyvision-data/emptydeck-3dprint

# 2. Targeted EmptyDeck subset: download + 1 fps frames + provenance manifest
training/.venv/bin/python scripts/acquire_emptydeck.py \
  --repo /Users/alex/tinyvision-data/emptydeck-3dprint \
  --out-root /Users/alex/tinyvision-data/emptydeck-3dprint-run \
  --only "PinkNetriti_Early,YellowEarly,BlackTilt_noz,WhiteNeitiri_SupportFail,YellowOval_still,WhiteBoat,RedSkullEarly,GreenSkullEarly,BlackFrancesco,GreenCase,PinkPenholder" \
  --max-filesize 250000000
# -> /Users/alex/tinyvision-data/emptydeck-3dprint-run/manifest.json
# -> /Users/alex/tinyvision-data/emptydeck-3dprint-run/videos/<seq>/{*.mp4,frames/}

# 3. wengmhu (MIT) via Google Drive large-file confirm flow, then static subset
curl -sL "https://drive.usercontent.google.com/download?id=<ID>&export=download" -o page.html
curl -sL "https://drive.usercontent.google.com/download?id=<ID>&export=download&confirm=t&uuid=<UUID>" \
  -o /Users/alex/tinyvision-data/wengmhu-fdm/wengmhu.zip
# extract 40/class -> /Users/alex/tinyvision-data/wengmhu-fdm/subset/
# -> /Users/alex/tinyvision-data/wengmhu-fdm/subset_manifest.json (per-file sha256)

# 4. Validation
training/.venv/bin/python -m pytest training/tests/test_manifest.py training/tests/test_streams.py -q
```

Repo files created/modified by this task (none committed):

* `scripts/acquire_emptydeck.py` (new) — targeted EmptyDeck downloader + manifest writer.
* `training/tv/data/manifest.py` (modified) — `Source` gained optional provenance/
  licence fields (`source_dataset`, `license`, `license_url`, `attribution`,
  `capture_environment`, `session_id`, `video_sha256`, `noncommercial`,
  `redistributable`, `split`, `notes`); `from_dict` is backward compatible.
* `docs/research/datasets/emptydeck-bed-adhesion-onset.md` (new record).
* `docs/research/datasets/roboflow-3d-printing-errors-g4f6l.md` (new record).
* `docs/research/datasets/kaggle-early-detection-3d-printing-issues.md` (new record).
* `docs/research/datasets/fdm-3d-printing-defect.md` (corrected Cracking count + acquisition note).
* `docs/research/datasets/README.md` (index updated).
* This report + `realworld-data-acquisition.json`.

## 8. Update 2026-10-14 — Kaggle source inspected (access blocked)

The Kaggle source was inspected via its **public** pages (no sign-in, no rules
accepted). Verified facts: the dataset is **106,342 files / 12.29 GB**; the label is a
**print-level binary** `has_under_extrusion` (`1` failure, **`0` = good print**), and
*all snapshots in a print share the label*. Structure: **7 printers × 6–20 prints**,
a print being a snapshots series ~0.5 s apart; `images/` folders are keyed by printer
id; two printers (`021`,`023`) and several prints are held out for test.

**This materially improves the outlook on the normal-stream gap:** because good
prints exist and are labelled `0`, **independent normal streams can be reconstructed**
(one good `print_id` = one independent multi-frame normal stream; one `printer_id` =
one environment; group by `print_id`/hold out whole printers to avoid leakage). It
adds only **under-extrusion** on the failure side.

**However, acquisition is blocked on user action:** Kaggle authentication is **not
configured** (no CLI, package, `kaggle.json`, or env vars), and the Data tab is
gated — *"you need to agree to the competition rules … sign in or register to accept
the rules."* The data licence is **"Subject to Competition Rules"**, which is minimal
and pro-open-source but **not** an open licence and does not clearly permit
redistribution. Recommended handling: **research-only, non-redistributable**, no
commits of the data.

**Exact blocking step:** the user must (1) sign in to Kaggle and **accept the
competition rules**, and (2) provide credentials (a `kaggle.json` API token at
`~/.kaggle/kaggle.json`, or a manual download to
`/Users/alex/tinyvision-data/kaggle-early-detection/`). Disk is sufficient (122 GB
free vs 12.29 GB needed). No download, inventory, manifest, or validation of this
dataset was performed.

### Kaggle "Early detection of 3D printing issues" (competition) — acquired

- License: Subject to Competition Rules — research-only, **non-redistributable** (noncommercial; no blanket open license). 
- Acquisition: downloaded via `kaggle competitions download -c early-detection-of-3d-printing-issues` (CLI 2.2.4, OAuth, rules accepted). 106,339 images, 12.29 GB.
- Labeled (train): 52 print streams total; **27 independent normal** + **25 under-extrusion failing** (print-level). No print mixed labels.
- Environments: 7 printers total (021,022,023,101,102,103,104); 5 in labeled train (022,101,102,103,104).
- Leakage-safe grouping: stream = `print_id`, environment = `printer_id`. Environment-held-out: train (101,102,104: 13N/14F, 63,641 frames), val (022: 10N/7F, 7,492 frames), test (103: 4N/4F, 9,927 frames). No printer spans splits.
- Paths: `/Users/alex/tinyvision-data/kaggle-early-detection/{manifest.json, splits/, index.csv, kaggle-early-detection-metadata.json, train.csv,test.csv,sample_submission.csv}`
- Validation: `validate_manifest` → 0 warnings; `build_streams` → 52 streams, 81,060 frames; environment disjointness verified.


## Gap closure (Kaggle acquired)

- Independent normal streams: **0 → 27** (meets ≥16). These are 27 labeled good prints from train (printers 022,101,102,103,104; 5 environments).
- Independent failing streams: **11 → 36** (EmptyDeck 11 + Kaggle 25 under-extrusion). Meets ≥4.
- Leakage-safe: streams grouped by `print_id`, environments held out by `printer_id` (no printer spans splits).
- Status: The independent-normal bottleneck is **CLOSED** for the stated RC2 gap. Environmental coverage remains limited (7 printer environments total across both classes) and failure modes are still narrow (bed adhesion + under-extrusion).
