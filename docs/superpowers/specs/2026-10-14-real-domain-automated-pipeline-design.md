# Design — Automated real-domain data pipeline (no printer, minimal labelling)

Date: 2026-10-14
Status: approved (design), implementation pending
Author: TinyVision audit session

## 1. Problem

RC1 (`rc1_release_candidate_shufflenet_v2_x1_0`) fails on real footage because
it was trained on Trier photos, not the target domain (webcam framing, real
lighting, real spools). On real frames its `P(failing)` is high on almost
everything: all 5 VLM-normal videos alarm (video-level FAR = 1.0), and every
Stage-14 idle frame scores > 0.5 (median 0.92). The only real fix is real
in-domain training data. The operator has **no printer** and **no time to label
by hand**, so both acquisition and labelling must be automated.

## 2. Goal

An automated, local, reproducible pipeline that turns publicly available 3D-print
footage into an in-domain, source-isolated dataset with automatic labelling, and
trains a model that measurably reduces real-world false alarms relative to RC1 —
evaluated honestly through the existing reliability gate.

## 3. Success criteria (measurable, honest)

On a **held-out set of real videos** (source-level split; never trained on):

| metric | target | RC1 today |
|---|---|---|
| video-level false-alarm rate on normal clips | ≤ 0.2 | 1.0 |
| known failing clips detected | ≥ 1 of 2 | (separate measure) |

* Recorded in `docs/research/` with the config, manifest, and gate output.
* An honest gate **FAIL** is an acceptable outcome; a fabricated PASS is not.
* Fully reproducible: same sources + same config ⇒ same split and same numbers.

## 4. Non-goals

* Production/release approval; printer control or pause integration.
* Redistributing or committing scraped media.
* State-of-the-art accuracy, or fixing the domain gap without real data.
* Replacing the reliability gate or loosening any policy threshold.

## 5. Constraints (from AGENTS.md and the mission)

* **Research-only** footage: cached under a gitignored dir, never committed or
  redistributed; every source recorded with URL, retrieval date, and a
  `research_only` licence note.
* **No fabricated labels**: VLM output is a *proposal*; `uncertain` is preserved
  and never silently converted. Human review remains the only source of verified
  labels, and is reduced to a small, **optional/deferrable** QA pass (≤ 5% of
  frames on the review queue).
* **Vision dependency**: the implementing agent cannot view images, so the human
  QA pass is the operator's (or is deferred). The pipeline must be fully usable
  with auto-accepted + weak labels alone; QA only upgrades trust, it is never
  required to run the pipeline or the gate.
* **Source-level isolation**: a video that appears in one split never appears in
  another (leakage guard; already proven in this repo).
* **Local, small**: inference and training on the operator's M1 (CPU/MPS);
  no cloud, no large model downloads.
* **Deterministic CLI** over notebooks.

## 6. Approaches considered

* **A — Acquisition engine only.** Build acquire + weak-label + VLM proposal
  tooling, produce no trained model. Unblocks everything but delivers nothing
  runnable today. *Rejected as the sole deliverable.*
* **B — Reconstruct + retrain (chosen).** Re-acquire the known real videos from
  the YouTube IDs already recorded in the repo, plus a few keyword searches;
  weak-label + VLM-label automatically; train in-domain with source-level
  splits; evaluate on held-out videos. Produces a working artifact now.
* **C — Normal-only anomaly detector.** Train only on normal footage; failure =
  deviation. Fewer labels but a larger research bet; kept as a **second arm**
  for comparison, not the primary path.

Chosen: **B now, C as a second arm (B+)** so the two decision strategies can be
compared on the same held-out videos.

## 7. Architecture

New, independently testable units. Each has one purpose and a narrow interface.

```
training/tv/data/acquire.py     download + frame extraction + probing
training/tv/data/weaklabel.py   title/description -> weak video-level label
training/tv/data/manifest.py    dataset manifest read/write/validate
training/tv/data/autolabel.py   VLM proposals + confidence gating + QA queue
training/tv/data/streams.py     source-isolated train/val/test split builder
training/tv/cli.py              new subcommands: acquire, autolabel, streams
training/configs/planned/rc2_realdomain.yaml   training config (NOT auto-run)
training/configs/reliability/suite-realdomain.yaml  gate suite (held-out videos)
```

### 7.1 `acquire.py`

```python
def probe(video_path: Path) -> dict            # duration, resolution via ffprobe
def fetch_video(source: str, out_dir: Path,
                max_seconds: int | None = None,
                runner=subprocess.run) -> AcquiredVideo   # yt-dlp wrapper
def extract_frames(video_path: Path, out_dir: Path, fps: float = 1.0,
                   max_side: int | None = None,
                   runner=subprocess.run) -> list[Path]      # ffmpeg 1fps grid
```

* `source` is a URL or a bare YouTube id.
* Idempotent/resumable: an already-downloaded video or extracted frame set is
  reused, not re-fetched.
* All external commands go through an injectable `runner` so tests mock them;
  no network or ffmpeg in unit tests.
* Failures raise `AcquireError` with the command and stderr; a failed source is
  recorded, never silently dropped.

### 7.2 `weaklabel.py`

```python
def weak_label(title: str, description: str = "",
               keywords: dict | None = None) -> str   # 'normal' | 'failing' | 'uncertain'
```

* Curated, versioned keyword sets: failing terms (`spaghetti`, `fail`, `failed`,
  `layer shift`, `blob`, `warped`, `adhesion`, `stringing`, `clog`, `mess`) and
  normal terms (`timelapse`, `success`, `finished`, `printing`, `asmr`).
* Ambiguous/no signal ⇒ `uncertain`. Video-level only; frame labels come from
  `autolabel.py`. Keyword sets live in the module (single source of truth).

### 7.3 `manifest.py`

```python
@dataclass Source:    id, url, title, uploader, duration, resolution,
                      retrieved_at, license_note, weak_label, frames_dir,
                      frame_count, failure_type | None
@dataclass Manifest:  version, created_at, sources: list[Source]
def load_manifest(path) -> Manifest
def save_manifest(manifest, path) -> None
def validate_manifest(manifest) -> list[str]   # warnings/errors
```

* JSON, human-readable, diff-friendly.
* `validate_manifest` flags missing frame dirs, zero-frame sources, and
  duplicate ids.

### 7.4 `autolabel.py`

```python
def propose_labels(manifest, client, *, model: str,
                   confidence_threshold: float = 0.9) -> LabelResult
```

* Reuses the **existing** `tv/vlm_review.py` proposal machinery / prompt
  discipline and the LM Studio endpoint (`http://127.0.0.1:1234`,
  `qwen/qwen3-vl-8b`).
* For each frame, a `normal`/`failing`/`uncertain` proposal + confidence.
* `confidence >= threshold` and label agrees with `weak_label` (when not
  `uncertain`) ⇒ **auto-accepted**; everything else ⇒ **review queue**.
* Emits `proposals.json` and `review_queue.json`. Never writes verified labels.
* VLM errors are recorded per frame; never guessed.

### 7.5 `streams.py`

```python
def build_splits(manifest, labels, *, ratios=(0.7, 0.15, 0.15),
                 seed: int = 0) -> Splits
def write_splits(splits, out_dir) -> None   # path,label CSVs + splits.json
```

* Splits by **source** (video), never by frame (prevents temporal leakage).
* Guard: raises if any source id appears in more than one split.
* Output CSVs match the existing `datasets/processed/*/splits/{train,val,test}.csv`
  format so the existing trainer/gate consume them unchanged.

## 8. Data flow

```
sources.json (known ids + search queries)
        │  tv acquire
        ▼
cache/<id>/video.mp4 ──ffmpeg──► cache/<id>/frames/*.jpg
        │  weak_label(metadata)
        ▼
manifest.json ──tv autolabel (LM Studio)──► proposals.json + review_queue.json
        │  (human QA on the queue only)
        ▼
verified ∪ auto-accepted ──tv streams──► splits/{train,val,test}.csv
        │
        ▼
tv train (rc2_realdomain.yaml) ─► best.pt ─► export onnx/int8
        │
        ▼
tv reliability evaluate (suite-realdomain.yaml) ─► held-out FAR/recall report
```

## 9. Error handling

* Missing network / yt-dlp / ffmpeg ⇒ explicit error naming the tool and the
  source; the pipeline is resumable and skips already-complete work.
* LM Studio down ⇒ `autolabel` reports the error and produces no guesses.
* Corrupt/zero-length video ⇒ recorded as a failed source, excluded from splits.
* A new surface with no data ⇒ the gate returns INCONCLUSIVE, never PASS.

## 10. Testing (test-driven)

* `weak_label`: failing/normal/uncertain keyword cases + ambiguous ⇒ uncertain.
* `manifest`: round-trip; validation catches missing dir, zero frames, dup ids.
* `acquire`: mocked `runner` asserts exact yt-dlp/ffmpeg argv and idempotent
  reuse; failure path raises `AcquireError`.
* `autolabel`: mocked HTTP; threshold gating; uncertain never auto-accepted;
  error path records, never guesses.
* `streams`: source-level isolation guard raises on overlap; ratios/seed
  deterministic; CSV format matches the existing loader.
* Integration (no network): `ffmpeg testsrc` synthetic clip → `extract_frames`
  → `build_splits` → CSVs load via the existing `CSVImageDataset`.

## 11. Risks

| risk | mitigation |
|---|---|
| YouTube ToS / licensing | research-only, cached locally, never committed/redistributed |
| Tiny dataset ⇒ overfitting | source-level splits, heavy augmentation, honest gate |
| VLM label noise | auto-accept only high-confidence+agreeing; small human QA; trusted holdout |
| Compute | small model, few epochs, M1 MPS/CPU |
| Downloaded media lost | deterministic re-acquire from recorded URLs |

## 12. Vertical slice (first implementation milestone)

Prove the path end-to-end on 2 known videos (1 normal: `0TY6LVvMpUo`;
1 failing: `87gsL-Luvis`): acquire → frames → weak label → manifest → split →
confirm CSVs load. Only then scale to more sources and add VLM auto-labelling.

## 13. Open decisions resolved

* Scraper vs retrain scope → **B+ (retrain + anomaly arm)**.
* Label strategy → **VLM proposals + weak labels + ≤5% human QA**.
* Storage → gitignored `datasets/external-cache/` (research-only).
* Model → fine-tune RC1 (primary), normal-only anomaly (arm C).
