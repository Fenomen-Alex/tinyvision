# Dataset Record: Bed-Adhesion Failure-Onset Dataset for FDM 3D Printing (EmptyDeck)

- **Name**: Bed-Adhesion Failure-Onset Dataset for FDM 3D Printing
- **Source**: GitHub (EmptyDeck); timelapse videos hosted on YouTube
- **URL**: https://github.com/EmptyDeck/3DPrint-Failure-Video-Dataset
- **License**: **CC BY-NC 4.0** (noncommercial). The GitHub API reports no license
  field; the authoritative statement is the repository README, which places the
  annotations, masks and metadata under CC BY-NC 4.0 and states the externally
  hosted videos are "provided for research use under the same terms".
- **Image count**: 30 print sequences (20 environment **P1**, 10 environment **P2**).
  A targeted subset of 11 sequences (28,469 frames at fps=1) is acquired locally.
- **Annotation format**: per-sequence failure **onset** (`onset_frame`); ground-truth
  masks covering the pre-onset interval `[0, onset_frame)`. This is a video change-point
  / segmentation annotation, not a per-frame classification of every frame.
- **Classes**: bed-adhesion failure onset (the pre-onset interval is the print's own
  *normal baseline*).
- **Resolution**: P1 1024×768 (bed-mounted), P2 1920×1080; cadence 1 fps.
- **Collection methodology**: bed-mounted camera timelapses of FFF prints that
  eventually fail by bed adhesion (object detaches / moves).
- **Known limitations**:
  - Every failure is **bed adhesion** — a single failure type.
  - Only **two** capture environments (P1, P2).
  - The `n_normal_baseline` frames are the pre-onset segment of **the same failing
    print**, so they are **not** independent normal streams.
  - **CC BY-NC 4.0**: research-only; not for commercial use or redistribution.
- **Intended usage**: independent *failing* streams for research evaluation. Must
  **never** enter a release-bound or training split. Source media is not redistributed.
- **Retrieval date**: 2026-10-14.

## Acquisition record

- Annotation repo cloned to `/Users/alex/tinyvision-data/emptydeck-3dprint` (547 MB).
- 11 shortest sequences downloaded (360p) and decoded to 1 fps frames with
  `scripts/acquire_emptydeck.py`:
  `P1__PinkNetriti_Early, P1__YellowEarly, P1__BlackTilt_noz,
  P1__WhiteNeitiri_SupportFail, P1__YellowOval_still, P1__WhiteBoat,
  P2__RedSkullEarly, P2__GreenSkullEarly, P2__GreenCase, P2__BlackFrancesco,
  P2__PinkPenholder` → `/Users/alex/tinyvision-data/emptydeck-3dprint-run/`
  (**1.6 GB**, manifest at `…/manifest.json`).
- Manifest entries set `noncommercial=true`, `redistributable=false`, and record the
  video sha256, `capture_environment`, `onset_frame` and attribution.
