# Dataset Records for 3D-Print Failure Detection

This directory contains provenance records for public datasets relevant to 3D-print
failure detection, in line with the TinyVision dataset policy. Each record is a
structured Markdown file.

## Records

- [caxton-cambridge.md](./caxton-cambridge.md) — "Caxton" dataset (University of Cambridge), layer-wise nozzle-camera images of 1,272,273 extrusion-printing images.
- [emptydeck-bed-adhesion-onset.md](./emptydeck-bed-adhesion-onset.md) — "Bed-Adhesion Failure-Onset Dataset" (EmptyDeck), 30 timelapse prints with failure onset; **CC BY-NC 4.0 (research-only)**.
- [fdm-3d-printing-defect.md](./fdm-3d-printing-defect.md) — wengmhu FDM defect classification dataset (1,912 images, MIT).
- [3d-printing-data-jszydlo.md](./3d-printing-data-jszydlo.md) — Szydlo et al. sensor/anomaly dataset (CC-BY-4.0).
- [kaggle-3d-printer-defected.md](./kaggle-3d-printer-defected.md) — "3D-Printer Defected Dataset" (Kaggle, binary classification).
- [kaggle-early-detection-3d-printing-issues.md](./kaggle-early-detection-3d-printing-issues.md) — Kaggle competition, snapshots from 7 printers × 6–20 prints; licence unverified, needs auth.
- [roboflow-3d-printing-errors-g4f6l.md](./roboflow-3d-printing-errors-g4f6l.md) — "3d-printing-errors" Roboflow object-detection dataset (224 images, 7 classes incl. `normal`, CC BY 4.0).
- [roboflow-seprinder.md](./roboflow-seprinder.md) — "detect-failure-in-3d-printing" Roboflow object-detection dataset (~3.6k images, CC BY 4.0).
- [roboflow-sylucauc.md](./roboflow-sylucauc.md) — "3d-printing-failure-detection" Roboflow object-detection dataset (~5.8k images, CC BY 4.0).
- [huggingface-javiai.md](./huggingface-javiai.md) — "failures-3D-print" (Hugging Face, 73 rows, license unknown).
- [nilshagenbeyer-3d-printing-errors.md](./nilshagenbeyer-3d-printing-errors.md) — "3d printing errors" (NilsHagenBeyer / nimbus200, ~10.9k–16.3k images, Apache 2.0). **Used for Experiment 1.**

## Caveats

- Several Roboflow Universe entries are community-created and may contain
  re-hosted/derived images with uncertain provenance despite a CC-BY-4.0 claim.
  Licensing of the *underlying images* must be resolved before any commercial use.
- Kaggle datasets use Kaggle's terms of service; verify the specific license shown
  on each dataset page before redistribution or commercial use.
- License/quality information was gathered during the research phase and should be
  re-verified at retrieval time (per the dataset policy).
