# Dataset Record: 3D Printing Errors (NilsHagenBeyer)

- **Name**: 3d printing errors
- **Source**: Kaggle (NilsHagenBeyer / nimbus200) and Hugging Face mirror
- **URL**:
  - Kaggle: https://www.kaggle.com/datasets/nimbus200/3d-printing-errors
  - Hugging Face: https://huggingface.co/datasets/NilsHagenBeyer/3D_printing_errors
- **License**: Apache 2.0 (verified on both Kaggle page and Hugging Face dataset card)
- **Image count**:
  - Stated per class (Kaggle card, original set): GOOD 5,069; UNDEREXTRUSION 2,962; STRINGING 2,798; SPAGHETTI 134 (total ≈ 10,963).
  - `all_images_no_filter.csv` (the full, unfiltered download used for Experiment 1): 16,287 images total. This includes additional recordings beyond the stated original counts (silver-bed and oblique-camera test image subgroups).
  - `all_images.csv` (no spaghetti rows): not used in Experiment 1.
- **Annotation format**: CSV metadata linked to images (`image` filename ↔ `class` label); classification labels only (no bounding boxes/segmentation).
- **Classes** (class_list.txt): 0=GOOD, 1=UNDEREXTRUSION, 2=STRINGING, 4=SPAGHETTI (note: index 3 is skipped).
- **Resolution**: 256×256 PNG (main set `all_images256`); also includes `test_images_silver265` (265×265) and `test_images_oblique256` (256×256). Source camera is ELP-USB13MAFKV76 (Sony IMX214, 3264×2448), downscaled.
- **Collection methodology**: Recorded while printing PLA parts on a customized Creality Ender-3 Pro with a fixed autofocus USB webcam (position 1). Metadata (layer, nozzle, extrusion multiplier, retraction, layer height, filament color, shape, `recording` session id, print-bed color) collected via an OctoPrint macro recorder. Under-extrusion was partially injected synthetically via gcode modification. 34 geometries ("shape" 1–34).
- **Known limitations**:
  - Printed on a single printer model (Ender-3 Pro) with a single filament type/color (PLA, gray) and predominantly a black print-bed — domain narrowness.
  - Class imbalance (SPAGHETTI underrepresented at 134).
  - Consecutive frames of the same recording are highly correlated (temporal redundancy); requires source-level splitting (by `recording`) to avoid leakage.
  - Oblique-camera images have significant motion blur; silver-bed images differ in background.
  - Binary mapping for Experiment 1 maps GOOD→normal and all three error classes→failing; the multiple error modes are collapsed, so per-mode confusion is not separated.
- **Intended usage**: Visual error detection in FDM printing (author's FDM_error_detection project, ResNet/baseCNN). For TinyVision: Experiment 1 binary classification (normal vs failing).
- **Retrieval date**: 2026-09-03 (downloaded `datasets/Printing_Errors/`).

## Experiment 1 use

Prepared to Layout A in `datasets/processed/printing-errors/` (gitignored) with
binary labels: `normal` (GOOD) and `failing` (UNDEREXTRUSION + STRINGING +
SPAGHETTI). Split at the `recording` level (70/15/15, seed 42) to prevent leakage.
Audit: 16,287 images, 0 corrupted, 0 exact duplicates, uniform 256×256.

## Licensing notes
Apache 2.0 permits commercial use, modification, and distribution with attribution.
Verified via the Kaggle page license badge and the Hugging Face dataset card
("License: apache-2.0"). The underlying printer/camera are the author's own,
so no third-party image copyright concern beyond the author's releases under Apache 2.0.
