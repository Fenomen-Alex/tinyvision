# Dataset Record: FDM 3D Printing Defect Dataset (wengmhu)

- **Name**: FDM-3D-Printing-Defect-Dataset
- **Source**: GitHub (wengmhu) and Kaggle mirror
- **URL**: https://github.com/wengmhu/FDM-3D-Printing-Defect-Dataset ; Kaggle: https://www.kaggle.com/datasets/wengmhu/fdm-3d-printing-defect-dataset
- **License**: MIT (GitHub repo)
- **Image count**: 1,912 total
  - Cracking: 427
  - Layer_shifting: 364
  - Off_platform: 91
  - Stringing: 447
  - Warping: 538
- **Annotation format**: Image classification by folder (defect type); no bounding boxes.
- **Classes**: cracking, layer_shifting, off_platform, stringing, warping
- **Resolution**: Not stated in record.
- **Collection methodology**: FFF/FDM printer defect images (per the accompanying paper).
- **Known limitations**: Classification-only (no box/segmentation labels); single-class per image; relatively small (~1.9k images); "off_platform" class underrepresented (91 images).
- **Intended usage**: Defect classification; the authors' paper uses it for real-time FFF defect detection with a lightweight model.
- **Retrieval date**: 2026-09-02 (research phase).

## Licensing notes
MIT license permits commercial use with attribution/copyright notice. Verified via the GitHub repository LICENSE file. The Kaggle mirror should be checked for its own terms.
