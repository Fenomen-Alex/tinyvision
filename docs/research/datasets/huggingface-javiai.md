# Dataset Record: failures-3D-print (Hugging Face)

- **Name**: failures-3D-print
- **Source**: Hugging Face Datasets (Javiai)
- **URL**: https://huggingface.co/datasets/Javiai/failures-3D-print
- **License**: **Unknown** (dataset card lists license as "unknown").
- **Image count**: 73 rows across ~375–1,600 px image widths.
- **Annotation format**: Object detection (bounding boxes). Format: `bbox` (x_center, y_center, width, height) + `categories` per image.
- **Classes (4)**: error (general error), extrusor (extruder base), part (the printed part), spagheti (spaghetti).
- **Resolution**: Varies; widths ~375–1,600 px.
- **Collection methodology**: Not documented on the card.
- **Known limitations**: Very small (73 images); class naming has typos/ambiguity (e.g. "spagheti"); license unknown.
- **Intended usage**: Fine-tuning object-detection models for 3D print failure detection.
- **Retrieval date**: 2026-09-02 (research phase).

## Licensing notes
**License unknown — NOT usable** until licensing is resolved, per the TinyVision dataset policy. A fine-tuned model snapshot (Javiai/3dprintfails-yolo5vs) exists on Hugging Face.
