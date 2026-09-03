# Dataset Record: Caxton (University of Cambridge)

- **Name**: Caxton — dataset for "Generalisable 3D printing error detection and correction via multi-head neural networks"
- **Source**: University of Cambridge Apollo repository
- **URL**: https://www.repository.cam.ac.uk/items/6d77cd6d-8569-4bf4-9d5f-311ad2a49ac8 (DOI: 10.17863/CAM.84082)
- **License**: Attribution 4.0 International (CC BY 4.0)
- **Image count**: 1,272,273 labelled images (948,396 after filtering). CSVs range ~110–144 MB.
- **Annotation format**: Per-image CSV labels including flow rate, lateral speed, Z offset, hotend temperature, hotend target temperature, bed temperature, timestamp, nozzle tip x/y coordinates. Not traditional bounding-box object detection labels.
- **Classes**: Not labelled with discrete "failure" classes; instead continuous process parameters and an anomaly/failure status suitable for regression and anomaly detection (per the accompanying paper).
- **Resolution**: Not stated in the record; camera mounted next to the nozzle.
- **Collection methodology**: Automated pipeline on a fleet of 8 extrusion printers; 192 parts; varied geometries, material colours, and lighting; Marlin 1.1.9 firmware.
- **Known limitations**: Primarily a parameter-label dataset for closed-loop control / error detection, not an image classification or object-detection set with discrete classes. Built for multi-head neural network training.
- **Intended usage**: Real-time error detection, closed-loop control, parameter prediction.
- **Retrieval date**: 2026-09-02 (research phase; record created from repository metadata).

## Licensing notes
CC BY 4.0 permits commercial use with attribution. Verified via the repository
"Rights and licensing" field.
