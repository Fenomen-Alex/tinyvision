# Dataset Record: 3D-Printing-Data (Szydlo et al.)

- **Name**: Dataset for Anomalies Detection in 3D Printing
- **Source**: GitHub (joanna-) / Springer ICCS 2021 paper
- **URL**: https://github.com/joanna-/3D-Printing-Data ; paper DOI: https://doi.org/10.1007/978-3-030-77970-2_50
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Image count**: Not an image dataset. Contains timeseries sensor data (acceleration via two LSM9DS1 IMUs, tension) plus printer-interface JSON logs for several printing scenarios.
- **Annotation format**: CSV (`accel.txt`) and JSON files per scenario.
- **Classes / scenarios**: proper (reference), plastic_lack, unstick, bowden (bowden tube fallout), plus documented scenarios: wrong retraction setup, arm failure, missing tower (part removed during printing).
- **Resolution**: N/A (sensor data).
- **Collection methodology**: Collected using custom modular hardware (FogDevices4 project) on a MonkeyFab Spire printer; IMU sensors on print platform (accel0) and print head (accel1).
- **Known limitations**: Sensor/timeseries data, not camera images; single printer model; not directly a computer-vision dataset.
- **Intended usage**: Anomaly detection in 3D printing (potentially machine-learning / sensor fusion).
- **Retrieval date**: 2026-09-02 (research phase).

## Licensing notes
CC BY 4.0 permits commercial use with attribution. Verified via the GitHub LICENSE.
