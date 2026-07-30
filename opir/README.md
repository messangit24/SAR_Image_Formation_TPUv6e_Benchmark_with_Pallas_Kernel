# NOAA-20 (JPSS-1) VIIRS Overhead Persistent Infrared (OPIR) Processing Engine

This module provides end-to-end processing of **authentic NOAA-20 satellite Level 0 raw telemetry packet streams** and **Level 1B (SDR) calibrated OPIR products**.

## 🛰️ Satellite & Sensor Specifications

- **Satellite Platform:** NOAA-20 (JPSS-1 / Joint Polar Satellite System-1), Sun-Synchronous Polar Orbit (~824 km altitude).
- **Sensor Payload:** VIIRS (Visible Infrared Imaging Radiometer Suite).
- **MWIR Band M12 (3.70 µm):** Primary OPIR active thermal fire/hotspot detection band (750 m GSD, 3200 cross-track pixels x 768 in-track lines).
- **MWIR High-Res Band I4 (3.74 µm):** High-resolution imaging band (375 m GSD).
- **LWIR Band M15 (10.76 µm):** Long-wave infrared thermal background baseline.
- **Precision Geolocation (GMTCO):** WGS84 terrain-corrected latitude and longitude arrays for every individual pixel detector.

## 📦 Ingested Satellite Data (`opir/data/`)

1. **`RNSCT_j01_d20260529_t1539410.h5`** (7.05 MB): Authentic Level 0 raw downlinked CCSDS binary telemetry packet stream (`RawApplicationPackets_0`, APID `0x230`).
2. **`SVM12_j01_d20211202_t1438358.h5`** (5.40 MB): Level 1B Sensor Data Record (SDR) for MWIR Band M12 (2,457,600 pixels). Contains 12-bit detector A/D readout counts $q_{\text{raw}}$, gain factors, and calibrated brightness temperatures.
3. **`SVI04_j01_d20211202_t1438358.h5`** (22.26 MB): Level 1B SDR for MWIR Band I4 (375m resolution).
4. **`SVM15_j01_d20211202_t1438358.h5`** (5.32 MB): Level 1B SDR for LWIR Band M15 (10.76 µm).
5. **`GMTCO_j01_d20211202_t1438358.h5`** (23.54 MB): WGS84 Terrain-Corrected Precision Geolocation (Latitude & Longitude arrays).

## 🚀 Execution & Verification

Run the main pipeline:
```bash
python3 opir/noaa_viirs_l0_l1b_pipeline.py
```

## 📐 Calibration Formulas

### 1. Radiometric Calibration
$$L_\lambda = \text{Scale}_{\text{rad}} \times q_{\text{raw}} + \text{Offset}_{\text{rad}}$$

### 2. Analytical Planck Spectral Inversion
$$T = \frac{h c}{\lambda k_B \ln\left(1 + \frac{2 h c^2}{\lambda^5 L_\lambda}\right)}$$
