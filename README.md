# Remote Sensing & Payload Processing Benchmark Repository

This repository contains multi-modal remote sensing image formation and payload processing engines for **Synthetic Aperture Radar (SAR)** and **Overhead Persistent Infrared (OPIR)** datasets.

---

## 📁 Repository Organization

```
.
├── sar/                                # Synthetic Aperture Radar (SAR) Module
│   ├── bpa.py                          # Backprojection Algorithm (BPA) CPU/TPU Engine
│   ├── csa.py                          # Chirp Scaling Algorithm (CSA)
│   ├── pfa.py                          # Polar Format Algorithm (PFA)
│   ├── rma.py                          # Range Migration Algorithm (RMA) / Omega-K
│   ├── cphd_reader.py                  # CPHD Raw Phase History Data Parser
│   ├── render_multi_method_comparison.py # Multi-Method Verification Script
│   └── tpu_bpa.py                      # TPU Accelerated BPA Engine
├── opir/                               # Overhead Persistent Infrared (OPIR) Module
│   ├── data/                           # Ingested Authentic NOAA-20 Satellite HDF5 Files
│   │   ├── RNSCT_j01_d20260529_t1539410.h5  # Level 0 Raw Telemetry Packet Stream (7.05 MB CCSDS)
│   │   ├── SVM12_j01_d20211202_t1438358.h5  # Level 1B SDR MWIR Band M12 (3.7 µm Fire Band, 2.45M Pixels)
│   │   ├── SVI04_j01_d20211202_t1438358.h5  # Level 1B SDR MWIR Band I4 (3.74 µm High-Res 375m)
│   │   ├── SVM15_j01_d20211202_t1438358.h5  # Level 1B SDR LWIR Band M15 (10.76 µm Thermal Baseline)
│   │   ├── GMTCO_j01_d20211202_t1438358.h5  # Precision WGS84 Geolocation (Latitude & Longitude)
│   │   └── noaa_viirs_opir_benchmark.png    # Benchmark Figure Output
│   ├── noaa_viirs_l0_l1b_pipeline.py   # NOAA-20 VIIRS Processing Pipeline Script
│   ├── true_l0_pipeline.py             # True L0 Noise & Spike Filter Simulation Engine
│   ├── geolocation_evaluation.py       # Geostationary Un-Projection & GSD Distortion Engine
│   └── README.md                       # OPIR Module Documentation
├── docs/                               # SAR Benchmarks & Specifications
├── tpu_bpa.py                          # Top-Level SAR BPA Wrapper Script
├── tpu_ffbp.py                         # Top-Level SAR FFBP Wrapper Script
├── requirements.txt
└── README.md
```

---

## 🚀 Execution Instructions

### 1. Execute NOAA-20 VIIRS OPIR Telemetry & Calibration Pipeline
```bash
python3 opir/noaa_viirs_l0_l1b_pipeline.py
```

### 2. Execute SAR BPA Benchmark
```bash
python3 tpu_bpa.py
```
