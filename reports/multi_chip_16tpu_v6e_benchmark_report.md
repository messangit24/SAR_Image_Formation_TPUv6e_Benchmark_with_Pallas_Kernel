# 16-Chip Cloud TPU v6e (Trillium) Multimodal Benchmark Report

**SAR Dataset**: `2023-01-18-07-20-11_UMBRA-05_CPHD.cphd` | **Grid**: `1024x1024` | **Pulses**: `1024`  
**MODIS Dataset**: 2030 lines x 1354 frames (16 Thermal Emissive Bands = 43.98M Pixels)  
**Cluster**: `mb-tpu-benchmark` (Region: `europe-west4`)  
**Node Pool**: `mb-tpu-v6e-multi-16` (4 Nodes x ct6e-standard-4t = 16 Physical TPU v6e Chips)  
**Interconnect Mesh**: 4x4 Torus Optical ICI Mesh (1,600 Gbps cross-host bandwidth)  

---

## 🏆 Part 1: Real Umbra SAR 16-Chip Performance Scaling & Point-Target Quality

| Metric / Parameter | JAX TPU BPA (Single-Chip Base) | Pallas BPA (Single-Chip Opt) | Multi-Chip Pallas BPA (16-Chip) | Multi-Chip Pallas FFBP (16-Chip) |
|:---|:---:|:---:|:---:|:---:|
| **Latency (seconds)** | **79.296 s** | **35.370 s** | **9.954 s** | **4.600 s** |
| **Throughput (MP/s)** | 0.0132 MP/s | 0.0296 MP/s | **0.1053 MP/s** | **0.2280 MP/s** |
| **Speedup vs Baseline** | 1.00x | 2.24x | **7.97x** | **17.24x** |
| **Speedup vs Single-Chip Pallas** | - | 1.00x | **3.55x** | **7.69x** |
| **Median Coherence (γ_med)** | 1.0000 | 0.9967 | **0.9967** | **0.9967** |
| **High-Signal Coherence** | 1.0000 | 0.9973 | **0.9973** | **0.9973** |
| **Normalized Cross-Corr (NCC)** | 1.0000 | 0.9934 | **0.9933** | **0.9933** |
| **Impulse Response Resolution** | 11.528 m / 0.459 m | 11.528 m / 0.438 m | **11.528 m / 0.438 m** | **11.528 m / 0.438 m** |
| **Range PSLR (Peak Sidelobe Ratio)** | -0.01 dB | -0.03 dB | **-0.02 dB** | **-0.02 dB** |
| **Azimuth PSLR (Peak Sidelobe Ratio)** | -5.33 dB | -4.67 dB | **-4.67 dB** | **-4.67 dB** |
| **Range ISLR (Integrated Sidelobe)** | 10.02 dB | 11.14 dB | **11.15 dB** | **11.15 dB** |
| **Azimuth ISLR (Integrated Sidelobe)** | -0.98 dB | -1.05 dB | **-1.05 dB** | **-1.05 dB** |
| **Numerical Parity (L2 Error)** | Reference | 8.37e-02 | **8.47e-02** | **8.47e-02** |

---

## 🛰️ Part 2: Real MODIS 16-Chip Multispectral Thermal Inversion

| Workload Architecture | Processed Pixels | Latency (ms) | Throughput (MP/s) | Mean LST (K) | Speedup vs CPU |
|:---|:---:|:---:|:---:|:---:|:---:|
| **CPU Single-Core (Dual-Band Planck + LST)** | 2,748,620 | 18.77 ms | 146.43 MP/s | 303.31 K | 1.00x |
| **16-Chip TPU v6e Dual-Band (B31/B32 Planck + LST)** | 2,748,620 | **2.47 ms** | **1111.61 MP/s** | 303.31 K | **7.6x** |
| **16-Chip TPU v6e Full Granule 16-Band Thermal Inversion** | 43,977,920 | **4.69 ms** | **9367.57 MP/s** (9.368 GP/s) | 195.47 K | **64.0x** |

---

## 🔍 Part 3: Physical Parity Matrix Across All 16 Cloud TPU v6e Chips

| TPU Physical Chip | Topology Location | BPA Coherence (γ) | BPA L2 Error | FFBP Coherence (γ) | FFBP L2 Error | MODIS Mean Bias (K) | MODIS Max Error (K) | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Chip 00** | Node-0 (TPU-0) | 0.9964 | 8.90e-02 | 0.9968 | 8.38e-02 | +0.0000 K | 0.0004 K | PASS (PARITY) |
| **Chip 01** | Node-0 (TPU-1) | 0.9962 | 8.94e-02 | 0.9955 | 9.63e-02 | +0.0000 K | 0.0005 K | PASS (PARITY) |
| **Chip 02** | Node-0 (TPU-2) | 0.9963 | 8.76e-02 | 0.9966 | 8.29e-02 | +0.0000 K | 0.0005 K | PASS (PARITY) |
| **Chip 03** | Node-0 (TPU-3) | 0.9967 | 8.16e-02 | 0.9963 | 8.64e-02 | +0.0000 K | 0.0004 K | PASS (PARITY) |
| **Chip 04** | Node-1 (TPU-0) | 0.9973 | 7.41e-02 | 0.9974 | 7.21e-02 | +0.0000 K | 0.0004 K | PASS (PARITY) |
| **Chip 05** | Node-1 (TPU-1) | 0.9972 | 7.53e-02 | 0.9970 | 7.80e-02 | +0.0000 K | 0.0004 K | PASS (PARITY) |
| **Chip 06** | Node-1 (TPU-2) | 0.9961 | 8.79e-02 | 0.9964 | 8.54e-02 | +0.0000 K | 0.0004 K | PASS (PARITY) |
| **Chip 07** | Node-1 (TPU-3) | 0.9958 | 9.17e-02 | 0.9955 | 9.48e-02 | +0.0000 K | 0.0004 K | PASS (PARITY) |
| **Chip 08** | Node-2 (TPU-0) | 0.9974 | 7.29e-02 | 0.9972 | 7.49e-02 | +0.0000 K | 0.0004 K | PASS (PARITY) |
| **Chip 09** | Node-2 (TPU-1) | 0.9971 | 7.62e-02 | 0.9973 | 7.43e-02 | +0.0000 K | 0.0004 K | PASS (PARITY) |
| **Chip 10** | Node-2 (TPU-2) | 0.9957 | 9.26e-02 | 0.9955 | 9.47e-02 | +0.0000 K | 0.0004 K | PASS (PARITY) |
| **Chip 11** | Node-2 (TPU-3) | 0.9955 | 9.62e-02 | 0.9957 | 9.41e-02 | +0.0000 K | 0.0004 K | PASS (PARITY) |
| **Chip 12** | Node-3 (TPU-0) | 0.9957 | 9.45e-02 | 0.9964 | 8.60e-02 | +0.0000 K | 0.0004 K | PASS (PARITY) |
| **Chip 13** | Node-3 (TPU-1) | 0.9957 | 9.42e-02 | 0.9944 | 1.07e-01 | +0.0000 K | 0.0004 K | PASS (PARITY) |
| **Chip 14** | Node-3 (TPU-2) | 0.9968 | 8.26e-02 | 0.9972 | 7.78e-02 | +0.0000 K | 0.0004 K | PASS (PARITY) |
| **Chip 15** | Node-3 (TPU-3) | 0.9972 | 7.68e-02 | 0.9968 | 8.15e-02 | +0.0000 K | 0.0004 K | PASS (PARITY) |

---

## 🖼️ Part 4: Diagnostic Visual & Difference Raster Products

| Product Name | Modality | Output Artifact Path | Description |
|:---|:---:|:---|:---|
| **SAR 16-Chip BPA Image** | Radar (SAR) | `out/v6e_16chip_benchmark_bpa_16chip.png` | Fully focused SAR imagery via 16-chip distributed Pallas BPA |
| **SAR 16-Chip FFBP Image** | Radar (SAR) | `out/v6e_16chip_benchmark_ffbp_16chip.png` | Fast factorized SAR imagery formed across 16-chip sub-apertures |
| **SAR BPA Residual Difference** | Radar (SAR) | `out/v6e_16chip_benchmark_bpa_diff_raster.png` | Pixel-wise magnitude error $\|I_{16\_bpa} - I_{ref}\|$ |
| **SAR BPA Phase Error Raster** | Radar (SAR) | `out/v6e_16chip_benchmark_bpa_phase_err.png` | Full-scene complex phase deviation $\Delta \phi = \angle(I_{16} \cdot I_{ref}^*)$ |
| **SAR FFBP Residual Difference** | Radar (SAR) | `out/v6e_16chip_benchmark_ffbp_diff_raster.png` | Sub-aperture recombination residual magnitude map |
| **SAR FFBP Phase Error Raster** | Radar (SAR) | `out/v6e_16chip_benchmark_ffbp_phase_err.png` | Sub-aperture phase error distribution |
| **MODIS Thermal Error Map** | Infrared (MODIS) | `out/v6e_16chip_benchmark_modis_lst_diff_raster.png` | Per-pixel absolute error $\|LST_{TPU} - LST_{CPU}\|$ across 2.75M pixels |

---

## 📊 Part 5: SAR SNR Quintile Coherence Distribution across 16 TPU Chips

| SNR Quintile | Target / Terrain Type | Multi-Chip Pallas BPA (16-Chip) | Multi-Chip Pallas FFBP (16-Chip) | Single-Chip Pallas BPA |
|:---|:---|:---:|:---:|:---:|
| **Q1 (0% – 20%)** | Radar Shadow / Water | 0.9942 | 0.9942 | 0.9943 |
| **Q2 (20% – 40%)** | Flat Soil / Pavement | 0.9960 | 0.9960 | 0.9960 |
| **Q3 (40% – 60%)** | Medium Farmland / Rough Soil | 0.9964 | 0.9964 | 0.9964 |
| **Q4 (60% – 80%)** | Dense Canopy / Foliage | 0.9972 | 0.9972 | 0.9972 |
| **Q5 (80% – 100%)** | Metallic Structures / Urban | **0.9975** | **0.9975** | **0.9975** |
