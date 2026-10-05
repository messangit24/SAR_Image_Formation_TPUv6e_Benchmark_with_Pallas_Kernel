# Pallas vs JAX Comprehensive Benchmark Report

Generated from automated execution on real Umbra SAR dataset.

## 🏆 Master Algorithm Benchmark Matrix

| Metric / Parameter                     |  Pallas BPA (New)  | JAX TPU BPA (Baseline) | Pallas FFBP (New)  | JAX TPU FFBP (Baseline) |
|:---------------------------------------|:-------------------:|:-----------------------:|:-------------------:|:-----------------------:|
| Median Complex Coherence (γ_median)    |       1.0000       |         1.0000         |       1.0000       |         1.0000         |
| High-Signal Target Coherence           |       1.0000       |         1.0000         |       1.0000       |         1.0000         |
| 90th Percentile Coherence (p90)        |       1.0000       |         1.0000         |       1.0000       |         1.0000         |
| Peak Tile Coherence (γ_max)            |       1.0000       |         1.0000         |       1.0000       |         1.0000         |
| Normalized Cross-Correlation (NCC)     |       1.0000       |         1.0000         |       1.0000       |         1.0000         |
| Residual Scene Rotation Angle          |       0.000°       |         0.000°         |       0.000°       |         0.000°         |
| Measured Range IPR -3dB Res            |      28.603 m      |        28.603 m        |      28.603 m      |        28.603 m        |
| Measured Azimuth IPR -3dB Res          |      3.970 m       |        3.970 m         |      3.970 m       |        3.970 m         |
| Numerical Parity vs Baseline           |     < 1.00e-15     |       Reference        |      7.54e-16      |        7.54e-16        |
| Grid Native Frame                      |  Slant Pixel Grid  |    Slant Pixel Grid    |  Slant Pixel Grid  |    Slant Pixel Grid    |
| Latency (seconds)                      |      44.380 s      |        7.471 s         |      15.869 s      |        3.630 s         |
| Throughput (Megapixels/s)              |     0.01 MP/s      |       0.04 MP/s        |     0.02 MP/s      |       0.07 MP/s        |

## 📊 Coherence vs. Backscatter SNR Quintile Analysis

| SNR Quintile                                       |    Pallas BPA    |   JAX TPU BPA    |   Pallas FFBP    |
|:---------------------------------------------------|:-----------------:|:-----------------:|:-----------------:|
| Q1 (0% – 20%): Radar Shadow / Deep Pits            |      1.0000      |      1.0000      |      1.0000      |
| Q2 (20% – 40%): Smooth Soil / Dark Ground          |      1.0000      |      1.0000      |      1.0000      |
| Q3 (40% – 60%): Medium Terrain / Gravel            |      1.0000      |      1.0000      |      1.0000      |
| Q4 (60% – 80%): High Backscatter Terrain           |      1.0000      |      1.0000      |      1.0000      |
| Q5 (80% – 100%): Metallic Structures / Buildings   |      1.0000      |      1.0000      |      1.0000      |
