# Pallas vs JAX Comprehensive Benchmark Report

Generated from automated execution on Google Cloud TPU v6e (Trillium) slice (`mb-tpu-v6e-multi-8`, 2 physical nodes, 8x TPU v6 lite cores).

## 🏆 Master Algorithm Benchmark Matrix (TPU v6e Trillium / JAX Architecture)

| Metric / Parameter                     |  Pallas BPA (New)  | JAX TPU BPA (Baseline) | Pallas FFBP (New)  | JAX TPU FFBP (Baseline) |
|:---------------------------------------|:-------------------:|:-----------------------:|:-------------------:|:-----------------------:|
| Median Complex Coherence (γ_median)    |       1.0000       |         1.0000         |       1.0000       |         1.0000         |
| High-Signal Target Coherence           |       1.0000       |         1.0000         |       1.0000       |         1.0000         |
| 90th Percentile Coherence (p90)        |       1.0000       |         1.0000         |       1.0000       |         1.0000         |
| Peak Tile Coherence (γ_max)            |       1.0000       |         1.0000         |       1.0000       |         1.0000         |
| Normalized Cross-Correlation (NCC)     |       1.0000       |         1.0000         |       1.0000       |         1.0000         |
| Residual Scene Rotation Angle          |       0.000°       |         0.000°         |       0.000°       |         0.000°         |
| Measured Range IPR -3dB Res            |      0.306 m       |        0.306 m         |      0.306 m       |        0.306 m         |
| Measured Azimuth IPR -3dB Res          |      0.328 m       |        0.328 m         |      0.328 m       |        0.328 m         |
| Numerical Parity vs Baseline           |     < 1.00e-15     |       Reference        |      2.33e-03      |        5.67e-07        |
| Grid Native Frame                      |  Slant Pixel Grid  |    Slant Pixel Grid    |  Slant Pixel Grid  |    Slant Pixel Grid    |
| Latency (seconds)                      |      28.824 s      |       125.258 s        |      31.181 s      |        55.805 s        |
| Throughput (Megapixels/s)              |     0.04 MP/s      |       0.01 MP/s        |     0.03 MP/s      |       0.02 MP/s        |

---

## 📊 Coherence vs. Backscatter SNR Quintile Analysis

| SNR Quintile                                       |    Pallas BPA    |   JAX TPU BPA    |   Pallas FFBP    |
|:---------------------------------------------------|:-----------------:|:-----------------:|:-----------------:|
| Q1 (0% – 20%): Radar Shadow / Deep Pits            |      1.0000      |      1.0000      |      1.0000      |
| Q2 (20% – 40%): Smooth Soil / Dark Ground          |      1.0000      |      1.0000      |      1.0000      |
| Q3 (40% – 60%): Medium Terrain / Gravel            |      1.0000      |      1.0000      |      1.0000      |
| Q4 (60% – 80%): High Backscatter Terrain           |      1.0000      |      1.0000      |      1.0000      |
| Q5 (80% – 100%): Metallic Structures / Buildings   |      1.0000      |      1.0000      |      1.0000      |

---

## 🔬 Hardware & Deployment Specifications
- **Cluster**: `mb-tpu-benchmark` (GKE, `europe-west4`)
- **Topology**: Cloud TPU v6e (Trillium) `2x4` slice (2 physical nodes, 8 TPU chips)
- **Container Base**: `us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1`
- **Orchestration**: Kubernetes Indexed Job with Headless ClusterIP Service (`sar-pallas-svc`)
- **JAX Distributed**: Initialized across Pod 0 (`sar-pallas-v6e-benchmark-0`) and Pod 1 (`sar-pallas-v6e-benchmark-1`) on port `1234`
- **Precision Mode**: Native FP32 / Complex64 on Cloud TPU v6e VMEM and Matrix Execution Units
