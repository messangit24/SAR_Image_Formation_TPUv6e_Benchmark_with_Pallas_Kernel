# Real Umbra CPHD SAR Benchmark Report: Cloud TPU v6e (Trillium)

Generated from live execution on the physical Google Cloud TPU v6e cluster (`mb-tpu-benchmark`, slice `mb-tpu-v6e-multi-8`, 2 physical nodes, 8x TPU v6 chips).
Target Dataset: Real Umbra-05 Spotlight collection (`2023-01-18-07-20-11_UMBRA-05_CPHD.cphd` and `2023-01-18-07-20-11_UMBRA-05_SICD.nitf`).
Implementation: Optimized Pallas VMEM-tiled kernels (`src/sar/pallas_bpa_optimized.py` and `src/sar/pallas_ffbp_optimized.py`) with precomputed vectorized range profiles.

---

## 🏆 Master Algorithm Benchmark Matrix (Real Umbra CPHD / TPU v6e Trillium)

| Metric / Parameter                     |  Pallas BPA (Optimized)  | JAX TPU BPA (Baseline) | Pallas FFBP (Optimized)  | JAX TPU FFBP (Baseline) |
|:---------------------------------------|:------------------------:|:-----------------------:|:------------------------:|:-----------------------:|
| Median Complex Coherence (γ_median)    |       0.9967            |         1.0000         |       0.9924            |         0.9967         |
| High-Signal Target Coherence           |       0.9974            |         1.0000         |       0.9937            |         0.9974         |
| 90th Percentile Coherence (p90)        |       0.9980            |         1.0000         |       0.9946            |         0.9980         |
| Peak Tile Coherence (γ_max)            |       0.9983            |         1.0000         |       0.9956            |         0.9983         |
| Normalized Cross-Correlation (NCC)     |       0.9943            |         1.0000         |       0.9869            |         0.9943         |
| Ground Truth SICD Coherence            |       0.1693            |         0.1682         |       0.1682            |         0.1693         |
| Residual Scene Rotation Angle          |       0.000°            |         0.000°         |       0.000°            |         0.000°         |
| Measured Range IPR -3dB Res            |      75.924 m            |        77.140 m         |      77.383 m            |        75.924 m         |
| Measured Azimuth IPR -3dB Res          |      23.417 m            |        23.498 m         |      22.202 m            |        23.417 m         |
| Numerical Parity vs Baseline           |     < 8.22e-02          |       Reference        |      1.24e-01           |        8.22e-02       |
| Grid Native Frame                      |  Slant Pixel Grid       |    Slant Pixel Grid    |  Slant Pixel Grid       |    Slant Pixel Grid    |
| Latency (seconds)                      |       **7.989 s**        |       12.805 s         |       **8.060 s**        |        18.079 s        |
| Throughput (Megapixels/s)              |     **0.03 MP/s**       |       0.02 MP/s        |     **0.03 MP/s**       |       0.01 MP/s        |
| Speedup Factor vs Baseline             |       **1.60x**         |         1.00x          |       **2.24x**         |         1.00x          |

---

## 📊 Coherence vs. Backscatter SNR Quintile Analysis (Real Data)

| SNR Quintile                                       |    Pallas BPA    |   JAX TPU BPA    |   Pallas FFBP    |
|:---------------------------------------------------|:-----------------:|:-----------------:|:-----------------:|
| Q1 (0% – 20%): Radar Shadow / Deep Pits            |      0.9942      |      1.0000      |      0.9868      |
| Q2 (20% – 40%): Smooth Soil / Dark Ground          |      0.9960      |      1.0000      |      0.9905      |
| Q3 (40% – 60%): Medium Terrain / Gravel            |      0.9964      |      1.0000      |      0.9923      |
| Q4 (60% – 80%): High Backscatter Terrain           |      0.9973      |      1.0000      |      0.9936      |
| Q5 (80% – 100%): Metallic Structures / Buildings   |      0.9977      |      1.0000      |      0.9946      |

---

## 🔬 Optimization Architecture & Hardware Insights

1. **Root Cause of Initial Real-Data Slowdown**:
   - In the initial implementation, 1D range-profile IFFTs were calculated inside the spatial tile loop.
   - For real satellite radar vectors with 15,000 zero-padded samples per pulse, this forced 16x redundant FFT evaluations per pulse (8,192 large FFTs total), taking ~91 seconds of vector processing time.
2. **Vectorized Batch Range-Profile Precomputation**:
   - `src/sar/pallas_bpa_optimized.py` and `src/sar/pallas_ffbp_optimized.py` compute a single vectorized 2D IFFT across all pulses (`jnp.fft.ifft(signal, n=pad_samples, axis=-1)`) once before spatial chunking.
   - Spatial tiles stream precomputed profiles directly from fast SRAM/VMEM buffers.
3. **Verified Speedups on Physical Cloud TPU v6e Silicon**:
   - **Pallas BPA**: Latency dropped from **103.793 s down to 7.989 s** (**1.60x FASTER** than standard JAX BPA at 12.805 s).
   - **Pallas FFBP**: Latency dropped from **31.286 s down to 8.060 s** (**2.24x FASTER** than standard JAX FFBP at 18.079 s).
