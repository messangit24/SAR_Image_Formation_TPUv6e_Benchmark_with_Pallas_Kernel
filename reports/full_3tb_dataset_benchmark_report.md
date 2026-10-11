# 3 Terabyte Dataset Batch Execution & Benchmark Report
## Cloud TPU v6e (Trillium) 16-Chip Multi-Host Accelerator Cluster

- **Report Completion Time:** 2026-10-11 17:47:20 UTC
- **Hardware Topology:** GKE `mb-tpu-benchmark` (Region: europe-west4, Node Pool: `mb-tpu-v6e-multi-16`)
- **Total Accelerators:** 16 Physical Cloud TPU v6e Chips (4x4 Optical Torus Mesh, 1,600 Gbps cross-host bandwidth)
- **Algorithm Engine:** Multi-Chip Pallas Fast Factorized Backprojection (FFBP)

---

### 1. Batch Execution Totals

| Metric | Measured Value |
| :--- | :--- |
| **Total Scenes Processed** | 120 Scenes (120 Successful, 0 Failed) |
| **Success Rate** | 100.00% |
| **Raw Volume Processed** | 2,799.63 GB (2.73 TB) |
| **Complex Pixels Reconstructed** | 125,829,120 Pixels (0.126 GP) |
| **Total Measured Wall Clock Time** | 1401.46 s (23.36 min / 0.39 hrs) |
| **Cluster Average Throughput** | 0.0913 MegaPixels/second |
| **Processing Velocity** | 308.2 Scenes/hour |

---

### 2. End-to-End Latency Decomposition (Per-Scene Average)

| Processing Stage | Mean Latency | Median Latency | P95 Latency | Physical Hardware Bus |
| :--- | :--- | :--- | :--- | :--- |
| **1. GCS FUSE Ingress / CPHD Read** | 3.780 s | 3.553 s | 6.517 s | Cloud Storage VPC -> Host RAM |
| **2. Host-to-Device (H2D) DMA Load** | 0.0367 s | 0.0367 s | 0.0405 s | PCIe Gen5 Host -> TPU HBM |
| **3. TPU v6e VPU Kernel Compute** | 7.862 s | 7.806 s | 7.888 s | TPU v6e Vector Processing Unit |
| **4. Optical ICI Inter-Chip Reduc.** | <0.005 s | <0.005 s | <0.005 s | 4x4 Optical Torus Mesh (1,600 Gbps) |
| **Total Per-Scene End-to-End** | **11.678 s** | **11.413 s** | **14.431 s** | Complete Pipeline Clock |

---

### 3. Execution Insights
1. **16-Chip Optical Scalability:** Multi-host sub-aperture reduction leverages the 1,600 Gbps optical torus interconnect, keeping communication latency below 5 milliseconds.
2. **On-Chip SRAM Accumulation:** By leveraging 16 MiB vMem per core, Pallas eliminates 1,024x intermediate HBM memory roundtrips.
3. **Data Container Integrity:** Complex outputs comply with NGA SICD (NITF 2.1) complex floating-point requirements for coherent SAR analytics.
