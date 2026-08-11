# Active Backlog & Action Items — GoldThread / TPUbench (AGO TO2)

Current active development tasks and planned stories for **AGO-TO2**.

---

## 🚀 Active & Up Next

- [x] **[TO2-MODIS-02] MWIR Thermal Anomaly Detection & Multi-Granule Pipeline**:
  - Modular MWIR detection engine with contextual & absolute triggers (`src/modis/mwir_detector.py`).
  - Solar glint discrimination filter using NIR Band 2 (858 nm) reflectance.
  - Dozier dual-channel sub-pixel temperature/area non-linear solver and Fire Radiative Power (FRP).
  - 8-connected spatial clustering and multi-granule GeoJSON/JSON catalog export (`src/modis/multi_granule_pipeline.py`).

- [ ] **[TO2-SAR-01] Benchmark JAX FFBP on Cloud TPU v5e Slices**:
  - Run `src/sar/tpu_ffbp.py` across 1x1, 2x2, and 4x4 TPU topologies.
  - Measure kernel memory footprint and Roofline arithmetic intensity.
  - Target: Sub-0.5s full-scene reconstruction.

- [ ] **[TO2-INFRA-01] Automated GKE Cluster Turnup & Worker Helm Chart**:
  - Create Kubernetes deployment / Job manifests for SAR and MODIS benchmark workers.
  - Automate node pool scale-up on Job dispatch and scale-down to zero upon completion.

---

## ✅ Completed Tasks

- [x] **[TO2] Adopt uv for Python package and virtualenv management** ([`personal/goldthread#11`](https://depot.code.corp.goog/personal/goldthread/issues/11), PR [`#12`](https://depot.code.corp.goog/personal/goldthread/pull/12), [`b/545157875`](https://b.corp.google.com/issues/545157875))
- [x] **[TO2] Establish CI/CD Pipeline and PR Quality Gates** ([`personal/goldthread#10`](https://depot.code.corp.goog/personal/goldthread/issues/10), PR [`#9`](https://depot.code.corp.goog/personal/goldthread/pull/9), [`b/545154550`](https://b.corp.google.com/issues/545154550))
- [x] **[TO2] MODIS L0 Decommutation & L1B Radiometric Calibration Test Suite** ([`personal/goldthread#7`](https://depot.code.corp.goog/personal/goldthread/issues/7), PR [`#8`](https://depot.code.corp.goog/personal/goldthread/pull/8), [`b/545151068`](https://b.corp.google.com/issues/545151068))
- [x] **[TO2] Create Terraform resources** ([`personal/goldthread#2`](https://depot.code.corp.goog/personal/goldthread/issues/2), [`b/545136760`](https://b.corp.google.com/issues/545136760))
- [x] **[TO2] Support ability to manually scale clusters up/down** ([`personal/goldthread#3`](https://depot.code.corp.goog/personal/goldthread/issues/3), [`b/545136440`](https://b.corp.google.com/issues/545136440))
- [x] **[TO2] SAR BPA & FFBP JAX Acceleration** ([`personal/goldthread#5`](https://depot.code.corp.goog/personal/goldthread/issues/5), [`b/545133622`](https://b.corp.google.com/issues/545133622))
