# Active Backlog & Action Items — GoldThread / TPUbench (AGO TO2)

Current active development tasks and planned stories for **AGO-TO2**.

---

## 🚀 Active & Up Next

- [ ] **[TO2-SAR-01] Benchmark JAX FFBP on Cloud TPU v5e Slices**:
  - Run `src/sar/tpu_ffbp.py` across 1x1, 2x2, and 4x4 TPU topologies.
  - Measure kernel memory footprint and Roofline arithmetic intensity.
  - Target: Sub-0.5s full-scene reconstruction.

- [ ] **[TO2-MODIS-01] Finish L1B Radiometric Calibration Suite**:
  - Complete two-point calibration in `src/modis/modis_l1b_recon.py`.
  - Validate radiance calculations across Band 31 and Band 32 against NASA reference `MOD021KM`.

- [ ] **[TO2-INFRA-01] Automated GKE Cluster Turnup & Worker Helm Chart**:
  - Create Kubernetes deployment / Job manifests for SAR and MODIS benchmark workers.
  - Automate node pool scale-up on Job dispatch and scale-down to zero upon completion.

---

## ✅ Completed Tasks

- [x] **[TO2] Create Terraform resources** ([`personal/goldthread#2`](https://depot.code.corp.goog/personal/goldthread/issues/2), [`b/545136760`](https://b.corp.google.com/issues/545136760))
- [x] **[TO2] Support ability to manually scale clusters up/down** ([`personal/goldthread#3`](https://depot.code.corp.goog/personal/goldthread/issues/3), [`b/545136440`](https://b.corp.google.com/issues/545136440))
- [x] **[TO2] SAR BPA & FFBP JAX Acceleration** ([`personal/goldthread#5`](https://depot.code.corp.goog/personal/goldthread/issues/5), [`b/545133622`](https://b.corp.google.com/issues/545133622))
