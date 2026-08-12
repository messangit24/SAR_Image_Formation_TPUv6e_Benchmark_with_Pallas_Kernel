# Feature Plan: [TO2-MODIS-03] MODIS L1B Calibration Validation vs MOD021KM Reference

## 1. Overview & Objectives
The goal of **TO2-MODIS-03** is to provide a comprehensive, automated, and mathematically rigorous validation engine for MODIS Level-1B (L1B) radiometric calibration against standard reference datasets (`MOD021KM` HDF products).

This module bridges the gap between raw counts (L0/L1A) and Level-1B calibrated radiance across all 16 Thermal Emissive Bands (TEB), quantifying calibration fidelity, detector uniformity, Response-Versus-Scan-Angle (RVS) characteristics, and brightness temperature residuals.

---

## 2. Target Files & Architecture

```
TPUbench/
├── src/modis/
│   ├── l1b_validator.py            # Core validation engine & multi-band metrics evaluator
│   ├── modis_l1b_recon.py          # Radiometric calibration engine (reused & imported)
│   └── __init__.py                 # Export validator symbols
├── tests/
│   ├── test_modis_l1b_validation.py# Comprehensive unit & integration test suite
│   └── test_modis_radiometry.py    # Existing Planck/radiometry unit tests
├── docs/
│   ├── modis-l1b-validation-plan.md# This implementation plan
│   ├── TODO.md                     # Backlog tracking
│   └── ROADMAP.md                  # Milestone status
└── reports/                        # JSON/Markdown validation artifact output directory
```

---

## 3. Mathematical & Algorithmic Specifications

### 3.1 Radiance and Temperature Inversion
Given spectral wavelength $\lambda$ in $\mu\text{m}$, Planck's law converts temperature $T$ (Kelvin) to spectral radiance $L$ ($\text{W}\cdot\text{m}^{-2}\cdot\text{sr}^{-1}\cdot\mu\text{m}^{-1}$):
$$L(\lambda, T) = \frac{C_1}{\lambda^5 \left(\exp\left(\frac{C_2}{\lambda T}\right) - 1\right)}$$
where $C_1 = 1.191042953 \times 10^8 \text{ W}\cdot\mu\text{m}^4\cdot\text{m}^{-2}\cdot\text{sr}^{-1}$ and $C_2 = 1.4387770 \times 10^4 \mu\text{m}\cdot\text{K}$.

The brightness temperature $T_b$ is derived via the inverse Planck function:
$$T_b(\lambda, L) = \frac{C_2}{\lambda \ln\left(\frac{C_1}{L \lambda^5} + 1\right)}$$

### 3.2 Error & Agreement Metrics
For reference radiance $L_{\text{ref}}$ and reconstructed radiance $L_{\text{recon}}$ over valid pixels $N$:
1. **Root Mean Square Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left(L_{\text{recon}, i} - L_{\text{ref}, i}\right)^2}$$
2. **Relative RMSE (% Error)**:
   $$\text{RelRMSE} = 100 \times \frac{\text{RMSE}}{\bar{L}_{\text{ref}}}$$
3. **Mean Bias Error (MBE)**:
   $$\text{MBE} = \frac{1}{N} \sum_{i=1}^N \left(L_{\text{recon}, i} - L_{\text{ref}, i}\right)$$
4. **Coefficient of Determination ($R^2$)**:
   $$R^2 = 1 - \frac{\sum (L_{\text{recon}, i} - L_{\text{ref}, i})^2}{\sum (L_{\text{ref}, i} - \bar{L}_{\text{ref}})^2}$$
5. **Brightness Temperature Residuals ($\Delta T_b$)**:
   $$\Delta T_b = T_b(\lambda, L_{\text{recon}}) - T_b(\lambda, L_{\text{ref}})$$
   Metrics include mean $\Delta T_b$, RMS $\Delta T_b$, and 95th percentile $|\Delta T_b|$.
6. **Onboard Blackbody Consistency**:
   - Implied blackbody radiance $L_{BB}$ mean, standard deviation, and relative spread ($\%$) across the 20 (10 detectors $\times$ 2 mirror sides) groups.
   - Derived blackbody physical temperature $T_{BB}$.

---

## 4. Verification & Quality Gates

1. **Unit Tests**:
   - Synthetic ground-truth generator testing known gains, offsets, and nonlinearities.
   - Exact mathematical invariants ($R^2=1.0, \text{RMSE}=0.0, \text{Bias}=0.0$ on identical data).
   - Robust handling of NaNs, saturated detector counts, out-of-range pixels, zero masks.
   - Multi-band iteration across all 16 Thermal Emissive Bands.
2. **Quality Checks**:
   - `uv run ruff check src tests scripts`
   - `uv run pytest -q`
   - `uv run python .github/pr-reviewer/test_review.py`
3. **Report Generation**:
   - JSON report schema validation for automated downstream ingestion.
