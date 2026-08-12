# Story MODIS.1.3: MODIS L1B Calibration Validation Suite vs MOD021KM Reference

Status: done
GitHub-Issue: #15 (https://depot.code.corp.goog/personal/goldthread/issues/15)
GitHub-PR: #16 (https://depot.code.corp.goog/personal/goldthread/pull/16)
Buganizer-Issue: 545360026 (https://b.corp.google.com/issues/545360026)
Component: 2011243
Hotlist: 8806410

## Story Description
Implement a mathematically rigorous and automated validation and quality gate suite (`src/modis/l1b_validator.py`) for MODIS Level-1B (L1B) radiometric calibration. The suite compares independently reconstructed radiances (derived from L0/L1A detector counts via onboard blackbody radiometry) against NASA operational `MOD021KM` reference products across all 16 Thermal Emissive Bands (TEB), quantifying calibration fidelity, detector uniformity, Response-Versus-Scan-Angle (RVS) characteristics, and brightness temperature residuals ($\Delta T_b$).

## Acceptance Criteria
- [x] **AC1 — Multi-Band Error & Agreement Metrics**: Compute per-band Root Mean Square Error (RMSE), Normalized Relative RMSE (%), Pearson $R^2$, Mean Bias Error (MBE), and percentile error spreads ($p50, p90, p99$) for reconstructed vs reference radiances.
- [x] **AC2 — Brightness Temperature Residuals ($\Delta T_b$)**: Invert spectral radiances to equivalent brightness temperatures via inverse Planck function and calculate $\Delta T_b$ RMSE, mean bias, and peak deviations in Kelvin.
- [x] **AC3 — Onboard Blackbody Uniformity & Stability**: Quantify implied blackbody radiance ($L_{BB}$) mean, standard deviation, and relative spread (%) across the 20 (10 detectors $\times$ 2 mirror sides) calibration groups, and derive effective physical $T_{BB}$.
- [x] **AC4 — Multi-Mode Calibration Support**: Support independent blackbody-referenced reconstruction, per-detector affine recovery, and empirical Response-versus-Scan-Angle (RVS) correction modes.
- [x] **AC5 — Report & Telemetry Export**: Generate formatted console/Markdown summary tables (`reports/modis_l1b_validation.md`) and structured machine-readable JSON reports (`reports/modis_l1b_validation.json`).
- [x] **AC6 — Hermetic Test Suite & Quality Gates**: Implement `tests/test_modis_l1b_validation.py` with synthetic L1A/L1B test data generators, verifying exact match invariants, noise perturbations, NaN/invalid mask handling, JSON export, and CLI execution. All quality gates (`ruff`, `pytest`, `test_review.py`) must pass 100% green.

## Verification & Test Results
- `uv run ruff check src tests scripts`: **All checks passed! (0 errors)**
- `uv run pytest -q`: **42 passed in 0.83s (100% green)**
- `uv run python .github/pr-reviewer/test_review.py`: **94 passed in 0.017s (OK)**
- `uv run python -m src.modis.l1b_validator --synthetic --strict`: **Mean $R^2 = 0.99741$, Relative RMSE $= 1.31\%$, All Quality Gates Passed (✅ YES)**
