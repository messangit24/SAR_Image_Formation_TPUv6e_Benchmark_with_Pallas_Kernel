# Story MODIS.1.1: MODIS L0 Decommutation & L1B Radiometric Calibration Test Suite

Status: done
GitHub-Issue: #7 (https://depot.code.corp.goog/personal/goldthread/issues/7)

## Story Description
Establish a comprehensive, hermetic unit and integration test suite for the MODIS satellite telemetry ingestion (`src/modis/modis_l0_decom.py`) and radiometric calibration reconstruction (`src/modis/modis_l1b_recon.py`) modules, ensuring algorithmic parity, Planck function numerical stability, and bit-level unpack verification without requiring external NASA network downloads.

## Acceptance Criteria
- [x] **AC1 — 12-Bit Sample Decommutation**: Implement synthetic packet generation to verify big-endian 12-bit word unpack (`_words`) bit-exactness.
- [x] **AC2 — Telemetry Header Decoding**: Verify header bitfield parsing (`seq`, `is_cal`, `efc`, `cal_type`, `cal_fcnt`) across edge-case packets.
- [x] **AC3 — Planck & Radiometric Inversion**: Verify forward and inverse Planck spectral radiance functions across all MODIS thermal emissive bands (Bands 20-36) with relative tolerance $< 10^{-6}$.
- [x] **AC4 — SAR Coherence Mathematical Invariants**: Verify complex coherence function $|\gamma| \in [0, 1]$, identity self-coherence $\gamma(X, X) = 1.0$, and spatial shift invariance.
- [x] **AC5 — Test Suite Execution**: `pytest -q` runs cleanly and reports all 26 tests passing.
- [x] **AC6 — Linting Gate**: `ruff check src tests scripts` passes cleanly with zero errors.

## Verification & Test Results
- `pytest -q`: **26/26 tests passed**.
- `test_review.py`: **94/94 unit tests passed**.
- `ruff check src tests scripts`: **All checks passed**.
