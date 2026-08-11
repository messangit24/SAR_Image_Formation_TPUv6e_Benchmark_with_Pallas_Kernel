# Story MODIS.1.2: MWIR Thermal Anomaly Detection & Multi-Granule Ingestion Pipeline

Status: done
Component: 2011243
Hotlist: 8806410

## Story Description
Implement a modular, high-performance mid-wave infrared (MWIR) thermal anomaly detection engine (`src/modis/mwir_detector.py`) and multi-granule ingestion pipeline (`src/modis/multi_granule_pipeline.py`). The pipeline identifies sub-pixel high-temperature thermal sources (e.g. industrial flares, rocket plumes, wildland fires) using MODIS 3.9 µm (Bands 21/22) and 11 µm (Band 31) radiometry, discriminates real emitters from solar glint using NIR Band 2 (858 nm) reflectance, solves the non-linear Dozier dual-channel sub-pixel temperature/area equations, and aggregates multi-granule detections into structured GeoJSON/JSON catalogs.

## Acceptance Criteria
- [x] **AC1 — Contextual & Absolute Dual-Band Detection**: Implement contextual and absolute thermal anomaly tests combining $T_{21}$, $T_{31}$, and excess $\Delta T_{21-31}$ with adaptive background statistics and cloud/water filtering.
- [x] **AC2 — Solar Glint & False Alarm Discrimination**: Implement daytime solar glint rejection using NIR Band 2 (858 nm) reflectance thresholds to separate reflective surfaces from true thermal emitters.
- [x] **AC3 — High Dynamic Range Handling**: Support dual-gain 3.9 µm processing seamlessly transitioning from high-gain Band 22 ($T < 331\text{ K}$) to low-gain Band 21 (up to $500\text{ K}$) without signal saturation.
- [x] **AC4 — Dozier Subpixel Temperature & Area Inversion**: Implement the dual-channel non-linear Dozier solver $(p, T_t)$ and Fire Radiative Power (FRP) estimation with physical bounds ($p \in (0, 1]$, $T_t > T_b$).
- [x] **AC5 — Multi-Granule Ingestion & Spatial Clustering**: Implement multi-granule ingestion that groups contiguous active pixels into spatial thermal clusters and aggregates events across granule time-series into GeoJSON/JSON formats.
- [x] **AC6 — Comprehensive Test Suite & Quality Gates**: Implement `tests/test_modis_mwir.py` verifying all detection modes, Dozier numerical accuracy, and serialization. `uv run ruff check` and `uv run pytest -q` must pass with 100% green status.

## Verification & Test Results
- `uv run ruff check src tests scripts`: **All checks passed! (0 errors)**
- `uv run pytest -q`: **33 passed in 1.70s (100% green)**
- `uv run python .github/pr-reviewer/test_review.py`: **94 tests passed in 0.020s (OK)**
