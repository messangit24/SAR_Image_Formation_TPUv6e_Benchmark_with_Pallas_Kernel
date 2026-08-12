# Story MODIS.1.4: Distributed Multi-Granule Batch Processing Pipeline

Status: done
GitHub-Issue: (To be assigned upon Issue creation)
GitHub-PR: (To be assigned upon PR creation)
Buganizer-Issue: (To be assigned upon Buganizer task creation)
Component: 2011243
Hotlist: 8806410

## Story Description
Implement a high-throughput, multi-worker distributed batch ingestion engine (`src/modis/distributed_batch.py`) for scaling MODIS Level-0/Level-1B calibration and MWIR thermal anomaly detection across multi-core systems and compute nodes. The pipeline features process-pool worker dispatch, per-task fault isolation, orbit-track cluster stitching across consecutive granule boundaries along satellite tracks, real-time performance telemetry (granules/s, MP/s), and unified GeoJSON/JSON catalog export.

## Acceptance Criteria
- [x] **AC1 — Multi-Process Worker Pool Execution**: Implement parallel task execution using `ProcessPoolExecutor` with dynamic worker allocation and configurable timeout/chunking.
- [x] **AC2 — Fault Isolation & Resilience**: Ensure individual corrupted or invalid granule tasks log error diagnostics without aborting batch execution across remaining granules.
- [x] **AC3 — Cross-Granule Orbit-Track Stitching**: Implement `link_orbit_clusters` to merge contiguous thermal anomaly clusters occurring along adjacent granule boundaries within temporal ($\le 5\text{ min}$) and spatial ($\le 2.5\text{ km}$) thresholds.
- [x] **AC4 — Performance Telemetry & Metrics**: Compute batch throughput (granules/s, Megapixels/s), total active pixels, aggregate Fire Radiative Power (FRP in MW), and cluster counts.
- [x] **AC5 — CLI & Catalog Export**: Implement CLI entrypoint generating structured GeoJSON spatial catalogs (`reports/modis_batch_catalog.geojson`) and JSON execution summaries.
- [x] **AC6 — Comprehensive Verification & Tests**: Implement `tests/test_modis_distributed.py` testing parallel execution, fault tolerance, spatial cluster stitching, empty batches, and CLI runs. All quality gates (`ruff`, `pytest`, `test_review.py`) pass 100% green.

## Verification & Test Results
- `uv run ruff check src tests scripts`: **All checks passed! (0 errors)**
- `uv run pytest -q`: **48 passed in 1.10s (100% green)**
- `uv run python .github/pr-reviewer/test_review.py`: **94 passed in 0.017s (OK)**
- CLI Synthetic Batch Execution: **4 granules processed at >150 MP/s, 0 failures**
