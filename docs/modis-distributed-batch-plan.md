# Feature Plan: [TO2-MODIS-04] Distributed Multi-Granule Ingestion Pipeline

## 1. Overview & Objectives
This feature scales the MODIS thermal anomaly detection and L1B processing workflows across multi-core CPU worker pools and distributed compute clusters. It provides batch ingestion, parallel granule decommutation/calibration, fault-tolerant execution, orbit-track cluster stitching across consecutive granules, and unified catalog export.

---

## 2. Architecture & File Structure

```
TPUbench/
├── src/modis/
│   ├── distributed_batch.py         # Multi-process & distributed batch ingestion engine
│   ├── multi_granule_pipeline.py    # Multi-granule catalog structures
│   ├── mwir_detector.py             # Core MWIR anomaly detector & Dozier solver
│   └── __init__.py                  # Package exports
├── tests/
│   └── test_modis_distributed.py    # Unit & integration tests for distributed pipeline
├── docs/bmad/stories/
│   └── story-modis-1.4-distributed-batch-pipeline.md # BMad story specification
└── reports/
    └── modis_batch_catalog.geojson  # Aggregated output catalog
```

---

## 3. Key Specifications

### 3.1 Parallel Execution & Worker Pools
- `ProcessPoolExecutor` with dynamic worker allocation (`max_workers=min(os.cpu_count(), 32)`).
- Chunked job dispatch with per-granule timeout and exception isolation.

### 3.2 Orbit-Track Cluster Stitching
- Detects contiguous thermal anomaly clusters across neighboring granules along the satellite ground track when timestamps differ by $\le 5$ minutes and spatial distance $< 2\text{ km}$.
- Recomputes unified cluster bounding boxes, total FRP, and peak temperatures.

---

## 4. Quality Gates
1. `uv run ruff check src tests scripts` -> clean.
2. `uv run pytest -q` -> all tests passing.
3. `uv run python .github/pr-reviewer/test_review.py` -> clean.
