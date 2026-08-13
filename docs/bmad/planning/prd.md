# GoldThread / TPUbench Product Requirements Document (PRD)

## 1. Executive Summary & Mission
GoldThread (AGO TO2) delivers high-throughput Synthetic Aperture Radar (SAR) image formation and automated Earth Observation (MODIS/VIIRS) thermal anomaly processing on Google Cloud TPU infrastructure.

## 2. Functional Requirements (FR)
- **FR1 (SAR Backprojection Engine)**: Time-domain Backprojection Algorithm (BPA) and Fast Factorized Backprojection (FFBP) on JAX Cloud TPU with $>97\%$ complex coherence against Umbra SICD products.
- **FR2 (MODIS Telemetry Ingestion)**: L0 CCSDS packet decommutation and L1B radiometric calibration for thermal emissive and reflective solar bands.
- **FR3 (Thermal Anomaly & MWIR Detection)**: Rapid detection of active wildfires, thermal plumes, and high-temperature events.
- **FR4 (GKE Distributed Worker Fleet)**: Cloud TPU v4/v5e orchestration with autoscaling and GCS staging.

## 3. Non-Functional Requirements (NFR)
- **NFR1 (Inference Latency)**: Sub-second ($<0.7\text{s}$) full scene SAR focusing.
- **NFR2 (Coherence Parity)**: Complex correlation coefficient $|\gamma| \ge 0.97$ across all range/azimuth bins.
- **NFR3 (Hermetic Verification)**: Zero external network dependencies for CI unit and calibration test suites.
