# GoldThread Epic Breakdown & Requirements Mapping

## Epic 1: MODIS Telemetry Decommutation & L1B Radiometric Calibration
Hermetic ingestion of raw CCSDS satellite packets, 12-bit sample decoding, thermal emissive band calibration, and Planck radiometric inversion.

### Stories:
- **Story MODIS.1.1**: MODIS L0 Decommutation & L1B Radiometric Calibration Test Suite (`story-modis-1.1-l1b-calibration-and-tests.md`)
- **Story MODIS.1.2**: MWIR / OPIR Thermal Anomaly Detection Pipeline (`story-modis-1.2-mwir-anomaly-pipeline.md`)
- **Story MODIS.1.3**: L1B Radiometric Calibration Quality Validation (`story-modis-1.3-l1b-calibration-validation.md`)
- **Story MODIS.1.4**: Distributed Batch Processing Pipeline (`story-modis-1.4-distributed-batch-pipeline.md`)

## Epic 2: Cloud TPU SAR Backprojection & Complex Coherence
JAX-accelerated time-domain backprojection (BPA), fast factorized backprojection (FFBP), and polar format algorithms on Cloud TPU v4/v5e.

## Epic 3: GKE Distributed Worker Fleet & Cloud Infrastructure
Terraform provisioning of custom VPC, GKE CPU/TPU clusters, and automated batch worker scaling.

### Stories:
- **Story INFRA.1.1**: GKE Worker Orchestration & GCS Bucket Synchronization (`story-infra-1.1-gke-worker-orchestration.md`)
