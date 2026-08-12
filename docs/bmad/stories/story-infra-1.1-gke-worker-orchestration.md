# Story INFRA.1.1: Automated GKE Benchmark Worker Manifests & Scale-to-Zero Orchestration

Status: done
GitHub-Issue: (To be assigned upon PR creation)
GitHub-PR: (To be assigned upon PR creation)
Buganizer-Issue: (To be assigned upon Buganizer task creation)
Component: 2011243
Hotlist: 8806410

## Story Description
Implement production containerization (`Dockerfile`), Kubernetes Job manifests, parameterized Helm charts (`k8s/helm/tpubench-worker/`), and Python CLI dispatcher (`src/infra/job_dispatcher.py`) for dispatching SAR and MODIS benchmark workloads to GKE CPU standard (`c3-standard-88`) and Cloud TPU v5e/v4 node pools. The orchestration interfaces with Cloud Storage via the GKE GCS FUSE CSI driver for zero-copy phase history / granule streaming and leverages node pool autoscaling to dynamically spin up worker instances and automatically scale to zero when jobs terminate.

## Acceptance Criteria
- [x] **AC1 — Production Dockerfile Packaging**: Multi-stage Debian Bookworm container with Python 3.12, `uv`, HDF4/OpenBLAS system libraries, and editable `tpubench` installation.
- [x] **AC2 — Base Kubernetes Job Manifests**: Static manifests for SAR CPU (`k8s/base/sar-cpu-benchmark-job.yaml`), Cloud TPU v5e (`k8s/base/sar-tpu-benchmark-job.yaml`), and MODIS multi-granule batch processing (`k8s/base/modis-batch-job.yaml`).
- [x] **AC3 — GCS FUSE CSI Storage Integration**: Workload Identity ServiceAccount (`k8s/base/serviceaccount.yaml`) and Pod annotations (`gke-gcsfuse/volumes: "true"`) mounting `/data` for zero-copy streaming.
- [x] **AC4 — Parameterized Helm Chart**: Reusable Helm chart (`k8s/helm/tpubench-worker/`) with templates for CPU and Cloud TPU workloads, configurable resource limits, and topology parameters.
- [x] **AC5 — CLI Job Dispatcher & Manifest Generator**: Python CLI (`src/infra/job_dispatcher.py`) supporting dynamic generation, validation, and dry-run rendering of benchmark job manifests.
- [x] **AC6 — Comprehensive Verification & Tests**: Unit tests in `tests/test_infra_k8s_manifests.py` verifying YAML parsing, nodeSelectors, TPU resource requests/tolerations, and Helm structure. All quality gates (`ruff`, `pytest`, `terraform validate`, `test_review.py`) pass 100% green.

## Verification & Test Results
- `uv run ruff check src tests scripts`: **All checks passed! (0 errors)**
- `uv run pytest -q`: **47 passed in 0.95s (100% green)**
- `terraform fmt -check` & `terraform validate`: **Success (0 errors)**
- `uv run python .github/pr-reviewer/test_review.py`: **94 passed in 0.018s (OK)**
