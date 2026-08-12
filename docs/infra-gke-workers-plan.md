# Feature Plan: [TO2-INFRA-01] Automated GKE Benchmark Worker Manifests & Scale-to-Zero Orchestration

## 1. Overview & Objectives
This feature bridges the gap between infrastructure provisioned via Terraform (`terraform/cpu_cluster.tf`, `terraform/tpu_cluster.tf`) and execution of automated SAR/MODIS benchmark jobs on GKE.

It provides container packaging (`Dockerfile`), Kubernetes Job specifications, a Helm chart for parameterized dispatch, and a Python CLI dispatcher with GCS FUSE zero-copy storage integration and automatic node pool scale-to-zero lifecycle support.

---

## 2. Architecture & File Structure

```
TPUbench/
├── Dockerfile                               # Multi-stage production container image
├── k8s/
│   ├── base/
│   │   ├── namespace.yaml                  # Dedicated benchmark namespace
│   │   ├── serviceaccount.yaml             # Workload Identity GCS FUSE service account
│   │   ├── sar-cpu-benchmark-job.yaml      # High-compute CPU Job manifest
│   │   ├── sar-tpu-benchmark-job.yaml      # Cloud TPU v5e/v4 Job manifest
│   │   └── modis-batch-job.yaml            # MODIS multi-granule batch Job manifest
│   └── helm/tpubench-worker/
│       ├── Chart.yaml                      # Helm Chart definition
│       ├── values.yaml                     # Configurable defaults
│       └── templates/                      # Job & SA templates
├── src/infra/
│   ├── __init__.py
│   └── job_dispatcher.py                   # Python CLI dispatcher & manifest generator
├── tests/
│   └── test_infra_k8s_manifests.py         # K8s manifest & Helm validation tests
└── docs/bmad/stories/
    └── story-infra-1.1-gke-worker-orchestration.md # BMAD story spec
```

---

## 3. Key Specifications

### 3.1 Node Pool Targeting & Tolerations
- **CPU Cluster Jobs**:
  - `nodeSelector`: `tpubench.io/workload-type: cpu-benchmark` (or instance type `c3-standard-88`).
  - Resources: Request/Limit 64 vCPUs, 256Gi memory.
- **TPU Cluster Jobs**:
  - `nodeSelector`: `cloud.google.com/gke-tpu-accelerator: tpu-v5-lite-podslice` (or `v4-podslice`).
  - Resources: `google.com/tpu: "4"` (or `"8"`).
  - Tolerations: `google.com/tpu: "present", operator: "Exists"`.

### 3.2 Storage Integration (GCS FUSE CSI)
- Annotations: `gke-gcsfuse/volumes: "true"`.
- Zero-copy direct streaming of raw `.cphd` and MODIS `.NRT` granules from Cloud Storage buckets.

---

## 4. Quality Gates
1. `uv run ruff check src tests scripts` -> clean.
2. `uv run pytest -q` -> all tests passing.
3. `terraform fmt -check` and `terraform validate` -> valid.
4. `uv run python .github/pr-reviewer/test_review.py` -> clean.
