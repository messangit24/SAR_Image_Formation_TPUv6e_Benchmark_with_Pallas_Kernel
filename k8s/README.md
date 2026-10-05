# SAR JAX / Pallas Benchmark on GKE Cloud TPU v6e (Trillium)

This directory contains the Kubernetes deployment manifests and automation scripts to benchmark the **Pallas & VMEM-tiled SAR image formation kernels** on Google Kubernetes Engine (GKE) with Cloud TPU v6e hardware.

---

## 🎯 Target Hardware & Cluster Spec

* **GCP Project**: `mb-development-447000`
* **GKE Cluster**: `mb-tpu-benchmark`
* **Region**: `europe-west4`
* **Node Pool**: `mb-tpu-v6e-multi-8` (Cloud TPU v6e / Trillium slice, 4 chips per host)
* **Resource Type**: `google.com/tpu: "4"`

---

## 🚀 Quickstart: Running the Benchmark

### 1. Ensure GKE Credentials are Active
If your `gcloud` session has expired, re-authenticate and fetch credentials:
```bash
gcloud auth login
gcloud container clusters get-credentials mb-tpu-benchmark --region europe-west4 --project mb-development-447000
```

### 2. Run the Benchmark (Automated)
Run the one-click deployment script:
```bash
./k8s/deploy_benchmark.sh
```

This script will:
1. Verify cluster and node pool connectivity.
2. Package the latest SAR & Pallas code into a Kubernetes `ConfigMap` (`sar-pallas-code`).
3. Deploy the Kubernetes `Job` ([`k8s/tpu_v6e_job.yaml`](tpu_v6e_job.yaml)) configured for `mb-tpu-v6e-multi-8`.
4. Stream the real-time execution logs and output benchmark comparison tables.

---

## 📁 Key Files

* [`tpu_v6e_job.yaml`](tpu_v6e_job.yaml): Kubernetes Job specification targeting the TPU v6e node pool with `google.com/tpu: "4"`, `/dev/shm` IPC volume, and double-precision environment variables (`JAX_ENABLE_X64=True`).
* [`deploy_benchmark.sh`](deploy_benchmark.sh): Orchestration script for deploying, streaming logs, and monitoring.
* [`../src/sar/pallas_bpa.py`](../src/sar/pallas_bpa.py): Custom 2D spatial-tiled Pallas Backprojection engine.
* [`../src/sar/pallas_ffbp.py`](../src/sar/pallas_ffbp.py): Custom streaming tree-fused Fast Factorized Backprojection engine.
* [`../src/sar/run_pallas_benchmark.py`](../src/sar/run_pallas_benchmark.py): Automated benchmark comparison runner.

---

## 🔍 Monitoring & Troubleshooting

### Check Pod Status
```bash
kubectl get pods -l app=sar-pallas-benchmark
kubectl describe pod -l app=sar-pallas-benchmark
```

### View Live Logs
```bash
kubectl logs -f -l app=sar-pallas-benchmark -c tpu-benchmark-runner
```

### Delete the Job
```bash
kubectl delete job sar-pallas-v6e-benchmark
```
