#!/usr/bin/env bash
# =============================================================================
# Deploy 3TB Catalog Batch Benchmark on 16 Cloud TPU v6e Chips
# =============================================================================
set -euo pipefail

CLUSTER_NAME="mb-tpu-benchmark"
REGION="europe-west4"
PROJECT_ID="mb-development-447000"
JOB_NAME="sar-pallas-v6e-3tb-batch-benchmark"
CONFIGMAP_NAME="sar-pallas-3tb-batch-code"

echo "=== Deploying 3TB Batch Benchmark on 16 Cloud TPU v6e Chips ==="
echo "Project: $PROJECT_ID | Cluster: $CLUSTER_NAME | Region: $REGION"
echo "Target Node Pool: mb-tpu-v6e-multi-16 (4 nodes x ct6e-standard-4t = 16 TPU v6e chips)"

# 1. Refresh GKE Auth Token
TOKEN=$(gcloud auth print-access-token)
kubectl config set-credentials gke-token --token="$TOKEN" >/dev/null
kubectl config set-context --current --user=gke-token >/dev/null

# 2. Cleanup previous batch job and configmap
echo "[*] Cleaning up previous batch job and configmap if existing..."
kubectl delete job "$JOB_NAME" --ignore-not-found=true
kubectl delete configmap "$CONFIGMAP_NAME" --ignore-not-found=true

# Wait for termination
sleep 3

# 3. Create ConfigMap with all required source files and catalog manifest
echo "[*] Packaging source files into ConfigMap $CONFIGMAP_NAME..."
kubectl create configmap "$CONFIGMAP_NAME" \
  --from-file=run_3tb_catalog_batch_benchmark.py=src/sar/run_3tb_catalog_batch_benchmark.py \
  --from-file=pallas_bpa_multi_chip.py=src/sar/pallas_bpa_multi_chip.py \
  --from-file=pallas_ffbp_multi_chip.py=src/sar/pallas_ffbp_multi_chip.py \
  --from-file=pallas_bpa_optimized.py=src/sar/pallas_bpa_optimized.py \
  --from-file=pallas_ffbp_optimized.py=src/sar/pallas_ffbp_optimized.py \
  --from-file=pallas_bpa.py=src/sar/pallas_bpa.py \
  --from-file=pallas_ffbp.py=src/sar/pallas_ffbp.py \
  --from-file=tpu_bpa.py=src/sar/tpu_bpa.py \
  --from-file=tpu_ffbp.py=src/sar/tpu_ffbp.py \
  --from-file=cphd_reader.py=src/sar/cphd_reader.py \
  --from-file=quality_evaluator.py=quality_evaluator.py \
  --from-file=viz.py=src/sar/viz.py \
  --from-file=coherence.py=src/sar/coherence.py \
  --from-file=cphd_3tb_catalog_manifest.json=cphd_3tb_catalog_manifest.json

# 4. Deploy 3TB Batch Service & Job
echo "[*] Applying k8s/tpu_v6e_3tb_batch_job.yaml..."
kubectl apply -f k8s/tpu_v6e_3tb_batch_job.yaml

echo "[*] Waiting for batch benchmark pods to initialize..."
sleep 5
kubectl get pods -l job-name="$JOB_NAME" -o wide

echo "=== 3TB Batch Benchmark Successfully Dispatched on 16 TPUs ==="
