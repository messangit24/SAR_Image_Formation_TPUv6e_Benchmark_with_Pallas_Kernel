#!/bin/bash
# =============================================================================
# Automated Benchmark Deployment for Cloud TPU v6e (Trillium) on GKE
# Cluster: mb-tpu-benchmark (europe-west4)
# Node Pool: mb-tpu-v6e-multi-8
# =============================================================================

set -e

export CLOUDSDK_CONTAINER_USE_APPLICATION_DEFAULT_CREDENTIALS=false

CLUSTER_NAME="mb-tpu-benchmark"
REGION="europe-west4"
PROJECT_ID="mb-development-447000"
NODE_POOL="mb-tpu-v6e-multi-8"
JOB_NAME="sar-pallas-v6e-benchmark"
CONFIGMAP_NAME="sar-pallas-code"

echo "======================================================================"
echo "🛰️  Deploying SAR JAX/Pallas Benchmark to GKE TPU Cluster"
echo "Project:   $PROJECT_ID"
echo "Cluster:   $CLUSTER_NAME ($REGION)"
echo "Node Pool: $NODE_POOL"
echo "======================================================================"

# 1. Verify GKE credentials and cluster connectivity
echo -e "\n[*] Step 1: Checking GKE cluster context..."
CURRENT_CTX=$(kubectl config current-context 2>/dev/null || echo "")

if [[ "$CURRENT_CTX" != *"$CLUSTER_NAME"* ]]; then
  echo "[!] Updating kubeconfig to cluster $CLUSTER_NAME..."
  gcloud container clusters get-credentials "$CLUSTER_NAME" --region "$REGION" --project "$PROJECT_ID"
else
  echo "[✓] Connected to GKE context: $CURRENT_CTX"
fi

# 2. Check TPU nodes in the target node pool
echo -e "\n[*] Step 2: Checking TPU node availability in pool '$NODE_POOL'..."
TPU_NODES=$(kubectl get nodes -l cloud.google.com/gke-nodepool="$NODE_POOL" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")

if [ -z "$TPU_NODES" ]; then
  echo "[!] Warning: No ready nodes detected in node pool '$NODE_POOL'."
  echo "    If the node pool is scaled to zero, GKE autoscaler will provision nodes upon Job submission."
else
  echo "[✓] Found TPU v6e nodes: $TPU_NODES"
fi

export CLOUDSDK_CONTAINER_USE_APPLICATION_DEFAULT_CREDENTIALS=false

# 3. Create / Update the ConfigMap with the benchmark code
echo -e "\n[*] Step 3: Packaging Pallas and SAR code into Kubernetes ConfigMap '$CONFIGMAP_NAME'..."
kubectl delete configmap "$CONFIGMAP_NAME" --ignore-not-found=true

kubectl create configmap "$CONFIGMAP_NAME" \
  --from-file=pallas_bpa.py=src/sar/pallas_bpa.py \
  --from-file=pallas_ffbp.py=src/sar/pallas_ffbp.py \
  --from-file=tpu_bpa.py=src/sar/tpu_bpa.py \
  --from-file=tpu_ffbp.py=src/sar/tpu_ffbp.py \
  --from-file=tpu_pfa.py=src/sar/tpu_pfa.py \
  --from-file=quality_evaluator.py=quality_evaluator.py \
  --from-file=viz.py=src/sar/viz.py \
  --from-file=coherence.py=src/sar/coherence.py \
  --from-file=run_pallas_benchmark.py=src/sar/run_pallas_benchmark.py

echo "[✓] ConfigMap '$CONFIGMAP_NAME' updated."

# 4. Clean up any existing job
echo -e "\n[*] Step 4: Submitting Kubernetes Benchmark Job..."
kubectl delete job "$JOB_NAME" --ignore-not-found=true
kubectl delete service sar-pallas-svc --ignore-not-found=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
kubectl apply -f "$SCRIPT_DIR/tpu_v6e_job.yaml"

echo "[✓] Job submitted. Waiting for pod to initialize..."

# 5. Stream logs
POD_NAME=""
for i in {1..30}; do
  POD_NAME=$(kubectl get pods -l app=sar-pallas-benchmark --sort-by=.metadata.name -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  if [ -n "$POD_NAME" ]; then
    break
  fi
  sleep 2
done

if [ -n "$POD_NAME" ]; then
  echo "[*] Waiting for pod $POD_NAME to start running..."
  for i in {1..60}; do
    PHASE=$(kubectl get pod "$POD_NAME" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
    if [[ "$PHASE" == "Running" || "$PHASE" == "Succeeded" || "$PHASE" == "Failed" ]]; then
      break
    fi
    sleep 2
  done
  echo "[*] Streaming logs from TPU benchmark pod ($POD_NAME)..."
  kubectl logs -f "$POD_NAME" -c tpu-benchmark-runner || true
else
  echo "[!] Timed out waiting for pod to start. Check 'kubectl get pods' for status."
fi

echo -e "\n======================================================================"
echo "🎯 Deployment Script Finished"
echo "To inspect pod status:   kubectl describe pod -l app=sar-pallas-benchmark"
echo "To check job status:     kubectl get job $JOB_NAME"
echo "======================================================================"
