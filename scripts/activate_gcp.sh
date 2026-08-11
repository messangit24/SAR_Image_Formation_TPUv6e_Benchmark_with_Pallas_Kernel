#!/usr/bin/env bash
# Quick-switch script to activate GCP credentials & configuration for TPUbench (AGO TO2)

CONFIG_NAME="goldthread-to2"
PROJECT_ID="${1:-to2-goldthread}"
REGION="us-central1"
ZONE="us-central1-c"

echo "=== Activating GCP Configuration: ${CONFIG_NAME} (${PROJECT_ID}) ==="

# 1. Create configuration if it doesn't exist
if ! gcloud config configurations describe "${CONFIG_NAME}" >/dev/null 2>&1; then
    echo "Creating new gcloud configuration: ${CONFIG_NAME}..."
    gcloud config configurations create "${CONFIG_NAME}"
fi

# 2. Activate the configuration
gcloud config configurations activate "${CONFIG_NAME}"

# 3. Set project, compute region, and compute zone
gcloud config set project "${PROJECT_ID}"
gcloud config set compute/region "${REGION}"
gcloud config set compute/zone "${ZONE}"

# 4. Set environment variables for the current shell session
export CLOUDSDK_ACTIVE_CONFIG_NAME="${CONFIG_NAME}"
export CLOUDSDK_CORE_PROJECT="${PROJECT_ID}"
export GOOGLE_CLOUD_PROJECT="${PROJECT_ID}"
export TF_VAR_project_id="${PROJECT_ID}"

echo "✅ Active Configuration: $(gcloud config configurations list | grep True | awk '{print $1}')"
echo "✅ Active Project:       ${GOOGLE_CLOUD_PROJECT}"
echo "✅ Active Region/Zone:   ${REGION} / ${ZONE}"
echo "--------------------------------------------------------"
echo "Tip: Run 'gcloud auth login' or 'gcloud auth application-default login' if you need to refresh tokens."
