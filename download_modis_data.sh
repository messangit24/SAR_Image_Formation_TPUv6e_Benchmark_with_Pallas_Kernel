#!/bin/bash
# Script to download MODIS Terra L0/L1A/L1B/GEO granule dataset from GCS or NASA MODAPS/LANCE

set -e

DATA_DIR="${1:-./data/modis}"
GCS_BUCKET="gs://umbra-sar-dataset-75tb-mb-development-447000/modis-data"
TOKEN_FILE="${EARTHDATA_TOKEN_FILE:-$HOME/.edl_token}"

mkdir -p "$DATA_DIR"

echo "=== MODIS Terra Granule Data Download Helper ==="
echo "Target Directory: $DATA_DIR"

# Clean mTLS / cert configs that can slow down gcloud
unset GOOGLE_API_CERTIFICATE_CONFIG CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE

# 1. First priority: High-throughput Google Cloud Storage mirror
if command -v gcloud &>/dev/null && [ "${FORCE_NASA_DOWNLOAD:-0}" != "1" ]; then
    echo "Fetching MODIS Granule A2026213.0710 directly from GCS bucket ($GCS_BUCKET)..."
    gcloud storage cp "${GCS_BUCKET}/*" "$DATA_DIR/"
    echo "=== MODIS Download Complete from GCS ==="
    echo "Files downloaded to: $DATA_DIR"
    ls -lh "$DATA_DIR"
    exit 0
elif command -v gsutil &>/dev/null && [ "${FORCE_NASA_DOWNLOAD:-0}" != "1" ]; then
    echo "Fetching MODIS Granule A2026213.0710 directly from GCS via gsutil ($GCS_BUCKET)..."
    gsutil -m cp "${GCS_BUCKET}/*" "$DATA_DIR/"
    echo "=== MODIS Download Complete from GCS ==="
    echo "Files downloaded to: $DATA_DIR"
    ls -lh "$DATA_DIR"
    exit 0
fi

# 2. Fallback: Direct NASA Earthdata / LANCE download
echo "GCS CLI not available or fallback requested; checking NASA Earthdata credentials..."
if [ -f "$TOKEN_FILE" ]; then
    TOKEN=$(cat "$TOKEN_FILE" | tr -d '\r\n')
elif [ -n "$EARTHDATA_TOKEN" ]; then
    TOKEN="$EARTHDATA_TOKEN"
else
    echo "Note: NASA MODAPS/LANCE requires a free NASA Earthdata Bearer Token."
    echo "To download automatically, generate an Earthdata token at https://urs.earthdata.nasa.gov/"
    echo "and set EARTHDATA_TOKEN=<your_token> or save it to ~/.edl_token"
    exit 1
fi

echo "Downloading MODIS Granule A2026213.0710 from NASA Earthdata..."

BASE_URL="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61"

FILES=(
    "MOD00F/2026/213/MOD00F.A2026213.0710.20262130706.000.PDS.xz"
    "MOD00F/2026/213/MOD00F.A2026213.0710.20262130706.001.PDS.xz"
    "MOD021KM/2026/213/MOD021KM.A2026213.0710.061.2026216005446.hdf"
    "MOD03/2026/213/MOD03.A2026213.0710.061.2026216003448.hdf"
)

for rel_path in "${FILES[@]}"; do
    filename=$(basename "$rel_path")
    echo "Fetching $filename ..."
    curl -C - -H "Authorization: Bearer $TOKEN" -o "$DATA_DIR/$filename" "$BASE_URL/$rel_path"
done

echo "=== MODIS Download Complete ==="
echo "Files downloaded to: $DATA_DIR"
