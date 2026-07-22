#!/bin/bash
# Script to download official open-source Umbra-05 CPHD and SICD dataset from AWS S3 (No AWS Login Required)

set -e

DATA_DIR="${1:-./tpu_sar_data}"
mkdir -p "$DATA_DIR"

echo "=== Downloading Umbra Open Data Catalog Sample Dataset ==="
echo "Target Directory: $DATA_DIR"

# Base AWS S3 Direct HTTP URL for Texas Disaster Relief Spotlight Scene (2023-07-28)
BASE_URL="https://umbra-open-data-catalog.s3.amazonaws.com/sar-data/tasks/ship_detection_testdata/3fcc50e4-413f-4782-b75a-cf3a0b662c12/2023-01-18-07-20-11_UMBRA-05"

CPHD_FILE="2023-01-18-07-20-11_UMBRA-05_CPHD.cphd"
SICD_FILE="2023-01-18-07-20-11_UMBRA-05_SICD.nitf"

echo "1. Downloading CPHD Phase History File ($CPHD_FILE)..."
curl -C - -o "$DATA_DIR/$CPHD_FILE" "$BASE_URL/$CPHD_FILE"

echo "2. Downloading SICD Reference Image File ($SICD_FILE)..."
curl -C - -o "$DATA_DIR/$SICD_FILE" "$BASE_URL/$SICD_FILE"

echo "=== Download Complete ==="
echo "CPHD: $DATA_DIR/$CPHD_FILE"
echo "SICD: $DATA_DIR/$SICD_FILE"
