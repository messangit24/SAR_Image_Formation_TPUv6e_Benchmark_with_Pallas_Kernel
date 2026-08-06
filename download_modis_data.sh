#!/bin/bash
# Script to download MODIS Terra L0/L1A/L1B/GEO granule dataset from NASA MODAPS/LANCE

set -e

DATA_DIR="${1:-./data/modis}"
TOKEN_FILE="${EARTHDATA_TOKEN_FILE:-$HOME/.edl_token}"

mkdir -p "$DATA_DIR"

echo "=== MODIS Terra Granule Data Download Helper ==="
echo "Target Directory: $DATA_DIR"

if [ -f "$TOKEN_FILE" ]; then
    TOKEN=$(cat "$TOKEN_FILE" | tr -d '\r\n')
elif [ -n "$EARTHDATA_TOKEN" ]; then
    TOKEN="$EARTHDATA_TOKEN"
else
    echo "Note: NASA MODAPS/LANCE requires a free NASA Earthdata Bearer Token."
    echo "To download automatically, generate an Earthdata token at https://urs.earthdata.nasa.gov/"
    echo "and set EARTHDATA_TOKEN=<your_token> or save it to ~/.edl_token"
    echo ""
    echo "Example curl command with token:"
    echo "  curl -H \"Authorization: Bearer <TOKEN>\" -O \"https://nrt3.modaps.eosdis.nasa.gov/api/v2/content/archives/allData/61/MOD021KM/2026/213/MOD021KM.A2026213.0710.061.NRT.hdf\""
    exit 1
fi

echo "Downloading MODIS Granule A2026213.0710 (Terra, 2026-08-01 07:10 UTC)..."

BASE_URL="https://nrt3.modaps.eosdis.nasa.gov/api/v2/content/archives/allData/61"

# Granule file definitions
FILES=(
    "MOD00F/2026/213/MOD00F.A2026213.0710.061.P.000.NRT"
    "MOD00F/2026/213/MOD00F.A2026213.0710.061.P.001.NRT"
    "MOD01/2026/213/MOD01.A2026213.0710.061.2026213081621.NRT.hdf"
    "MOD021KM/2026/213/MOD021KM.A2026213.0710.061.2026213082055.NRT.hdf"
    "MOD03/2026/213/MOD03.A2026213.0710.061.2026213081852.NRT.hdf"
)

for rel_path in "${FILES[@]}"; do
    filename=$(basename "$rel_path")
    echo "Fetching $filename ..."
    curl -C - -H "Authorization: Bearer $TOKEN" -o "$DATA_DIR/$filename" "$BASE_URL/$rel_path"
done

echo "=== MODIS Download Complete ==="
echo "Files downloaded to: $DATA_DIR"
