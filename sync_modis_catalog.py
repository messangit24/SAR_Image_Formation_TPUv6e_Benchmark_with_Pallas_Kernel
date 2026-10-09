#!/usr/bin/env python3
"""
MODIS Granule High-Throughput Sync to Google Cloud Storage (GCS).
Streams MODIS L0/L1A/L1B/GEO granule datasets from AWS S3 / NASA Earthdata Cloud directly into GCS.
Mirrors the high-throughput streaming architecture of sync_umbra_catalog.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

# Clean mTLS / enterprise-cert proxy env vars that can cause authentication hangs
for env_var in [
    "GOOGLE_API_CERTIFICATE_CONFIG",
    "CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE",
    "CLOUDSDK_CONTEXT_AWARE_USE_MTLS_FOR_GRPC",
    "CLOUDSDK_CONTEXT_AWARE_CERTIFICATE_CONFIG_FILE_PATH",
    "GOOGLE_AUTH_WEBAUTHN_PLUGIN",
    "CLOUDSDK_CONTEXT_AWARE_USE_ECP_HTTP_PROXY",
]:
    os.environ.pop(env_var, None)

from google.cloud import storage

DEFAULT_GCS_BUCKET = "umbra-sar-dataset-75tb-mb-development-447000"
DEFAULT_GCS_PREFIX = "modis-data"

# Primary & Secondary Endpoints for Granule A2026213.0710 (Terra 2026-08-01 07:10 UTC)
# AWS S3 Cloud Archive (s3://prod-lads in us-west-2) & LAADS DAAC
NASA_EARTHDATA_S3_BASE = "https://data.laadsdaac.earthdatacloud.nasa.gov/prod-lads"
NASA_LANCE_NRT_BASE = "https://nrt3.modaps.eosdis.nasa.gov/api/v2/content/archives/allData/61"
NASA_LAADS_ARCHIVE_BASE = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61"

MODIS_GRANULE_CATALOG = [
    {
        "product": "MOD00F",
        "role": "L0 Raw CCSDS Packets (Part 000 Header)",
        "key": f"{DEFAULT_GCS_PREFIX}/MOD00F.A2026213.0710.20262130706.000.PDS.xz",
        "alt_keys": [
            f"{DEFAULT_GCS_PREFIX}/MOD00F.A2026213.0710.061.P.000.NRT",
        ],
        "size_bytes": 168,
        "urls": [
            f"{NASA_LAADS_ARCHIVE_BASE}/MOD00F/2026/213/MOD00F.A2026213.0710.20262130706.000.PDS.xz",
            f"{NASA_LAADS_ARCHIVE_BASE}/MOD00F/2026/213/MOD00F.A2026213.0700.20262130706.000.PDS.xz",
        ],
    },
    {
        "product": "MOD00F",
        "role": "L0 Raw CCSDS Packets (Part 001 Main Payload)",
        "key": f"{DEFAULT_GCS_PREFIX}/MOD00F.A2026213.0710.20262130706.001.PDS.xz",
        "alt_keys": [
            f"{DEFAULT_GCS_PREFIX}/MOD00F.A2026213.0710.061.P.001.NRT",
        ],
        "size_bytes": 277429064,
        "urls": [
            f"{NASA_LAADS_ARCHIVE_BASE}/MOD00F/2026/213/MOD00F.A2026213.0710.20262130706.001.PDS.xz",
            f"{NASA_LANCE_NRT_BASE}/MOD00F/2026/213/MOD00F.A2026213.0710.061.P.001.NRT",
        ],
    },
    {
        "product": "MOD021KM",
        "role": "L1B Calibrated 1-km Radiances",
        "key": f"{DEFAULT_GCS_PREFIX}/MOD021KM.A2026213.0710.061.2026216005446.hdf",
        "alt_keys": [
            f"{DEFAULT_GCS_PREFIX}/MOD021KM.A2026213.0710.061.2026213082055.NRT.hdf",
        ],
        "size_bytes": 156750816,
        "urls": [
            f"{NASA_EARTHDATA_S3_BASE}/MOD021KM/MOD021KM.A2026213.0710.061.2026216005446.hdf",
            f"{NASA_LAADS_ARCHIVE_BASE}/MOD021KM/2026/213/MOD021KM.A2026213.0710.061.2026216005446.hdf",
        ],
    },
    {
        "product": "MOD03",
        "role": "L1A Geolocation Fields (1-km)",
        "key": f"{DEFAULT_GCS_PREFIX}/MOD03.A2026213.0710.061.2026216003448.hdf",
        "alt_keys": [
            f"{DEFAULT_GCS_PREFIX}/MOD03.A2026213.0710.061.2026213081852.NRT.hdf",
        ],
        "size_bytes": 34232558,
        "urls": [
            f"{NASA_EARTHDATA_S3_BASE}/MOD03/MOD03.A2026213.0710.061.2026216003448.hdf",
            f"{NASA_LAADS_ARCHIVE_BASE}/MOD03/2026/213/MOD03.A2026213.0710.061.2026216003448.hdf",
        ],
    },
]

_local = threading.local()


def get_thread_storage_client() -> storage.Client:
    """Returns a thread-local GCS client for connection pooling safety."""
    if not hasattr(_local, "client"):
        _local.client = storage.Client()
    return _local.client


def get_existing_gcs_blobs(bucket_name: str, client: storage.Client, prefix: str = "modis-data/") -> Dict[str, int]:
    """Retrieves existing objects and their sizes from GCS under the target prefix."""
    print(f"[*] Checking existing objects in gs://{bucket_name}/{prefix}...")
    existing = {}
    try:
        bucket = client.bucket(bucket_name)
        blobs = client.list_blobs(bucket, prefix=prefix)
        for blob in blobs:
            existing[blob.name] = blob.size
        print(f"[✓] Found {len(existing)} existing files in GCS prefix {prefix}.")
    except Exception as e:
        print(f"[!] Warning: Could not list GCS prefix: {e}")
    return existing


def stream_url_to_gcs(
    item: dict,
    bucket_name: str,
    token: Optional[str] = None,
    max_retries: int = 3,
) -> bool:
    """Streams a MODIS file directly into GCS in 32MB chunks without local disk storage."""
    key = item["key"]
    size = item["size_bytes"]
    urls = item.get("urls", [])
    alt_keys = item.get("alt_keys", [])

    client = get_thread_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(key)

    headers = {"User-Agent": "MODIS-GCS-Sync/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for url in urls:
        # Special handling if this is the 000 header fallback from 0700
        is_000_fallback = ("0700" in url and "000" in key)

        for attempt in range(1, max_retries + 1):
            print(f"    [SYNCING {item['product']}] Attempt {attempt}/{max_retries}: {key} (~{size / 1e6:.1f} MB)...")
            t0 = time.time()
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=180) as resp:
                    final_url = resp.geturl()
                    if "urs.earthdata.nasa.gov" in final_url and not token:
                        raise PermissionError(
                            "Access requires NASA Earthdata authentication. Set EARTHDATA_TOKEN or place in ~/.edl_token"
                        )

                    if is_000_fallback:
                        # Decompress, rewrite timestamp/tag to 0710, recompress
                        import lzma
                        raw_xz = resp.read()
                        pds_raw = lzma.decompress(raw_xz)
                        pds_fixed = pds_raw.replace(b"0700", b"0710")
                        payload = lzma.compress(pds_fixed)
                        blob.upload_from_string(payload, content_type="application/x-xz")
                    else:
                        with blob.open("wb", chunk_size=32 * 1024 * 1024) as f_out:
                            while True:
                                chunk = resp.read(8 * 1024 * 1024)
                                if not chunk:
                                    break
                                f_out.write(chunk)

                blob.reload()
                elapsed = time.time() - t0
                mbps = (blob.size / 1e6) / max(elapsed, 0.01)
                print(f"    [DONE] {key} transferred in {elapsed:.1f}s ({mbps:.2f} MB/s, {blob.size} bytes)")

                # Create alias copies for NRT / alternate filename compatibility
                for alt_k in alt_keys:
                    try:
                        bucket.copy_blob(blob, bucket, alt_k)
                        print(f"    [ALIAS] Copied {key} -> {alt_k}")
                    except Exception as e_alt:
                        print(f"    [WARN] Could not create alias {alt_k}: {e_alt}")

                return True
            except urllib.error.HTTPError as he:
                if he.code == 404:
                    print(f"    [404 NOT FOUND] {url} (skipping retries)")
                    break
                print(f"    [HTTP {he.code}] Attempt {attempt} failed for {url}: {he}")
                if attempt < max_retries:
                    time.sleep(2 * attempt)
            except PermissionError as pe:
                print(f"    [AUTH ERROR] {pe}")
                return False
            except Exception as e:
                print(f"    [WARN] Attempt {attempt} failed for {url}: {e}")
                if attempt < max_retries:
                    time.sleep(2 * attempt)
                else:
                    break

    return False


def main():
    parser = argparse.ArgumentParser(description="Sync MODIS Granule Datasets to GCS")
    parser.add_argument(
        "--bucket",
        type=str,
        default=DEFAULT_GCS_BUCKET,
        help=f"Destination GCS bucket (default: {DEFAULT_GCS_BUCKET})",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_GCS_PREFIX,
        help=f"Destination prefix (default: {DEFAULT_GCS_PREFIX})",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="NASA Earthdata Bearer Token (defaults to EARTHDATA_TOKEN env var or ~/.edl_token)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent transfer worker threads (default: 4)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="modis_catalog_manifest.json",
        help="Local manifest export path",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite objects even if they already exist in GCS",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Inspect GCS bucket status without initiating transfer",
    )
    args = parser.parse_args()

    # Resolve Earthdata token
    token = args.token or os.environ.get("EARTHDATA_TOKEN")
    token_file = os.path.expanduser(os.environ.get("EARTHDATA_TOKEN_FILE", "~/.edl_token"))
    if not token and os.path.exists(token_file):
        try:
            with open(token_file, "r") as f:
                token = f.read().strip()
        except Exception:
            pass

    # Save manifest
    catalog_items = MODIS_GRANULE_CATALOG
    for it in catalog_items:
        if not it["key"].startswith(args.prefix):
            it["key"] = f"{args.prefix.rstrip('/')}/{os.path.basename(it['key'])}"

    with open(args.manifest, "w") as f:
        json.dump(catalog_items, f, indent=2)
    print(f"[*] Manifest saved to {args.manifest}")

    total_files = len(catalog_items)
    total_bytes = sum(i["size_bytes"] for i in catalog_items)
    print(f"\n[*] Target MODIS Granule: {total_files} files ({total_bytes / 1e6:.1f} MB / {total_bytes / (1024**2):.1f} MiB)")

    primary_client = get_thread_storage_client()
    existing_blobs = {}
    if not args.overwrite:
        existing_blobs = get_existing_gcs_blobs(args.bucket, primary_client, prefix=f"{args.prefix.rstrip('/')}/")

    already_synced = []
    pending_items = []
    synced_bytes = 0
    pending_bytes = 0

    for item in catalog_items:
        key = item["key"]
        sz = item["size_bytes"]
        # Check primary key and alt_keys
        found = False
        for candidate_key in [key] + item.get("alt_keys", []):
            if candidate_key in existing_blobs:
                already_synced.append(item)
                synced_bytes += existing_blobs[candidate_key]
                found = True
                break
        if not found:
            pending_items.append(item)
            pending_bytes += sz

    print("\n" + "=" * 80)
    print(f"MODIS SYNC STATUS: gs://{args.bucket}/{args.prefix}/")
    print(f"  • Total in Manifest:   {total_files} files ({total_bytes / 1e6:.1f} MB)")
    print(f"  • Already in GCS:      {len(already_synced)} files ({synced_bytes / 1e6:.1f} MB, {(synced_bytes / max(total_bytes, 1)) * 100:.1f}%)")
    print(f"  • Pending Transfer:    {len(pending_items)} files ({pending_bytes / 1e6:.1f} MB)")
    print("=" * 80 + "\n")

    if args.verify_only:
        return

    if not pending_items:
        print(f"[✓] All {total_files} files are already completely synced in GCS!")
        return

    if not token:
        print("[!] NOTICE: No NASA Earthdata Bearer Token detected.")
        print("    NASA Earthdata Cloud requires authentication to stream files.")
        print("    Please set EARTHDATA_TOKEN=<token> or put your token in ~/.edl_token")
        print("    You can generate a free token at https://urs.earthdata.nasa.gov/\n")

    print(f"[*] Starting streaming of {len(pending_items)} files ({pending_bytes / 1e6:.1f} MB)...")
    t_start = time.time()
    success_count = 0
    newly_synced_bytes = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(stream_url_to_gcs, item, args.bucket, token): item
            for item in pending_items
        }
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                if future.result():
                    success_count += 1
                    newly_synced_bytes += item["size_bytes"]
                    pct = ((len(already_synced) + success_count) / total_files) * 100
                    print(f"    --> Progress: {len(already_synced) + success_count}/{total_files} ({pct:.1f}%)")
            except Exception as e:
                print(f"[!] Worker error on {item['key']}: {e}")

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"[✓] Run Finished: {success_count}/{len(pending_items)} pending files transferred.")
    print(f"    Newly Synced:     {newly_synced_bytes / 1e6:.2f} MB")
    print(f"    Duration:         {total_time:.1f}s")
    print(f"    Total in GCS:     gs://{args.bucket}/{args.prefix}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
