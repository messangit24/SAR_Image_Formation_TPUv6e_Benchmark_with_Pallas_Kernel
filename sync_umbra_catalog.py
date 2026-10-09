#!/usr/bin/env python3
"""
Umbra Open Data Catalog High-Throughput Sync to Google Cloud Storage (GCS).
Streams CPHD phase history and SICD reference data from AWS S3 directly into GCS.
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
import xml.etree.ElementTree as ET

# Clean mTLS / enterprise-cert proxy env vars that can cause authentication hangs on corporate workstations
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

S3_BASE_URL = "https://umbra-open-data-catalog.s3.amazonaws.com"
S3_XML_ENDPOINT = "https://umbra-open-data-catalog.s3.amazonaws.com/?list-type=2&prefix=sar-data/&max-keys=1000"
XML_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

_local = threading.local()


def get_thread_storage_client() -> storage.Client:
    """Returns a thread-local GCS client to ensure thread safety and avoid connection pool exhaustion."""
    if not hasattr(_local, "client"):
        _local.client = storage.Client()
    return _local.client


def list_umbra_catalog(
    max_total_gb: float = 30000.0,
    max_scenes: int = 5000,
    cphd_only: bool = False,
    sicd_only: bool = False,
) -> list[dict]:
    """Iterates through S3 XML pagination to catalog CPHD and SICD files up to max_total_gb."""
    mode_desc = "All (CPHD + SICD)"
    if cphd_only:
        mode_desc = "CPHD Phase History Only"
    elif sicd_only:
        mode_desc = "SICD NITF Only"

    print(f"[*] Discovering Umbra datasets ({mode_desc})...")
    print(f"[*] Target Cap: {max_total_gb:,.1f} GB ({max_total_gb/1024:.2f} TB), Max files: {max_scenes}...")
    scenes = []
    total_bytes = 0
    continuation_token = None
    page = 1

    while True:
        url = S3_XML_ENDPOINT
        if continuation_token:
            url += f"&continuation-token={urllib.parse.quote_plus(continuation_token)}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Umbra-GCS-Sync/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                xml_data = resp.read()
        except Exception as e:
            print(f"[!] Error querying S3 catalog page {page}: {e}")
            break

        root = ET.fromstring(xml_data)
        contents = root.findall("s3:Contents", XML_NS)
        if not contents:
            break

        for item in contents:
            key = item.find("s3:Key", XML_NS).text
            size = int(item.find("s3:Size", XML_NS).text)

            is_cphd = key.endswith("_CPHD.cphd")
            is_sicd = key.endswith("_SICD.nitf")

            if not (is_cphd or is_sicd):
                continue
            if cphd_only and not is_cphd:
                continue
            if sicd_only and not is_sicd:
                continue

            scenes.append({
                "key": key,
                "size_bytes": size,
                "size_mb": round(size / 1e6, 2),
                "url": f"{S3_BASE_URL}/{key}",
            })
            total_bytes += size

            if (total_bytes / 1e9) >= max_total_gb or len(scenes) >= max_scenes:
                print(f"[✓] Reached target catalog cap: {total_bytes / 1e12:.3f} TB across {len(scenes)} files.")
                return scenes

        is_truncated = root.find("s3:IsTruncated", XML_NS)
        if is_truncated is not None and is_truncated.text == "true":
            next_token = root.find("s3:NextContinuationToken", XML_NS)
            if next_token is not None:
                continuation_token = next_token.text
                page += 1
                if page % 2 == 0:
                    print(f"    Page {page}: Cataloged {len(scenes)} files ({total_bytes / 1e12:.3f} TB)...")
            else:
                break
        else:
            break

    print(f"[✓] Finished discovery: {len(scenes)} files ({total_bytes / 1e12:.3f} TB).")
    return scenes


def get_existing_gcs_blobs(bucket_name: str, client: storage.Client, prefix: str = "sar-data/") -> dict[str, int]:
    """Retrieves all existing objects and their sizes from GCS in a few fast paginated queries."""
    print(f"[*] Checking existing objects in gs://{bucket_name}/{prefix}...")
    existing = {}
    try:
        bucket = client.bucket(bucket_name)
        blobs = client.list_blobs(bucket, prefix=prefix)
        for blob in blobs:
            existing[blob.name] = blob.size
        print(f"[✓] Found {len(existing)} existing files in GCS bucket.")
    except Exception as e:
        print(f"[!] Warning: Could not bulk-list GCS bucket: {e}")
    return existing


def stream_s3_to_gcs(
    item: dict,
    bucket_name: str,
    max_retries: int = 3,
) -> bool:
    """Streams a single file from AWS S3 directly into a GCS bucket without saving to local disk."""
    key = item["key"]
    url = item["url"]
    size = item["size_bytes"]

    client = get_thread_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(key)

    for attempt in range(1, max_retries + 1):
        print(f"    [SYNCING attempt {attempt}/{max_retries}] {key} ({size / 1e6:.1f} MB)...")
        t0 = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Umbra-GCS-Sync/1.0"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                # 32MB chunks for high-throughput GCS resumable upload
                with blob.open("wb", chunk_size=32 * 1024 * 1024) as f_out:
                    while True:
                        chunk = resp.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        f_out.write(chunk)

            # Verification: reload blob metadata and verify size matches
            blob.reload()
            if blob.size != size:
                raise IOError(f"Size mismatch after upload: expected {size} bytes, got {blob.size} bytes")

            elapsed = time.time() - t0
            mbps = (size / 1e6) / max(elapsed, 0.01)
            print(f"    [DONE] {key} transferred in {elapsed:.1f}s ({mbps:.2f} MB/s)")
            return True
        except Exception as e:
            print(f"    [WARN] Attempt {attempt} failed for {key}: {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                print(f"    [ERROR] All {max_retries} attempts failed for {key}")
                return False


def main():
    parser = argparse.ArgumentParser(description="Sync Umbra Open Data Catalog to GCS")
    parser.add_argument(
        "--bucket",
        type=str,
        default="umbra-sar-dataset-75tb-mb-development-447000",
        help="Destination GCS bucket",
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=30000.0,
        help="Max total volume in GB (default: 30000 GB = 30 TB)",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=5000,
        help="Max number of files to catalog/sync",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent transfer threads (default: 8)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="umbra_catalog_manifest.json",
        help="Local manifest export path",
    )
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Force re-querying S3 catalog instead of using cached manifest",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only build manifest without downloading",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check what is already present in GCS vs the manifest without syncing",
    )
    parser.add_argument(
        "--cphd-only",
        action="store_true",
        help="Sync only raw phase history (CPHD) files",
    )
    parser.add_argument(
        "--sicd-only",
        action="store_true",
        help="Sync only detected/processed SAR (SICD) NITF files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files even if they already exist in GCS with matching size",
    )
    args = parser.parse_args()

    # Step 1: Load or generate manifest
    if os.path.exists(args.manifest) and not args.refresh_manifest and not args.discover_only:
        print(f"[*] Loading existing manifest from {args.manifest}...")
        with open(args.manifest, "r") as f:
            items = json.load(f)
        if args.cphd_only:
            items = [x for x in items if x["key"].endswith("_CPHD.cphd")]
        elif args.sicd_only:
            items = [x for x in items if x["key"].endswith("_SICD.nitf")]
        if args.max_scenes and len(items) > args.max_scenes:
            items = items[:args.max_scenes]
    else:
        items = list_umbra_catalog(
            max_total_gb=args.max_gb,
            max_scenes=args.max_scenes,
            cphd_only=args.cphd_only,
            sicd_only=args.sicd_only,
        )
        with open(args.manifest, "w") as f:
            json.dump(items, f, indent=2)
        print(f"[*] Manifest saved to {args.manifest}")

    total_files = len(items)
    total_bytes = sum(i["size_bytes"] for i in items)
    print(f"\n[*] Target manifest contains {total_files} files ({total_bytes / 1e12:.3f} TB / {total_bytes / (1024**4):.3f} TiB)")

    if args.discover_only:
        return

    # Step 2: Initialize GCS client and check existing objects
    primary_client = get_thread_storage_client()

    existing_blobs = {}
    if not args.overwrite:
        existing_blobs = get_existing_gcs_blobs(args.bucket, primary_client, prefix="sar-data/")

    # Step 3: Partition into already synced vs pending
    already_synced = []
    pending_items = []
    synced_bytes = 0
    pending_bytes = 0

    for item in items:
        key = item["key"]
        sz = item["size_bytes"]
        if not args.overwrite and key in existing_blobs and existing_blobs[key] == sz:
            already_synced.append(item)
            synced_bytes += sz
        else:
            pending_items.append(item)
            pending_bytes += sz

    print("\n" + "=" * 80)
    print(f"SAR SYNC STATUS: gs://{args.bucket}/")
    print(f"  • Total in Manifest:   {total_files} files ({total_bytes / 1e12:.3f} TB)")
    print(f"  • Already in GCS:      {len(already_synced)} files ({synced_bytes / 1e12:.3f} TB, {(synced_bytes / max(total_bytes, 1)) * 100:.1f}%)")
    print(f"  • Pending Transfer:    {len(pending_items)} files ({pending_bytes / 1e12:.3f} TB)")
    print("=" * 80 + "\n")

    if args.verify_only:
        return

    if not pending_items:
        print(f"[✓] All {total_files} files ({total_bytes / 1e12:.3f} TB) are already completely synced in GCS!")
        return

    # Step 4: Execute multi-threaded streaming for pending items
    print(f"[*] Starting streaming of {len(pending_items)} files ({pending_bytes / 1e12:.3f} TB) with {args.workers} concurrent workers...")
    t_start = time.time()
    success_count = 0
    newly_synced_bytes = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(stream_s3_to_gcs, item, args.bucket): item
            for item in pending_items
        }
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                if future.result():
                    success_count += 1
                    newly_synced_bytes += item["size_bytes"]
                    pct = ((len(already_synced) + success_count) / total_files) * 100
                    total_current_bytes = synced_bytes + newly_synced_bytes
                    elapsed = time.time() - t_start
                    speed_mbps = (newly_synced_bytes / 1e6) / max(elapsed, 0.1)
                    rem_bytes = pending_bytes - newly_synced_bytes
                    eta_sec = rem_bytes / max(speed_mbps * 1e6, 1)
                    eta_min = eta_sec / 60
                    print(
                        f"    --> Progress: {len(already_synced) + success_count}/{total_files} ({pct:.1f}%) | "
                        f"{total_current_bytes / 1e12:.3f} TB total | "
                        f"Speed: {speed_mbps:.1f} MB/s | ETA: {eta_min:.1f}m"
                    )
            except Exception as e:
                print(f"[!] Unhandled worker error on {item['key']}: {e}")

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"[✓] Run Finished: {success_count}/{len(pending_items)} pending files transferred.")
    print(f"    Newly Synced:     {newly_synced_bytes / 1e9:.2f} GB ({newly_synced_bytes / 1e12:.3f} TB)")
    print(f"    Duration:         {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"    Average Speed:    {((newly_synced_bytes / 1e6) / max(total_time, 1)):.2f} MB/s")
    print(f"    Total in Bucket:  {(synced_bytes + newly_synced_bytes) / 1e12:.3f} / {total_bytes / 1e12:.3f} TB")
    print("=" * 80)


if __name__ == "__main__":
    main()
