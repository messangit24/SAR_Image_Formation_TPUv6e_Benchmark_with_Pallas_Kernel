#!/usr/bin/env python3
"""
Umbra Datacenter Sync Monitor (Multi-Pod Indexed Support)
Inspects the live state of all parallel GKE sync pods, aggregate network throughput,
and GCS bucket committed storage volume.
"""

import concurrent.futures
import json
import os
import subprocess
import sys
import time

def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except subprocess.CalledProcessError as e:
        return ""

def get_pod_network_stats(pod_name):
    net_py = """
import time
def get_bytes():
    with open('/proc/net/dev') as f:
        for line in f:
            if 'eth0' in line:
                parts = line.split()
                return int(parts[1]), int(parts[9])
    return 0, 0

rx1, tx1 = get_bytes()
t1 = time.time()
time.sleep(1.5)
rx2, tx2 = get_bytes()
t2 = time.time()
dt = t2 - t1

with open('/proc/net/tcp') as f:
    active_sock = sum(1 for line in f.readlines()[1:] if line.strip().split()[3] == '01')

print(f'{rx2},{tx2},{(rx2-rx1)/dt},{(tx2-tx1)/dt},{active_sock}')
"""
    cmd = f'kubectl exec {pod_name} -c sync-worker -- python3 -c "{net_py}"'
    out = run_cmd(cmd)
    if out:
        try:
            parts = [float(x) for x in out.split(",")]
            return parts[0], parts[1], parts[2], parts[3], int(parts[4])
        except Exception:
            pass
    return 0.0, 0.0, 0.0, 0.0, 0

def ensure_gke_token():
    try:
        token_out = run_cmd("gcloud auth print-access-token")
        lines = [l.strip() for l in token_out.splitlines() if l.strip() and not l.startswith("Created ")]
        token = lines[-1] if lines else None
        if token:
            run_cmd(f'kubectl config set-credentials gke-token --token="{token}"')
            run_cmd('kubectl config set-context --current --user=gke-token')
            return token
    except Exception:
        pass
    return None

def main():
    print("=" * 80)
    print("       UMBRA S3 -> GCS DATACENTER SYNC MONITOR (GKE MULTI-POD INDEXED)       ")
    print("=" * 80)

    # 0. Ensure fresh credentials for kubectl and GCS
    cached_token = ensure_gke_token()

    # 1. Check pods (check indexed job first, fallback to single job)
    pod_info = run_cmd("kubectl get pods -l app=umbra-s3-to-gcs-sync-indexed -o json")
    if not pod_info or len(json.loads(pod_info).get("items", [])) == 0:
        pod_info = run_cmd("kubectl get pods -l app=umbra-s3-to-gcs-sync -o json")

    if not pod_info:
        print("[!] No sync pods found or kubectl error.")
        return

    pods = json.loads(pod_info).get("items", [])
    if not pods:
        print("[!] No active sync pods found.")
        return

    print(f"Total Pods: {len(pods)}")
    print(f"{'Pod Name':<38} {'Shard':<7} {'Status':<11} {'Node':<30}")
    print("-" * 80)

    running_pods = []
    for pod in sorted(pods, key=lambda p: p["metadata"]["name"]):
        pname = pod["metadata"]["name"]
        status = pod["status"]["phase"]
        node = pod["spec"].get("nodeName", "Pending")
        # Extract shard index from env or completion index
        shard = "N/A"
        for env in pod["spec"]["containers"][0].get("env", []):
            if env.get("name") == "JOB_COMPLETION_INDEX":
                shard = env.get("value", "N/A")
        if shard == "N/A":
            shard = pod["metadata"].get("labels", {}).get("batch.kubernetes.io/job-completion-index", "0")

        if len(node) > 28:
            node_disp = "..." + node[-25:]
        else:
            node_disp = node

        print(f"{pname:<38} {shard:<7} {status:<11} {node_disp:<30}")
        if status == "Running":
            running_pods.append(pname)

    print("-" * 80)

    # 2. Parallel Network Throughput Measurement
    if running_pods:
        print(f"[*] Measuring live throughput across {len(running_pods)} active worker pods...")
        tot_rx = 0.0
        tot_tx = 0.0
        tot_rx_rate = 0.0
        tot_tx_rate = 0.0
        tot_socks = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(running_pods)) as pool:
            results = list(pool.map(get_pod_network_stats, running_pods))

        for rx, tx, rx_r, tx_r, sock in results:
            tot_rx += rx
            tot_tx += tx
            tot_rx_rate += rx_r
            tot_tx_rate += tx_r
            tot_socks += sock

        print(f"\nAggregate Datacenter Network Activity:")
        print(f"  Total Ingested (RX from AWS S3): {tot_rx / 1e9:.2f} GB ({tot_rx / (1024**3):.2f} GiB)")
        print(f"  Total Uploaded (TX to GCS):     {tot_tx / 1e9:.2f} GB ({tot_tx / (1024**3):.2f} GiB)")
        print(f"  Aggregate Download Speed:       {tot_rx_rate / 1e6:.2f} MB/s ({tot_rx_rate * 8 / 1e6:.1f} Mbps)")
        print(f"  Aggregate Upload Speed:         {tot_tx_rate / 1e6:.2f} MB/s ({tot_tx_rate * 8 / 1e6:.1f} Mbps)")
        print(f"  Total Active TLS Connections:   {tot_socks} sockets across cluster")
    print("-" * 80)

    # 3. GCS Committed Storage State
    try:
        from google.cloud import storage
        import google.oauth2.credentials
        token = cached_token or ensure_gke_token()
        if token:
            creds = google.oauth2.credentials.Credentials(token)
            client = storage.Client(credentials=creds, project="mb-development-447000")
            bucket = client.bucket("umbra-sar-dataset-75tb-mb-development-447000")
            blobs = list(client.list_blobs(bucket, prefix="sar-data/"))
            total_bytes = sum(b.size for b in blobs if b.size)
            pct = (total_bytes / 30.007e12) * 100
            print(f"GCS Bucket Committed State:")
            print(f"  Committed Files:  {len(blobs)} / 1,872 files ({len(blobs)/1872*100:.1f}%)")
            print(f"  Committed Volume: {total_bytes / 1e9:.2f} GB ({total_bytes / (1024**3):.2f} GiB) / 30.01 TB ({pct:.2f}%)")

            # Overall ETA estimate if running
            if running_pods and tot_tx_rate > 1e5:
                rem_bytes = max(30.007e12 - total_bytes, 0)
                eta_hours = (rem_bytes / tot_tx_rate) / 3600
                print(f"  Estimated Full Catalog ETA: ~{eta_hours:.1f} hours at current cluster pace")

            print("\nLatest Committed Blobs:")
            for b in sorted(blobs, key=lambda x: x.updated or '', reverse=True)[:5]:
                print(f"  - {b.name.split('/')[-1]} ({b.size / 1e6:.1f} MB, {b.updated})")
    except Exception as e:
        print(f"[!] Could not query GCS: {e}")

    print("=" * 80)

if __name__ == "__main__":
    main()
