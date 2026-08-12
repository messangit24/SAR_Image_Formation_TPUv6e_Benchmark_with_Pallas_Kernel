"""TPUbench GKE Benchmark Job Dispatcher & Manifest Generator.

Generates validated Kubernetes Job specifications for dispatching SAR and MODIS
benchmarks to GKE CPU standard and Cloud TPU node pools.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import os
import sys
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None  # Fallback to basic JSON/YAML serialization if PyYAML is absent


@dataclass
class BenchmarkJobConfig:
    """Configuration for a GKE benchmark worker job."""

    job_name: str
    workload_type: str = "sar-cpu"  # "sar-cpu" | "sar-tpu" | "modis-batch"
    image: str = "us-docker.pkg.dev/PROJECT_ID/tpubench/worker:latest"
    namespace: str = "tpubench"
    service_account: str = "tpubench-benchmark-sa"
    gcs_bucket: str = "goldthread-raw-data"
    gcs_mount_path: str = "/data"
    backoff_limit: int = 2
    ttl_seconds_after_finished: int = 600

    # CPU settings
    cpu_requests: Optional[str] = None
    cpu_limits: Optional[str] = None
    memory_requests: Optional[str] = None
    memory_limits: Optional[str] = None

    # TPU settings
    tpu_chips: int = 4
    tpu_accelerator: str = "tpu-v5-lite-podslice"
    tpu_topology: str = "2x2x1"

    # Command override
    command: Optional[List[str]] = None
    args: Optional[List[str]] = None
    env: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


def generate_job_manifest(config: BenchmarkJobConfig) -> Dict[str, Any]:
    """Generate a validated Kubernetes Job manifest dictionary.

    Args:
        config: BenchmarkJobConfig specification.

    Returns:
        Dictionary representation of the Kubernetes Job resource.
    """
    labels = {
        "app.kubernetes.io/name": config.job_name,
        "app.kubernetes.io/component": "benchmark-worker",
        "app.kubernetes.io/part-of": "goldthread",
        "tpubench.io/workload-type": config.workload_type,
    }

    annotations = {
        "gke-gcsfuse/volumes": "true",
    }

    # Determine command and args
    if config.command is not None:
        cmd = config.command
        args = config.args or []
    elif config.workload_type == "sar-cpu":
        cmd = ["python", "-m", "src.sar.tpu_ffbp"]
        args = config.args if config.args is not None else ["--cphd", f"{config.gcs_mount_path}/sar/Bingham.cphd", "--device", "cpu"]
    elif config.workload_type == "sar-tpu":
        cmd = ["python", "-m", "src.sar.tpu_ffbp"]
        args = config.args if config.args is not None else ["--cphd", f"{config.gcs_mount_path}/sar/Bingham.cphd", "--device", "tpu"]
    elif config.workload_type == "modis-batch":
        cmd = ["python", "-m", "src.modis.distributed_batch"]
        args = config.args if config.args is not None else ["--synthetic-granules", "8"]
    else:
        cmd = ["python", "--version"]
        args = config.args or []

    container_env = []
    if config.workload_type == "sar-tpu":
        container_env.extend([
            {"name": "JAX_ENABLE_X64", "value": "True"},
            {"name": "LIBTPU_INIT_ARGS", "value": "--xla_tpu_enable_megacore=true"},
        ])
    for k, v in config.env.items():
        container_env.append({"name": k, "value": v})

    # Resource requests and limits
    if config.workload_type == "sar-tpu":
        cpu_req = config.cpu_requests or "16"
        mem_req = config.memory_requests or "64Gi"
        cpu_lim = config.cpu_limits or "32"
        mem_lim = config.memory_limits or "128Gi"
        resources = {
            "requests": {
                "google.com/tpu": str(config.tpu_chips),
                "cpu": cpu_req,
                "memory": mem_req,
            },
            "limits": {
                "google.com/tpu": str(config.tpu_chips),
                "cpu": cpu_lim,
                "memory": mem_lim,
            },
        }
        node_selector = {
            "cloud.google.com/gke-tpu-accelerator": config.tpu_accelerator,
            "cloud.google.com/gke-tpu-topology": config.tpu_topology,
        }
        tolerations = [
            {
                "key": "google.com/tpu",
                "operator": "Exists",
                "effect": "NoSchedule",
            }
        ]
    elif config.workload_type == "modis-batch":
        cpu_req = config.cpu_requests or "16"
        mem_req = config.memory_requests or "32Gi"
        cpu_lim = config.cpu_limits or "32"
        mem_lim = config.memory_limits or "64Gi"
        resources = {
            "requests": {
                "cpu": cpu_req,
                "memory": mem_req,
            },
            "limits": {
                "cpu": cpu_lim,
                "memory": mem_lim,
            },
        }
        node_selector = {
            "tpubench.io/workload-type": "cpu-benchmark",
        }
        tolerations = []
    else:
        cpu_req = config.cpu_requests or "64"
        mem_req = config.memory_requests or "256Gi"
        cpu_lim = config.cpu_limits or "80"
        mem_lim = config.memory_limits or "320Gi"
        resources = {
            "requests": {
                "cpu": cpu_req,
                "memory": mem_req,
            },
            "limits": {
                "cpu": cpu_lim,
                "memory": mem_lim,
            },
        }
        node_selector = {
            "tpubench.io/workload-type": "cpu-benchmark",
        }
        tolerations = []

    pod_spec: Dict[str, Any] = {
        "serviceAccountName": config.service_account,
        "restartPolicy": "Never",
        "nodeSelector": node_selector,
        "containers": [
            {
                "name": "worker",
                "image": config.image,
                "imagePullPolicy": "IfNotPresent",
                "command": cmd,
                "args": args,
                "resources": resources,
                "volumeMounts": [
                    {
                        "name": "gcs-data",
                        "mountPath": config.gcs_mount_path,
                        "readOnly": True,
                    },
                    {
                        "name": "output-dir",
                        "mountPath": "/workspace/out",
                    },
                ],
            }
        ],
        "volumes": [
            {
                "name": "gcs-data",
                "csi": {
                    "driver": "gcsfuse.csi.storage.gke.io",
                    "volumeAttributes": {
                        "bucketName": config.gcs_bucket,
                        "mountOptions": "implicit-dirs",
                    },
                },
            },
            {
                "name": "output-dir",
                "emptyDir": {},
            },
        ],
    }

    if container_env:
        pod_spec["containers"][0]["env"] = container_env

    if tolerations:
        pod_spec["tolerations"] = tolerations

    manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": config.job_name,
            "namespace": config.namespace,
            "labels": labels,
        },
        "spec": {
            "backoffLimit": config.backoff_limit,
            "ttlSecondsAfterFinished": config.ttl_seconds_after_finished,
            "template": {
                "metadata": {
                    "labels": labels,
                    "annotations": annotations,
                },
                "spec": pod_spec,
            },
        },
    }

    return manifest


def _dict_to_yaml(data: Any, indent: int = 0) -> str:
    """Minimal YAML serializer for nested dictionaries and lists."""
    lines = []
    prefix = " " * indent
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{prefix}{k}:")
                lines.append(_dict_to_yaml(v, indent + 2))
            elif isinstance(v, bool):
                lines.append(f"{prefix}{k}: {str(v).lower()}")
            elif isinstance(v, (int, float)):
                lines.append(f"{prefix}{k}: {v}")
            elif v is None:
                lines.append(f"{prefix}{k}: null")
            else:
                s_val = str(v)
                if any(c in s_val for c in [":", "{", "}", "[", "]", ",", "&", "*", "#", "?", "|", "<", ">", "=", "!", "%", "@", "`", " "]):
                    lines.append(f'{prefix}{k}: "{s_val}"')
                else:
                    lines.append(f"{prefix}{k}: {s_val}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                sub_lines = _dict_to_yaml(item, indent + 2).splitlines()
                if sub_lines:
                    first = sub_lines[0].lstrip()
                    lines.append(f"{prefix}- {first}")
                    lines.extend(sub_lines[1:])
            else:
                s_val = str(item)
                if any(c in s_val for c in [":", "{", "}", "[", "]", ",", "&", "*", "#", "?", "|", "<", ">", "=", "!", "%", "@", "`", " "]):
                    lines.append(f'{prefix}- "{s_val}"')
                else:
                    lines.append(f"{prefix}- {s_val}")
    return "\n".join(lines)


def render_manifest_yaml(config: BenchmarkJobConfig) -> str:
    """Render manifest dictionary to formatted YAML string."""
    manifest = generate_job_manifest(config)
    if yaml is not None:
        return yaml.dump(manifest, sort_keys=False)
    return _dict_to_yaml(manifest) + "\n"




def main() -> None:
    """CLI dispatcher entrypoint."""
    ap = argparse.ArgumentParser(description="TPUbench GKE Benchmark Job Dispatcher")
    ap.add_argument("--job-name", default="sar-benchmark-job", help="Kubernetes job name")
    ap.add_argument("--workload-type", choices=["sar-cpu", "sar-tpu", "modis-batch"], default="sar-cpu")
    ap.add_argument("--image", default="us-docker.pkg.dev/PROJECT_ID/tpubench/worker:latest")
    ap.add_argument("--gcs-bucket", default="goldthread-raw-data")
    ap.add_argument("--tpu-chips", type=int, default=4)
    ap.add_argument("--tpu-topology", default="2x2x1")
    ap.add_argument("--out-yaml", help="Path to write rendered YAML manifest")
    ap.add_argument("--dry-run", action="store_true", help="Print manifest without submitting")
    args = ap.parse_args()

    config = BenchmarkJobConfig(
        job_name=args.job_name,
        workload_type=args.workload_type,
        image=args.image,
        gcs_bucket=args.gcs_bucket,
        tpu_chips=args.tpu_chips,
        tpu_topology=args.tpu_topology,
    )

    yaml_text = render_manifest_yaml(config)
    if args.dry_run or not args.out_yaml:
        print(yaml_text)

    if args.out_yaml:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_yaml)), exist_ok=True)
        with open(args.out_yaml, "w", encoding="utf-8") as f:
            f.write(yaml_text)
        print(f"Wrote Kubernetes manifest to {args.out_yaml}", file=sys.stderr)


if __name__ == "__main__":
    main()
