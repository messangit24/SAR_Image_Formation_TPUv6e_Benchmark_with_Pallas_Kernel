"""Unit tests for GKE benchmark Kubernetes manifests, Helm charts, and job dispatcher."""
import os
import tempfile

from src.infra.job_dispatcher import (
    BenchmarkJobConfig,
    generate_job_manifest,
    render_manifest_yaml,
)


def test_generate_sar_cpu_job_manifest():
    """Verify SAR CPU benchmark job manifest structure, resources, and GCS mount."""
    config = BenchmarkJobConfig(
        job_name="sar-cpu-test",
        workload_type="sar-cpu",
        image="us-docker.pkg.dev/my-proj/tpubench/worker:v1",
        gcs_bucket="test-bucket",
    )
    manifest = generate_job_manifest(config)

    assert manifest["apiVersion"] == "batch/v1"
    assert manifest["kind"] == "Job"
    assert manifest["metadata"]["name"] == "sar-cpu-test"
    assert manifest["metadata"]["namespace"] == "tpubench"

    pod = manifest["spec"]["template"]["spec"]
    assert pod["serviceAccountName"] == "tpubench-benchmark-sa"
    assert pod["restartPolicy"] == "Never"
    assert pod["nodeSelector"]["tpubench.io/workload-type"] == "cpu-benchmark"

    container = pod["containers"][0]
    assert container["image"] == "us-docker.pkg.dev/my-proj/tpubench/worker:v1"
    assert container["resources"]["requests"]["cpu"] == "64"
    assert container["resources"]["requests"]["memory"] == "256Gi"
    assert "--device" in container["args"]
    assert "cpu" in container["args"]

    # Verify GCS FUSE volume
    gcs_vol = [v for v in pod["volumes"] if v["name"] == "gcs-data"][0]
    assert gcs_vol["csi"]["driver"] == "gcsfuse.csi.storage.gke.io"
    assert gcs_vol["csi"]["volumeAttributes"]["bucketName"] == "test-bucket"


def test_generate_sar_tpu_job_manifest():
    """Verify Cloud TPU benchmark job manifest structure, TPU resources, and tolerations."""
    config = BenchmarkJobConfig(
        job_name="sar-tpu-test",
        workload_type="sar-tpu",
        tpu_chips=4,
        tpu_accelerator="tpu-v5-lite-podslice",
        tpu_topology="2x2x1",
    )
    manifest = generate_job_manifest(config)

    pod = manifest["spec"]["template"]["spec"]
    assert pod["nodeSelector"]["cloud.google.com/gke-tpu-accelerator"] == "tpu-v5-lite-podslice"
    assert pod["nodeSelector"]["cloud.google.com/gke-tpu-topology"] == "2x2x1"

    # Tolerations for TPU node pool
    toleration = pod["tolerations"][0]
    assert toleration["key"] == "google.com/tpu"
    assert toleration["operator"] == "Exists"

    container = pod["containers"][0]
    assert container["resources"]["requests"]["google.com/tpu"] == "4"
    assert container["resources"]["limits"]["google.com/tpu"] == "4"
    assert "--device" in container["args"]
    assert "tpu" in container["args"]

    # Environment variables
    env_vars = {e["name"]: e["value"] for e in container["env"]}
    assert env_vars["JAX_ENABLE_X64"] == "True"


def test_generate_modis_batch_job_manifest():
    """Verify MODIS batch processing job manifest."""
    config = BenchmarkJobConfig(
        job_name="modis-batch-test",
        workload_type="modis-batch",
    )
    manifest = generate_job_manifest(config)

    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert "src.modis.distributed_batch" in container["command"][2]
    assert "--synthetic-granules" in container["args"]
    assert container["resources"]["requests"]["cpu"] == "16"
    assert container["resources"]["requests"]["memory"] == "32Gi"
    assert container["resources"]["limits"]["cpu"] == "32"
    assert container["resources"]["limits"]["memory"] == "64Gi"


def test_render_manifest_yaml_and_file_export():
    """Verify YAML rendering and file export."""
    config = BenchmarkJobConfig(
        job_name="export-test",
        workload_type="sar-cpu",
    )
    yaml_out = render_manifest_yaml(config)
    assert "export-test" in yaml_out
    assert "tpubench" in yaml_out

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "job.yaml")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(yaml_out)
        assert os.path.exists(out_file)
        assert os.path.getsize(out_file) > 100


def test_static_manifest_files_exist_and_labels_standardized():
    """Verify that all base Kubernetes manifests exist and use standardized labels."""
    base_dir = os.path.join(os.path.dirname(__file__), "..", "k8s")
    required_files = [
        os.path.join(base_dir, "base", "namespace.yaml"),
        os.path.join(base_dir, "base", "serviceaccount.yaml"),
        os.path.join(base_dir, "base", "sar-cpu-benchmark-job.yaml"),
        os.path.join(base_dir, "base", "sar-tpu-benchmark-job.yaml"),
        os.path.join(base_dir, "base", "modis-batch-job.yaml"),
        os.path.join(base_dir, "helm", "tpubench-worker", "Chart.yaml"),
        os.path.join(base_dir, "helm", "tpubench-worker", "values.yaml"),
        os.path.join(base_dir, "helm", "tpubench-worker", "templates", "job.yaml"),
    ]
    for rf in required_files:
        assert os.path.isfile(rf), f"Missing required file: {rf}"
        assert os.path.getsize(rf) > 0, f"Empty required file: {rf}"

    # Verify standardized workload-type label in base jobs
    for job_name in ["sar-cpu-benchmark-job.yaml", "sar-tpu-benchmark-job.yaml", "modis-batch-job.yaml"]:
        job_file = os.path.join(base_dir, "base", job_name)
        with open(job_file, "r", encoding="utf-8") as f:
            content = f.read()
        assert "tpubench.io/workload-type:" in content
        assert "tpubench.io/benchmark-type:" not in content


def test_custom_command_args_and_resource_overrides():
    """Verify custom command, args, and resource overrides in job manifest."""
    config = BenchmarkJobConfig(
        job_name="custom-worker",
        workload_type="modis-batch",
        command=["python", "custom_script.py"],
        args=["--custom-flag", "value"],
        cpu_requests="8",
        cpu_limits="16",
        memory_requests="16Gi",
        memory_limits="32Gi",
    )
    manifest = generate_job_manifest(config)
    container = manifest["spec"]["template"]["spec"]["containers"][0]

    assert container["command"] == ["python", "custom_script.py"]
    assert container["args"] == ["--custom-flag", "value"]
    assert container["resources"]["requests"]["cpu"] == "8"
    assert container["resources"]["requests"]["memory"] == "16Gi"
    assert container["resources"]["limits"]["cpu"] == "16"
    assert container["resources"]["limits"]["memory"] == "32Gi"


def test_tpu_ffbp_cli_execution(monkeypatch):
    """Verify src.sar.tpu_ffbp CLI entrypoint and benchmark output writing."""
    from src.sar.tpu_ffbp import main as tpu_ffbp_main
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        test_args = [
            "tpu_ffbp.py",
            "--cphd", "/nonexistent/path.cphd",
            "--chip", "32",
            "--device", "cpu",
            "--out-dir", tmpdir,
        ]
        monkeypatch.setattr("sys.argv", test_args)
        tpu_ffbp_main()

        metrics_file = os.path.join(tmpdir, "benchmark_metrics.json")
        complex_file = os.path.join(tmpdir, "sar_ffbp_complex.npy")
        png_file = os.path.join(tmpdir, "sar_ffbp.png")

        assert os.path.exists(metrics_file)
        assert os.path.exists(complex_file)
        assert os.path.exists(png_file)

        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        assert metrics["status"] == "SUCCESS"
        assert metrics["workload_type"] == "sar-ffbp"
        assert metrics["chip_size"] == [32, 32]
        assert metrics["megapixels_per_sec"] >= 0.0


def test_dockerfile_structure():
    """Verify Dockerfile contains multi-stage base, non-root user (Wiz compliance), uv, and TPU runtime."""
    dockerfile_path = os.path.join(os.path.dirname(__file__), "..", "Dockerfile")
    assert os.path.isfile(dockerfile_path), f"Dockerfile not found at {dockerfile_path}"
    with open(dockerfile_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "FROM python:3.12-slim-bookworm" in content
    assert "COPY --from=ghcr.io/astral-sh/uv" in content
    assert "useradd" in content and "tpubench" in content
    assert "USER 10001:10001" in content or "USER tpubench" in content
    assert "jax[tpu]" in content


def test_render_manifest_yaml_fallback_without_pyyaml(monkeypatch):
    """Verify render_manifest_yaml produces valid YAML even if PyYAML is unavailable."""
    import src.infra.job_dispatcher as jd

    monkeypatch.setattr(jd, "yaml", None)
    config = BenchmarkJobConfig(job_name="test-yaml", workload_type="sar-cpu")
    out = jd.render_manifest_yaml(config)
    assert "apiVersion: batch/v1" in out
    assert "name: test-yaml" in out
    assert "tpubench.io/workload-type: cpu-benchmark" in out



