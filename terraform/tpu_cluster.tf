# Dedicated TPU GKE Cluster
resource "google_container_cluster" "tpu_cluster" {
  name     = var.tpu_cluster_name
  location = var.zone_tpu

  # Remove default node pool immediately to manage dedicated TPU node pools cleanly
  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.benchmark_vpc.id
  subnetwork = google_compute_subnetwork.benchmark_subnet.id

  ip_allocation_policy {
    cluster_secondary_range_name  = "tpu-pods"
    services_secondary_range_name = "tpu-services"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  addons_config {
    http_load_balancing {
      disabled = false
    }
    horizontal_pod_autoscaling {
      disabled = true # Disabled to eliminate autoscaling interference during benchmarks
    }
    gcs_fuse_csi_driver_config {
      enabled = true # High-throughput direct Cloud Storage mount for raw CPHD datasets
    }
  }

  release_channel {
    channel = "RAPID"
  }

  maintenance_policy {
    daily_maintenance_window {
      start_time = "04:00"
    }
  }

  # Strict node isolation for deterministic benchmarking
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  deletion_protection = false

  depends_on = [
    google_compute_router_nat.nat_gateway,
    google_project_service.required_apis
  ]
}

# Dedicated TPU Node Pool (Scale-to-Zero Capable)
resource "google_container_node_pool" "tpu_node_pool" {
  name     = "tpu-benchmark-pool"
  location = var.zone_tpu
  cluster  = google_container_cluster.tpu_cluster.name

  # Explicit static node count (Autoscaling strictly disabled for apples-to-apples benchmarking)
  node_count = var.tpu_initial_node_count

  node_config {
    machine_type = var.tpu_machine_type
    disk_size_gb = 100
    disk_type    = "hyperdisk-balanced"

    service_account = google_service_account.sar_benchmark_sa.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    labels = {
      "workload-type"  = "sar-formation"
      "benchmark-tier" = "tpu-accelerator"
      "tpu-generation" = "v7-trillium"
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = false # Prevent mid-benchmark surprise node upgrades
  }

  lifecycle {
    ignore_changes = [
      # Allows manual scaling scripts (gcloud resize) to manage node count dynamically without Terraform drift conflicts
      node_count
    ]
  }
}
