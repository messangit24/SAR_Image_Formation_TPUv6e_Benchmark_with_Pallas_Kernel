output "project_id" {
  description = "The GCP Project ID."
  value       = var.project_id
}

output "region" {
  description = "GCP Region."
  value       = var.region
}

# output "sar_storage_bucket" {
#   description = "Cloud Storage bucket URI for SAR datasets (CPHD) and benchmark artifacts."
#   value       = "gs://${google_storage_bucket.sar_dataset_bucket.name}"
# }

output "service_account_email" {
  description = "Service account used by the benchmark pods via Workload Identity."
  value       = google_service_account.sar_benchmark_sa.email
}

# TPU Cluster Outputs
output "tpu_cluster_name" {
  description = "Name of the dedicated TPU GKE cluster."
  value       = google_container_cluster.tpu_cluster.name
}

output "tpu_cluster_location" {
  description = "Zone of the TPU GKE cluster."
  value       = google_container_cluster.tpu_cluster.location
}

output "tpu_get_credentials_command" {
  description = "Command to authenticate kubectl to the TPU GKE cluster."
  value       = "gcloud container clusters get-credentials ${google_container_cluster.tpu_cluster.name} --zone ${google_container_cluster.tpu_cluster.location} --project ${var.project_id}"
}

# CPU Cluster Outputs
output "cpu_cluster_name" {
  description = "Name of the dedicated CPU GKE cluster."
  value       = google_container_cluster.cpu_cluster.name
}

output "cpu_cluster_location" {
  description = "Zone of the CPU GKE cluster."
  value       = google_container_cluster.cpu_cluster.location
}

output "cpu_get_credentials_command" {
  description = "Command to authenticate kubectl to the CPU GKE cluster."
  value       = "gcloud container clusters get-credentials ${google_container_cluster.cpu_cluster.name} --zone ${google_container_cluster.cpu_cluster.location} --project ${var.project_id}"
}

output "quickstart_scaling_guide" {
  description = "Quick commands for scaling the benchmark clusters to zero or spinning them up."
  value       = <<-EOT
    ================================================================================
    SAR BENCHMARK CLUSTER SCALING QUICKSTART:
    --------------------------------------------------------------------------------
    To Scale UP before running a benchmark:
      ./scripts/scale_clusters.sh up 1

    To Scale DOWN to ZERO after benchmark completion (0 node cost):
      ./scripts/scale_clusters.sh down

    To Check Cluster & Node Pool Status:
      ./scripts/scale_clusters.sh status
    ================================================================================
  EOT
}
