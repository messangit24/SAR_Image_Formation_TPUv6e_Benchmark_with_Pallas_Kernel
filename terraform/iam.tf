# Random suffix for unique storage bucket naming
# resource "random_id" "bucket_suffix" {
#   byte_length = 4
# }

# Cloud Storage Bucket for SAR datasets (CPHD raw phase history & SICD products) and benchmark artifacts
# resource "google_storage_bucket" "sar_dataset_bucket" {
#   name                        = var.gcs_bucket_name != "" ? var.gcs_bucket_name : "sar-benchmark-data-${var.project_id}-${random_id.bucket_suffix.hex}"
#   location                    = var.gcs_bucket_location
#   force_destroy               = false
#   uniform_bucket_level_access = true

#   versioning {
#     enabled = false
#   }

#   lifecycle_rule {
#     condition {
#       age = 90
#     }
#     action {
#       type = "Delete"
#     }
#   }
# }

# Dedicated Service Account for SAR Benchmark Pods (Workload Identity)
resource "google_service_account" "sar_benchmark_sa" {
  account_id   = "sar-benchmark-runner"
  display_name = "SAR Benchmark Runner Service Account"
  description  = "Service account for SAR CPU/TPU benchmarking pods in GKE"
}

# Grant Storage Object Admin on the benchmark bucket to the Service Account
# resource "google_storage_bucket_iam_member" "sa_storage_admin" {
#   bucket = google_storage_bucket.sar_dataset_bucket.name
#   role   = "roles/storage.objectAdmin"
#   member = "serviceAccount:${google_service_account.sar_benchmark_sa.email}"
# }

# Grant Logging and Monitoring Metrics permissions
resource "google_project_iam_member" "sa_metric_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.sar_benchmark_sa.email}"
}

resource "google_project_iam_member" "sa_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.sar_benchmark_sa.email}"
}

# Workload Identity IAM bindings for K8s Service Account in both clusters
resource "google_service_account_iam_member" "wi_tpu_binding" {
  service_account_id = google_service_account.sar_benchmark_sa.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[default/sar-benchmark-sa]"
}
