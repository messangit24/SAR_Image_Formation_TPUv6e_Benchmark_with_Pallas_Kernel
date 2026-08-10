# Random suffix for unique storage bucket naming
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# Cloud Storage Bucket for SAR datasets (CPHD raw phase history & SICD products) and benchmark artifacts
resource "google_storage_bucket" "sar_dataset_bucket" {
  name                        = var.gcs_bucket_name != "" ? var.gcs_bucket_name : "sar-benchmark-data-${var.project_id}-${random_id.bucket_suffix.hex}"
  location                    = var.gcs_bucket_location
  force_destroy               = false
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = false
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
}
