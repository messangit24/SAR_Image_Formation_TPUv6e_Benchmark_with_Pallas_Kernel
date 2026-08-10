variable "project_id" {
  description = "The Google Cloud Platform project ID."
  type        = string
  default     = "to2-goldthread"
}

variable "region" {
  description = "The GCP region for the benchmark infrastructure (e.g., us-central2 or us-central1)."
  type        = string
  default     = "us-central1"
}

variable "zone_tpu" {
  description = "The GCP zone for the TPU GKE cluster and node pool (must support target TPU generation)."
  type        = string
  default     = "us-central1-c"
}

variable "zone_cpu" {
  description = "The GCP zone for the CPU GKE cluster and node pool."
  type        = string
  default     = "us-central1-c"
}

variable "network_name" {
  description = "Name of the dedicated VPC network for isolated benchmark communication."
  type        = string
  default     = "sar-benchmark-vpc"
}

variable "subnet_cidr" {
  description = "Primary CIDR block for the benchmark subnet."
  type        = string
  default     = "10.10.0.0/20"
}

variable "pods_cidr_tpu" {
  description = "Secondary CIDR range for TPU cluster pods."
  type        = string
  default     = "10.20.0.0/16"
}

variable "services_cidr_tpu" {
  description = "Secondary CIDR range for TPU cluster services."
  type        = string
  default     = "10.30.0.0/20"
}

variable "pods_cidr_cpu" {
  description = "Secondary CIDR range for CPU cluster pods."
  type        = string
  default     = "10.40.0.0/16"
}

variable "services_cidr_cpu" {
  description = "Secondary CIDR range for CPU cluster services."
  type        = string
  default     = "10.50.0.0/20"
}

variable "tpu_cluster_name" {
  description = "Name of the dedicated TPU GKE cluster."
  type        = string
  default     = "gke-sar-tpu-cluster"
}

variable "cpu_cluster_name" {
  description = "Name of the dedicated CPU GKE cluster."
  type        = string
  default     = "gke-sar-cpu-cluster"
}

variable "tpu_machine_type" {
  description = "TPU VM machine type for GKE (e.g., tpu7x-standard-4t for Trillium v7."
  type        = string
  default     = "tpu7x-standard-4t"
}

variable "cpu_machine_type" {
  description = "Compute-optimized CPU machine type (e.g., c3-standard-88 with Intel Sapphire Rapids / Emerald Rapids)."
  type        = string
  default     = "c3-standard-88"
}

variable "tpu_initial_node_count" {
  description = "Initial number of nodes for the TPU node pool (set to 0 for scale-to-zero safety)."
  type        = number
  default     = 0
}

variable "cpu_initial_node_count" {
  description = "Initial number of nodes for the CPU node pool (set to 0 for scale-to-zero safety)."
  type        = number
  default     = 0
}

variable "gcs_bucket_name" {
  description = "Name of the Google Cloud Storage bucket for CPHD phase history datasets and benchmark artifacts (leave blank to auto-generate)."
  type        = string
  default     = ""
}

variable "gcs_bucket_location" {
  description = "Location for the benchmark artifact GCS bucket."
  type        = string
  default     = "us-central1"
}
