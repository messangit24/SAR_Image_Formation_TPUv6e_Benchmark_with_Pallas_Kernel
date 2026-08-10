# GCP Provider Configuration
provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Enable required GCP APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "container.googleapis.com",
    "compute.googleapis.com",
    "storage.googleapis.com",
    "tpu.googleapis.com",
    "iam.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com"
  ])
  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

# Dedicated VPC for strict benchmark network isolation
resource "google_compute_network" "benchmark_vpc" {
  name                    = var.network_name
  auto_create_subnetworks = false
  description             = "Dedicated VPC for SAR CPU vs TPU performance benchmark clusters"
  depends_on              = [google_project_service.required_apis]
}

# Primary Subnet with Secondary Ranges for TPU and CPU Pods/Services
resource "google_compute_subnetwork" "benchmark_subnet" {
  name                     = "${var.network_name}-subnet"
  ip_cidr_range            = var.subnet_cidr
  region                   = var.region
  network                  = google_compute_network.benchmark_vpc.id
  private_ip_google_access = true

  secondary_ip_range {
    range_name    = "tpu-pods"
    ip_cidr_range = var.pods_cidr_tpu
  }

  secondary_ip_range {
    range_name    = "tpu-services"
    ip_cidr_range = var.services_cidr_tpu
  }

  secondary_ip_range {
    range_name    = "cpu-pods"
    ip_cidr_range = var.pods_cidr_cpu
  }

  secondary_ip_range {
    range_name    = "cpu-services"
    ip_cidr_range = var.services_cidr_cpu
  }
}

# Cloud Router and NAT for outbound container image and dependency fetching
resource "google_compute_router" "nat_router" {
  name    = "${var.network_name}-router"
  region  = var.region
  network = google_compute_network.benchmark_vpc.id
}

resource "google_compute_router_nat" "nat_gateway" {
  name                               = "${var.network_name}-nat"
  router                             = google_compute_router.nat_router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}
