---
name: terraform-infra-review
description: Reviews Terraform GCP Infrastructure-as-Code for security, parameterization, and cost efficiency.
phase: review
user-invocable: true
---

# Terraform GCP Infrastructure Review

Perform a focused review on GCP Terraform modules and configurations under `terraform/`.

## Key Review Areas

### 1. Cost Optimization & Scaling
- **Zero-Node Downscaling**: Verify node pools (CPU & TPU v4/v5) can be scaled to zero nodes (`initial_node_count = 0` / variable min nodes) to avoid idle cluster costs.
- **Spot / Preemptible Options**: Check whether benchmark worker node pools support Spot VMs for cost savings.

### 2. Security & Network Isolation
- **Private Clusters**: Verify GKE control plane and nodes use private IPs behind Cloud NAT.
- **Least-Privilege IAM**: Ensure benchmark service accounts are granted only necessary permissions (`roles/storage.objectViewer`, `roles/artifactregistry.reader`, `roles/logging.logWriter`).

### 3. Parameterization & Maintainability
- Ensure no hardcoded project IDs, subnet CIDRs, or cluster zones in `.tf` resources.
- Centralize all configurations in `variables.tf` and expose key cluster outputs in `outputs.tf`.
