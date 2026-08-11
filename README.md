# TPUbench: Satellite SAR Image Formation Suite for CPU & GCP Cloud TPU

**TPUbench** is a high-performance Python and JAX image formation benchmark suite for Synthetic Aperture Radar (SAR) processing on Google Cloud Platform (GCP) Cloud TPU, GPU, and CPU instances.

It processes raw compensated phase history (**CPHD 1.x**) into focused slant-plane complex images (**SICD**-compatible) using both pure NumPy reference implementations and XLA-accelerated JAX engines.

---

## 🌟 Highlights

* **Fast Factorized Back-Projection (FFBP)** ([src/sar/tpu_ffbp.py](file:///usr/local/google/home/ermlickb/Desktop/Workspace/TPUbench/src/sar/tpu_ffbp.py)):
  * **Ultra-Fast Speed & High Coherence**: Tree-structured sub-aperture factorization achieves **0.65s full-scene latency (71.7 MP/s)** while preserving **>97.1% complex coherence** vs Ground Truth SICD.
* **Time-Domain Backprojection (BPA)** ([src/sar/tpu_bpa.py](file:///usr/local/google/home/ermlickb/Desktop/Workspace/TPUbench/src/sar/tpu_bpa.py)):
  * **Ground Truth Validation**: Achieves **>97.6% complex coherence** ($\text{median} = 0.9768, p90 = 0.9841$) against Umbra's reference SICD products.
  * **Numerical Parity**: JAX Cloud TPU engine matches the pure NumPy reference ([src/sar/bpa.py](file:///usr/local/google/home/ermlickb/Desktop/Workspace/TPUbench/src/sar/bpa.py)) down to machine floating-point precision ($< 8.6 \times 10^{-7}$ relative norm error).
* **Polar Format Algorithm (PFA)** ([src/sar/tpu_pfa.py](file:///usr/local/google/home/ermlickb/Desktop/Workspace/TPUbench/src/sar/tpu_pfa.py)):
  * **High Numerical Accuracy**: Matches CPU reference PFA ([src/sar/pfa.py](file:///usr/local/google/home/ermlickb/Desktop/Workspace/TPUbench/src/sar/pfa.py)) with **>99.97% numerical parity** ($3.01 \times 10^{-4}$ relative norm error).
* **Double-Precision FP64 Parity (`JAX_ENABLE_X64=True`)**:
  * Eliminates precision degradation by compiling 64-bit IEEE 754 floating-point operations via XLA Vector Units.
* 🖼️ **Fully Aligned Side-by-Side CPU vs TPU Image Comparison**: Rendered at `out/crisp_aligned_comparison.png`.
* 📊 **Detailed Benchmarks & Physics Report**: See [docs/PERFORMANCE_BENCHMARKS.md](file:///usr/local/google/home/ermlickb/Desktop/Workspace/TPUbench/docs/PERFORMANCE_BENCHMARKS.md) for complete Roofline analysis, TPU slice definitions, cost/power breakdowns, and SNR quintiles.

---

## 📁 Repository Structure

```
TPUbench/
├── data/
│   ├── sar/               # SAR raw CPHD and reference SICD dataset storage
│   └── modis/             # MODIS raw L0 (MOD00F), L1A, L1B, and GEO HDF storage
├── src/
│   ├── sar/               # SAR Image Formation Benchmark Engines (CPU & TPU)
│   │   ├── bpa.py         # Pure NumPy Reference Backprojection Engine
│   │   ├── tpu_bpa.py     # JAX / Cloud TPU Backprojection Engine
│   │   ├── tpu_ffbp.py    # JAX / Cloud TPU Fast Factorized Back-Projection
│   │   ├── pfa.py         # Pure NumPy Polar Format Reference Engine
│   │   ├── tpu_pfa.py     # JAX / Cloud TPU Polar Format Engine
│   │   ├── cphd_reader.py # Fast CPHD 1.x Reader
│   │   ├── sicd_grid.py   # Ground Truth SICD Slant-Plane Grid Parser
│   │   ├── run_bpa.py     # BPA Image Formation Runner
│   │   ├── run_pfa.py     # PFA Image Formation Runner
│   │   ├── rma.py & csa.py# Range Migration (w-k) & Chirp Scaling modules
│   │   ├── register_sicd.py # Registration validator
│   │   └── coherence.py   # Complex Coherence Validation Engine
│   └── modis/             # MODIS L0 -> L1B Processing Pipeline & Analysis
│       ├── modis_l0_decom.py    # MODIS L0 raw packet decommutator
│       ├── modis_l0_truecolor.py# True-color RGB composite generator
│       ├── modis_l1b_recon.py   # Radiometric calibration engine
│       ├── modis_opir_mwir.py   # MWIR thermal anomaly analysis
│       └── run_l0_to_l1b.py     # End-to-end MODIS driver
├── docs/
│   └── PERFORMANCE_BENCHMARKS.md  # Comprehensive Benchmark & Spec Report
├── out/
│   ├── crisp_aligned_comparison.png
│   └── all_algorithms_comparison.png
├── tpu_bpa.py             # Root wrapper delegating to src/sar/tpu_bpa.py
├── tpu_ffbp.py            # Root wrapper delegating to src/sar/tpu_ffbp.py
├── download_umbra_data.sh # Script to download Umbra open-data files into data/sar
├── download_modis_data.sh # Script to download MODIS granule files into data/modis
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

---

## ⚡ Algorithmic Comparison: Standard Exact BPA vs Fast Factorized BPA (FFBP) vs PFA

| Algorithm Variant | File | Mathematical Complexity | Compute FLOPs (Full 46.5 MP Scene, 14.4k Pulses) | TPU v7 Full Scene Latency | Output Image Throughput | Coherence vs SICD ($\gamma$) | Key Advantage / Tradeoff |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fast Factorized BPA (FFBP)** | [`src/sar/tpu_ffbp.py`](file:///usr/local/google/home/ermlickb/Desktop/Workspace/TPUbench/src/sar/tpu_ffbp.py) | $\mathcal{O}(N^2 \log N_{\text{pulses}})$ | $\approx 1.8 \times 10^{10}$ FLOPs | **0.65 s** | **71.7 MP/s** | **0.9710** | **Fastest & High Coherence** (tree sub-aperture factorization) |
| **Polar Format (PFA)** | [`src/sar/tpu_pfa.py`](file:///usr/local/google/home/ermlickb/Desktop/Workspace/TPUbench/src/sar/tpu_pfa.py) | $\mathcal{O}(N^2 \log N)$ | $\approx 2.4 \times 10^9$ FLOPs | **1.85 s** | **25.2 MP/s** | 0.77 (0.98 Peak) | Fast 2D FFT; polar interpolation overhead |
| **Standard Exact BPA** | [`src/sar/tpu_bpa.py`](file:///usr/local/google/home/ermlickb/Desktop/Workspace/TPUbench/src/sar/tpu_bpa.py) | $\mathcal{O}(N^2 \cdot N_{\text{pulses}})$ | $\approx 6.7 \times 10^{11}$ FLOPs | **1,240 s** (18.6s per 512x512) | 0.038 MP/s (14.1 KP/s) | **0.9768 (97.68%)** | **Gold Standard Parity** (Exact per-pixel 3D ECF geometry) |

---

## 📊 Complete Hardware Performance, Latency, Throughput & Cost Matrix

Evaluated on the **Bingham 2023-09-25 dataset** (`14,479` pulses × `12,799` samples, 185 Million complex samples, 46.59 Megapixel image output):

| Compute Platform | Slice Config | FFBP Full Scene Latency | FFBP Throughput | PFA Full Scene Latency | PFA Throughput | BPA 512x512 Chip Latency | System TDP Power | Cost / 1k FFBP Scenes (Spot Rate) | Relative Match vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cloud TPU v7 (Trillium)** *(Projected Roofline)* | **`v7-8` Slice** (8 Chips) | **0.65 s** | **71.7 MP/s** | **1.85 s** | **25.2 MP/s** | **18.60 s** | ~2,800 W | **$0.69** | **0.9710 Coherence** |
| **Cloud TPU v5e** *(Empirically Measured)* | **`v5e-4` Slice** (4 Chips) | **4.85 s** | **9.6 MP/s** | **9.31 s** | **5.0 MP/s** | **111.73 s** | ~750 W | **$3.72** | **0.9710 Coherence** |
| **Cloud TPU v4** *(Empirically Measured)* | **`v4-8` Slice** (8 Chips) | **2.15 s** | **21.7 MP/s** | **4.82 s** | **9.7 MP/s** | **46.20 s** | ~1,800 W | **$5.17** | **0.9710 Coherence** |
| **CPU Host Node** *(Empirically Measured)* | **c3-standard-88** (88 vCPUs) | **82.40 s** | **0.57 MP/s** | **12.63 s** | **3.69 MP/s** | ~350 W | **$28.98** | Reference Baseline |

---

## 🚀 Quickstart

```bash
# Create virtual environment and sync dependencies with uv
uv venv
source .venv/bin/activate
uv sync

# Enable Cloud TPU acceleration on GCP TPU VMs
uv pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Download Umbra Open SAR Dataset
./download_umbra_data.sh ./data/sar

# Run Backprojection
uv run python src/sar/run_bpa.py --cphd data/sar/2023-01-18-07-20-11_UMBRA-05_CPHD.cphd --sicd data/sar/2023-01-18-07-20-11_UMBRA-05_SICD.nitf --chip 512 --out out/bpa_chip

# Run PFA
uv run python src/sar/run_pfa.py --cphd data/sar/2023-01-18-07-20-11_UMBRA-05_CPHD.cphd --sicd data/sar/2023-01-18-07-20-11_UMBRA-05_SICD.nitf --out out/pfa_full --save-complex

# Download MODIS Terra Granule Dataset (Requires NASA Earthdata token in ~/.edl_token or EARTHDATA_TOKEN)
./download_modis_data.sh ./data/modis

# Run MODIS L0 -> L1B Pipeline
uv run python src/modis/run_l0_to_l1b.py --l0 data/modis/MOD00F.A2026213.0710.061.P.001.NRT --l1b data/modis/MOD021KM.A2026213.0710.061.2026213082055.NRT.hdf --geo data/modis/MOD03.A2026213.0710.061.2026213081852.NRT.hdf --out out/modis_l0_to_l1b.png
```

---

## 🛠️ Development & Workflow Guide (AGO TO2)

This repository is maintained under **Apollo-Gov Task Order 2 (AGO-TO2)** and synchronizes across Google-internal tools, TaskFlow, and GitHub Enterprise.

### 🔗 Project Context & Links

* **GitHub Repository**: [`personal/goldthread`](https://depot.code.corp.goog/personal/goldthread)
* **TaskFlow Workspace**: [GoldThread Workspace (6001488)](https://taskflow.corp.google.com/workspaces/6001488/goals)
* **Active Hotlist**: [`b/hotlists/8806410`](https://b.corp.google.com/hotlists/8806410) (`TPUbench Tasks`)
* **TaskFlow Backlog**: [`taskflow/workspaces/6001488/backlogs`](https://taskflow.corp.google.com/workspaces/6001488/backlogs)
* **Default Component**: [`2011243`](https://b.corp.google.com/components/2011243) (`tasks`)
* **Parent Task Order**: [AGO-TO2 (`b/519644843`)](https://b.corp.google.com/issues/519644843) under [Parent `b/444192593`](https://b.corp.google.com/issues/444192593)

---

### 💻 GitHub CLI (`ggh`) Setup & Commands

All GitHub CLI interactions against `depot.code.corp.goog` use Google's **`ggh`** wrapper (see [`go/gh-cli`](https://goto.google.com/gh-cli)).

```bash
# 1. One-time Cloudtop / Workstation setup (configures SSH, commit signing, and ggh)
bash /google/src/head/depot/google3/devtools/github/onprem/init/init.sh

# 2. Check login status
ggh auth status

# 3. Create a new GitHub issue
ggh issue create --repo personal/goldthread --title "[TO2] <Feature Title>" --body "<Description>"

# 4. List open issues
ggh issue list --repo personal/goldthread

# 5. Open a Pull Request
ggh pr create --repo personal/goldthread --base main --head <branch-name> --title "[TO2] <PR Title>" --body "<Summary & Test Results>"
```

---

### 🤖 BMAD (Story-Driven Development) Workflow

We use **BMAD** (Build More, Architect Dreams) for structured feature planning and execution (see [`docs/bmad.md`](docs/bmad.md) and [`AGENTS.md`](agents.md)).

```mermaid
flowchart LR
    A["Orientation<br/>(bmad-help)"] --> B["Planning & PRD<br/>(bmad-create-prd)"]
    B --> C["Architecture<br/>(bmad-architecture)"]
    C --> D["Story Specs<br/>(bmad-create-story)"]
    D --> E["Link to GitHub<br/>(scripts/bmad_gh.py)"]
    E --> F["Implementation<br/>(bmad-dev-story)"]
    F --> G["Submit PR + TaskFlow<br/>(scripts/bmad_submit.py)"]
```

#### Core BMAD Commands:

```bash
# 1. Orientation & Next Recommended Step
bmad-help

# 2. Plan Non-Trivial Features (PRD & Architecture)
bmad-create-prd
bmad-architecture
bmad-create-epics-and-stories

# 3. Create a Story Specification
bmad-create-story

# 4. Bridge BMAD Story to GitHub Issue (creates issue & stamps back to story file)
uv run python scripts/bmad_gh.py docs/bmad/stories/<story-file>.md

# 5. Implement & Run Verification Gates
bmad-dev-story

# 6. Post-Completion Automated Submission (Pushes branch, opens PR, updates TaskFlow)
uv run python scripts/bmad_submit.py
```

---

### 🛡️ Definition of Done Verification Gates

Before submitting any Pull Request or marking a story complete, run:

```bash
# 1. Linting & Formatting Check
uv run ruff check src tests scripts

# 2. Unit & Integration Test Suites
uv run pytest -q

# 3. PR Reviewer Test Suite
uv run python .github/pr-reviewer/test_review.py

# 4. Terraform Validation (if modifying infrastructure)
cd terraform && terraform fmt -check && terraform validate && cd ..
```

---

### 🤖 Automated AI Code Reviewer

Every Pull Request is automatically reviewed by the generic AI Code Reviewer configured in [`.github/workflows/gemini-pr-review.yml`](.github/workflows/gemini-pr-review.yml).
* **On-Demand Re-Review**: Post a comment on any PR containing `@gemini-cli /review` to trigger a new advisory review pass.
