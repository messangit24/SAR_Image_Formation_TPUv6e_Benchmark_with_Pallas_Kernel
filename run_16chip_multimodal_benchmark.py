"""
Root entrypoint for 16-Chip Cloud TPU v6e Multimodal Benchmark Suite.
Dispatches Real Umbra SAR CPHD/SICD Scaled Image Formation and MODIS Thermal Inversion.
"""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "sar"))

from src.sar.run_16chip_multimodal_benchmark import main

if __name__ == "__main__":
    main()
