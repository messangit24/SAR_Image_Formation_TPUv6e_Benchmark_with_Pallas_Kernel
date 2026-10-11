"""
Root entrypoint for 3TB Catalog Batch Benchmark on 16 Cloud TPU v6e Chips.
Executes multi-chip Pallas FFBP and BPA kernels on a calibrated ~3TB subset
(120 scenes) of the Umbra open SAR archive.
"""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "sar"))

from src.sar.run_3tb_catalog_batch_benchmark import main

if __name__ == "__main__":
    main()
