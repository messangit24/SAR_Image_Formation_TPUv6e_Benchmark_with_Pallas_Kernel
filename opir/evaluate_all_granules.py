#!/usr/bin/env python3
"""
Multi-Granule Evaluation & Generalization Benchmark for OPIR Pipeline
"""

import os
import glob
import numpy as np
from download_and_process_granule import process_granule

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def run_multi_granule_eval():
    print("=" * 90)
    print("        OPIR PIPELINE MULTI-GRANULE GENERALIZATION EVALUATION")
    print("=" * 90)

    # Identify matching triplet files in DATA_DIR
    m12_files = sorted(glob.glob(os.path.join(DATA_DIR, "SVM12_*.h5")))
    
    granules = []
    for m12_path in m12_files:
        base = os.path.basename(m12_path).replace('.h5', '')
        # Extract timestamp string e.g. d20211202_t1438358 or d20260729_t0001036
        parts = base.split('_')
        d_stamp = [p for p in parts if p.startswith('d20')][0]
        t_stamp = [p for p in parts if p.startswith('t0') or p.startswith('t1') or p.startswith('t2')][0]
        stamp = f"{d_stamp}_{t_stamp}"

        # Find matching M15 and GMTCO files
        m15_candidates = glob.glob(os.path.join(DATA_DIR, f"SVM15_*{stamp}*.h5"))
        geo_candidates = glob.glob(os.path.join(DATA_DIR, f"GMTCO_*{stamp}*.h5"))

        if m15_candidates and geo_candidates:
            granules.append((stamp, m12_path, m15_candidates[0], geo_candidates[0]))

    print(f"\nFound {len(granules)} complete satellite granules for multi-image benchmark evaluation:")
    for idx, (stamp, m12, m15, geo) in enumerate(granules):
        print(f"  Granule #{idx+1}: Stamp {stamp}")

    summary_metrics = []
    for idx, (stamp, m12, m15, geo) in enumerate(granules):
        plot_path = os.path.join(DATA_DIR, f"eval_plot_{stamp}.png")
        res = process_granule(m12, m15, geo, output_plot_path=plot_path)
        res['stamp'] = stamp
        summary_metrics.append(res)

    print("\n" + "=" * 90)
    print("                  MULTI-GRANULE EVALUATION SUMMARY TABLE")
    print("=" * 90)
    header = f"{'Granule Stamp':<22} | {'Valid Pixels':<12} | {'Mono MAE (°C)':<14} | {'SRF MAE (°C)':<14} | {'SRF RMSE (°C)':<14} | {'Pearson r':<14} | {'Error Reduction':<15}"
    print(header)
    print("-" * len(header))
    
    for m in summary_metrics:
        ratio = m['mae_mono'] / m['mae_srf']
        row = f"{m['stamp']:<22} | {m['pixels']:<12,} | {m['mae_mono']:<14.6f} | {m['mae_srf']:<14.6f} | {m['rmse_srf']:<14.6f} | {m['r_srf']:<14.10f} | {ratio:<15.1f}x"
        print(row)
    print("=" * 90)

if __name__ == '__main__':
    run_multi_granule_eval()
