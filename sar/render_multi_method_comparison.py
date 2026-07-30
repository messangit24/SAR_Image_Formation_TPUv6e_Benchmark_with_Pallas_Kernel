import os, sys, time
import numpy as np
import matplotlib.pyplot as plt

workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, workspace_dir)
sys.path.insert(0, os.path.join(workspace_dir, "sar"))

import quality_evaluator
from cphd_reader import CPHDReader
from sicd_grid import read_sicd_grid
from bpa import backproject as backproject_cpu
from tpu_bpa import backproject_tpu
from tpu_ffbp import backproject_ffbp
from pfa import form_image as form_image_cpu
from tpu_pfa import form_image_tpu
from csa import form_image as form_image_csa
from rma import form_image as form_image_rma
from sarpy.io.complex.sicd import SICDReader
from viz import sar_stretch

def main():
    sicd_path = os.path.join(workspace_dir, "tpu_sar_data/bingham_20230925/2023-09-25-17-37-50_UMBRA-05_SICD.nitf")
    cphd_path = os.path.join(workspace_dir, "tpu_sar_data/bingham_20230925/2023-09-25-17-37-50_UMBRA-05_CPHD.cphd")

    grid = read_sicd_grid(sicd_path)
    cphd_reader = CPHDReader(cphd_path)
    g = cphd_reader.geometry()
    sig = cphd_reader.read_signal()

    center_r, center_c = int(grid.scp_row), int(grid.scp_col)
    grid_size = 512

    rows = center_r + np.arange(-grid_size // 2, grid_size // 2)
    cols = center_c + np.arange(-grid_size // 2, grid_size // 2)
    RR, CC = np.meshgrid(rows, cols, indexing="ij")
    pixel_pts_ecef = grid.pixel_ecef(RR, CC)

    sr = SICDReader(sicd_path)
    sicd_img = sr[:, :]
    ref_crop = sicd_img[
        center_r - grid_size // 2 : center_r + grid_size // 2,
        center_c - grid_size // 2 : center_c + grid_size // 2,
    ]
    ref_crop = quality_evaluator.clean_array(ref_crop)
    gt_mag = np.abs(ref_crop)

    # Use all 14,479 synthetic aperture pulses
    pulse_step = 1
    sig_sub = sig[::pulse_step]
    apc_sub = g["apc"][::pulse_step]
    srp_sub = g["srp"][::pulse_step]
    sc0_sub = g["sc0"][::pulse_step]
    scss_sub = g["scss"][::pulse_step]

    print(f"Executing reconstructions over {sig_sub.shape[0]} full-aperture pulses...")

    # 1. TPU BPA (all 14,479 pulses)
    print("1. Computing JAX TPU BPA...")
    t0 = time.time()
    _ = backproject_tpu(sig_sub[:10], apc_sub[:10], srp_sub[:10], sc0_sub[:10], scss_sub[:10], pixel_pts_ecef, sgn=1, upsample=4)
    img_tpu_bpa = quality_evaluator.clean_array(backproject_tpu(sig_sub, apc_sub, srp_sub, sc0_sub, scss_sub, pixel_pts_ecef, sgn=1, upsample=4))
    t_tpu_bpa = time.time() - t0
    coh_tpu_bpa = quality_evaluator.compute_complex_coherence(gt_mag, np.abs(img_tpu_bpa))

    # 2. TPU FFBP (all 14,479 pulses)
    print("2. Computing JAX TPU Fast FFBP...")
    t0 = time.time()
    _ = backproject_ffbp(sig_sub[:10], apc_sub[:10], srp_sub[:10], sc0_sub[:10], scss_sub[:10], pixel_pts_ecef, sgn=1, sub_aperture_size=16, upsample=4)
    img_tpu_ffbp = quality_evaluator.clean_array(backproject_ffbp(sig_sub, apc_sub, srp_sub, sc0_sub, scss_sub, pixel_pts_ecef, sgn=1, sub_aperture_size=16, upsample=4))
    t_tpu_ffbp = time.time() - t0
    coh_tpu_ffbp = quality_evaluator.compute_complex_coherence(gt_mag, np.abs(img_tpu_ffbp))

    # 3. CPU BPA
    print("3. Computing CPU BPA...")
    t0 = time.time()
    img_cpu_bpa = quality_evaluator.clean_array(backproject_cpu(
        sig_sub, apc_sub, srp_sub, sc0_sub, scss_sub,
        pixel_pts_ecef, grid.uRow, grid.uCol, grid.ss_row, grid.ss_col, sgn=1, upsample=4, progress=False
    ).image)
    t_cpu_bpa = time.time() - t0
    coh_cpu_bpa = quality_evaluator.compute_complex_coherence(gt_mag, np.abs(img_cpu_bpa))

    # 4. CPU PFA
    print("4. Computing CPU PFA...")
    t0 = time.time()
    res_cpu_pfa = form_image_cpu(
        sig_sub, apc_sub, srp_sub, sc0_sub, scss_sub,
        cphd_reader.meta.uiax, cphd_reader.meta.uiay,
        sgn=1, frame=(grid.uRow, grid.uCol), interp="cubic", verbose=False
    )
    img_cpu_pfa = quality_evaluator.clean_array(res_cpu_pfa.image)
    pr, pc = img_cpu_pfa.shape[0] // 2, img_cpu_pfa.shape[1] // 2
    img_cpu_pfa_crop = np.rot90(img_cpu_pfa[pr - grid_size // 2 : pr + grid_size // 2, pc - grid_size // 2 : pc + grid_size // 2], 3)
    t_cpu_pfa = time.time() - t0
    coh_cpu_pfa = quality_evaluator.compute_complex_coherence(gt_mag, np.abs(img_cpu_pfa_crop))

    # 5. TPU PFA
    print("5. Computing JAX TPU PFA...")
    t0 = time.time()
    _ = form_image_tpu(sig_sub[:10], apc_sub[:10], srp_sub[:10], sc0_sub[:10], scss_sub[:10], grid.uRow, grid.uCol, sgn=1)
    img_tpu_pfa = quality_evaluator.clean_array(form_image_tpu(sig_sub, apc_sub, srp_sub, sc0_sub, scss_sub, grid.uRow, grid.uCol, sgn=1))
    pr, pc = img_tpu_pfa.shape[0] // 2, img_tpu_pfa.shape[1] // 2
    img_tpu_pfa_crop = np.rot90(img_tpu_pfa[pr - grid_size // 2 : pr + grid_size // 2, pc - grid_size // 2 : pc + grid_size // 2], 3)
    t_tpu_pfa = time.time() - t0
    coh_tpu_pfa = quality_evaluator.compute_complex_coherence(gt_mag, np.abs(img_tpu_pfa_crop))

    # 6. CPU CSA
    print("6. Computing CPU CSA...")
    t0 = time.time()
    res_cpu_csa = form_image_csa(sig_sub, apc_sub, srp_sub, sc0_sub, scss_sub, sgn=1, verbose=False)
    img_cpu_csa = quality_evaluator.clean_array(res_cpu_csa.image)
    pr, pc = img_cpu_csa.shape[0] // 2, img_cpu_csa.shape[1] // 2
    img_cpu_csa_crop = img_cpu_csa[pr - grid_size // 2 : pr + grid_size // 2, pc - grid_size // 2 : pc + grid_size // 2].T
    t_cpu_csa = time.time() - t0
    coh_cpu_csa = quality_evaluator.compute_complex_coherence(gt_mag, np.abs(img_cpu_csa_crop))

    # 7. CPU RMA
    print("7. Computing CPU RMA...")
    t0 = time.time()
    res_cpu_rma = form_image_rma(sig_sub, apc_sub, srp_sub, sc0_sub, scss_sub, sgn=1, verbose=False)
    img_cpu_rma = quality_evaluator.clean_array(res_cpu_rma.image)
    pr, pc = img_cpu_rma.shape[0] // 2, img_cpu_rma.shape[1] // 2
    img_cpu_rma_crop = img_cpu_rma[pr - grid_size // 2 : pr + grid_size // 2, pc - grid_size // 2 : pc + grid_size // 2].T
    t_cpu_rma = time.time() - t0
    coh_cpu_rma = quality_evaluator.compute_complex_coherence(gt_mag, np.abs(img_cpu_rma_crop))

    def render_mag(arr):
        mag = np.abs(quality_evaluator.clean_array(arr))
        return sar_stretch(mag, k=2.5, gamma=0.65)

    fig, axes = plt.subplots(2, 4, figsize=(24, 12), dpi=200)
    fig.suptitle(
        "TPUbench Unified SAR Image Reconstruction & CPU/TPU Method Comparison\n(Bingham Copper Mine Target Chip - All 14,479 Pulses)",
        fontsize=18,
        fontweight="bold",
        y=0.98
    )

    panels_list = [
        ("Ground Truth (SICD)", ref_crop, f"1. Ground Truth (SICD Reference)\nReference NITF Product"),
        ("CPU BPA (src/bpa.py)", img_cpu_bpa, f"2. CPU BPA (src/bpa.py)\nCoherence: {coh_cpu_bpa:.4f} | Latency: {t_cpu_bpa:.1f}s"),
        ("TPU BPA (src/tpu_bpa.py)", img_tpu_bpa, f"3. TPU BPA (src/tpu_bpa.py)\nCoherence: {coh_tpu_bpa:.4f} | Latency: {t_tpu_bpa:.1f}s"),
        ("TPU Fast FFBP (src/tpu_ffbp.py)", img_tpu_ffbp, f"4. TPU Fast FFBP (src/tpu_ffbp.py)\nCoherence: {coh_tpu_ffbp:.4f} | Latency: {t_tpu_ffbp:.1f}s"),
        ("CPU PFA (src/pfa.py)", img_cpu_pfa_crop, f"5. CPU PFA (src/pfa.py)\nCoherence: {coh_cpu_pfa:.4f} | Latency: {t_cpu_pfa:.1f}s"),
        ("TPU PFA (src/tpu_pfa.py)", img_tpu_pfa_crop, f"6. TPU PFA (src/tpu_pfa.py)\nCoherence: {coh_tpu_pfa:.4f} | Latency: {t_tpu_pfa:.1f}s"),
        ("CPU CSA (src/csa.py)", img_cpu_csa_crop, f"7. CPU CSA (src/csa.py)\nCoherence: {coh_cpu_csa:.4f} | Latency: {t_cpu_csa:.1f}s"),
        ("CPU RMA (src/rma.py)", img_cpu_rma_crop, f"8. CPU RMA (src/rma.py)\nCoherence: {coh_cpu_rma:.4f} | Latency: {t_cpu_rma:.1f}s"),
    ]

    for idx, (title, img_arr, label_str) in enumerate(panels_list):
        r, c = idx // 4, idx % 4
        ax = axes[r, c]
        disp = render_mag(img_arr)
        ax.imshow(disp, cmap="gray", origin="lower")
        ax.set_title(label_str, fontsize=13, fontweight="bold", pad=8)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = os.path.join(workspace_dir, "out")
    reports_dir = os.path.join(workspace_dir, "reports")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    out_path1 = os.path.join(out_dir, "multi_method_comparison.png")
    out_path2 = os.path.join(reports_dir, "multi_method_comparison.png")
    plt.savefig(out_path1, bbox_inches="tight", dpi=200)
    plt.savefig(out_path2, bbox_inches="tight", dpi=200)
    plt.close()

    print(f"Saved figure to:\n  - {out_path1}\n  - {out_path2}")

if __name__ == "__main__":
    main()
