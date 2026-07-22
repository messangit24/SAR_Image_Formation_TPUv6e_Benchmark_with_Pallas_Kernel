"""BPA Coherence Optimizer for Bingham Mine 9-25 Dataset.

Tests:
1. Baseline BPA unaligned correlation.
2. Fine 2D Sub-pixel shift registration.
3. Taylor windowing SLL parameter sweep.
4. 16x Range upsampling vs 8x.
5. Frequency spectrum centering & phase calibration.
"""

import sys
import time
import numpy as np
import scipy.ndimage
import scipy.optimize
import scipy.signal.windows
import jax
import jax.numpy as jnp

import cphd_parser
import sicd_parser
import quality_evaluator

jax.config.update("jax_enable_x64", True)


def run_bpa_kernel(
    sig_jax, tx_jax, rcv_jax, srp_jax, sc0_jax, scss_jax, tropo_jax, pts_jax,
    multiplier=8, center_freq_mode="fc"
):
    num_vectors, num_samples = sig_jax.shape
    c = 299792458.0
    pad_samples = num_samples * multiplier

    mean_scss = jnp.mean(scss_jax)
    dt_sample = 1.0 / (num_samples * mean_scss)
    range_grid_res = c * dt_sample / (2.0 * multiplier)

    grid_shape = (pts_jax.shape[0], pts_jax.shape[1])
    bpa_init = jnp.zeros(grid_shape, dtype=jnp.complex128)

    def vector_step(bpa_acc, i):
        fc = sc0_jax[i] + 0.5 * (num_samples - 1) * scss_jax[i]
        
        if center_freq_mode == "fc":
            f_ref = fc
        elif center_freq_mode == "sc0":
            f_ref = sc0_jax[i]
        else:
            f_ref = fc

        pulse_signal = sig_jax[i, :] * jnp.exp(-2j * jnp.pi * f_ref * tropo_jax[i])

        # 1D Range IFFT
        range_profile = jnp.fft.fftshift(jnp.fft.ifft(pulse_signal, n=pad_samples))

        tx_i, rcv_i, srp_i = tx_jax[i], rcv_jax[i], srp_jax[i]
        r_srp = 0.5 * (jnp.linalg.norm(tx_i - srp_i) + jnp.linalg.norm(rcv_i - srp_i))

        dist_tx = jnp.linalg.norm(pts_jax - tx_i[None, None, :], axis=-1)
        dist_rcv = jnp.linalg.norm(pts_jax - rcv_i[None, None, :], axis=-1)
        r_pixel = 0.5 * (dist_tx + dist_rcv)

        delta_r = r_pixel - r_srp
        bin_idx = (pad_samples // 2) + (delta_r / range_grid_res)

        valid_mask = (bin_idx >= 0) & (bin_idx < pad_samples - 1)
        idx0 = jnp.floor(bin_idx).astype(jnp.int32)
        frac = bin_idx - idx0

        idx0_clamped = jnp.clip(idx0, 0, pad_samples - 2)
        idx1_clamped = idx0_clamped + 1

        interp_val = (1.0 - frac) * range_profile[idx0_clamped] + frac * range_profile[idx1_clamped]
        interp_val = jnp.where(valid_mask, interp_val, 0.0 + 0.0j)

        phase_factor = jnp.exp(4j * jnp.pi * (fc / c) * delta_r)
        return bpa_acc + interp_val * phase_factor, None

    bpa_final, _ = jax.lax.scan(vector_step, bpa_init, jnp.arange(num_vectors))
    return bpa_final


def evaluate_subpixel_registration(bpa_img, ref_img):
    bpa_mag = np.abs(quality_evaluator.clean_array(bpa_img))
    ref_mag = np.abs(quality_evaluator.clean_array(ref_img))

    ref_norm = ref_mag - np.mean(ref_mag)
    ref_sq_sum = np.sum(ref_norm**2)

    # 1. Unaligned correlation
    bpa_norm = bpa_mag - np.mean(bpa_mag)
    denom = np.sqrt(ref_sq_sum * np.sum(bpa_norm**2))
    unaligned_corr = float(np.sum(ref_norm * bpa_norm) / denom) if denom > 0 else 0.0

    # 2. Coarse integer shift
    best_corr = unaligned_corr
    best_shift = (0.0, 0.0)
    for r_shift in range(-15, 16):
        for c_shift in range(-15, 16):
            shifted = scipy.ndimage.shift(bpa_mag, (r_shift, c_shift), order=3, mode='constant', cval=0.0)
            shifted_norm = shifted - np.mean(shifted)
            d = np.sqrt(ref_sq_sum * np.sum(shifted_norm**2))
            c = float(np.sum(ref_norm * shifted_norm) / d) if d > 0 else 0.0
            if c > best_corr:
                best_corr = c
                best_shift = (float(r_shift), float(c_shift))

    # 3. Fine sub-pixel shift grid search
    r_center, c_center = best_shift
    fine_step = 0.05
    for dr in np.arange(-1.0, 1.05, fine_step):
        for dc in np.arange(-1.0, 1.05, fine_step):
            r_test = r_center + dr
            c_test = c_center + dc
            shifted = scipy.ndimage.shift(bpa_mag, (r_test, c_test), order=3, mode='constant', cval=0.0)
            shifted_norm = shifted - np.mean(shifted)
            d = np.sqrt(ref_sq_sum * np.sum(shifted_norm**2))
            c = float(np.sum(ref_norm * shifted_norm) / d) if d > 0 else 0.0
            if c > best_corr:
                best_corr = c
                best_shift = (r_test, c_test)

    return unaligned_corr, best_shift, best_corr


def main():
    print("=== BPA Coherence Optimizer for Bingham Mine 9-25 ===", flush=True)
    sicd_path = "tpu_sar_data/bingham_20230925/2023-09-25-17-37-50_UMBRA-05_SICD.nitf"
    cphd_path = "tpu_sar_data/bingham_20230925/2023-09-25-17-37-50_UMBRA-05_CPHD.cphd"

    print(f"Loading reference SICD: {sicd_path}", flush=True)
    sicd = sicd_parser.parse_sicd_nitf(sicd_path)
    print(f"Loading CPHD dataset: {cphd_path}", flush=True)
    cphd = cphd_parser.load_cphd(cphd_path)

    pvp = cphd.pvp_array
    signal_memmap = cphd.signal_memmap

    tx_pos = np.column_stack((pvp["TxPos_X"], pvp["TxPos_Y"], pvp["TxPos_Z"]))
    rcv_pos = np.column_stack((pvp["RcvPos_X"], pvp["RcvPos_Y"], pvp["RcvPos_Z"]))
    tx_pos = np.where(np.isnan(tx_pos), rcv_pos, tx_pos)
    srp_pos = np.column_stack((pvp["SRPPos_X"], pvp["SRPPos_Y"], pvp["SRPPos_Z"]))
    sc0, scss = pvp["SC0"], pvp["SCSS"]
    tropo_delay = pvp["TDTropoSRP"]

    root = sicd.xml_tree.getroot()
    grid_node = sicd_parser.find_elem(root, "Grid")
    row_node = sicd_parser.find_elem(grid_node, "Row")
    col_node = sicd_parser.find_elem(grid_node, "Col")
    row_ss = float(sicd_parser.find_elem(row_node, "SS").text)
    col_ss = float(sicd_parser.find_elem(col_node, "SS").text)

    ur = sicd_parser.find_elem(row_node, "UVectECF")
    uc = sicd_parser.find_elem(col_node, "UVectECF")
    row_vec = np.array([
        float(sicd_parser.find_elem(ur, "X").text),
        float(sicd_parser.find_elem(ur, "Y").text),
        float(sicd_parser.find_elem(ur, "Z").text),
    ])
    col_vec = np.array([
        float(sicd_parser.find_elem(uc, "X").text),
        float(sicd_parser.find_elem(uc, "Y").text),
        float(sicd_parser.find_elem(uc, "Z").text),
    ])

    scp_node = sicd_parser.find_elem(root, "SCPPixel")
    center_r = int(float(sicd_parser.find_elem(scp_node, "Row").text))
    center_c = int(float(sicd_parser.find_elem(scp_node, "Col").text))
    print(f"Crop center: Row={center_r}, Col={center_c}", flush=True)

    grid_size = 512
    ref_crop = sicd.image_array[
        center_r - grid_size // 2 : center_r + grid_size // 2,
        center_c - grid_size // 2 : center_c + grid_size // 2,
    ]
    ref_crop = quality_evaluator.clean_array(ref_crop)

    # 3D ECEF Grid construction
    row_idx = np.arange(grid_size) - grid_size // 2
    col_idx = np.arange(grid_size) - grid_size // 2
    grid_r, grid_c = np.meshgrid(row_idx, col_idx, indexing="ij")
    mean_srp = np.mean(srp_pos, axis=0)
    pixel_pts_ecef = (
        mean_srp
        + grid_r[:, :, None] * row_ss * row_vec
        + grid_c[:, :, None] * col_ss * col_vec
    )

    num_pulses = len(tx_pos)
    num_samples = signal_memmap.shape[1]

    tx_j = jnp.array(tx_pos, dtype=jnp.float64)
    rcv_j = jnp.array(rcv_pos, dtype=jnp.float64)
    srp_j = jnp.array(srp_pos, dtype=jnp.float64)
    sc0_j = jnp.array(sc0, dtype=jnp.float64)
    scss_j = jnp.array(scss, dtype=jnp.float64)
    tropo_j = jnp.array(tropo_delay, dtype=jnp.float64)
    pts_j = jnp.array(pixel_pts_ecef, dtype=jnp.float64)

    chunk_size = 5000

    # Experiment Configurations to test
    experiments = [
        {"name": "Baseline (sll=35dB, mult=8)", "sll": 35, "mult": 8},
        {"name": "Taylor sll=30dB, mult=8", "sll": 30, "mult": 8},
        {"name": "Taylor sll=38dB, mult=8", "sll": 38, "mult": 8},
        {"name": "Taylor sll=40dB, mult=8", "sll": 40, "mult": 8},
        {"name": "Taylor sll=35dB, mult=16", "sll": 35, "mult": 16},
        {"name": "Taylor sll=38dB, mult=16", "sll": 38, "mult": 16},
    ]

    results = []

    for exp in experiments:
        exp_name = exp["name"]
        sll = exp["sll"]
        mult = exp["mult"]

        print(f"\n--- Experiment: {exp_name} ---", flush=True)
        range_win = scipy.signal.windows.taylor(num_samples, nbar=4, sll=sll)
        az_win = scipy.signal.windows.taylor(num_pulses, nbar=4, sll=sll)

        img_bpa = np.zeros((grid_size, grid_size), dtype=np.complex128)
        t0 = time.time()
        for start_idx in range(0, num_pulses, chunk_size):
            end_idx = min(start_idx + chunk_size, num_pulses)
            print(f"  Processing pulses {start_idx} to {end_idx}...", flush=True)
            chunk_signal = np.array(signal_memmap[start_idx:end_idx, :])
            chunk_az = az_win[start_idx:end_idx]
            chunk_signal_win = chunk_signal * range_win[None, :] * chunk_az[:, None]

            sig_j = jnp.array(chunk_signal_win, dtype=jnp.complex128)
            chunk_img = run_bpa_kernel(
                sig_j,
                tx_j[start_idx:end_idx],
                rcv_j[start_idx:end_idx],
                srp_j[start_idx:end_idx],
                sc0_j[start_idx:end_idx],
                scss_j[start_idx:end_idx],
                tropo_j[start_idx:end_idx],
                pts_j,
                multiplier=mult,
            )
            img_bpa += np.array(chunk_img)

        elapsed = time.time() - t0
        unaligned_corr, best_shift, fine_corr = evaluate_subpixel_registration(img_bpa, ref_crop)

        print(f"  Execution Time: {elapsed:.2f}s", flush=True)
        print(f"  Unaligned Correlation: {unaligned_corr:.4f}", flush=True)
        print(f"  Best Sub-pixel Shift: {best_shift}", flush=True)
        print(f"  Shift-Aligned Correlation: {fine_corr:.4f}", flush=True)

        results.append({
            "name": exp_name,
            "sll": sll,
            "mult": mult,
            "unaligned_corr": unaligned_corr,
            "shift": best_shift,
            "aligned_corr": fine_corr,
        })

    print("\n================ SUMMARY RESULTS ================", flush=True)
    print(f"{'Experiment Name':<35s} | {'Unaligned':<10s} | {'Subpixel Shift':<16s} | {'Aligned Coherence':<18s}", flush=True)
    print("-" * 88, flush=True)
    for r in results:
        shift_str = f"({r['shift'][0]:.2f}, {r['shift'][1]:.2f})"
        print(f"{r['name']:<35s} | {r['unaligned_corr']:<10.4f} | {shift_str:<16s} | {r['aligned_corr']:<18.4f}", flush=True)


if __name__ == "__main__":
    main()
