"""Experimental script to test and improve BPA coherence on Bingham 9-25 dataset."""

import os
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
    multiplier=8, use_freq_shift=False
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
        pulse_signal = sig_jax[i, :] * jnp.exp(-2j * jnp.pi * fc * tropo_jax[i])

        if use_freq_shift:
            # Shift frequency center (fc) to zero index before IFFT
            pad_sig = jnp.zeros(pad_samples, dtype=jnp.complex128)
            pad_sig = pad_sig.at[0:num_samples].set(pulse_signal)
            # Center the signal spectrum in the padded array
            pad_sig = jnp.fft.fftshift(pad_sig)
            range_profile = jnp.fft.ifft(pad_sig)
        else:
            # Standard implementation in tpu_bpa.py
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


def find_subpixel_shift(bpa_mag, ref_mag):
    ref_norm = ref_mag - np.mean(ref_mag)
    ref_sq_sum = np.sum(ref_norm**2)
    
    # 1. Coarse integer search
    best_corr = -1.0
    best_shift = (0.0, 0.0)
    for r_shift in range(-15, 16):
        for c_shift in range(-15, 16):
            shifted = scipy.ndimage.shift(bpa_mag, (r_shift, c_shift), order=3, mode='constant', cval=0.0)
            shifted_norm = shifted - np.mean(shifted)
            denom = np.sqrt(ref_sq_sum * np.sum(shifted_norm**2))
            corr = np.sum(ref_norm * shifted_norm) / denom if denom > 0 else 0.0
            if corr > best_corr:
                best_corr = corr
                best_shift = (float(r_shift), float(c_shift))

    # 2. Fine sub-pixel search around best integer shift
    r_best, c_best = best_shift
    fine_step = 0.1
    for r_sub in np.arange(-1.0, 1.05, fine_step):
        for c_sub in np.arange(-1.0, 1.05, fine_step):
            r_test = r_best + r_sub
            c_test = c_best + c_sub
            shifted = scipy.ndimage.shift(bpa_mag, (r_test, c_test), order=3, mode='constant', cval=0.0)
            shifted_norm = shifted - np.mean(shifted)
            denom = np.sqrt(ref_sq_sum * np.sum(shifted_norm**2))
            corr = np.sum(ref_norm * shifted_norm) / denom if denom > 0 else 0.0
            if corr > best_corr:
                best_corr = corr
                best_shift = (r_test, c_test)

    return best_shift, best_corr


def main():
    sicd_path = "tpu_sar_data/bingham_20230925/2023-09-25-17-37-50_UMBRA-05_SICD.nitf"
    cphd_path = "tpu_sar_data/bingham_20230925/2023-09-25-17-37-50_UMBRA-05_CPHD.cphd"

    print(f"Loading reference SICD from {sicd_path}...")
    sicd = sicd_parser.parse_sicd_nitf(sicd_path)
    print(f"Loading CPHD dataset from {cphd_path}...")
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
    print(f"Crop center: Row={center_r}, Col={center_c}")

    grid_size = 512
    ref_crop = sicd.image_array[
        center_r - grid_size // 2 : center_r + grid_size // 2,
        center_c - grid_size // 2 : center_c + grid_size // 2,
    ]
    ref_crop = quality_evaluator.clean_array(ref_crop)
    ref_mag = np.abs(ref_crop)

    # Construct 3D ECEF grid
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

    # Convert JAX inputs once
    tx_j = jnp.array(tx_pos, dtype=jnp.float64)
    rcv_j = jnp.array(rcv_pos, dtype=jnp.float64)
    srp_j = jnp.array(srp_pos, dtype=jnp.float64)
    sc0_j = jnp.array(sc0, dtype=jnp.float64)
    scss_j = jnp.array(scss, dtype=jnp.float64)
    tropo_j = jnp.array(tropo_delay, dtype=jnp.float64)
    pts_j = jnp.array(pixel_pts_ecef, dtype=jnp.float64)

    # Test Window configurations and upsampling parameters
    print("\n--- Running BPA coherence test experiments ---")
    chunk_size = 5000

    sll_options = [30, 35, 38, 40]
    mult_options = [8, 16]

    for sll in sll_options:
        for mult in mult_options:
            print(f"\nTesting SLL={sll} dB, Range Upsampling={mult}x ...")
            range_win = scipy.signal.windows.taylor(num_samples, nbar=4, sll=sll)
            az_win = scipy.signal.windows.taylor(num_pulses, nbar=4, sll=sll)

            img_bpa = np.zeros((grid_size, grid_size), dtype=np.complex128)
            t0 = time.time()
            for start_idx in range(0, num_pulses, chunk_size):
                end_idx = min(start_idx + chunk_size, num_pulses)
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
                    use_freq_shift=False,
                )
                img_bpa += np.array(chunk_img)

            dt = time.time() - t0
            bpa_clean = quality_evaluator.clean_array(img_bpa)
            bpa_mag = np.abs(bpa_clean)

            # Integer shift correlation
            ref_norm = ref_mag - np.mean(ref_mag)
            bpa_norm = bpa_mag - np.mean(bpa_mag)
            denom = np.sqrt(np.sum(ref_norm**2) * np.sum(bpa_norm**2))
            unaligned_corr = np.sum(ref_norm * bpa_norm) / denom if denom > 0 else 0.0

            best_shift, fine_corr = find_subpixel_shift(bpa_mag, ref_mag)
            print(f"  Time: {dt:.2f}s | Unaligned Corr: {unaligned_corr:.4f} | Subpixel Shift: {best_shift} -> Fine Corr: {fine_corr:.4f}")


if __name__ == "__main__":
    main()
