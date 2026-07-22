"""Fast 5k-pulse Coherence Test and Parameter Optimization for Bingham Mine 9-25."""

import sys
import time
import numpy as np
import scipy.ndimage
import scipy.optimize
import scipy.signal.windows
import jax
import jax.numpy as jnp
from functools import partial

import cphd_parser
import sicd_parser
import quality_evaluator

jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=("multiplier",))
def bpa_pulse_chunk(
    sig_chunk, tx_chunk, rcv_chunk, srp_chunk, sc0_chunk, scss_chunk, tropo_chunk, pts_jax, multiplier=8
):
    num_vectors, num_samples = sig_chunk.shape
    c = 299792458.0
    pad_samples = num_samples * multiplier

    mean_scss = jnp.mean(scss_chunk)
    dt_sample = 1.0 / (num_samples * mean_scss)
    range_grid_res = c * dt_sample / (2.0 * multiplier)

    grid_shape = (pts_jax.shape[0], pts_jax.shape[1])
    bpa_init = jnp.zeros(grid_shape, dtype=jnp.complex128)

    def vector_step(bpa_acc, i):
        fc = sc0_chunk[i] + 0.5 * (num_samples - 1) * scss_chunk[i]
        pulse_signal = sig_chunk[i, :] * jnp.exp(-2j * jnp.pi * fc * tropo_chunk[i])

        range_profile = jnp.fft.fftshift(jnp.fft.ifft(pulse_signal, n=pad_samples))

        tx_i, rcv_i, srp_i = tx_chunk[i], rcv_chunk[i], srp_chunk[i]
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


def find_fast_subpixel_shift(bpa_mag, ref_mag):
    ref_norm = ref_mag - np.mean(ref_mag)
    bpa_norm = bpa_mag - np.mean(bpa_mag)
    ref_sq = np.sum(ref_norm**2)
    bpa_sq = np.sum(bpa_norm**2)

    unaligned_denom = np.sqrt(ref_sq * bpa_sq)
    unaligned_corr = float(np.sum(ref_norm * bpa_norm) / unaligned_denom) if unaligned_denom > 0 else 0.0

    F_ref = np.fft.fft2(ref_norm)
    F_bpa = np.fft.fft2(bpa_norm)
    cross_corr = np.fft.ifft2(F_ref * np.conj(F_bpa))
    cross_corr_mag = np.abs(np.fft.fftshift(cross_corr))

    h, w = bpa_mag.shape
    r_center, c_center = h // 2, w // 2

    max_idx = np.unravel_index(np.argmax(cross_corr_mag), cross_corr_mag.shape)
    peak_r = max_idx[0] - r_center
    peak_c = max_idx[1] - c_center

    def loss(shift):
        dr, dc = shift
        shifted = scipy.ndimage.shift(bpa_mag, (dr, dc), order=3, mode='constant', cval=0.0)
        shifted_norm = shifted - np.mean(shifted)
        denom = np.sqrt(ref_sq * np.sum(shifted_norm**2))
        corr = np.sum(ref_norm * shifted_norm) / denom if denom > 0 else 0.0
        return -corr

    res = scipy.optimize.minimize(loss, [peak_r, peak_c], method='Nelder-Mead', options={'maxiter': 35})
    best_shift = (float(res.x[0]), float(res.x[1]))
    best_corr = -float(res.fun)

    return unaligned_corr, best_shift, best_corr


def main():
    sicd_path = "tpu_sar_data/bingham_20230925/2023-09-25-17-37-50_UMBRA-05_SICD.nitf"
    cphd_path = "tpu_sar_data/bingham_20230925/2023-09-25-17-37-50_UMBRA-05_CPHD.cphd"

    print("=== Fast 5k-pulse BPA Coherence Optimizer (Bingham 9-25) ===", flush=True)
    sicd = sicd_parser.parse_sicd_nitf(sicd_path)
    cphd = cphd_parser.load_cphd(cphd_path)

    pvp = cphd.pvp_array
    signal_memmap = cphd.signal_memmap

    # Use central 5,000 pulses for fast hyperparameter sweep
    num_total_pulses = len(pvp["TxPos_X"])
    num_pulses = 5000
    start_idx = (num_total_pulses - num_pulses) // 2
    end_idx = start_idx + num_pulses

    print(f"Using pulse subset: {start_idx} to {end_idx} (out of {num_total_pulses} pulses)", flush=True)

    tx_pos = np.column_stack((pvp["TxPos_X"], pvp["TxPos_Y"], pvp["TxPos_Z"]))[start_idx:end_idx, :]
    rcv_pos = np.column_stack((pvp["RcvPos_X"], pvp["RcvPos_Y"], pvp["RcvPos_Z"]))[start_idx:end_idx, :]
    tx_pos = np.where(np.isnan(tx_pos), rcv_pos, tx_pos)
    srp_pos = np.column_stack((pvp["SRPPos_X"], pvp["SRPPos_Y"], pvp["SRPPos_Z"]))[start_idx:end_idx, :]
    sc0 = pvp["SC0"][start_idx:end_idx]
    scss = pvp["SCSS"][start_idx:end_idx]
    tropo_delay = pvp["TDTropoSRP"][start_idx:end_idx]
    signal_subset = np.array(signal_memmap[start_idx:end_idx, :])

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

    grid_size = 512
    ref_crop = sicd.image_array[
        center_r - grid_size // 2 : center_r + grid_size // 2,
        center_c - grid_size // 2 : center_c + grid_size // 2,
    ]
    ref_crop = quality_evaluator.clean_array(ref_crop)
    ref_mag = np.abs(ref_crop)

    # Grid
    row_idx = np.arange(grid_size) - grid_size // 2
    col_idx = np.arange(grid_size) - grid_size // 2
    grid_r, grid_c = np.meshgrid(row_idx, col_idx, indexing="ij")
    mean_srp = np.mean(srp_pos, axis=0)
    pixel_pts_ecef = (
        mean_srp
        + grid_r[:, :, None] * row_ss * row_vec
        + grid_c[:, :, None] * col_ss * col_vec
    )

    num_samples = signal_subset.shape[1]

    tx_j = jnp.array(tx_pos, dtype=jnp.float64)
    rcv_j = jnp.array(rcv_pos, dtype=jnp.float64)
    srp_j = jnp.array(srp_pos, dtype=jnp.float64)
    sc0_j = jnp.array(sc0, dtype=jnp.float64)
    scss_j = jnp.array(scss, dtype=jnp.float64)
    tropo_j = jnp.array(tropo_delay, dtype=jnp.float64)
    pts_j = jnp.array(pixel_pts_ecef, dtype=jnp.float64)

    experiments = [
        {"name": "Rectangular (no window), mult=8", "win_type": "rect", "sll": 0, "mult": 8},
        {"name": "Taylor sll=30dB, mult=8", "win_type": "taylor", "sll": 30, "mult": 8},
        {"name": "Taylor sll=35dB, mult=8", "win_type": "taylor", "sll": 35, "mult": 8},
        {"name": "Taylor sll=38dB, mult=8", "win_type": "taylor", "sll": 38, "mult": 8},
        {"name": "Taylor sll=40dB, mult=8", "win_type": "taylor", "sll": 40, "mult": 8},
        {"name": "Taylor sll=38dB, mult=16", "win_type": "taylor", "sll": 38, "mult": 16},
    ]

    for exp in experiments:
        exp_name = exp["name"]
        win_type = exp["win_type"]
        sll = exp["sll"]
        mult = exp["mult"]

        print(f"\n--- Running Experiment: {exp_name} ---", flush=True)
        t0 = time.time()

        if win_type == "taylor":
            range_win = scipy.signal.windows.taylor(num_samples, nbar=4, sll=sll)
            az_win = scipy.signal.windows.taylor(num_pulses, nbar=4, sll=sll)
            signal_win = signal_subset * range_win[None, :] * az_win[:, None]
        else:
            signal_win = signal_subset

        sig_j = jnp.array(signal_win, dtype=jnp.complex128)

        img_bpa = bpa_pulse_chunk(
            sig_j, tx_j, rcv_j, srp_j, sc0_j, scss_j, tropo_j, pts_j, multiplier=mult
        )
        img_bpa = np.array(img_bpa)

        dt = time.time() - t0
        bpa_clean = quality_evaluator.clean_array(img_bpa)
        bpa_mag = np.abs(bpa_clean)

        unaligned_corr, best_shift, fine_corr = find_fast_subpixel_shift(bpa_mag, ref_mag)

        print(f"  Execution Time: {dt:.2f}s", flush=True)
        print(f"  Unaligned Correlation: {unaligned_corr:.4f}", flush=True)
        print(f"  Sub-pixel Shift: ({best_shift[0]:.3f}, {best_shift[1]:.3f})", flush=True)
        print(f"  Shift-Aligned Coherence: {fine_corr:.4f}", flush=True)


if __name__ == "__main__":
    main()
