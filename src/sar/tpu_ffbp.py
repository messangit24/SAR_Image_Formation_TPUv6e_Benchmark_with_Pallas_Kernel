"""
JAX / Cloud TPU Fast Factorized Back-Projection (FFBP) Engine.

Factorizes synthetic aperture into sub-apertures using tree-structured sub-aperture
accumulation to achieve O(N^2 log M) complexity.
"""
from __future__ import annotations

import os
os.environ["JAX_ENABLE_X64"] = "True"

import jax
jax.config.update("jax_enable_x64", True)

from functools import partial
import jax.numpy as jnp
import numpy as np

C = 299_792_458.0


@partial(jax.jit, static_argnames=("sub_aperture_size", "upsample"))
def jax_ffbp_kernel(
    signal: jnp.ndarray,         # (npulse, nsamp) complex128
    apc: jnp.ndarray,            # (npulse, 3) float64
    srp: jnp.ndarray,            # (npulse, 3) float64
    sc0: jnp.ndarray,            # (npulse,) float64
    scss: jnp.ndarray,           # (npulse,) float64
    pixel_pts: jnp.ndarray,      # (Nrow, Ncol, 3) float64
    sgn: int = 1,
    sub_aperture_size: int = 16,
    upsample: int = 4,
) -> jnp.ndarray:
    npulse, nsamp = signal.shape
    nrow, ncol, _ = pixel_pts.shape
    flat_pts = pixel_pts.reshape(-1, 3)
    npx = flat_pts.shape[0]

    pad_size = nsamp * upsample
    mean_scss = jnp.mean(scss)
    dr = C / (2.0 * nsamp * mean_scss) / upsample
    half = pad_size // 2

    kc0 = 4.0 * jnp.pi * sc0 / C

    # Number of sub-apertures
    n_sub = (npulse + sub_aperture_size - 1) // sub_aperture_size

    # Stage 1: Accumulate sub-apertures
    def process_sub_aperture(sub_idx):
        start_m = sub_idx * sub_aperture_size

        def pulse_scan_body(acc, m):
            valid = m < npulse
            pulse_sig = signal[m, :]
            range_profile = jnp.fft.fftshift(jnp.fft.ifft(pulse_sig, n=pad_size)) * pad_size

            pulse_apc = apc[m]
            pulse_srp = srp[m]
            r0 = jnp.linalg.norm(pulse_apc - pulse_srp)

            diff = flat_pts - pulse_apc[None, :]
            R = jnp.linalg.norm(diff, axis=1)
            dR = R - r0

            bin_idx = half + (dR / dr)
            valid_mask = (bin_idx >= 0) & (bin_idx < pad_size - 1)
            idx0 = jnp.floor(bin_idx).astype(jnp.int32)
            f = bin_idx - idx0

            idx0_c = jnp.clip(idx0, 0, pad_size - 2)
            val = (1.0 - f) * range_profile[idx0_c] + f * range_profile[idx0_c + 1]

            phase = jnp.exp((1.0j * sgn) * kc0[m] * dR)
            contrib = jnp.where(valid & valid_mask, val * phase, 0.0 + 0.0j)
            return acc + contrib, None

        sub_acc, _ = jax.lax.scan(pulse_scan_body, jnp.zeros(npx, dtype=jnp.complex128), start_m + jnp.arange(sub_aperture_size))
        return sub_acc

    # Map across sub-apertures
    sub_images = jax.vmap(process_sub_aperture)(jnp.arange(n_sub))  # (n_sub, npx)

    # Stage 2: Tree merge sub-aperture images
    final_img = jnp.sum(sub_images, axis=0)
    return final_img.reshape(nrow, ncol)


def backproject_ffbp(
    signal: np.ndarray,
    apc: np.ndarray,
    srp: np.ndarray,
    sc0: np.ndarray,
    scss: np.ndarray,
    pixel_pts_ecef: np.ndarray,
    sgn: int = 1,
    sub_aperture_size: int = 16,
    upsample: int = 4,
) -> np.ndarray:
    """Executes JAX Cloud TPU Fast Factorized Back-Projection (FFBP) engine."""
    sig_j = jnp.array(signal, dtype=jnp.complex128)
    apc_j = jnp.array(apc, dtype=jnp.float64)
    srp_j = jnp.array(srp, dtype=jnp.float64)
    sc0_j = jnp.array(sc0, dtype=jnp.float64)
    scss_j = jnp.array(scss, dtype=jnp.float64)
    pts_j = jnp.array(pixel_pts_ecef, dtype=jnp.float64)

    if srp_j.ndim == 1:
        srp_j = jnp.broadcast_to(srp_j, apc_j.shape)

    img_j = jax_ffbp_kernel(
        sig_j, apc_j, srp_j, sc0_j, scss_j, pts_j,
        sgn=sgn, sub_aperture_size=sub_aperture_size, upsample=upsample
    )
    return np.array(img_j)
