"""
JAX / Cloud TPU Time-Domain Backprojection (BPA) Engine.

Mathematically aligned with src/bpa.py:
  * Uses start frequency SC0 for carrier phase restoration: exp(+j * 4pi*SC0/c * dR)
  * Supports zero-padded range profile upsampling (default 8x)
  * Memory-efficient pulse loop using jax.lax.scan
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

# Enable float64 for exact ECEF geometry calculations
jax.config.update("jax_enable_x64", True)

C = 299_792_458.0


@partial(jax.jit, static_argnames=("upsample",))
def jax_bpa_kernel(
    signal: jnp.ndarray,       # (npulse, nsamp) complex128 FX-domain
    apc: jnp.ndarray,          # (npulse, 3) float64 ECEF phase centre
    srp: jnp.ndarray,          # (npulse, 3) float64 SRP
    sc0: jnp.ndarray,          # (npulse,) float64 start frequency
    scss: jnp.ndarray,         # (npulse,) float64 freq step
    pixel_pts: jnp.ndarray,    # (Naz, Nrg, 3) float64 ECEF pixel grid
    sgn: int = 1,
    upsample: int = 8,
) -> jnp.ndarray:
    npulse, nsamp = signal.shape
    pad_samples = nsamp * upsample

    mean_scss = jnp.mean(scss)
    dr = C / (2.0 * nsamp * mean_scss) / upsample
    half = pad_samples // 2

    Naz, Nrg, _ = pixel_pts.shape
    bpa_init = jnp.zeros((Naz, Nrg), dtype=jnp.complex128)

    kc0_all = 4.0 * jnp.pi * sc0 / C

    def vector_step(bpa_acc, i):
        pulse_sig = signal[i, :]
        range_profile = jnp.fft.fftshift(jnp.fft.ifft(pulse_sig, n=pad_samples)) * pad_samples

        apc_i = apc[i]
        srp_i = srp[i]
        r_srp = jnp.linalg.norm(apc_i - srp_i)

        dist = jnp.linalg.norm(pixel_pts - apc_i[None, None, :], axis=-1)
        delta_r = dist - r_srp

        bin_idx = half + (delta_r / dr)
        valid_mask = (bin_idx >= 0) & (bin_idx < pad_samples - 1)
        idx0 = jnp.floor(bin_idx).astype(jnp.int32)
        frac = bin_idx - idx0

        idx0_clamped = jnp.clip(idx0, 0, pad_samples - 2)
        idx1_clamped = idx0_clamped + 1

        interp_val = (1.0 - frac) * range_profile[idx0_clamped] + frac * range_profile[idx1_clamped]
        interp_val = jnp.where(valid_mask, interp_val, 0.0 + 0.0j)

        phase_factor = jnp.exp((1j * sgn) * kc0_all[i] * delta_r)
        return bpa_acc + interp_val * phase_factor, None

    bpa_final, _ = jax.lax.scan(vector_step, bpa_init, jnp.arange(npulse))
    return bpa_final


def backproject_tpu(
    signal: np.ndarray,
    apc: np.ndarray,
    srp: np.ndarray,
    sc0: np.ndarray,
    scss: np.ndarray,
    pixel_pts: np.ndarray,
    sgn: int = 1,
    upsample: int = 8,
) -> np.ndarray:
    """Executes JAX Cloud TPU / GPU Backprojection kernel."""
    sig_j = jnp.array(signal, dtype=jnp.complex128)
    apc_j = jnp.array(apc, dtype=jnp.float64)
    srp_j = jnp.array(srp, dtype=jnp.float64)
    sc0_j = jnp.array(sc0, dtype=jnp.float64)
    scss_j = jnp.array(scss, dtype=jnp.float64)
    pts_j = jnp.array(pixel_pts, dtype=jnp.float64)

    if srp_j.ndim == 1:
        srp_j = jnp.broadcast_to(srp_j, apc_j.shape)

    img_j = jax_bpa_kernel(
        sig_j, apc_j, srp_j, sc0_j, scss_j, pts_j, sgn=sgn, upsample=upsample
    )
    return np.array(img_j)
