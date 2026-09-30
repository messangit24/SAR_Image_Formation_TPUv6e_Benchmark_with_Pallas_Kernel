"""
JAX / Cloud TPU Polar Format Algorithm (PFA) Engine.

Resamples polar phase history (K_r, phi) to Cartesian (K_x, K_y) grid using 4-tap cubic
convolution interpolation, applies optional k-space zero-padding oversampling, and executes 2D FFT.
Mathematically aligned with src/pfa.py.
"""
from __future__ import annotations

import os
os.environ["JAX_ENABLE_X64"] = "True"

import jax
jax.config.update("jax_enable_x64", True)

from dataclasses import dataclass
from functools import partial
import jax.numpy as jnp
import numpy as np

C = 299_792_458.0


def jax_cubic_weights(f: jnp.ndarray):
    f2 = f * f
    f3 = f2 * f
    w_1 = -0.5 * f3 + f2 - 0.5 * f
    w0 = 1.5 * f3 - 2.5 * f2 + 1.0
    w1 = -1.5 * f3 + 2.0 * f2 + 0.5 * f
    w2 = 0.5 * f3 - 0.5 * f2
    return w_1, w0, w1, w2


@partial(jax.jit, static_argnames=("nkx", "nky", "sgn", "oversample_r", "oversample_az"))
def jax_pfa_kernel(
    signal: jnp.ndarray,     # (npulse, nsamp) complex128
    apc: jnp.ndarray,        # (npulse, 3) float64
    srp: jnp.ndarray,        # (npulse, 3) float64
    sc0: jnp.ndarray,        # (npulse,) float64
    scss: jnp.ndarray,       # (npulse,) float64
    r_hat: jnp.ndarray,      # (3,) float64
    c_hat: jnp.ndarray,      # (3,) float64
    sgn: int = 1,
    nkx: int = 0,
    nky: int = 0,
    oversample_r: float = 1.3,
    oversample_az: float = 1.4,
) -> jnp.ndarray:
    npulse, nsamp = signal.shape
    nkx = nsamp if nkx <= 0 else nkx
    nky = npulse if nky <= 0 else nky

    look = apc - srp
    look = look / jnp.linalg.norm(look, axis=1, keepdims=True)

    mean_look = jnp.mean(look, axis=0)
    dot_r = jnp.dot(mean_look, r_hat)
    r_hat_c = jnp.where(dot_r < 0, -r_hat, r_hat)

    a = jnp.dot(look, r_hat_c)
    b = jnp.dot(look, c_hat)
    t = b / a

    kr0 = 4.0 * jnp.pi * sc0 / C
    dkr = 4.0 * jnp.pi * scss / C
    kr_lo = kr0
    kr_hi = kr0 + (nsamp - 1) * dkr

    kx_min = jnp.max(kr_lo * a)
    kx_max = jnp.min(kr_hi * a)
    kx = jnp.linspace(kx_min, kx_max, nkx)

    # Stage 1: Range interpolation along Kx
    kr_needed = kx[None, :] / a[:, None]
    frac1 = (kr_needed - kr0[:, None]) / dkr[:, None]

    def interp_pulse(i):
        pulse_sig = signal[i]
        frac_i = frac1[i]
        i0 = jnp.floor(frac_i).astype(jnp.int32)
        f = frac_i - i0
        w_1, w0, w1, w2 = jax_cubic_weights(f)

        out = jnp.zeros_like(frac_i, dtype=jnp.complex128)
        weights = (w_1, w0, w1, w2)
        offsets = (-1, 0, 1, 2)

        for off, w in zip(offsets, weights):
            idx = i0 + off
            ok = (idx >= 0) & (idx < nsamp)
            idxc = jnp.clip(idx, 0, nsamp - 1)
            out = out + w * jnp.where(ok, pulse_sig[idxc], 0.0 + 0.0j)

        return out

    stage1 = jax.vmap(interp_pulse)(jnp.arange(npulse))  # (npulse, nkx)

    # Stage 2: Azimuth interpolation along Ky
    kxc = 0.5 * (kx_min + kx_max)
    is_asc = t[-1] >= t[0]
    t_ord = jnp.where(is_asc, t, t[::-1])
    stage1_ord = jnp.where(is_asc, stage1, stage1[::-1, :])

    ky_min = kxc * t_ord[0]
    ky_max = kxc * t_ord[-1]
    ky = jnp.linspace(ky_min, ky_max, nky)
    pulse_index = jnp.arange(npulse, dtype=jnp.float64)

    q = ky[:, None] / kx[None, :]  # (nky, nkx)
    frac2 = jnp.interp(q.ravel(), t_ord, pulse_index, left=-1.0, right=npulse).reshape(q.shape)

    def interp_col(j):
        col_sig = stage1_ord[:, j]
        frac_j = frac2[:, j]
        i0 = jnp.floor(frac_j).astype(jnp.int32)
        f = frac_j - i0
        w_1, w0, w1, w2 = jax_cubic_weights(f)

        out = jnp.zeros_like(frac_j, dtype=jnp.complex128)
        weights = (w_1, w0, w1, w2)
        offsets = (-1, 0, 1, 2)

        for off, w in zip(offsets, weights):
            idx = i0 + off
            ok = (idx >= 0) & (idx < npulse)
            idxc = jnp.clip(idx, 0, npulse - 1)
            out = out + w * jnp.where(ok, col_sig[idxc], 0.0 + 0.0j)

        return out

    stage2 = jax.vmap(interp_col, in_axes=0, out_axes=1)(jnp.arange(nkx))  # (nky, nkx)

    # K-space Zero-padding for oversampling
    Nx = int(round(nkx * oversample_r))
    Ny = int(round(nky * oversample_az))
    
    if Nx > nkx or Ny > nky:
        y0 = (Ny - nky) // 2
        x0 = (Nx - nkx) // 2
        pad_grid = jnp.zeros((Ny, Nx), dtype=jnp.complex128)
        stage2 = pad_grid.at[y0 : y0 + nky, x0 : x0 + nkx].set(stage2)

    fft = jnp.fft.ifft2 if sgn == 1 else jnp.fft.fft2
    image = jnp.fft.fftshift(fft(jnp.fft.ifftshift(stage2)))
    return image


def form_image_tpu(
    signal: np.ndarray,
    apc: np.ndarray,
    srp: np.ndarray,
    sc0: np.ndarray,
    scss: np.ndarray,
    r_hat: np.ndarray,
    c_hat: np.ndarray,
    sgn: int = 1,
    nkx: int = 0,
    nky: int = 0,
    oversample: tuple = (1.3, 1.4),
) -> np.ndarray:
    """Executes JAX Cloud TPU / GPU Polar Format Algorithm (PFA) kernel."""
    sig_j = jnp.array(signal, dtype=jnp.complex128)
    apc_j = jnp.array(apc, dtype=jnp.float64)
    srp_j = jnp.array(srp, dtype=jnp.float64)
    sc0_j = jnp.array(sc0, dtype=jnp.float64)
    scss_j = jnp.array(scss, dtype=jnp.float64)
    r_j = jnp.array(r_hat, dtype=jnp.float64)
    c_j = jnp.array(c_hat, dtype=jnp.float64)

    if srp_j.ndim == 1:
        srp_j = jnp.broadcast_to(srp_j, apc_j.shape)

    img_j = jax_pfa_kernel(
        sig_j, apc_j, srp_j, sc0_j, scss_j, r_j, c_j,
        sgn=sgn, nkx=nkx, nky=nky, oversample_r=float(oversample[0]), oversample_az=float(oversample[1])
    )
    return np.array(img_j)
