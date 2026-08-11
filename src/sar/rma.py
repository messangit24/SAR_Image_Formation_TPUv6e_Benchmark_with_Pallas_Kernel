"""
Omega-k / Range Migration Algorithm (RMA) for spotlight CPHD.

The frequency-domain migration method: azimuth-FFT to the wavenumber domain,
then a Stolt interpolation that remaps the range-frequency axis onto the
Cartesian output wavenumber via the dispersion relation

    Kx = sqrt(kr^2 - Ky^2)          (kr = 4*pi*f/c round-trip wavenumber)

followed by a 2-D IFFT.  Unlike PFA (two separable polar->Cartesian interps)
the migration is handled by the single nonlinear Stolt map.  For this collect
the wavefront-curvature (QPE) is negligible, so a bulk reference at the scene
range plus Stolt focuses the whole chip.

Output is the aperture (slant) plane, same as PFA -- so it validates the same
way (coherence.py / register_sicd.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C = 299_792_458.0


@dataclass
class RMAResult:
    image: np.ndarray
    dx_range: float
    dy_cross: float
    r_hat: np.ndarray
    c_hat: np.ndarray
    n_hat: np.ndarray
    srp: np.ndarray
    res_range: float
    res_cross: float


def _cubic_axis0(samples, frac):
    """cubic-convolution interp of complex `samples` along axis0 at fractional
    row indices `frac` (2-D, column-aligned)."""
    n = samples.shape[0]
    i0 = np.floor(frac).astype(np.int64)
    f = (frac - i0)
    f2 = f * f; f3 = f2 * f
    w = ((-0.5 * f3 + f2 - 0.5 * f), (1.5 * f3 - 2.5 * f2 + 1.0),
         (-1.5 * f3 + 2.0 * f2 + 0.5 * f), (0.5 * f3 - 0.5 * f2))
    out = np.zeros(frac.shape, samples.dtype)
    for off, wk in zip((-1, 0, 1, 2), w):
        idx = np.clip(i0 + off, 0, n - 1)
        ok = ((i0 + off) >= 0) & ((i0 + off) < n)
        out += wk * np.where(ok, np.take_along_axis(samples, idx, axis=0), 0)
    return out


def form_image(signal, apc, srp, sc0, scss, sgn=-1,
               oversample=(1.3, 1.4), verbose=True):
    sig = np.ascontiguousarray(signal).astype(np.complex64)
    npulse, nsamp = sig.shape
    srp = np.broadcast_to(np.asarray(srp, float), (npulse, 3))
    apc = np.asarray(apc, float)

    # aperture-plane frame (identical to PFA)
    look = apc - srp
    look = look / np.linalg.norm(look, axis=1, keepdims=True)
    ml = look.mean(0); r_hat = ml / np.linalg.norm(ml)
    sweep = look[-1] - look[0]; c_in = sweep - r_hat * (sweep @ r_hat)
    c_hat = c_in / np.linalg.norm(c_in); n_hat = np.cross(r_hat, c_hat)

    from scipy.ndimage import map_coordinates
    a = np.einsum("ij,j->i", look, r_hat)          # cos(theta_m)
    b = np.einsum("ij,j->i", look, c_hat)          # sin(theta_m)
    theta = np.arctan2(b, a)
    kr = 4.0 * np.pi * (sc0.mean() + np.arange(nsamp) * scss.mean()) / C   # (nsamp,)

    # --- 1. resample onto a uniform aspect-angle axis (polar grid) -----------
    order = 1 if theta[-1] >= theta[0] else -1
    th = theta[::order]; s_th = sig[::order]
    thu = np.linspace(th[0], th[-1], npulse)
    fidx = np.interp(thu, th, np.arange(npulse)).astype(np.float64)
    Su = _cubic_axis0(s_th, np.broadcast_to(fidx[:, None], (npulse, nsamp)).copy())
    dth = thu[1] - thu[0]; dkr = kr[1] - kr[0]

    # --- 2. STOLT: resample polar (kr, theta) -> uniform Cartesian (Kx, Ky) ---
    #     via the exact dispersion  kr = sqrt(Kx^2 + Ky^2),  theta = atan2(Ky,Kx).
    krc = kr[nsamp // 2]
    nkx, nky = nsamp, npulse
    Kx = np.linspace(kr[0], kr[-1], nkx)                       # range wavenumber
    ky_max = krc * np.tan(max(abs(thu[0]), abs(thu[-1])))
    Ky = np.linspace(-ky_max, ky_max, nky)                     # cross-range wavenumber
    KX, KY = np.meshgrid(Kx, Ky)                               # (nky, nkx)
    kr_t = np.sqrt(KX ** 2 + KY ** 2)
    th_t = np.arctan2(KY, KX)
    coords = np.array([(th_t - thu[0]) / dth, (kr_t - kr[0]) / dkr])
    Sc = (map_coordinates(Su.real, coords, order=3, mode="constant")
          + 1j * map_coordinates(Su.imag, coords, order=3, mode="constant")).astype(np.complex64)

    # --- 3. oversample + 2-D FFT ---------------------------------------------
    dkx = (Kx[-1] - Kx[0]) / (nkx - 1)
    dky = (Ky[-1] - Ky[0]) / (nky - 1)
    Nx = int(round(nkx * oversample[0])); Ny = int(round(nky * oversample[1]))
    pad = np.zeros((Ny, Nx), np.complex64)
    pad[(Ny - nky) // 2:(Ny - nky) // 2 + nky,
        (Nx - nkx) // 2:(Nx - nkx) // 2 + nkx] = Sc
    fft = np.fft.fft2 if sgn < 0 else np.fft.ifft2
    image = np.fft.fftshift(fft(np.fft.ifftshift(pad)))

    dx_range = 2 * np.pi / (Nx * dkx)
    dy_cross = 2 * np.pi / (Ny * dky)
    res_range = 2 * np.pi / (Kx[-1] - Kx[0])
    res_cross = 2 * np.pi / (Ky[-1] - Ky[0])
    if verbose:
        print(f"  [rma] image {image.shape} pixel {dx_range:.3f}x{dy_cross:.3f} m "
              f"res {res_range:.3f}x{res_cross:.3f}")
    return RMAResult(image, dx_range, dy_cross, r_hat, c_hat, n_hat, srp[0],
                     res_range, res_cross)
