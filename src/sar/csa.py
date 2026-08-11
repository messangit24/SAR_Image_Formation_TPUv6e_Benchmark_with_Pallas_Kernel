"""
Chirp Scaling Algorithm (CSA) -- attempt + honest verdict for this data.

WHY PURE CSA DOES NOT APPLY HERE
--------------------------------
CSA (Raney et al. 1994) is a *stripmap* range-Doppler-family processor whose
defining trick is "chirp scaling": a quadratic phase multiply, keyed to the
range **chirp rate**, that equalises range-cell-migration (RCM) curvature across
range so it can then be removed by a single bulk multiply (no interpolation).

Two of its premises are absent in Umbra's CPHD:
  1. The data is in the **FX (frequency) domain** -- i.e. already range
     de-ramped / compressed.  There is no range chirp to scale, so the chirp-
     scaling step itself is undefined (equivalently, chirp rate -> infinity).
  2. It is **spotlight**, motion-compensated to a fixed SRP.  The azimuth signal
     is not a stripmap Doppler chirp (its azimuth FM rate ~ 0 -- azimuth is
     compressed by a single FFT), and the migration is the polar/keystone
     geometry, not the stripmap hyperbola CSA models.

WHAT REMAINS, AND WHAT WE DO
----------------------------
The one real migration effect is a *linear range walk* of ~+-10 m across the
1.8 deg aperture (dR ~ -sin(theta)*y).  That is exactly the range-azimuth
coupling the polar/Stolt resample corrects.  So the CSA *family* reduces here to
Range-Doppler processing with a **keystone** range-walk correction (the
interpolation-based sibling of chirp scaling).  We implement that -- it focuses
and reaches PFA/RMA-level coherence -- and note that the *pure* chirp-scaling
variant is not applicable to de-ramped spotlight data.  (This is why ω-k/RMA is
the delivered frequency-domain method: see rma.py.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C = 299_792_458.0


@dataclass
class CSAResult:
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
    n = samples.shape[0]
    i0 = np.floor(frac).astype(np.int64); f = frac - i0
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
    """CSA-family (range-Doppler + keystone RCMC) image formation."""
    sig = np.ascontiguousarray(signal).astype(np.complex64)
    npulse, nsamp = sig.shape
    srp = np.broadcast_to(np.asarray(srp, float), (npulse, 3))
    apc = np.asarray(apc, float)

    look = apc - srp
    look = look / np.linalg.norm(look, axis=1, keepdims=True)
    ml = look.mean(0); r_hat = ml / np.linalg.norm(ml)
    sweep = look[-1] - look[0]; c_in = sweep - r_hat * (sweep @ r_hat)
    c_hat = c_in / np.linalg.norm(c_in); n_hat = np.cross(r_hat, c_hat)

    a = np.einsum("ij,j->i", look, r_hat)
    b = np.einsum("ij,j->i", look, c_hat)
    theta = np.arctan2(b, a)
    kr = 4.0 * np.pi * (sc0.mean() + np.arange(nsamp) * scss.mean()) / C     # (nsamp,)
    krc = kr[nsamp // 2]

    # 1. uniform azimuth (aspect angle) axis
    order = 1 if theta[-1] >= theta[0] else -1
    th = theta[::order]; s_th = sig[::order]
    thu = np.linspace(th[0], th[-1], npulse)
    fidx = np.interp(thu, th, np.arange(npulse)).astype(np.float64)
    Su = _cubic_axis0(s_th, np.broadcast_to(fidx[:, None], (npulse, nsamp)).copy())

    # 2. KEYSTONE range-walk correction: the target's Ky = kr*sin(theta) couples
    #    range (kr) and azimuth; resample the azimuth angle per range bin by
    #    krc/kr so the cross-range wavenumber becomes range-independent.  This is
    #    the interpolation sibling of chirp scaling's RCMC.
    tanu = np.tan(thu)
    ky = krc * tanu                                   # target cross-range wavenumber axis
    Sk = np.empty_like(Su)
    idx_grid = np.arange(npulse)
    for n in range(nsamp):
        # angle that gives cross-range wavenumber ky on this range bin: tan = ky/kr[n]
        want = np.arctan2(ky, kr[n])
        fi = np.interp(want, thu, idx_grid, left=0, right=npulse - 1)
        Sk[:, n] = _cubic_axis0(Su[:, n:n + 1], fi[:, None])[:, 0]

    # 3. azimuth compression = FFT over the (now decoupled) azimuth axis;
    #    range compression = IFFT over range frequency.  Oversample + 2-D FFT.
    dky = (ky[1] - ky[0]) if npulse > 1 else 1.0
    dkx = kr[1] - kr[0]
    nkx, nky = nsamp, npulse
    Nx = int(round(nkx * oversample[0])); Ny = int(round(nky * oversample[1]))
    pad = np.zeros((Ny, Nx), np.complex64)
    pad[(Ny - nky) // 2:(Ny - nky) // 2 + nky,
        (Nx - nkx) // 2:(Nx - nkx) // 2 + nkx] = Sk
    fft = np.fft.fft2 if sgn < 0 else np.fft.ifft2
    image = np.fft.fftshift(fft(np.fft.ifftshift(pad)))

    dx_range = 2 * np.pi / (Nx * dkx)
    dy_cross = 2 * np.pi / (Ny * abs(dky))
    res_range = 2 * np.pi / (kr[-1] - kr[0])
    res_cross = 2 * np.pi / abs(ky[-1] - ky[0])
    if verbose:
        print(f"  [csa] image {image.shape} pixel {dx_range:.3f}x{dy_cross:.3f} m "
              f"res {res_range:.3f}x{res_cross:.3f}")
    return CSAResult(image, dx_range, dy_cross, r_hat, c_hat, n_hat, srp[0],
                     res_range, res_cross)
