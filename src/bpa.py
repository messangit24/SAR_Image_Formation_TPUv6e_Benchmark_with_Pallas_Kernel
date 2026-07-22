"""
Back-Projection Algorithm (BPA) for spotlight CPHD.

The time-domain "gold standard": for every output pixel, coherently sum the
range-compressed contribution of every pulse at that pixel's exact two-way
delay.  Unlike PFA it makes NO planar-wavefront approximation -- it uses the
exact geometry per pixel -- so it is the natural reference to cross-check PFA
against.  Cost is O(pixels x pulses), so we form a chip on the SICD's grid.

Data specifics (Umbra CPHD):
  * FX (frequency) domain  -> range compression is just an IFFT over frequency,
    giving a complex range profile vs differential range (relative to the SRP).
  * motion-compensated to a fixed SRP -> the phase reference is |APC_m - SRP|.

For pixel p and pulse m:
    dR = |APC_m - p| - |APC_m - SRP|               (differential range)
    image(p) += envelope_m(dR) * exp(+j * 4pi*f_c/c * dR)
where envelope_m is the baseband (band-centred) range profile, interpolated at
dR, and the exp restores the carrier phase at the exact delay (interpolating the
smooth baseband envelope, not the fast-oscillating full signal, keeps it clean).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

C = 299_792_458.0


@dataclass
class BPAResult:
    image: np.ndarray
    dx_range: float
    dy_cross: float
    r_hat: np.ndarray
    c_hat: np.ndarray
    n_hat: np.ndarray
    srp: np.ndarray


def _lininterp_complex(prof: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Linear interp of 1-D complex `prof` at fractional indices `idx`; OOB -> 0."""
    n = prof.shape[0]
    i0 = np.floor(idx).astype(np.int64)
    f = (idx - i0)
    ok = (i0 >= 0) & (i0 < n - 1)
    i0c = np.clip(i0, 0, n - 2)
    lo = prof[i0c]; hi = prof[i0c + 1]
    out = lo * (1.0 - f) + hi * f
    out[~ok] = 0
    return out


def backproject(
    signal: np.ndarray,      # (npulse, nsamp) complex FX-domain
    apc: np.ndarray,         # (npulse, 3) ECEF phase centre
    srp: np.ndarray,         # (3,) or (npulse,3)
    sc0: np.ndarray,         # (npulse,) start freq
    scss: np.ndarray,        # (npulse,) freq step
    pix_ecef: np.ndarray,    # (Naz, Nrg, 3) ECEF pixel positions to image
    r_hat, c_hat, dx_range, dy_cross,
    sgn: int = -1,
    fc: float | None = None,
    pulse_step: int = 1,
    upsample: int = 4,       # range-profile oversampling (finer delay interp)
    progress: bool = True,
) -> BPAResult:
    sig = np.ascontiguousarray(signal).astype(np.complex64)
    npulse, nsamp = sig.shape
    srp = np.broadcast_to(np.asarray(srp, float), (npulse, 3))
    apc = np.asarray(apc, float)

    Naz, Nrg, _ = pix_ecef.shape
    P = pix_ecef.reshape(-1, 3)                      # (Npix, 3) float64
    Px, Py, Pz = P[:, 0].copy(), P[:, 1].copy(), P[:, 2].copy()

    Nup = nsamp * upsample
    dr = C / (2.0 * nsamp * float(scss.mean())) / upsample      # profile bin (m), oversampled
    half = Nup // 2
    # per-pulse carrier reference = SC0 (the first-sample frequency).  ifft(sig)
    # leaves the target value as sigma*exp(-j 4pi*SC0*dR/c); we compensate with
    # exp(+j 4pi*SC0*dR/c) (SGN=-1 -> +j) so all pulses add coherently.  Using SC0
    # (not band-centre f_c) is essential: any offset leaves a per-pulse linear
    # phase in dR that smears azimuth.
    kc0 = 4.0 * np.pi * sc0 / C                                 # (npulse,)

    img = np.zeros(P.shape[0], np.complex64)
    Rsrp_all = np.linalg.norm(apc - srp, axis=1)

    idxs = range(0, npulse, pulse_step)
    total = len(idxs)
    for c, m in enumerate(idxs):
        prof = np.fft.fftshift(np.fft.ifft(sig[m], n=Nup)) * Nup   # range-compressed
        ax, ay, az = apc[m]
        dx = Px - ax; dy = Py - ay; dz = Pz - az
        R = np.sqrt(dx * dx + dy * dy + dz * dz)               # |APC - p|
        dR = R - Rsrp_all[m]                                    # differential range
        val = _lininterp_complex(prof, dR / dr + half)
        img += (val * np.exp((1j * sgn) * kc0[m] * dR)).astype(np.complex64)
        if progress and (c % 1000 == 0 or c == total - 1):
            print(f"  bpa pulse {c+1}/{total}", flush=True)

    image = img.reshape(Naz, Nrg)
    n_hat = np.cross(r_hat, c_hat)
    return BPAResult(image=image, dx_range=dx_range, dy_cross=dy_cross,
                     r_hat=np.asarray(r_hat), c_hat=np.asarray(c_hat),
                     n_hat=n_hat, srp=srp[0])
