"""Display helpers + point-target focus (IPR) analysis for SAR magnitude images."""
from __future__ import annotations

import numpy as np


def sar_stretch(mag: np.ndarray, k: float = 2.6, gamma: float = 0.7) -> np.ndarray:
    """
    Classic SAR amplitude stretch: normalise by mean, clip bright returns, gamma.
    Produces good land/structure contrast without a few bright scatterers
    saturating the whole scene (which naive percentile-dB scaling suffers from).
    """
    a = mag.astype(np.float64)
    a = a / (k * (a.mean() + 1e-12))
    a = np.clip(a, 0, 1)
    return a ** gamma


def measure_ipr(chip: np.ndarray, pixel_m: float, oversample: int = 16) -> dict:
    """
    Measure the -3 dB impulse response width (metres) of a 1-D cut through the
    peak of a point-target `chip` by FFT interpolation.  Also returns peak
    sidelobe ratio (PSLR, dB).
    """
    c = np.abs(chip).astype(np.float64)
    n = c.size
    spec = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(c)))
    padded = np.zeros(n * oversample, complex)
    lo = (n * oversample - n) // 2
    padded[lo:lo + n] = spec
    fine = np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(padded)))) * oversample
    fine /= fine.max()
    pk = int(np.argmax(fine))
    half = np.where(fine >= 10 ** (-3 / 20))[0]
    width_px = (half.max() - half.min()) / oversample
    # PSLR: highest peak outside the mainlobe
    mainlo = half.min()
    mainhi = half.max()
    mask = np.ones_like(fine, bool)
    mask[mainlo:mainhi + 1] = False
    pslr = 20 * np.log10(fine[mask].max() + 1e-12) if mask.any() else np.nan
    return dict(res_m=width_px * pixel_m, pslr_db=pslr, fine=fine, peak=pk)


def find_point_target(mag: np.ndarray, margin: int = 200):
    """Return (row, col) of a bright, locally-isolated point target."""
    m = mag.copy()
    m[:margin] = 0; m[-margin:] = 0
    m[:, :margin] = 0; m[:, -margin:] = 0
    # prefer targets that are locally dominant (peak / local mean)
    idx = np.argsort(m.ravel())[::-1][:2000]
    rows, cols = np.unravel_index(idx, m.shape)
    best, best_score = None, -1
    for r, cc in zip(rows, cols):
        loc = mag[r - 20:r + 21, cc - 20:cc + 21]
        score = mag[r, cc] / (loc.mean() + 1e-9)
        if score > best_score:
            best_score, best = score, (r, cc)
    return best
