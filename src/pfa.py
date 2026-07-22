"""
Polar Format Algorithm (PFA) image formation for spotlight CPHD.

Umbra's CPHD is already:
  * motion-compensated to a fixed Scene Reference Point (SRP / IARP), and
  * stored in the frequency (FX) domain, i.e. each sample IS a sample of the
    range-frequency spectrum for that pulse.

So the collected data lives on a *polar* grid in wavenumber (k) space: each
pulse is a radial spoke (constant look direction, swept over frequency), and
successive pulses rotate the spoke as the platform flies by.  PFA resamples
that polar grid onto a uniform Cartesian (Kx, Ky) grid and 2-D-FFTs it to an
image.  No autofocus is applied — for a 4 km scene at 663 km range the planar
wavefront approximation PFA makes is more than good enough, which is exactly
why Umbra can publish a focused SICD "without autofocus".

Reference: Carrara, Goodman & Majewski, "Spotlight SAR: Signal Processing
Algorithms" (1995), ch. 3 (polar format processing).

The two classic 1-D interpolation stages:

  Stage 1 (range / radial): each pulse's samples lie on a ray through the k
  origin at angle phi[m]; resample each pulse from its native radial-frequency
  grid onto a common uniform Kx (range wavenumber) grid.

  Stage 2 (azimuth / angular -> Cartesian): after stage 1 the cross-range
  wavenumber of column j, pulse m sits at Ky = Kx_j * tan(phi[m]); resample
  across pulses onto a uniform Ky grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C = 299_792_458.0


@dataclass
class PFAResult:
    image: np.ndarray        # complex image, [range (Kx-conjugate), cross-range (Ky-conjugate)]
    dx_range: float          # metres/pixel along range axis (r_hat)
    dy_cross: float          # metres/pixel along cross-range axis (c_hat)
    r_hat: np.ndarray        # (3,) ground-plane range unit vector (ECEF)
    c_hat: np.ndarray        # (3,) ground-plane cross-range unit vector (ECEF)
    n_hat: np.ndarray        # (3,) ground-plane normal (ECEF)
    srp: np.ndarray          # (3,) scene reference point (ECEF)
    res_range: float         # range resolution (m), = 2*pi/Kx_extent
    res_cross: float         # cross-range resolution (m)


def _cubic_weights(f: np.ndarray):
    """Catmull-Rom (Keys, a=-1/2) cubic-convolution weights for the 4 taps
    at offsets (-1, 0, +1, +2) given fractional position f in [0, 1)."""
    f2 = f * f
    f3 = f2 * f
    w_1 = -0.5 * f3 + f2 - 0.5 * f
    w0 = 1.5 * f3 - 2.5 * f2 + 1.0
    w1 = -1.5 * f3 + 2.0 * f2 + 0.5 * f
    w2 = 0.5 * f3 - 0.5 * f2
    return w_1, w0, w1, w2


def _cubic_interp_axis0(samples: np.ndarray, frac_idx: np.ndarray) -> np.ndarray:
    """
    Cubic-convolution interpolation of `samples` (2-D, uniform grid along axis 0)
    at fractional row positions `frac_idx` (same #columns as samples), with
    column-aligned gather.  Much lower interpolation error than linear -> the
    resampled speckle stays coherent with a reference processor.  Out-of-range
    taps are zeroed.  Complex-safe, vectorised.
    """
    n = samples.shape[0]
    i0 = np.floor(frac_idx).astype(np.int64)
    f = (frac_idx - i0).astype(np.float32)
    w_1, w0, w1, w2 = _cubic_weights(f)
    out = np.zeros(frac_idx.shape, samples.dtype)
    for off, w in ((-1, w_1), (0, w0), (1, w1), (2, w2)):
        idx = i0 + off
        ok = (idx >= 0) & (idx < n)
        idxc = np.clip(idx, 0, n - 1)
        vals = np.take_along_axis(samples, idxc, axis=0)
        out += w * np.where(ok, vals, 0)
    return out


def _cubic_interp_1d(samples: np.ndarray, frac_idx: np.ndarray) -> np.ndarray:
    """Cubic-convolution interpolation of a 1-D uniform `samples` at `frac_idx`."""
    n = samples.shape[0]
    i0 = np.floor(frac_idx).astype(np.int64)
    f = (frac_idx - i0).astype(np.float32)
    w_1, w0, w1, w2 = _cubic_weights(f)
    out = np.zeros(frac_idx.shape, samples.dtype)
    for off, w in ((-1, w_1), (0, w0), (1, w1), (2, w2)):
        idx = i0 + off
        ok = (idx >= 0) & (idx < n)
        out += w * np.where(ok, samples[np.clip(idx, 0, n - 1)], 0)
    return out


def form_image(
    signal: np.ndarray,      # (num_pulses, num_samples) complex, FX-domain
    apc: np.ndarray,         # (num_pulses, 3) effective phase-centre ECEF
    srp: np.ndarray,         # (num_pulses, 3) or (3,) scene reference point ECEF
    sc0: np.ndarray,         # (num_pulses,) start frequency (Hz) of sample 0
    scss: np.ndarray,        # (num_pulses,) frequency step (Hz) between samples
    uiax: np.ndarray,        # (3,) image-area X axis (defines ground plane)
    uiay: np.ndarray,        # (3,) image-area Y axis (defines ground plane)
    sgn: int = -1,           # CPHD Fourier sign convention
    nkx: int | None = None,  # k-space resampling grid size along range (default = num_samples)
    nky: int | None = None,  # k-space resampling grid size along azimuth (default = num_pulses)
    oversample: tuple = (1.3, 1.4),  # (range, azimuth) image oversample via k-space zero-pad
    taylor: tuple | None = None,  # (nbar, sll_dB) optional aperture weighting
    verbose: bool = True,
) -> PFAResult:
    sig = np.ascontiguousarray(signal).astype(np.complex64)
    npulse, nsamp = sig.shape
    srp = np.broadcast_to(np.asarray(srp, float), (npulse, 3)) if np.ndim(srp) == 1 else np.asarray(srp, float)

    # ---- imaging frame: the APERTURE (slant) plane -------------------------
    # Form the image in the plane spanned by the look direction and the aperture
    # sweep, NOT the ground plane.  In the ground plane the cos(graze)
    # foreshortening varies ~2% across the aperture (the platform's grazing
    # angle changes as it flies past the fixed SRP), which smears the common
    # range-frequency band and wrecks resolution.  In the aperture plane the
    # radial (range) component is ~constant -> a clean polar annulus and full
    # bandwidth, matching the plane the SICD is formed in.
    look = apc - srp
    look = look / np.linalg.norm(look, axis=1, keepdims=True)      # u_hat[m]
    mean_look = look.mean(axis=0)
    r_hat = mean_look / np.linalg.norm(mean_look)                  # range axis (LOS)
    sweep = look[-1] - look[0]                                     # aperture rotation dir
    c_in = sweep - r_hat * (sweep @ r_hat)                        # make it perp to r_hat
    c_hat = c_in / np.linalg.norm(c_in)                          # cross-range axis
    n_hat = np.cross(r_hat, c_hat)                                # aperture-plane normal

    # in-plane projection of each pulse look vector
    a = np.einsum("ij,j->i", look, r_hat)   # range component (~cos graze), >0
    b = np.einsum("ij,j->i", look, c_hat)   # cross-range component (small)
    t = b / a                               # tan(phi[m]); monotonic over aperture

    # per-pulse radial-frequency (round-trip) grid: kr = 4*pi*f/c
    kr0 = 4.0 * np.pi * sc0 / C             # (npulse,) kr at sample 0
    dkr = 4.0 * np.pi * scss / C            # (npulse,) kr step
    kr_lo = kr0
    kr_hi = kr0 + (nsamp - 1) * dkr

    # ------------------------------------------------------------------------
    # Stage 1: resample each pulse onto a common uniform Kx (range) grid.
    #   A point at range-wavenumber Kx on pulse m needs radial freq kr = Kx / a[m]
    #   (because Kx = kr * a[m]).  Choose the Kx grid as the band all pulses share.
    # ------------------------------------------------------------------------
    nkx = nkx or nsamp
    kx_min = np.max(kr_lo * a)              # conservative common lower edge
    kx_max = np.min(kr_hi * a)              # conservative common upper edge
    kx = np.linspace(kx_min, kx_max, nkx)  # (nkx,)

    # fractional sample index for every (pulse, Kx): idx = (Kx/a - kr0)/dkr
    kr_needed = kx[None, :] / a[:, None]                       # (npulse, nkx)
    frac = (kr_needed - kr0[:, None]) / dkr[:, None]
    stage1 = np.empty((npulse, nkx), np.complex64)
    for m in range(npulse):                                    # per-pulse (cheap, uniform grid)
        stage1[m] = _cubic_interp_1d(sig[m], frac[m])
    if verbose:
        print(f"  [stage1] range interp (cubic) -> {stage1.shape}  Kx=[{kx_min:.2f},{kx_max:.2f}] rad/m")

    # ------------------------------------------------------------------------
    # Stage 2: for each Kx column j the cross-range wavenumber is Ky = Kx_j*t[m].
    #   Resample across pulses onto a uniform Ky grid.  Reference the Ky support
    #   to the band-centre column (standard PFA), so Ky_grid = Kxc * [t_min,t_max].
    #   The pulse samples are (nearly) uniform in t, so we map each output Ky to a
    #   fractional pulse index through the monotonic t axis, then cubic-interpolate
    #   along pulses (column-aligned).  Chunked over columns to bound memory.
    # ------------------------------------------------------------------------
    nky = nky or npulse
    kxc = 0.5 * (kx_min + kx_max)
    order = 1 if t[-1] >= t[0] else -1
    t_ord = t[::order]
    ky_min = kxc * t_ord[0]
    ky_max = kxc * t_ord[-1]
    ky = np.linspace(ky_min, ky_max, nky)                      # (nky,)
    pulse_index = np.arange(npulse, dtype=np.float64)

    stage2 = np.empty((nky, nkx), np.complex64)
    s1_ord = np.ascontiguousarray(stage1[::order])            # pulses in increasing-t order
    CH = 1024
    for c0 in range(0, nkx, CH):
        c1 = min(c0 + CH, nkx)
        # target t for each (Ky, column) = Ky / Kx_col ; then -> fractional pulse index
        q = ky[:, None] / kx[None, c0:c1]                     # (nky, chunk) target t values
        frac_m = np.interp(q.ravel(), t_ord, pulse_index,
                           left=-1.0, right=npulse).reshape(q.shape)
        stage2[:, c0:c1] = _cubic_interp_axis0(s1_ord[:, c0:c1], frac_m)
    if verbose:
        print(f"  [stage2] azimuth interp (cubic) -> {stage2.shape}  Ky=[{ky_min:.3f},{ky_max:.3f}] rad/m")

    # optional aperture weighting (sidelobe control) on the uniform k grid
    if taylor is not None:
        from scipy.signal.windows import taylor as taylor_win
        wr = taylor_win(nkx, nbar=taylor[0], sll=taylor[1], sym=True)
        wc = taylor_win(nky, nbar=taylor[0], sll=taylor[1], sym=True)
        stage2 = stage2 * (wc[:, None] * wr[None, :])

    # ------------------------------------------------------------------------
    # Zero-pad the k-space (keeping the sample spacing dk fixed) so the image is
    # oversampled -> finer pixels than the resolution cell.  Without this the
    # azimuth pixel (~0.49 m) is coarser than the azimuth resolution (~0.44 m),
    # i.e. the mainlobe is undersampled/aliased.
    # ------------------------------------------------------------------------
    dkx = (kx_max - kx_min) / (nkx - 1)
    dky = (ky_max - ky_min) / (nky - 1)
    Nx = int(round(nkx * oversample[0]))
    Ny = int(round(nky * oversample[1]))
    if Nx > nkx or Ny > nky:
        pad = np.zeros((Ny, Nx), np.complex64)
        y0 = (Ny - nky) // 2
        x0 = (Nx - nkx) // 2
        pad[y0:y0 + nky, x0:x0 + nkx] = stage2
        stage2 = pad

    dx_range = 2.0 * np.pi / (Nx * dkx)       # metres/pixel, range
    dy_cross = 2.0 * np.pi / (Ny * dky)       # metres/pixel, cross-range
    res_range = 2.0 * np.pi / (kx_max - kx_min)
    res_cross = 2.0 * np.pi / (ky_max - ky_min)

    fft = np.fft.fft2 if sgn < 0 else np.fft.ifft2
    image = np.fft.fftshift(fft(np.fft.ifftshift(stage2)))
    if verbose:
        print(f"  [image ] {image.shape}  pixel {dx_range:.3f} m (range) x {dy_cross:.3f} m (cross)")
        print(f"  [res   ] range {res_range:.3f} m  cross {res_cross:.3f} m")

    return PFAResult(
        image=image, dx_range=dx_range, dy_cross=dy_cross,
        r_hat=r_hat, c_hat=c_hat, n_hat=n_hat, srp=srp[0],
        res_range=res_range, res_cross=res_cross,
    )
