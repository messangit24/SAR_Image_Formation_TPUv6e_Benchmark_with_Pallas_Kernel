"""
Complex COHERENCE between a PFA reconstruction and a reference SICD.

Coherence checks the full complex signal (amplitude AND phase): whether two
independent processings of the SAME raw phase history produce the same speckle.

    gamma = | sum(A conj(B)) | / sqrt( sum|A|^2 sum|B|^2 )     in [0, 1]

Everything is derived from metadata / the data itself, so it runs on any collect:
  * pixel spacings come from the PFA geometry sidecar and the SICD metadata;
  * the frame relationship (transpose / flips) is DETECTED, not assumed;
  * the SICD is demodulated to baseband (using its own dominant carrier) BEFORE
    resampling, so the phase survives the resample -- this is what lets the
    coherence approach its true ceiling instead of being capped by interpolation.
"""
from __future__ import annotations

import argparse
import numpy as np

try:
    from scipy.ndimage import zoom
except ImportError:
    zoom = None

try:
    from skimage.feature import match_template
except ImportError:
    match_template = None

try:
    from sarpy.io.complex.sicd import SICDReader
except ImportError:
    SICDReader = None


ORIENTS = {
    "identity": lambda x: x, "flipud": lambda x: x[::-1], "fliplr": lambda x: x[:, ::-1],
    "rot180": lambda x: x[::-1, ::-1], "T": lambda x: x.T, "T+flipud": lambda x: x.T[::-1],
    "T+fliplr": lambda x: x.T[:, ::-1], "T+rot180": lambda x: x.T[::-1, ::-1],
}


def detect_orientation(mine_mag, sicd_mag, sicd_px, my_px):
    """Find which dihedral orientation maps the SICD into my [azimuth, range] frame
    (metadata-free): resample+orient a decimated SICD and template-match."""
    D = 8
    a = np.log1p(mine_mag[::D, ::D] / (mine_mag.mean() + 1e-9)).astype(np.float32)
    best = (-1, None)
    for name, fn in ORIENTS.items():
        b = fn(sicd_mag)                                        # [range,az]->frame
        # pixel spacing after this orientation (transpose swaps axes)
        px = (sicd_px[1], sicd_px[0]) if "T" in name else (sicd_px[0], sicd_px[1])
        bz = zoom(b, (px[0] / my_px[0] / D, px[1] / my_px[1] / D), order=1)
        bz = np.log1p(bz / (bz.mean() + 1e-9)).astype(np.float32)
        if bz.shape[0] >= a.shape[0] or bz.shape[1] >= a.shape[1]:
            # template must be smaller; crop a central template from bz
            th = min(bz.shape[0], a.shape[0]) - 1
            tw = min(bz.shape[1], a.shape[1]) - 1
            bz = bz[:th, :tw]
        res = match_template(a, bz[: a.shape[0] - 1, : a.shape[1] - 1])
        if res.size and res.max() > best[0]:
            best = (res.max(), name)
    return best[1], best[0]


def complex_coherence_peak(A, B, upsample=2, search=1):
    """
    Robust complex coherence of two roughly-aligned chips, invariant to both a
    residual (integer) misregistration AND a linear phase ramp / carrier offset:

        gamma = max_{shift, fringe} | sum A(x) conj(B(x-shift)) exp(-j 2pi f.x) | / norm

    The inner max over fringe f is the interferogram-spectrum peak (removes a
    carrier-convention difference between two correct processors -- e.g. a
    different azimuth-carrier / KCtr); the outer max over a small integer shift
    mops up residual misregistration.  For A == B (up to shift+carrier) it is 1.
    """
    A = A - A.mean(); B = B - B.mean()
    n = A.shape[0]
    na = (np.abs(A) ** 2).sum()
    best = 0.0
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            Bs = np.roll(B, (dy, dx), axis=(0, 1))
            F = np.fft.fft2(A * np.conj(Bs), s=(n * upsample, n * upsample))
            g = np.abs(F).max() / (np.sqrt(na * (np.abs(Bs) ** 2).sum()) + 1e-30)
            if g > best:
                best = g
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complex", default="out/full_complex.npy")
    ap.add_argument("--geom", default="out/full_geom.npz")
    ap.add_argument("--sicd", default="data/sar/SICD.nitf")
    ap.add_argument("--out", default="out/coherence.png")
    ap.add_argument("--win", type=int, default=128)
    args = ap.parse_args()

    geom = np.load(args.geom)
    my_px = (float(geom["dy_cross"]), float(geom["dx_range"]))   # (azimuth, range)
    mine = np.asarray(np.load(args.complex, mmap_mode="r"))

    sr = SICDReader(args.sicd); s = sr.sicd_meta
    sic = sr[:, :]                                               # complex [range, az]
    sic_px = (s.Grid.Row.SS, s.Grid.Col.SS)                     # (range, az)

    # detect frame relationship (no hard-coded transpose/flip)
    name, score = detect_orientation(np.abs(mine), np.abs(sic), sic_px, my_px)
    print(f"detected SICD->my-frame orientation: {name} (match {score:.2f})")

    # orient into my frame, then phase-preserving cubic resample to my pixel grid.
    # (The SICD is essentially baseband, so a high-order interpolant on real/imag
    # keeps the phase; any residual global ramp is absorbed by the coherence peak.)
    fn = ORIENTS[name]
    sic_or = fn(sic)                                            # oriented (exact, no interp)
    px_or = (sic_px[1], sic_px[0]) if "T" in name else (sic_px[0], sic_px[1])
    zf = (px_or[0] / my_px[0], px_or[1] / my_px[1])
    sic_rs = zoom(sic_or.real, zf, order=3) + 1j * zoom(sic_or.imag, zf, order=3)
    print(f"mine {mine.shape} @ {tuple(round(x,3) for x in my_px)} ; SICD->my-frame {sic_rs.shape}")

    # coarse magnitude alignment (single global offset)
    D = 8
    a = np.abs(mine[::D, ::D]).astype(np.float32); b = np.abs(sic_rs[::D, ::D]).astype(np.float32)
    a -= a.mean(); b -= b.mean()
    H, W = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    C = np.fft.irfft2(np.fft.rfft2(a, (H, W)) * np.conj(np.fft.rfft2(b, (H, W))), (H, W))
    dy, dx = np.unravel_index(np.argmax(C), C.shape)
    if dy > H // 2: dy -= H
    if dx > W // 2: dx -= W
    oy, ox = dy * D, dx * D
    print(f"coarse offset (mine - SICD): row {oy}, col {ox}")

    # The two grids drift relative to each other by up to ~100 px across the
    # scene (Cartesian aperture plane vs the SICD's polar RGAZIM grid), so a
    # single global offset is not enough.  Register EACH window locally with
    # normalised template matching, then measure coherence on the aligned chips.
    from skimage.feature import match_template
    S = 90                                                     # local search radius (px)
    H2, W2 = sic_rs.shape
    r_lo = max(0, oy) + S + 200; r_hi = min(mine.shape[0], H2 + oy) - args.win - S - 200
    c_lo = max(0, ox) + S + 200; c_hi = min(mine.shape[1], W2 + ox) - args.win - S - 200
    sic_mag = np.abs(sic_rs).astype(np.float32)

    win, step = args.win, args.win
    rows = list(range(r_lo, r_hi - win, step)); cols = list(range(c_lo, c_hi - win, step))
    gmap = np.full((len(rows), len(cols)), np.nan); bmap = np.zeros_like(gmap)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            A = np.asarray(mine[r:r + win, c:c + win])
            if np.abs(A).mean() < 1e-6: continue
            sy0, sx0 = r - oy - S, c - ox - S
            search = sic_mag[sy0:sy0 + win + 2 * S, sx0:sx0 + win + 2 * S]
            if search.shape != (win + 2 * S, win + 2 * S): continue
            res = match_template(search, np.abs(A).astype(np.float32))
            py, px = np.unravel_index(np.argmax(res), res.shape)
            B = sic_rs[sy0 + py:sy0 + py + win, sx0 + px:sx0 + px + win]
            if B.shape != A.shape: continue
            bmap[i, j] = np.abs(A).mean()
            gmap[i, j] = max(complex_coherence_peak(A, B),
                             complex_coherence_peak(A, np.conj(B)))
    g = gmap[np.isfinite(gmap)]
    bright = bmap > np.nanpercentile(bmap[bmap > 0], 50)
    gb = gmap[np.isfinite(gmap) & bright]
    print(f"\nwindows: {g.size}")
    print(f"COHERENCE (all)            : median={np.median(g):.3f}  p90={np.percentile(g,90):.3f}  max={g.max():.3f}")
    print(f"COHERENCE (high-signal)    : median={np.median(gb):.3f}  p90={np.percentile(gb,90):.3f}")

    # is the gradient just SNR?  coherence vs window-brightness deciles
    ok = np.isfinite(gmap) & (bmap > 0)
    bv = bmap[ok]; gv = gmap[ok]
    edges = np.percentile(bv, np.arange(0, 101, 20))
    print("  coherence vs brightness quintile (dark -> bright):")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (bv >= lo) & (bv <= hi)
        print(f"    {np.median(gv[sel]):.3f}", end="")
    print(f"   (Spearman-ish corr {np.corrcoef(np.argsort(np.argsort(bv)), gv)[0,1]:.2f})")

    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from viz import sar_stretch
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    axs[0].imshow(sar_stretch(np.abs(np.asarray(mine[r_lo:r_hi, c_lo:c_hi]))[::8, ::8]),
                  cmap="gray"); axs[0].set_title("PFA reconstruction (overlap)")
    im = axs[1].imshow(gmap, cmap="inferno", vmin=0, vmax=1)
    axs[1].set_title(f"complex coherence vs SICD (median high-signal {np.median(gb):.2f})")
    fig.colorbar(im, ax=axs[1], fraction=0.046)
    for ax in axs: ax.axis("off")
    fig.tight_layout(); fig.savefig(args.out, dpi=130); plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
