"""Form a BPA image on the SICD grid over a chip and save complex + geom + quicklook."""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from cphd_reader import CPHDReader
from sicd_grid import read_sicd_grid
from bpa import backproject
from viz import sar_stretch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cphd", default="data/sar/umbra05_bingham.cphd")
    ap.add_argument("--sicd", default="data/sar/SICD.nitf")
    ap.add_argument("--chip", type=int, default=2048)
    ap.add_argument("--center", default="scp", help="'scp' or 'row,col'")
    ap.add_argument("--pulse-step", type=int, default=1)
    ap.add_argument("--upsample", type=int, default=4)
    ap.add_argument("--sgn", type=int, default=None, help="Fourier sign convention (default: from CPHD metadata)")
    ap.add_argument("--out", default="out/bpa")
    args = ap.parse_args()

    t0 = time.time()
    r = CPHDReader(args.cphd); m = r.meta
    sgn = args.sgn if args.sgn is not None else m.sgn
    grid = read_sicd_grid(args.sicd)
    if args.center == "scp":
        cr, cc = grid.scp_row, grid.scp_col
    else:
        cr, cc = (float(x) for x in args.center.split(","))
    N = args.chip
    rows = cr + np.arange(-N // 2, N // 2)
    cols = cc + np.arange(-N // 2, N // 2)
    RR, CC = np.meshgrid(rows, cols, indexing="ij")   # axis0=row(range), axis1=col(az)
    pix = grid.pixel_ecef(RR, CC)                      # (N,N,3)

    print(f"reading signal + geometry ...")
    sig = r.read_signal(pulse_step=args.pulse_step)
    g = r.geometry(pulse_step=args.pulse_step)
    print(f"backprojecting {N}x{N} px over {sig.shape[0]} pulses (upsample {args.upsample}, sgn={sgn})...")
    res = backproject(sig, g["apc"], g["srp"], g["sc0"], g["scss"], pix,
                      grid.uRow, grid.uCol, grid.ss_row, grid.ss_col,
                      sgn=sgn, upsample=args.upsample, progress=True)
    print(f"done in {time.time()-t0:.0f}s")

    # image is [row=range, col=azimuth]; store geom so coherence.py auto-orients.
    # coherence.py expects axis0=azimuth,axis1=range -> transpose to that convention.
    img = res.image.T                                   # -> [azimuth, range]
    np.save(args.out + "_complex.npy", img)
    np.savez(args.out + "_geom.npz", r_hat=res.r_hat, c_hat=res.c_hat, n_hat=res.n_hat,
             srp=res.srp, dx_range=res.dx_range, dy_cross=res.dy_cross,
             res_range=res.dx_range, res_cross=res.dy_cross, iarp_ecf=res.srp)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.imsave(args.out + ".png", sar_stretch(np.abs(img)), cmap="gray")
    print(f"wrote {args.out}_complex.npy / .png  ({img.shape})")


if __name__ == "__main__":
    main()
