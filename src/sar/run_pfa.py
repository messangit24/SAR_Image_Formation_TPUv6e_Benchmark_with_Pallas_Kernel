"""
Drive the PFA pipeline on an Umbra CPHD file:
    read CPHD  ->  polar-format image formation  ->  complex image + quicklook.

Examples:
    # fast decimated quicklook (every 4th pulse & sample)
    python src/run_pfa.py --pulse-step 4 --sample-step 4 --out out/quicklook
    # full resolution
    python src/run_pfa.py --out out/full
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from cphd_reader import CPHDReader
from pfa import form_image


def db_quicklook(mag: np.ndarray, path: str, title: str, flipud=False, fliplr=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = mag.copy()
    if flipud:
        img = img[::-1]
    if fliplr:
        img = img[:, ::-1]
    db = 20 * np.log10(img + 1e-6)
    lo, hi = np.percentile(db, [5, 99.7])
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(db, cmap="gray", vmin=lo, vmax=hi, aspect="equal", origin="upper")
    ax.set_title(title)
    # image axis0 = cross-range (Ky), axis1 = range (Kx)
    ax.set_xlabel("range (px)")
    ax.set_ylabel("cross-range / azimuth (px)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cphd", default="data/sar/umbra05_bingham.cphd")
    ap.add_argument("--pulse-step", type=int, default=1)
    ap.add_argument("--sample-step", type=int, default=1)
    ap.add_argument("--out", default="out/pfa")
    ap.add_argument("--interp", choices=["cubic", "sinc"], default="cubic",
                    help="k-space interpolation kernel (sinc=8-tap Lanczos-4)")
    ap.add_argument("--taylor-nbar", type=int, default=0, help="Taylor nbar (0=off)")
    ap.add_argument("--taylor-sll", type=float, default=-35.0)
    ap.add_argument("--save-complex", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    r = CPHDReader(args.cphd)
    m = r.meta
    print(f"CPHD {m.version} domain={m.domain_type} SGN={m.sgn} "
          f"vectors={m.num_vectors} samples={m.num_samples} channel={m.channel_id}")
    if m.domain_type != "FX":
        raise SystemExit(f"This PFA path expects FX (frequency) domain CPHD; got "
                         f"DomainType={m.domain_type}. (Umbra open-data CPHD is FX.)")

    print(f"reading signal (pulse_step={args.pulse_step}, sample_step={args.sample_step})...")
    sig = r.read_signal(pulse_step=args.pulse_step, sample_step=args.sample_step)
    g = r.geometry(pulse_step=args.pulse_step)
    # decimate the per-sample frequency step to match sample decimation
    scss = g["scss"] * args.sample_step
    print(f"  signal {sig.shape} ({sig.nbytes/1e6:.0f} MB)  read in {time.time()-t0:.1f}s")

    taylor = (args.taylor_nbar, args.taylor_sll) if args.taylor_nbar > 0 else None
    print("running PFA...")
    res = form_image(
        signal=sig, apc=g["apc"], srp=g["srp"], sc0=g["sc0"], scss=scss,
        uiax=m.uiax, uiay=m.uiay, sgn=m.sgn, taylor=taylor, interp=args.interp,
    )

    mag = np.abs(res.image)
    print(f"done in {time.time()-t0:.1f}s. image {res.image.shape} "
          f"pixel {res.dx_range:.3f}x{res.dy_cross:.3f} m, "
          f"res {res.res_range:.3f}x{res.res_cross:.3f} m")

    db_quicklook(mag, args.out + ".png",
                 f"Umbra PFA reconstruction ({sig.shape[0]}x{sig.shape[1]} phase history)")
    if args.save_complex:
        np.save(args.out + "_complex.npy", res.image)
        print(f"  wrote {args.out}_complex.npy")
    # save geometry sidecar for georeferencing / comparison
    np.savez(args.out + "_geom.npz",
             r_hat=res.r_hat, c_hat=res.c_hat, n_hat=res.n_hat, srp=res.srp,
             dx_range=res.dx_range, dy_cross=res.dy_cross,
             res_range=res.res_range, res_cross=res.res_cross,
             iarp_ecf=m.iarp_ecf)


if __name__ == "__main__":
    main()
