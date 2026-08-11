"""Drive CSA-family (range-Doppler + keystone RCMC): read CPHD -> image + quicklook."""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import csa as alg
from cphd_reader import CPHDReader
from viz import sar_stretch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cphd", default="data/sar/umbra05_bingham.cphd")
    ap.add_argument("--pulse-step", type=int, default=1)
    ap.add_argument("--sample-step", type=int, default=1)
    ap.add_argument("--out", default="out/csa")
    ap.add_argument("--save-complex", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    r = CPHDReader(args.cphd); m = r.meta
    if m.domain_type != "FX":
        raise SystemExit(f"expects FX-domain CPHD; got {m.domain_type}")
    sig = r.read_signal(pulse_step=args.pulse_step, sample_step=args.sample_step)
    g = r.geometry(pulse_step=args.pulse_step)
    scss = g["scss"] * args.sample_step
    print(f"signal {sig.shape}; running CSA ...")
    res = alg.form_image(sig, g["apc"], g["srp"], g["sc0"], scss, sgn=m.sgn)
    print(f"done in {time.time()-t0:.0f}s  pixel {res.dx_range:.3f}x{res.dy_cross:.3f} m")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.imsave(args.out + ".png", sar_stretch(np.abs(res.image)), cmap="gray")
    if args.save_complex:
        np.save(args.out + "_complex.npy", res.image)
    np.savez(args.out + "_geom.npz", r_hat=res.r_hat, c_hat=res.c_hat, n_hat=res.n_hat,
             srp=res.srp, dx_range=res.dx_range, dy_cross=res.dy_cross,
             res_range=res.res_range, res_cross=res.res_cross, iarp_ecf=m.iarp_ecf)
    print(f"wrote {args.out}.png" + (" + _complex.npy" if args.save_complex else ""))


if __name__ == "__main__":
    main()
