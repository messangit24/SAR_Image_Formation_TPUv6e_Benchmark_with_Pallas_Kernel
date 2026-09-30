"""End-to-end MODIS L0 -> L1B, entirely our own code.

  MOD00F (raw CCSDS packets)                         MOD021KM (reference L1B)
        |                                                     ^
        |  modis_l0_decom.decommutate_1km_night  (our unpack) |  compare
        v                                                     |
   detector counts  --->  modis_l1b_recon.calibrate  --->  radiance

The unpack is validated bit-exact vs MOD01; the calibration is the onboard-blackbody
two-point radiometry.  Nothing here comes from a NASA binary — MOD01 is used only to
prove the unpack, not to produce the answer.
"""
import argparse
import os
import sys
import numpy as np
from pyhdf.SD import SD, SDC

sys.path.insert(0, os.path.dirname(__file__))
from modis_l0_decom import decommutate_1km_night
from modis_l1b_recon import calibrate, stats, make_figure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l0", required=True, help="MOD00F .001 raw packet file")
    ap.add_argument("--l1b", required=True, help="MOD021KM (reference radiance)")
    ap.add_argument("--l1a", default=None, help="MOD01 (optional, to re-check the unpack)")
    ap.add_argument("--geo", default=None)
    ap.add_argument("--band", type=int, default=31)
    ap.add_argument("--out", default="out/modis_l0_to_l1b.png")
    args = ap.parse_args()

    # --- reference L1B radiance for the band ---
    L1B = SD(args.l1b, SDC.READ)
    be = list(L1B.select("Band_1KM_Emissive").get())
    bi = be.index(args.band)
    sds = L1B.select("EV_1KM_Emissive"); a = sds.attributes()
    SI = sds.get()[bi].astype(np.float64)
    Lref = a["radiance_scales"][bi] * (SI - a["radiance_offsets"][bi])
    NL, NF = SI.shape
    nscans = NL // 10

    # --- OUR L0 -> counts ---
    print(f"decommutating band {args.band} from L0 ({nscans} scans)...")
    ev, sv, bb = decommutate_1km_night(args.l0, args.band, nscans)
    ev = ev.astype(np.float64); sv = sv.astype(np.float64); bb = bb.astype(np.float64)

    if args.l1a:
        m = SD(args.l1a, SDC.READ); nb = args.band - 20
        et = m.select("EV_1km_night").get()[:NL, nb, :NF]
        print(f"  unpack check vs MOD01: EV bit-exact={np.array_equal(ev.astype(int), et.astype(int))}")

    valid = (SI > 0) & (SI < 32767) & (ev > 0)

    # --- calibrate our counts -> radiance ---
    c = calibrate(ev, sv, bb, Lref, valid, args.band)
    rms_a, pct_a, R2_a, _ = stats(c["recon_aff"], Lref, valid)
    rms_b, pct_b, R2_b, resid = stats(c["recon_bb"], Lref, valid)
    rms_r, pct_r, R2_r, _ = stats(c["recon_rvs"], Lref, valid)

    print(f"\n=== L0 -> L1B, band {args.band} ({c['lam']:.2f} um), OUR code end-to-end ===")
    print(f"independent BB two-point (SHOWN)  : RMS {rms_b:.4f} ({pct_b:.2f}%)  R^2 {R2_b:.5f}  <- our L1B, no L1B input")
    print(f"per-detector affine (fit to L1B)  : RMS {rms_a:.4f} ({pct_a:.2f}%)  R^2 {R2_a:.5f}")
    print(f"  + RVS (fit to L1B)              : RMS {rms_r:.4f} ({pct_r:.2f}%)  R^2 {R2_r:.5f}")
    print(f"onboard BB radiance across 20 groups: {c['LBB_mean']:.3f} +/- "
          f"{c['LBB_implied'].std():.3f} ({100*c['LBB_implied'].std()/c['LBB_mean']:.2f}%)"
          f"  => T_BB={c['T_BB']:.2f} K")

    make_figure(args, ev, Lref, valid, c, resid, rms_b, pct_b, R2_b)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
