"""True-color RGB straight from MODIS L0 packets.

Decommutates the three visible bands used for natural color -
  band 1 = red   (620-670 nm, 250 m)
  band 4 = green (545-565 nm, 500 m)
  band 3 = blue  (459-479 nm, 500 m)
- directly from MOD00F with our own unpack (src/modis_l0_decom.py), aggregates the
250 m red to the 500 m grid, and composites. Counts only (no radiometric calibration):
a per-channel percentile stretch is all a quicklook true-color needs.
"""
import argparse
import numpy as np
from modis_l0_decom import decommutate_ev_reflective


def stretch(x, lo=1, hi=99, gamma=0.6):
    m = x > 0
    a, b = np.percentile(x[m], [lo, hi])
    y = np.clip((x - a) / (b - a), 0, 1)
    return np.where(m, y ** gamma, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l0", required=True)
    ap.add_argument("--nscans", type=int, default=203)
    ap.add_argument("--out", default="out/modis_l0_truecolor.png")
    args = ap.parse_args()

    print("decommutating bands 1 (red,250m), 4 (green,500m), 3 (blue,500m) from L0...")
    red250 = decommutate_ev_reflective(args.l0, 1, args.nscans).astype(np.float64)
    green = decommutate_ev_reflective(args.l0, 4, args.nscans).astype(np.float64)
    blue = decommutate_ev_reflective(args.l0, 3, args.nscans).astype(np.float64)

    # aggregate 250m red (2x lines, 2x frames) down to the 500m grid
    NL, NF = green.shape
    red = red250[:2 * NL, :2 * NF].reshape(NL, 2, NF, 2).mean(axis=(1, 3))

    rgb = np.dstack([stretch(red), stretch(green), stretch(blue)])
    print(f"true-color RGB {rgb.shape[0]}x{rgb.shape[1]} at 500 m")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 15))
    ax.imshow(rgb, aspect="auto", origin="upper")
    ax.set_title("MODIS/Terra true-color from raw L0 packets\n"
                 "bands 1/4/3 (R/G/B), our own decommutation - A2026213.0710", fontsize=11)
    ax.set_xlabel("along-scan frame (500 m)"); ax.set_ylabel("along-track line")
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
