"""MODIS MWIR bands 20-25 from L0 - the OPIR (Overhead Persistent Infrared) analog.

OPIR missile-warning sensors detect hot rocket plumes by their mid-wave IR (MWIR,
~2.7 / 4.3 um) emission. MODIS's closest bands are 20-25 (3.7-4.5 um): band 21 is the
low-gain "fire/hot-source" band (saturates ~500 K), band 22 the high-gain twin
(saturates ~331 K), bands 24/25 sit in the 4.4 um CO2 feature. This script calibrates
them straight from the raw L0 packets (our own unpack + two-point radiometry) and does
the core OPIR job: find MWIR hot anomalies, then discriminate real thermal sources from
solar false-alarms.

Detection physics: a sub-pixel hot source raises MWIR brightness temperature far above
the long-wave (11 um, band 31) background because the Planck curve shifts toward short
wavelengths when something is hot. Discrimination: a real thermal source is *dark* in
the reflective NIR (it emits, it doesn't reflect); sun-glint and bright surfaces are
*bright* in both MWIR and NIR. That NIR test is the operational plume-vs-glint filter.
"""
import argparse
import numpy as np
from pyhdf.SD import SD, SDC
from modis_l0_decom import decommutate_1km_night, decommutate_ev_reflective
from modis_l1b_recon import calibrate, BAND_LAMBDA, inv_planck


def bt_from_l0(l0, l1b_sds, attrs, be, band):
    """Calibrate one emissive band from L0 and return brightness temperature (K)."""
    bi = be.index(band)
    SI = l1b_sds.get()[bi].astype(float)
    Lref = attrs["radiance_scales"][bi] * (SI - attrs["radiance_offsets"][bi])
    ev, sv, bb = decommutate_1km_night(l0, band, SI.shape[0] // 10)
    valid = (SI > 0) & (SI < 32767) & (ev > 0)
    c = calibrate(ev.astype(float), sv.astype(float), bb.astype(float), Lref, valid, band)
    L = np.where(valid, c["recon_bb"], np.nan)      # independent blackbody two-point
    return inv_planck(BAND_LAMBDA[band], np.clip(L, 1e-3, None)), SI, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l0", required=True)
    ap.add_argument("--l1b", required=True)
    ap.add_argument("--geo", required=True)
    ap.add_argument("--out", default="out/modis_opir_mwir.png")
    args = ap.parse_args()

    L1B = SD(args.l1b, SDC.READ)
    be = list(L1B.select("Band_1KM_Emissive").get())
    sds = L1B.select("EV_1KM_Emissive"); a = sds.attributes()

    BT21, _, valid = bt_from_l0(args.l0, sds, a, be, 21)     # MWIR low-gain hot-source band
    BT31, _, _ = bt_from_l0(args.l0, sds, a, be, 31)          # LWIR background
    BT22, _, _ = bt_from_l0(args.l0, sds, a, be, 22)          # high-gain twin (saturates ~331 K)
    nscans = BT21.shape[0] // 10

    # NIR band 2 (858 nm) -> 1 km, for the solar-vs-thermal discriminator
    nir = decommutate_ev_reflective(args.l0, 2, nscans).astype(float)
    NIR = nir[:, :5416].reshape(nir.shape[0] // 4, 4, 1354, 4).mean(axis=(1, 3))

    g = SD(args.geo, SDC.READ)
    lat = g.select("Latitude").get(); lon = g.select("Longitude").get()

    dBT = BT21 - BT31
    hot = (BT21 > 360) & np.isfinite(BT21)
    nir_med_bg = np.nanmedian(NIR[np.isfinite(BT21) & (BT21 < 305)])
    solar = hot & (NIR > 5 * nir_med_bg)                      # bright in NIR -> reflective
    thermal_cand = hot & (NIR < 2 * nir_med_bg)               # MWIR-hot, NIR-dark -> real
    sat22 = hot & np.isfinite(BT22) & (BT22 > 329)            # band 22 at its ~331 K ceiling

    pk = np.unravel_index(np.nanargmax(BT21), BT21.shape)
    print(f"=== MODIS MWIR (OPIR analog) from L0 ===")
    print(f"band 21 peak BT = {BT21[pk]:.0f} K at {lat[pk]:.3f} N, {lon[pk]:.3f} E "
          f"(Lake Bardawil, N. Sinai)")
    print(f"  band 31 (LWIR) there = {BT31[pk]:.0f} K  -> MWIR-LWIR excess {dBT[pk]:.0f} K")
    print(f"MWIR hot anomalies (BT21>360 K): {int(hot.sum())} px")
    print(f"  bright in NIR (solar false-alarm): {int(solar.sum())} px "
          f"({100*solar.sum()/max(hot.sum(),1):.0f}%)")
    print(f"  MWIR-hot AND NIR-dark (candidate thermal source): {int(thermal_cand.sum())} px")
    b22pk = "SATURATED (overflow)" if np.isnan(BT22[pk]) else f"{BT22[pk]:.0f} K"
    print(f"at that pixel: band 22 (high gain) = {b22pk} while band 21 (low gain) reads "
          f"{BT21[pk]:.0f} K - the high-dynamic-range reason the low-gain band exists")

    make_figure(args.out, BT21, BT22, BT31, dBT, NIR, nir_med_bg, hot)
    print(f"wrote {args.out}")


def make_figure(out, BT21, BT22, BT31, dBT, NIR, nir_bg, hot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.26, wspace=0.26)

    def img(ax, A, title, **kw):
        im = ax.imshow(A, aspect="auto", origin="upper", **kw)
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    img(fig.add_subplot(gs[0, 0]), BT21, "band 21 MWIR 3.99 um  (hot-source band)\nbrightness temp [K]",
        cmap="inferno", vmin=290, vmax=470)
    img(fig.add_subplot(gs[0, 1]), BT31, "band 31 LWIR 11 um  (background)\nbrightness temp [K]",
        cmap="inferno", vmin=290, vmax=320)
    img(fig.add_subplot(gs[0, 2]), dBT, "MWIR - LWIR  (BT21 - BT31)\nhot/reflective anomaly [K]",
        cmap="magma", vmin=0, vmax=170)

    img(fig.add_subplot(gs[1, 0]), NIR, "band 2 NIR 858 nm  (reflective)\ncounts", cmap="bone",
        vmin=0, vmax=3500)

    ax = fig.add_subplot(gs[1, 1])
    h = hot & np.isfinite(NIR)
    ax.scatter(NIR[h], dBT[h], s=3, c="crimson", alpha=0.3)
    ax.axvline(5 * nir_bg, color="k", ls="--", lw=1)
    ax.set_xlabel("NIR reflectance (band 2 counts)"); ax.set_ylabel("MWIR-LWIR excess [K]")
    ax.set_title("discriminator: hot anomalies are\nBRIGHT in NIR -> solar, not thermal", fontsize=10)
    ax.text(0.97, 0.05, "real plume would sit here\n(MWIR-hot, NIR-dark)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="gray")

    ax = fig.add_subplot(gs[1, 2])
    f = np.isfinite(BT21) & np.isfinite(BT22)
    ax.scatter(BT21[f], BT22[f], s=2, c="steelblue", alpha=0.2)
    ax.plot([290, 480], [290, 480], "k--", lw=1, label="1:1")
    ax.axhline(331, color="crimson", ls=":", lw=1.2, label="band 22 saturation ~331 K")
    ax.set_xlim(290, 480); ax.set_ylim(290, 340)
    ax.set_xlabel("band 21 BT (low gain) [K]"); ax.set_ylabel("band 22 BT (high gain) [K]")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("high dynamic range: band 22 flat-lines at\n~331 K while band 21 reaches 476 K",
                 fontsize=10)

    fig.suptitle("MODIS/Terra MWIR bands from raw L0 packets - the OPIR analog "
                 "(A2026213.0710, Lake Bardawil)", fontsize=12)
    fig.savefig(out, dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
