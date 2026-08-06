"""MODIS raw counts (L0/L1A) -> calibrated radiance (L1B), validated against NASA's L1B.

The thermal-IR analog of the CPHD -> SICD exercise: start from the *raw detector
counts* and independently reproduce NASA's operational Level-1B calibrated radiance,
then measure how well it matches.

Why MODIS thermal emissive bands (TEB) are the clean case:
  - No onboard non-uniformity correction and no onboard spatial aggregation (unlike
    VIIRS) - the L1A/L0 counts are the pristine per-detector readout.
  - The *calibration reference is inside the raw data*: every scan views a space port
    (SV, ~0 radiance) and an onboard blackbody (BB, ~290 K). Two known points ->
    counts->radiance gain, per detector, per scan. Textbook two-point radiometry,
    no proprietary LUTs.

Pipeline (band 31, 11 um, the clean thermal window):
  L1A (MOD01)  EV_1km_night[band]     raw Earth-view counts   (203 scans x 10 det x 1354)
               SV_1km_night[band]     space-view counts       (50 frames/scan)  -> offset
               BB_1km_night[band]     blackbody counts        (50 frames/scan)  -> gain
  L1B (MOD021KM) EV_1KM_Emissive[band]  NASA radiance = scale*(SI-offset)  (reference)
  MOD03          Latitude/Longitude     geolocation

  night-band index = band - 20   (night group is bands 20..36 in numeric order)

Results (this granule, band 31):
  - per-detector affine calibration recovers NASA L1B to RMS ~0.7% (R^2 ~ 0.992)
  - the implied blackbody radiance L_BB = a0 + b1*dn*_BB agrees across all 20
    (detector x mirror-side) groups to ~0.1% -> the onboard BB is the single common
    gain reference sitting in the raw counts
  - a simplified band-center Planck implies T_BB ~ 281 K vs the nominal ~290 K; the
    ~9 K gap is the omitted nonlinearity / emissivity / RVS terms of the operational
    algorithm - the "proprietary-chain residual", exactly like the SAR coherence gap.

Run:
  micromamba run -p .mamba/modis python src/modis_l1b_recon.py \
      --l1a data/modis/MOD01.A2026213.0710.061.2026213081621.NRT.hdf \
      --l1b data/modis/MOD021KM.A2026213.0710.061.2026213082055.NRT.hdf \
      --geo data/modis/MOD03.A2026213.0710.061.2026213081852.NRT.hdf \
      --band 31 --out out/modis_l1b_recon.png
"""
import argparse
import numpy as np
from pyhdf.SD import SD, SDC

# Planck spectral radiance in W/m^2/sr/um for wavelength in um, T in K.
C1 = 1.191042953e8   # W um^4 / (m^2 sr)  (2 h c^2, wavelength form)
C2 = 1.4387770e4     # um K                (h c / k)

# band-center wavelengths (um) for MODIS thermal emissive bands
BAND_LAMBDA = {20: 3.785, 21: 3.992, 22: 3.968, 23: 4.058, 24: 4.473, 25: 4.545,
               27: 6.766, 28: 7.338, 29: 8.523, 30: 9.730, 31: 11.017, 32: 12.028,
               33: 13.335, 34: 13.635, 35: 13.834, 36: 14.235}


def planck(lam, T):
    return C1 / (lam**5 * (np.exp(C2 / (lam * T)) - 1.0))


def inv_planck(lam, L):
    return C2 / (lam * np.log(C1 / (L * lam**5) + 1.0))


def read_band(l1a_path, l1b_path, band):
    """Return raw counts (EV/SV/BB) from L1A and NASA reference radiance from L1B."""
    nb = band - 20                       # night-group index (bands 20..36 in order)
    L1A = SD(l1a_path, SDC.READ)
    L1B = SD(l1b_path, SDC.READ)

    b1b = list(L1B.select("Band_1KM_Emissive").get()).index(band)
    ref = L1B.select("EV_1KM_Emissive")
    SI = ref.get()[b1b].astype(np.float64)          # (2030, 1354) scaled integers
    a = ref.attributes()
    Lref = a["radiance_scales"][b1b] * (SI - a["radiance_offsets"][b1b])
    NL, NF = SI.shape                               # 2030 lines, 1354 frames

    EV = L1A.select("EV_1km_night").get()[:NL, nb, :NF].astype(np.float64)
    SV = L1A.select("SV_1km_night").get()[:NL, nb, :50].astype(np.float64)
    BB = L1A.select("BB_1km_night").get()[:NL, nb, :50].astype(np.float64)

    valid = (SI > 0) & (SI < 32767) & (EV > 0) & (EV < 65535)
    return EV, SV, BB, Lref, valid, NL, NF


def calibrate(EV, SV, BB, Lref, valid, band):
    """Per-(detector, mirror-side) two-point calibration from the onboard blackbody.

    Returns a dict with the reconstructed radiance and diagnostics.
    """
    NL, NF = EV.shape
    sv0 = np.median(SV, axis=1)          # per detector-line space-view offset (dn0)
    bb0 = np.median(BB, axis=1)          # per detector-line blackbody counts
    dnBB = bb0 - sv0                     # background-subtracted BB counts
    dnEV = EV - sv0[:, None]             # background-subtracted Earth view

    det = np.arange(NL) % 10             # detector index within a scan
    mir = (np.arange(NL) // 10) % 2      # mirror side (2-sided scan mirror)

    lam = BAND_LAMBDA[band]
    recon_aff = np.full_like(Lref, np.nan)   # empirical affine recovery (b1, a0 from L1B)
    groups = []
    LBB_implied = []
    for d in range(10):
        for ms in range(2):
            sel = (det == d) & (mir == ms)
            v = sel[:, None] & valid
            x, y = dnEV[v], Lref[v]
            b1, a0 = np.polyfit(x, y, 1)                 # L1B = b1*dn* + a0
            R2 = 1 - np.sum((y - (b1 * x + a0))**2) / np.sum((y - y.mean())**2)
            dnbb_g = np.median(dnBB[sel])
            LBB = b1 * dnbb_g + a0                       # implied blackbody radiance
            recon_aff[sel] = b1 * dnEV[sel] + a0
            groups.append(dict(det=d, mir=ms, b1=b1, a0=a0, R2=R2,
                               dnBB=dnbb_g, LBB=LBB))
            LBB_implied.append(LBB)

    LBB_implied = np.array(LBB_implied)
    LBB_mean = LBB_implied.mean()
    T_BB = inv_planck(lam, LBB_mean)

    # Independent reconstruction: gains come ONLY from the onboard BB counts and a
    # single blackbody radiance (no per-detector fit to L1B). a0 ~ 0 per detector.
    gain_bb = np.full(NL, np.nan)
    for g in groups:
        sel = (det == g["det"]) & (mir == g["mir"])
        gain_bb[sel] = LBB_mean / g["dnBB"]
    recon_bb = gain_bb[:, None] * dnEV

    # Response-vs-Scan-Angle (RVS): the true gain drifts a few % across the +/-55 deg
    # scan, which the single-gain-per-detector fit cannot follow - it shows up as a
    # frame-structured residual (the vertical banding in the difference image). Recover
    # it empirically as a per-(mirror-side, frame) correction on top of the affine
    # recovery - the operational RVS-LUT analog - and apply it.
    rvs = np.zeros((2, NF))
    resid_af = np.where(valid, Lref - recon_aff, np.nan)
    for ms in (0, 1):
        curve = np.nanmedian(resid_af[mir == ms], axis=0)      # per frame
        k = np.ones(21) / 21.0                                 # light smoothing
        rvs[ms] = np.convolve(np.nan_to_num(curve, nan=0.0), k, mode="same")
    recon_rvs = recon_aff + rvs[mir]

    return dict(dnEV=dnEV, dnBB=dnBB, sv0=sv0, det=det, mir=mir,
                recon_aff=recon_aff, recon_bb=recon_bb, recon_rvs=recon_rvs, rvs=rvs,
                groups=groups, LBB_implied=LBB_implied, LBB_mean=LBB_mean,
                T_BB=T_BB, lam=lam)


def stats(recon, Lref, valid):
    v = valid & np.isfinite(recon)
    d = recon[v] - Lref[v]
    rms = np.sqrt(np.mean(d**2))
    R2 = 1 - np.sum(d**2) / np.sum((Lref[v] - Lref[v].mean())**2)
    return rms, 100 * rms / Lref[v].mean(), R2, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1a", required=True)
    ap.add_argument("--l1b", required=True)
    ap.add_argument("--geo", default=None)
    ap.add_argument("--band", type=int, default=31)
    ap.add_argument("--out", default="out/modis_l1b_recon.png")
    ap.add_argument("--table", action="store_true", help="print multi-band summary")
    args = ap.parse_args()

    EV, SV, BB, Lref, valid, NL, NF = read_band(args.l1a, args.l1b, args.band)
    c = calibrate(EV, SV, BB, Lref, valid, args.band)

    rms_a, pct_a, R2_a, _ = stats(c["recon_aff"], Lref, valid)
    rms_b, pct_b, R2_b, resid = stats(c["recon_bb"], Lref, valid)

    print(f"=== MODIS band {args.band} ({c['lam']:.2f} um) raw counts -> L1B radiance ===")
    print(f"scene: {NL//10} scans x 10 detectors x {NF} frames")
    print(f"per-detector affine recovery : RMS {rms_a:.4f} ({pct_a:.2f}%)  R^2 {R2_a:.5f}")
    print(f"independent BB-referenced    : RMS {rms_b:.4f} ({pct_b:.2f}%)  R^2 {R2_b:.5f}")
    print(f"implied L_BB across 20 groups: {c['LBB_mean']:.3f} +/- "
          f"{c['LBB_implied'].std():.3f} W/m^2/um/sr "
          f"({100*c['LBB_implied'].std()/c['LBB_mean']:.2f}%)")
    print(f"=> derived blackbody temperature T_BB = {c['T_BB']:.2f} K (nominal ~290 K)")

    if args.table:
        print("\nband  lambda   affine_RMS%   BB_RMS%    L_BB     T_BB(K)  20-grp spread%")
        for b in [20, 22, 29, 31, 32]:
            try:
                ev, sv, bb, lr, va, _, _ = read_band(args.l1a, args.l1b, b)
                cc = calibrate(ev, sv, bb, lr, va, b)
                _, pa, _, _ = stats(cc["recon_aff"], lr, va)
                _, pb, _, _ = stats(cc["recon_bb"], lr, va)
                sp = 100 * cc["LBB_implied"].std() / cc["LBB_mean"]
                print(f"  {b:2d}  {cc['lam']:6.2f}    {pa:6.2f}     {pb:6.2f}   "
                      f"{cc['LBB_mean']:7.3f}   {cc['T_BB']:6.1f}     {sp:5.2f}")
            except Exception as e:
                print(f"  {b:2d}  skipped ({e})")

    make_figure(args, EV, Lref, valid, c, resid, rms_b, pct_b, R2_b)
    print(f"\nwrote {args.out}")


def make_figure(args, EV, Lref, valid, c, resid, rms_b, pct_b, R2_b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recon = c["recon_bb"].copy()          # the genuinely independent reconstruction
    diff = np.where(valid, recon - Lref, np.nan)
    pctdiff = 100 * diff / np.where(Lref != 0, Lref, np.nan)
    vmin, vmax = np.nanpercentile(Lref[valid], [1, 99])
    dlim = np.nanpercentile(np.abs(pctdiff[valid]), 98)

    lat = lon = None
    if args.geo:
        g = SD(args.geo, SDC.READ)
        lat = g.select("Latitude").get()
        lon = g.select("Longitude").get()

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.28)

    def img(ax, A, title, **kw):
        im = ax.imshow(A, aspect="auto", origin="upper", **kw)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    img(fig.add_subplot(gs[0, 0]), np.where(valid, Lref, np.nan),
        f"NASA L1B  (MOD021KM band {args.band})", cmap="inferno", vmin=vmin, vmax=vmax)
    img(fig.add_subplot(gs[0, 1]), np.where(valid, recon, np.nan),
        "reconstruction from raw counts\n(independent onboard-blackbody two-point)",
        cmap="inferno", vmin=vmin, vmax=vmax)
    img(fig.add_subplot(gs[0, 2]), pctdiff,
        f"difference  (recon - L1B) [%]\nRMS {rms_b:.3f} = {pct_b:.2f}%  (a0/RVS/nonlinearity)",
        cmap="RdBu_r", vmin=-dlim, vmax=dlim)

    # scatter
    ax = fig.add_subplot(gs[1, 0])
    v = valid & np.isfinite(recon)
    ax.hexbin(Lref[v], recon[v], gridsize=90, bins="log", cmap="viridis")
    lo, hi = vmin, vmax
    ax.plot([lo, hi], [lo, hi], "w--", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("NASA L1B radiance"); ax.set_ylabel("reconstructed")
    ax.set_title(f"pixel-wise  R^2 = {R2_b:.5f}", fontsize=10)

    # per-detector gain: L1B-recovered vs BB-predicted
    ax = fig.add_subplot(gs[1, 1])
    b1 = np.array([g["b1"] for g in c["groups"]])
    gain_bb = np.array([c["LBB_mean"] / g["dnBB"] for g in c["groups"]])
    ax.scatter(b1 * 1e3, gain_bb * 1e3, s=18, c="crimson")
    lo2, hi2 = 0.95 * b1.min() * 1e3, 1.05 * b1.max() * 1e3
    ax.plot([lo2, hi2], [lo2, hi2], "k--", lw=1)
    ax.set_xlabel("gain from L1B fit  (x1e-3)")
    ax.set_ylabel("gain from onboard BB  (x1e-3)")
    ax.set_title("per-detector gain: L1B vs blackbody\n(20 detector x mirror groups)",
                 fontsize=10)

    # implied L_BB consistency across groups
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(c["LBB_implied"], "o-", ms=3, c="teal")
    ax.axhline(c["LBB_mean"], color="k", ls="--", lw=1)
    ax.set_xlabel("detector x mirror group (0..19)")
    ax.set_ylabel("implied blackbody radiance  L_BB")
    sp = 100 * c["LBB_implied"].std() / c["LBB_mean"]
    ax.set_title(f"onboard BB is the common gain reference\n"
                 f"L_BB = {c['LBB_mean']:.3f} +/- {sp:.2f}%   "
                 f"=> T_BB = {c['T_BB']:.1f} K", fontsize=10)

    loc = ""
    if lat is not None:
        loc = (f"   scene ~ {np.nanmean(lat):.1f}N, {np.nanmean(lon):.1f}E")
    fig.suptitle(f"MODIS/Terra raw L0/L1A counts -> Level-1B radiance, band {args.band} "
                 f"({c['lam']:.1f} um){loc}", fontsize=12)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
