"""MODIS L0 (MOD00F raw CCSDS packets) -> detector counts.  Our own decommutator.

This is the L0->L1A "unpack" (MOD_PR01) reimplemented from scratch in Python.
No black-box binaries: the packet -> (band, detector, frame) interleave was ported
from the open-source SDST/OCSSW `l1agen_modis` reference
(put_earth_data_in_scan.c / put_cal_data_in_scan.c, PH_pkt_hdr.h, PD_pkt_data.h)
and validated BIT-EXACT against NASA's official L1A (MOD01).

Packet stream (verified against MOD00F.A2026213.0710):
  - all packets APID 64, fixed 642 bytes, 6-byte CCSDS primary + 8-byte timestamp
    + 4-byte MODIS header (bytes 14-17) + science from byte 18.
  - 12-bit samples, 2 per 3 bytes, big-endian (unpack_packet_contents.c).
  - day packet = 415 samples = 5 IFOVs x 83 samples; per IFOV the order is
    250m[band x sample x det] (32) -> 500m (20) -> 1km-day (14) -> 1km-night (17).
  - detectors are REVERSED: line = 9 - scan_ifov;  scan_ifov = ifov (+5 for pkt 2/2).
  - a 1km-night band `band_num` (=band-20) sits at word  ifov*83 + 66 + band_num.

Header bit fields (PH_pkt_hdr.h), byte offsets into the 642-byte packet:
  seq_flag        = P[2] >> 6                      # 1=first pkt, 2=second pkt of frame
  source_id_flag  = P[15] >> 7                     # 0=Earth, 1=Cal
  earth_frame_cnt = ((P[15] & 0x7f) << 4) | (P[16] >> 4)     # 1..1354
  cal_type        = (P[15] >> 5) & 0x3             # 0=SD 1=SRCA 2=BB 3=SV
  cal_frame_cnt   = ((P[15] & 0x03) << 4) | (P[16] >> 4)     # 1..50

Scan structure (verified): each downlink "block" begins at an Earth packet with
earth_frame_cnt==1 and holds EV frames 1..1354 followed by the cal sectors.  The
cal sectors are viewed at scan start, so a block's cal belongs to the *next* block's
Earth view.  Hence, for official (MOD01/MOD021KM) scan s:
      EV(scan s)     = block s+1  (the first L0 block is a leading partial)
      BB/SV(scan s)  = block s  (its trailing cal packets)
"""
import numpy as np

PKT = 642
SCI = 18                      # science byte offset
IFOV_LEN = 83                 # samples per IFOV in a day packet
NIGHT_BASE = 32 + 20 + 14     # 1km-night group starts here within an IFOV (=66)


def read_packets(path):
    """Return the packet stream as an (N, 642) uint8 array."""
    b = np.fromfile(path, dtype=np.uint8)
    n = b.size // PKT
    return b[:n * PKT].reshape(n, PKT)


def _words(P_rows):
    """Vectorized 12-bit unpack (from byte 18) for an (M,642) block of packets.

    Returns (M, 416) uint16 — identical to l1agen_modis unpack_packet_contents.c.
    """
    pay = P_rows[:, SCI:SCI + 624].astype(np.uint16).reshape(-1, 208, 3)
    w = np.empty((P_rows.shape[0], 416), np.uint16)
    w[:, 0::2] = (pay[:, :, 0] << 4) | (pay[:, :, 1] >> 4)
    w[:, 1::2] = ((pay[:, :, 1] & 0x0F) << 8) | pay[:, :, 2]
    return w


def _hdr(P):
    """Decode header fields for every packet -> dict of (N,) arrays."""
    p2, p15, p16 = P[:, 2].astype(int), P[:, 15].astype(int), P[:, 16].astype(int)
    return dict(
        seq=p2 >> 6,
        is_cal=p15 >> 7,
        efc=((p15 & 0x7F) << 4) | (p16 >> 4),
        cal_type=(p15 >> 5) & 0x3,
        cal_fcnt=((p15 & 0x03) << 4) | (p16 >> 4),
    )


def _block_starts(H):
    """Packet indices that begin a downlink block (Earth packet, frame 1, first pkt)."""
    idx = np.where((H["is_cal"] == 0) & (H["efc"] == 1) & (H["seq"] == 1))[0]
    return idx


def decommutate_1km_night(path, band, nscans):
    """Decommutate one 1km-night band (e.g. 31) from L0 -> EV/SV/BB count arrays.

    Returns ev (10*nscans, 1354), sv (10*nscans, 50), bb (10*nscans, 50) in MOD01
    line/frame layout, built entirely from the raw MOD00F packets.
    """
    P = read_packets(path)
    H = _hdr(P)
    starts = _block_starts(H)
    band_num = band - 20                         # 1km-night index (0..16)
    off = NIGHT_BASE + band_num                  # within-IFOV word offset

    NL = 10 * nscans
    ev = np.full((NL, 1354), -1, np.int32)
    sv = np.full((NL, 50), -1, np.int32)
    bb = np.full((NL, 50), -1, np.int32)

    for s in range(nscans):
        # ---- Earth view: block s+1 ----
        b0 = starts[s + 1]
        # 1354 frames * 2 packets, contiguous from the block start
        ev_pkts = P[b0:b0 + 1354 * 2]
        w = _words(ev_pkts)                       # (2708, 416)
        seq = H["seq"][b0:b0 + 1354 * 2]
        for fr in range(1354):
            for half, off5 in ((0, 0), (1, 5)):   # packet 1 (ifov0-4), packet 2 (ifov5-9)
                row = 2 * fr + half
                for ifov in range(5):
                    ev[10 * s + (9 - (ifov + off5)), fr] = w[row, ifov * IFOV_LEN + off]

        # ---- Cal (BB=2, SV=3): trailing cal packets of block s ----
        lo, hi = starts[s], starts[s + 1]
        rng = np.arange(lo, hi)
        for ct, arr in ((2, bb), (3, sv)):
            sel = rng[(H["is_cal"][lo:hi] == 1) & (H["cal_type"][lo:hi] == ct)]
            if sel.size == 0:
                continue
            w = _words(P[sel])
            for j, pk in enumerate(sel):
                fc = H["cal_fcnt"][pk]
                if not (1 <= fc <= 50):
                    continue
                off5 = 0 if H["seq"][pk] == 1 else 5
                for ifov in range(5):
                    arr[10 * s + (9 - (ifov + off5)), fc - 1] = w[j, ifov * IFOV_LEN + off]
    return ev, sv, bb


# Sample-group parameters within a day-mode IFOV (from put_earth_data_in_scan.c).
#   base   = word offset of this group within the 83-sample IFOV
#   ndet   = detectors per IFOV, nsamp = along-scan subsamples, numdet = detectors/scan
#   fmul   = frames per 1km frame (=nsamp)
_RES = {
    "250": dict(first=1, last=2, base=0,  ndet=4, nsamp=4, numdet=40, fmul=4),
    "500": dict(first=3, last=7, base=32, ndet=2, nsamp=2, numdet=20, fmul=2),
}


def _res_of(band):
    if band in (1, 2):
        return "250"
    if 3 <= band <= 7:
        return "500"
    raise ValueError("decommutate_ev_reflective handles 250m/500m bands 1-7")


def decommutate_ev_reflective(path, band, nscans):
    """Decommutate a 250m (band 1-2) or 500m (band 3-7) Earth-view band from L0.

    Returns counts at native resolution: (numdet*nscans, 1354*fmul), MOD01 layout.
    Vectorized over frames.
    """
    P = read_packets(path)
    H = _hdr(P)
    starts = _block_starts(H)
    r = _RES[_res_of(band)]
    bn = band - r["first"]
    ndet, nsamp, numdet, base, fmul = r["ndet"], r["nsamp"], r["numdet"], r["base"], r["fmul"]
    grp = ndet * nsamp                       # words per band per IFOV
    NL, NF = numdet * nscans, 1354 * fmul
    ev = np.full((NL, NF), -1, np.int32)

    for s in range(nscans):
        b0 = starts[s + 1]
        w = _words(P[b0:b0 + 1354 * 2])       # (2708, 416)
        for half, off5 in ((0, 0), (1, 5)):
            wh = w[half::2][:1354]             # (1354, 416) — one packet per frame
            for ifov in range(5):
                si = ifov + off5
                for sample in range(nsamp):
                    for det in range(ndet):
                        pos = ifov * IFOV_LEN + base + bn * grp + sample * ndet + det
                        line = (numdet - 1) - (si * ndet + det)
                        ev[numdet * s + line, sample::fmul][:1354] = wh[:, pos]
    return ev


if __name__ == "__main__":
    import argparse
    from pyhdf.SD import SD, SDC
    ap = argparse.ArgumentParser(description="Decommutate MODIS L0 and validate vs MOD01")
    ap.add_argument("--l0", required=True, help="MOD00F .001 packet file")
    ap.add_argument("--l1a", required=True, help="MOD01 (official L1A) for validation")
    ap.add_argument("--band", type=int, default=31)
    args = ap.parse_args()

    m = SD(args.l1a, SDC.READ)
    nb = args.band - 20
    ev_t = m.select("EV_1km_night").get()[:, nb, :]
    nscans = ev_t.shape[0] // 10 if ev_t.shape[0] % 10 == 0 else 203
    nscans = 203
    ev, sv, bb = decommutate_1km_night(args.l0, args.band, nscans)

    NL = 10 * nscans
    ev_t = m.select("EV_1km_night").get()[:NL, nb, :1354].astype(np.int32)
    sv_t = m.select("SV_1km_night").get()[:NL, nb, :50].astype(np.int32)
    bb_t = m.select("BB_1km_night").get()[:NL, nb, :50].astype(np.int32)
    print(f"=== L0 -> counts, band {args.band}, {nscans} scans (our decommutator vs MOD01) ===")
    for name, a, t in [("EV", ev, ev_t), ("SV", sv, sv_t), ("BB", bb, bb_t)]:
        mism = int((a != t).sum())
        print(f"  {name}: bit-exact={np.array_equal(a, t)}  mismatches={mism}/{a.size}")
