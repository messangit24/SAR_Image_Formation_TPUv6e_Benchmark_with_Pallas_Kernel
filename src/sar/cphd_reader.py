"""
Self-contained reader for CPHD 1.x files (tested on Umbra open-data CPHD 1.1.0).

A CPHD file is:
    [ASCII text header]\f\n
    [XML metadata block]         (product/collection/geometry description)
    [support arrays]             (antenna patterns, ...)
    [PVP block]                  (Per-Vector Parameters: one record per pulse)
    [signal block]               (the phase-history samples themselves)

The text header tells us the byte offset/size of every block; the XML tells us
the array dimensions, the per-vector-parameter layout, and the collection
geometry.  All binary payloads are big-endian (NGA convention).

We avoid heavyweight dependencies and read the signal / PVP directly with
numpy memmaps so that huge files (GBs) are never fully loaded unless asked for,
and so decimation is cheap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from xml.etree import ElementTree as ET

import numpy as np

C = 299_792_458.0  # speed of light, m/s


# -----------------------------------------------------------------------------
# header / xml parsing
# -----------------------------------------------------------------------------
def _parse_text_header(raw: bytes) -> dict:
    """Parse the leading 'KEY := VALUE' ASCII header (terminated by \\f\\n)."""
    end = raw.find(b"\f\n")
    if end < 0:
        raise ValueError("not a CPHD file: no form-feed header terminator found")
    text = raw[:end].decode("latin-1")
    lines = text.splitlines()
    version = lines[0].strip()  # e.g. "CPHD/1.1.0"
    fields = {}
    for line in lines[1:]:
        if ":=" in line:
            k, v = line.split(":=", 1)
            fields[k.strip()] = v.strip()
    fields["_VERSION"] = version
    return fields


def _strip_ns(xml_text: str) -> ET.Element:
    """Remove the default namespace so we can use plain tag paths."""
    xml_text = re.sub(r'\sxmlns="[^"]+"', "", xml_text, count=1)
    return ET.fromstring(xml_text)


# PVP field format -> (numpy dtype, number of 8-byte words)
_PVP_FMT = {"F8": (">f8", 1), "I8": (">i8", 1), "U8": (">u8", 1), "CI16": (">i8", 1)}


@dataclass
class PVPField:
    name: str
    offset_words: int     # offset within a vector record, in 8-byte words
    size_words: int       # number of 8-byte words
    subfields: list        # ['X','Y','Z'] for vectors, [] for scalars


@dataclass
class CPHDMeta:
    version: str
    domain_type: str          # 'FX' (frequency) or 'TOA'
    sgn: int                  # Fourier sign convention (+1 / -1)
    num_vectors: int          # pulses (azimuth)
    num_samples: int          # range-frequency samples
    signal_format: str        # 'CF8' -> complex64
    num_bytes_pvp: int
    channel_id: str           # reference channel identifier
    signal_ch_offset: int     # channel's byte offset within the signal block
    pvp_ch_offset: int        # channel's byte offset within the PVP block
    fxmin: float
    fxmax: float
    # geometry / scene
    iarp_ecf: np.ndarray      # (3,) scene reference point, ECEF metres
    uiax: np.ndarray          # (3,) image-area X axis (ground plane)
    uiay: np.ndarray          # (3,) image-area Y axis (ground plane)
    ref_vector_index: int
    fxc: float
    fxbw: float
    slant_range: float
    graze_deg: float
    dca_deg: float            # Doppler cone angle
    side_of_track: str
    xml: ET.Element           # full parsed XML root (namespace-stripped)


class CPHDReader:
    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            head = f.read(4096)
        self.header = _parse_text_header(head)

        self._xml_off = int(self.header["XML_BLOCK_BYTE_OFFSET"])
        self._xml_sz = int(self.header["XML_BLOCK_SIZE"])
        self._pvp_off = int(self.header["PVP_BLOCK_BYTE_OFFSET"])
        self._sig_off = int(self.header["SIGNAL_BLOCK_BYTE_OFFSET"])

        with open(path, "rb") as f:
            f.seek(self._xml_off)
            xml_text = f.read(self._xml_sz).decode("utf-8")
        self.root = _strip_ns(xml_text)
        self.meta = self._build_meta()
        self._pvp_layout = self._parse_pvp_layout()

    # ---- helpers to pull values out of the XML -----------------------------
    def _t(self, path):
        e = self.root.find(path)
        return e.text.strip() if e is not None and e.text else None

    def _xyz(self, path, keys=("X", "Y", "Z")):
        e = self.root.find(path)
        if e is None:
            return None
        return np.array([float(e.find(k).text) for k in keys])

    def _build_meta(self) -> CPHDMeta:
        r = self.root
        # pick the reference channel named by Channel/RefChId (fall back to first)
        ref_id = self._t("Channel/RefChId")
        channels = r.findall("Data/Channel")
        ch = next((c for c in channels if c.findtext("Identifier") == ref_id), channels[0])

        def _chint(tag, default=0):
            e = ch.find(tag)
            return int(e.text) if e is not None and e.text else default

        return CPHDMeta(
            version=self.header["_VERSION"],
            domain_type=self._t("Global/DomainType"),
            sgn=int(self._t("Global/SGN")),
            num_vectors=int(ch.find("NumVectors").text),
            num_samples=int(ch.find("NumSamples").text),
            signal_format=self._t("Data/SignalArrayFormat"),
            num_bytes_pvp=int(self._t("Data/NumBytesPVP")),
            channel_id=ch.findtext("Identifier"),
            signal_ch_offset=_chint("SignalArrayByteOffset"),
            pvp_ch_offset=_chint("PVPArrayByteOffset"),
            fxmin=float(self._t("Global/FxBand/FxMin")),
            fxmax=float(self._t("Global/FxBand/FxMax")),
            iarp_ecf=self._xyz("SceneCoordinates/IARP/ECF"),
            uiax=self._xyz("SceneCoordinates/ReferenceSurface/Planar/uIAX"),
            uiay=self._xyz("SceneCoordinates/ReferenceSurface/Planar/uIAY"),
            ref_vector_index=int(self._t("Channel/Parameters/RefVectorIndex")),
            fxc=float(self._t("Channel/Parameters/FxC")),
            fxbw=float(self._t("Channel/Parameters/FxBW")),
            slant_range=float(self._t("ReferenceGeometry/Monostatic/SlantRange")),
            graze_deg=float(self._t("ReferenceGeometry/Monostatic/GrazeAngle")),
            dca_deg=float(self._t("ReferenceGeometry/Monostatic/DopplerConeAngle")),
            side_of_track=self._t("ReferenceGeometry/Monostatic/SideOfTrack"),
            xml=r,
        )

    def _parse_pvp_layout(self) -> dict:
        layout = {}
        pvp = self.root.find("PVP")
        for fld in pvp:
            off = int(fld.find("Offset").text)
            size = int(fld.find("Size").text)
            fmt = fld.find("Format").text.strip()
            subs = []
            if ";" in fmt and "=" in fmt:  # e.g. "X=F8;Y=F8;Z=F8;"
                subs = [p.split("=")[0] for p in fmt.split(";") if p]
            layout[fld.tag] = PVPField(fld.tag, off, size, subs)
        return layout

    # ---- binary payload access ---------------------------------------------
    @property
    def signal_dtype(self):
        return {"CF8": ">c8", "CF16": ">c16"}[self.meta.signal_format]

    def pvp_memmap(self) -> np.ndarray:
        """(num_vectors, num_bytes_pvp/8) big-endian float64 view of the PVP block."""
        nwords = self.meta.num_bytes_pvp // 8
        return np.memmap(self.path, dtype=">f8", mode="r",
                         offset=self._pvp_off + self.meta.pvp_ch_offset,
                         shape=(self.meta.num_vectors, nwords))

    def read_pvp(self, name: str, pulse_step: int = 1) -> np.ndarray:
        """Return a named per-vector parameter, decimated by pulse_step."""
        f = self._pvp_layout[name]
        pvp = self.pvp_memmap()[::pulse_step]
        cols = pvp[:, f.offset_words: f.offset_words + f.size_words]
        arr = np.ascontiguousarray(cols).astype(np.float64)
        return arr[:, 0] if f.size_words == 1 else arr

    def signal_memmap(self) -> np.ndarray:
        return np.memmap(self.path, dtype=self.signal_dtype, mode="r",
                         offset=self._sig_off + self.meta.signal_ch_offset,
                         shape=(self.meta.num_vectors, self.meta.num_samples))

    def read_signal(self, pulse_step: int = 1, sample_step: int = 1,
                    pulse_slice: slice | None = None,
                    sample_slice: slice | None = None) -> np.ndarray:
        """Read (decimated / cropped) signal into a native complex64 array."""
        sig = self.signal_memmap()
        ps = pulse_slice or slice(None)
        ss = sample_slice or slice(None)
        sub = sig[ps, ss][::pulse_step, ::sample_step]
        return np.ascontiguousarray(sub).astype(np.complex64)

    # ---- convenience: everything PFA needs ---------------------------------
    def geometry(self, pulse_step: int = 1):
        """Per-pulse APC (monostatic phase centre), SRP, and frequency sampling."""
        tx = self.read_pvp("TxPos", pulse_step)
        rcv = self.read_pvp("RcvPos", pulse_step)
        srp = self.read_pvp("SRPPos", pulse_step)
        sc0 = self.read_pvp("SC0", pulse_step)
        scss = self.read_pvp("SCSS", pulse_step)
        if np.isnan(tx).any():
            valid = ~np.isnan(tx[:, 0])
            if valid.any():
                idxs = np.arange(len(tx))
                deg = min(3, int(valid.sum()) - 1)
                tx_clean = tx.copy()
                for dim in range(3):
                    p = np.poly1d(np.polyfit(idxs[valid], tx[valid, dim], deg=deg))
                    tx_clean[~valid, dim] = p(idxs[~valid])
                tx = tx_clean
            else:
                tx = rcv.copy()
        apc = 0.5 * (tx + rcv)          # effective monostatic phase centre
        return dict(apc=apc, srp=srp, sc0=sc0, scss=scss,
                    txpos=tx, rcvpos=rcv)


if __name__ == "__main__":
    import sys
    r = CPHDReader(sys.argv[1] if len(sys.argv) > 1 else "data/sar/umbra05_bingham.cphd")
    m = r.meta
    print(f"CPHD {m.version}  domain={m.domain_type} SGN={m.sgn}")
    print(f"  vectors={m.num_vectors} samples={m.num_samples} fmt={m.signal_format}")
    print(f"  Fx {m.fxmin/1e9:.4f}-{m.fxmax/1e9:.4f} GHz  FxC={m.fxc/1e9:.4f} BW={m.fxbw/1e6:.1f} MHz")
    print(f"  slant_range={m.slant_range/1e3:.2f} km graze={m.graze_deg:.2f} DCA={m.dca_deg:.2f} side={m.side_of_track}")
    print(f"  IARP ECF={m.iarp_ecf}")
    print(f"  uIAX={m.uiax}\n  uIAY={m.uiay}")
    g = r.geometry(pulse_step=2000)
    print(f"  APC[0]={g['apc'][0]}  SC0={g['sc0'][0]:.6e} SCSS={g['scss'][0]:.6e}")
