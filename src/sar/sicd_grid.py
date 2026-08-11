"""Read a SICD's output grid geometry so other formers can produce images on the
exact same slant-plane grid (axes, pixel spacing, scene-centre pixel)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sarpy.io.complex.sicd import SICDReader


@dataclass
class SICDGrid:
    uRow: np.ndarray     # (3,) ECEF unit vector, increasing row  (range)
    uCol: np.ndarray     # (3,) ECEF unit vector, increasing col  (azimuth)
    ss_row: float        # m/pixel, range
    ss_col: float        # m/pixel, azimuth
    scp_ecf: np.ndarray  # (3,) scene centre point, ECEF
    scp_row: float       # scene-centre pixel (row)
    scp_col: float       # scene-centre pixel (col)
    n_rows: int
    n_cols: int

    def pixel_ecef(self, rows, cols):
        """ECEF positions for arrays of (row, col) pixel coordinates."""
        rows = np.asarray(rows)[..., None]
        cols = np.asarray(cols)[..., None]
        return (self.scp_ecf
                + (rows - self.scp_row) * self.ss_row * self.uRow
                + (cols - self.scp_col) * self.ss_col * self.uCol)


def read_sicd_grid(path: str) -> SICDGrid:
    s = SICDReader(path).sicd_meta
    g = s.Grid
    return SICDGrid(
        uRow=np.array([g.Row.UVectECF.X, g.Row.UVectECF.Y, g.Row.UVectECF.Z]),
        uCol=np.array([g.Col.UVectECF.X, g.Col.UVectECF.Y, g.Col.UVectECF.Z]),
        ss_row=float(g.Row.SS), ss_col=float(g.Col.SS),
        scp_ecf=np.array([s.GeoData.SCP.ECF.X, s.GeoData.SCP.ECF.Y, s.GeoData.SCP.ECF.Z]),
        scp_row=float(s.ImageData.SCPPixel.Row), scp_col=float(s.ImageData.SCPPixel.Col),
        n_rows=int(s.ImageData.NumRows), n_cols=int(s.ImageData.NumCols),
    )
