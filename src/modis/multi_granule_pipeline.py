"""Multi-Granule MODIS Ingestion & MWIR Thermal Anomaly Aggregation Pipeline.

Ingests sequential or batched MODIS L0/L1B telemetry granules, runs dual-band thermal anomaly
detection, performs 8-connected spatial clustering on active pixels, and aggregates temporal
time-series into structured GeoJSON and JSON detection catalogs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.ndimage import label

from .mwir_detector import (
    DetectionConfig,
    ThermalDetection,
    detect_thermal_anomalies,
)


@dataclass
class ThermalCluster:
    """Spatially aggregated cluster of contiguous active thermal anomaly pixels."""
    cluster_id: int
    granule_id: str
    pixel_count: int
    center_lat: float | None
    center_lon: float | None
    peak_bt21: float
    mean_bt31: float
    peak_delta_bt: float
    total_frp_mw: float
    is_solar_false_alarm: bool
    is_thermal_candidate: bool
    pixel_coords: list[tuple[int, int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GranuleResult:
    """Processing results for a single MODIS granule."""
    granule_id: str
    n_scans: int
    n_frames: int
    total_detections: int
    thermal_candidate_count: int
    solar_false_alarm_count: int
    total_frp_mw: float
    clusters: list[ThermalCluster]
    detections: list[ThermalDetection]

    def to_dict(self) -> dict[str, Any]:
        return {
            "granule_id": self.granule_id,
            "n_scans": self.n_scans,
            "n_frames": self.n_frames,
            "total_detections": self.total_detections,
            "thermal_candidate_count": self.thermal_candidate_count,
            "solar_false_alarm_count": self.solar_false_alarm_count,
            "total_frp_mw": self.total_frp_mw,
            "clusters": [c.to_dict() for c in self.clusters],
            "detections": [d.to_dict() for d in self.detections],
        }


class MultiGranuleCatalog:
    """Aggregated catalog of thermal anomalies across multi-granule time-series."""

    def __init__(self) -> None:
        self.granule_results: list[GranuleResult] = []

    def add_granule_result(self, result: GranuleResult) -> None:
        self.granule_results.append(result)

    @property
    def total_granules(self) -> int:
        return len(self.granule_results)

    @property
    def total_detections(self) -> int:
        return sum(g.total_detections for g in self.granule_results)

    @property
    def total_thermal_candidates(self) -> int:
        return sum(g.thermal_candidate_count for g in self.granule_results)

    @property
    def total_solar_false_alarms(self) -> int:
        return sum(g.solar_false_alarm_count for g in self.granule_results)

    @property
    def total_frp_mw(self) -> float:
        return sum(g.total_frp_mw for g in self.granule_results)

    @property
    def all_clusters(self) -> list[ThermalCluster]:
        clusters = []
        for g in self.granule_results:
            clusters.extend(g.clusters)
        return clusters

    def to_geojson(self) -> dict[str, Any]:
        """Export all thermal candidate clusters and detections as standard GeoJSON FeatureCollection."""
        features = []
        for g in self.granule_results:
            for c in g.clusters:
                if c.center_lat is not None and c.center_lon is not None:
                    geom = {
                        "type": "Point",
                        "coordinates": [float(c.center_lon), float(c.center_lat)],
                    }
                else:
                    geom = None

                features.append(
                    {
                        "type": "Feature",
                        "geometry": geom,
                        "properties": {
                            "granule_id": c.granule_id,
                            "cluster_id": c.cluster_id,
                            "pixel_count": c.pixel_count,
                            "peak_bt21_k": c.peak_bt21,
                            "mean_bt31_k": c.mean_bt31,
                            "peak_delta_bt_k": c.peak_delta_bt,
                            "total_frp_mw": c.total_frp_mw,
                            "is_thermal_candidate": c.is_thermal_candidate,
                            "is_solar_false_alarm": c.is_solar_false_alarm,
                        },
                    }
                )

        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "total_granules": self.total_granules,
                "total_detections": self.total_detections,
                "total_thermal_candidates": self.total_thermal_candidates,
                "total_frp_mw": round(self.total_frp_mw, 2),
            },
        }

    def save_geojson(self, filepath: str | Path) -> None:
        """Write catalog as GeoJSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_geojson(), f, indent=2)

    def save_json(self, filepath: str | Path) -> None:
        """Write full catalog as JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([g.to_dict() for g in self.granule_results], f, indent=2)


def cluster_thermal_detections(
    detections: list[ThermalDetection],
    shape: tuple[int, int],
    granule_id: str,
) -> list[ThermalCluster]:
    """Group 8-connected active thermal anomaly pixels into spatial clusters."""
    if not detections:
        return []

    grid = np.zeros(shape, dtype=int)
    det_map: dict[tuple[int, int], ThermalDetection] = {}
    for d in detections:
        grid[d.pixel_row, d.pixel_col] = 1
        det_map[(d.pixel_row, d.pixel_col)] = d

    # 8-connected structuring element
    struct = np.ones((3, 3), dtype=int)
    labeled, num_clusters = label(grid, structure=struct)

    clusters: list[ThermalCluster] = []
    for cid in range(1, num_clusters + 1):
        c_rows, c_cols = np.where(labeled == cid)
        c_dets = [det_map[(r, c)] for r, c in zip(c_rows, c_cols)]

        # Cluster aggregations
        lats = [d.lat for d in c_dets if d.lat is not None]
        lons = [d.lon for d in c_dets if d.lon is not None]
        center_lat = float(np.mean(lats)) if lats else None
        center_lon = float(np.mean(lons)) if lons else None

        peak_bt21 = max(d.bt21 for d in c_dets)
        mean_bt31 = float(np.mean([d.bt31 for d in c_dets]))
        peak_delta_bt = max(d.delta_bt for d in c_dets)

        frp_vals = [d.frp_mw for d in c_dets if d.frp_mw is not None]
        total_frp = float(sum(frp_vals)) if frp_vals else 0.0

        # If any pixel is a genuine thermal candidate, mark cluster as candidate
        is_thermal = any(d.is_thermal_candidate for d in c_dets)
        is_solar = all(d.is_solar_false_alarm for d in c_dets)

        clusters.append(
            ThermalCluster(
                cluster_id=int(cid),
                granule_id=granule_id,
                pixel_count=len(c_dets),
                center_lat=center_lat,
                center_lon=center_lon,
                peak_bt21=float(peak_bt21),
                mean_bt31=float(mean_bt31),
                peak_delta_bt=float(peak_delta_bt),
                total_frp_mw=float(total_frp),
                is_solar_false_alarm=bool(is_solar),
                is_thermal_candidate=bool(is_thermal),
                pixel_coords=[(int(r), int(c)) for r, c in zip(c_rows, c_cols)],
            )
        )

    return clusters


def process_granule_arrays(
    bt21: np.ndarray,
    bt31: np.ndarray,
    granule_id: str,
    nir: np.ndarray | None = None,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    water_mask: np.ndarray | None = None,
    config: DetectionConfig | None = None,
    is_day: bool = True,
) -> GranuleResult:
    """Process in-memory 2D arrays for a single MODIS granule."""
    detections, _ = detect_thermal_anomalies(
        bt21=bt21,
        bt31=bt31,
        nir=nir,
        lat=lat,
        lon=lon,
        water_mask=water_mask,
        config=config,
        is_day=is_day,
    )

    clusters = cluster_thermal_detections(detections, bt21.shape, granule_id)

    thermal_candidates = [d for d in detections if d.is_thermal_candidate]
    solar_alarms = [d for d in detections if d.is_solar_false_alarm]
    frp_total = sum(d.frp_mw for d in thermal_candidates if d.frp_mw is not None)

    return GranuleResult(
        granule_id=granule_id,
        n_scans=bt21.shape[0] // 10,
        n_frames=bt21.shape[1],
        total_detections=len(detections),
        thermal_candidate_count=len(thermal_candidates),
        solar_false_alarm_count=len(solar_alarms),
        total_frp_mw=float(frp_total),
        clusters=clusters,
        detections=detections,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Granule MODIS MWIR Thermal Anomaly Detection Pipeline")
    parser.add_argument("--l0", nargs="+", help="One or more MOD00F L0 raw telemetry files")
    parser.add_argument("--l1b", nargs="+", help="One or more MOD021KM L1B HDF files")
    parser.add_argument("--geo", nargs="+", help="One or more MOD03 Geolocation HDF files")
    parser.add_argument("--out-geojson", default="out/modis_thermal_catalog.geojson", help="Output GeoJSON catalog")
    parser.add_argument("--out-json", default="out/modis_thermal_catalog.json", help="Output JSON catalog")
    args = parser.parse_args()

    print("=== Multi-Granule MODIS MWIR Ingestion Pipeline ===")
    catalog = MultiGranuleCatalog()

    # If L0 files provided with L1B/GEO references
    if args.l0 and args.l1b and args.geo:
        from pyhdf.SD import SD, SDC
        from .modis_l0_decom import decommutate_ev_reflective
        from .modis_opir_mwir import bt_from_l0

        for l0_f, l1b_f, geo_f in zip(args.l0, args.l1b, args.geo):
            granule_id = Path(l0_f).stem
            print(f"Processing granule {granule_id}...")

            l1b_sd = SD(l1b_f, SDC.READ)
            be = list(l1b_sd.select("Band_1KM_Emissive").get())
            sds = l1b_sd.select("EV_1KM_Emissive")
            attrs = sds.attributes()

            bt21, _, _ = bt_from_l0(l0_f, sds, attrs, be, 21)
            bt31, _, _ = bt_from_l0(l0_f, sds, attrs, be, 31)
            nscans = bt21.shape[0] // 10

            nir = decommutate_ev_reflective(l0_f, 2, nscans).astype(float)
            nir_1km = nir[:, :5416].reshape(nir.shape[0] // 4, 4, 1354, 4).mean(axis=(1, 3))

            geo_sd = SD(geo_f, SDC.READ)
            lat = geo_sd.select("Latitude").get()
            lon = geo_sd.select("Longitude").get()

            result = process_granule_arrays(
                bt21=bt21,
                bt31=bt31,
                granule_id=granule_id,
                nir=nir_1km,
                lat=lat,
                lon=lon,
            )
            catalog.add_granule_result(result)
            print(f"  -> {result.total_detections} detections, {result.thermal_candidate_count} thermal candidates, {len(result.clusters)} clusters")

    catalog.save_geojson(args.out_geojson)
    catalog.save_json(args.out_json)
    print(f"\nSaved GeoJSON catalog to {args.out_geojson}")
    print(f"Saved JSON catalog to {args.out_json}")
    print(f"Total Granules: {catalog.total_granules}, Total FRP: {catalog.total_frp_mw:.1f} MW")
    return 0


if __name__ == "__main__":
    sys.exit(main())
