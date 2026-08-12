"""Distributed Multi-Granule Batch Processing Engine for MODIS & OPIR Workloads.

Scales raw granule decommutation, radiometric calibration, and MWIR thermal anomaly
detection across multi-process worker pools and distributed compute nodes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .multi_granule_pipeline import (
    GranuleResult,
    MultiGranuleCatalog,
    ThermalCluster,
    process_granule_arrays,
)
from .mwir_detector import DetectionConfig, ThermalDetection


@dataclass
class DistributedBatchConfig:
    """Configuration parameters for distributed batch granule processing."""

    max_workers: Optional[int] = None
    chunk_size: int = 1
    timeout_per_granule_sec: float = 60.0
    stitch_orbit_tracks: bool = True
    max_orbit_time_gap_sec: float = 300.0
    max_orbit_spatial_gap_km: float = 2.5
    detection_config: DetectionConfig = field(default_factory=DetectionConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


@dataclass
class BatchProcessingSummary:
    """Performance and telemetry metrics from a distributed batch execution."""

    total_granules: int
    successful_granules: int
    failed_granules: int
    total_active_pixels: int
    total_clusters: int
    total_frp_mw: float
    total_runtime_sec: float
    granules_per_sec: float
    megapixels_processed: float
    megapixels_per_sec: float
    failure_log: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return asdict(self)


def _worker_process_granule(task_data: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """Isolated worker process task executing granule ingestion & detection."""
    granule_id = task_data["granule_id"]
    try:
        bt21 = np.asarray(task_data["t21"], dtype=np.float64)
        bt31 = np.asarray(task_data["t31"], dtype=np.float64)
        nir = np.asarray(task_data.get("r2"), dtype=np.float64) if task_data.get("r2") is not None else None
        lat = np.asarray(task_data.get("lat"), dtype=np.float64) if task_data.get("lat") is not None else None
        lon = np.asarray(task_data.get("lon"), dtype=np.float64) if task_data.get("lon") is not None else None
        water_mask = np.asarray(task_data.get("water_mask"), dtype=bool) if task_data.get("water_mask") is not None else None

        config_val = task_data.get("config")
        if isinstance(config_val, DetectionConfig):
            det_config = config_val
        elif isinstance(config_val, dict):
            det_config = DetectionConfig(**config_val)
        else:
            det_config = DetectionConfig()

        granule_res = process_granule_arrays(
            bt21=bt21,
            bt31=bt31,
            granule_id=granule_id,
            nir=nir,
            lat=lat,
            lon=lon,
            water_mask=water_mask,
            config=det_config,
            is_day=task_data.get("is_day", True),
        )
        res_dict = granule_res.to_dict()
        if "timestamp" in task_data:
            res_dict["timestamp"] = task_data["timestamp"]
        if "metadata" in task_data:
            res_dict["metadata"] = task_data["metadata"]
        return granule_id, res_dict, None
    except Exception as exc:
        return granule_id, None, str(exc)


def _extract_granule_time(granule: Any) -> Optional[float]:
    """Extract timestamp in epoch seconds from a GranuleResult."""
    t_val = getattr(granule, "timestamp", None)
    if t_val is None and hasattr(granule, "metadata") and isinstance(granule.metadata, dict):
        t_val = granule.metadata.get("timestamp")
    if t_val is None:
        return None
    if isinstance(t_val, (int, float)):
        return float(t_val)
    if isinstance(t_val, str):
        try:
            from datetime import datetime
            s = t_val.replace("Z", "+00:00")
            return datetime.fromisoformat(s).timestamp()
        except Exception:
            pass
    return None


def link_orbit_clusters(
    catalog: MultiGranuleCatalog,
    max_time_gap_sec: float = 300.0,
    max_spatial_gap_km: float = 2.5,
) -> MultiGranuleCatalog:
    """Stitch contiguous thermal clusters across consecutive granules along an orbit track.

    Args:
        catalog: MultiGranuleCatalog containing processed granules.
        max_time_gap_sec: Maximum time difference between consecutive granules (5 min).
        max_spatial_gap_km: Maximum centroid distance for contiguous track stitching.

    Returns:
        New MultiGranuleCatalog with stitched cross-granule clusters.
    """
    if len(catalog.granule_results) < 2:
        return catalog

    flat_clusters: List[Tuple[int, ThermalCluster]] = []
    for g_idx, g in enumerate(catalog.granule_results):
        for c in g.clusters:
            flat_clusters.append((g_idx, c))

    num_clusters = len(flat_clusters)
    if num_clusters < 2:
        return catalog

    # Disjoint Set Union (DSU)
    parent = list(range(num_clusters))

    def find(i: int) -> int:
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            if root_i < root_j:
                parent[root_j] = root_i
            else:
                parent[root_i] = root_j

    granule_times = [_extract_granule_time(g) for g in catalog.granule_results]

    for i in range(num_clusters):
        g_idx_i, c1 = flat_clusters[i]
        lat1, lon1 = c1.center_lat, c1.center_lon
        if lat1 is None or lon1 is None:
            continue

        for j in range(i + 1, num_clusters):
            g_idx_j, c2 = flat_clusters[j]
            if g_idx_i == g_idx_j:
                continue

            lat2, lon2 = c2.center_lat, c2.center_lon
            if lat2 is None or lon2 is None:
                continue

            # Enforce max_time_gap_sec if timestamps exist
            t1, t2 = granule_times[g_idx_i], granule_times[g_idx_j]
            if t1 is not None and t2 is not None:
                if abs(t1 - t2) > max_time_gap_sec:
                    continue

            # Spatial distance with antimeridian longitude wrapping
            d_lat = lat1 - lat2
            d_lon = ((lon1 - lon2 + 180.0) % 360.0) - 180.0
            mean_lat = (lat1 + lat2) / 2.0
            dist_deg = np.sqrt(d_lat**2 + (np.cos(np.radians(mean_lat)) * d_lon)**2)
            dist_km = dist_deg * 111.32

            if dist_km <= max_spatial_gap_km:
                union(i, j)

    # Group clusters by connected component
    components: Dict[int, List[Tuple[int, ThermalCluster]]] = {}
    for idx in range(num_clusters):
        root = find(idx)
        components.setdefault(root, []).append(flat_clusters[idx])

    # Assign clusters to granules
    granule_assigned_clusters: Dict[int, List[ThermalCluster]] = {
        g_idx: [] for g_idx in range(len(catalog.granule_results))
    }

    for group in components.values():
        if len(group) == 1:
            g_idx, c = group[0]
            granule_assigned_clusters[g_idx].append(c)
        else:
            primary_g_idx, c_primary = group[0]
            all_coords = []
            for _, c in group:
                all_coords.extend(c.pixel_coords)

            total_px = sum(c.pixel_count for _, c in group)
            sum_frp = sum(c.total_frp_mw for _, c in group)
            peak_t21 = max(c.peak_bt21 for _, c in group)
            mean_t31 = float(np.mean([c.mean_bt31 for _, c in group]))
            peak_dbt = max(c.peak_delta_bt for _, c in group)

            lats = [c.center_lat for _, c in group if c.center_lat is not None]
            lons = [c.center_lon for _, c in group if c.center_lon is not None]
            c_lat = float(np.mean(lats)) if lats else None
            if lons:
                angles = np.radians(lons)
                c_lon = float(np.degrees(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))))
            else:
                c_lon = None

            try:
                cid = int(c_primary.cluster_id)
            except (ValueError, TypeError):
                cid = 1

            stitched_c = ThermalCluster(
                cluster_id=cid,
                granule_id=c_primary.granule_id,
                pixel_count=total_px,
                peak_bt21=peak_t21,
                mean_bt31=mean_t31,
                peak_delta_bt=peak_dbt,
                total_frp_mw=sum_frp,
                center_lat=c_lat,
                center_lon=c_lon,
                is_thermal_candidate=any(c.is_thermal_candidate for _, c in group),
                is_solar_false_alarm=all(c.is_solar_false_alarm for _, c in group),
                pixel_coords=all_coords,
            )
            granule_assigned_clusters[primary_g_idx].append(stitched_c)

    # Reconstruct final catalog preserving all granules and secondary clusters
    final_catalog = MultiGranuleCatalog()
    for g_idx, g in enumerate(catalog.granule_results):
        g_clusters = granule_assigned_clusters[g_idx]
        updated_g = GranuleResult(
            granule_id=g.granule_id,
            n_scans=g.n_scans,
            n_frames=g.n_frames,
            total_detections=g.total_detections,
            thermal_candidate_count=g.thermal_candidate_count,
            solar_false_alarm_count=g.solar_false_alarm_count,
            total_frp_mw=sum(c.total_frp_mw for c in g_clusters),
            clusters=g_clusters,
            detections=g.detections,
        )
        if hasattr(g, "timestamp"):
            updated_g.timestamp = g.timestamp
        if hasattr(g, "metadata"):
            updated_g.metadata = g.metadata
        final_catalog.add_granule_result(updated_g)

    return final_catalog


def process_granules_distributed(
    granule_tasks: List[Dict[str, Any]],
    config: Optional[DistributedBatchConfig] = None,
) -> Tuple[MultiGranuleCatalog, BatchProcessingSummary]:
    """Execute distributed parallel ingestion and detection over a list of granule tasks.

    Args:
        granule_tasks: List of dictionary specifications for each granule.
        config: DistributedBatchConfig controlling worker count, timeouts, and stitching.

    Returns:
        Tuple of (MultiGranuleCatalog, BatchProcessingSummary).
    """
    if config is None:
        config = DistributedBatchConfig()

    t_start = time.perf_counter()
    num_granules = len(granule_tasks)
    workers = config.max_workers or min(os.cpu_count() or 4, 32, max(1, num_granules))

    catalog = MultiGranuleCatalog()
    failures: Dict[str, str] = {}
    total_pixels_processed = 0

    if num_granules == 0:
        summary = BatchProcessingSummary(
            total_granules=0,
            successful_granules=0,
            failed_granules=0,
            total_active_pixels=0,
            total_clusters=0,
            total_frp_mw=0.0,
            total_runtime_sec=0.0,
            granules_per_sec=0.0,
            megapixels_processed=0.0,
            megapixels_per_sec=0.0,
        )
        return catalog, summary

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {}
        for task in granule_tasks:
            task_payload = dict(task)
            if task_payload.get("config") is None and config.detection_config is not None:
                if isinstance(config.detection_config, DetectionConfig):
                    task_payload["config"] = asdict(config.detection_config)
                elif isinstance(config.detection_config, dict):
                    task_payload["config"] = config.detection_config
            future = executor.submit(_worker_process_granule, task_payload)
            future_to_task[future] = task["granule_id"]

        total_timeout = config.timeout_per_granule_sec * max(1, (num_granules + workers - 1) // workers) + 30.0
        try:
            for future in as_completed(future_to_task, timeout=total_timeout):
                gid = future_to_task[future]
                try:
                    granule_id, result_dict, err = future.result(timeout=config.timeout_per_granule_sec)
                    if err is not None or result_dict is None:
                        failures[gid] = err or "Unknown worker failure"
                    else:
                        # Reconstruct ThermalDetection and ThermalCluster records
                        detections = [
                            ThermalDetection(
                                pixel_row=d["pixel_row"],
                                pixel_col=d["pixel_col"],
                                lat=d.get("lat"),
                                lon=d.get("lon"),
                                bt21=d["bt21"],
                                bt31=d["bt31"],
                                delta_bt=d["delta_bt"],
                                nir_count=d.get("nir_count"),
                                is_solar_false_alarm=d.get("is_solar_false_alarm", False),
                                is_thermal_candidate=d.get("is_thermal_candidate", True),
                                is_water_pixel=d.get("is_water_pixel", False),
                                confidence=d.get("confidence", 0.0),
                                target_temp_k=d.get("target_temp_k"),
                                target_fraction=d.get("target_fraction"),
                                frp_mw=d.get("frp_mw"),
                            )
                            for d in result_dict.get("detections", [])
                        ]
                        clusters = []
                        for c in result_dict.get("clusters", []):
                            raw_cid = c.get("cluster_id", 1)
                            try:
                                cid_int = int(float(raw_cid))
                            except (ValueError, TypeError):
                                cid_int = 1
                            clusters.append(
                                ThermalCluster(
                                    cluster_id=cid_int,
                                    granule_id=c["granule_id"],
                                    pixel_count=c["pixel_count"],
                                    peak_bt21=c["peak_bt21_k"] if "peak_bt21_k" in c else c.get("peak_bt21", 0.0),
                                    mean_bt31=c["mean_bt31_k"] if "mean_bt31_k" in c else c.get("mean_bt31", 0.0),
                                    peak_delta_bt=c["peak_delta_bt_k"] if "peak_delta_bt_k" in c else c.get("peak_delta_bt", 0.0),
                                    total_frp_mw=c["total_frp_mw"],
                                    center_lat=c.get("center_lat"),
                                    center_lon=c.get("center_lon"),
                                    is_thermal_candidate=c.get("is_thermal_candidate", True),
                                    is_solar_false_alarm=c.get("is_solar_false_alarm", False),
                                    pixel_coords=[tuple(coord) for coord in c.get("pixel_coords", [])],
                                )
                            )
                        g_res = GranuleResult(
                            granule_id=result_dict["granule_id"],
                            n_scans=result_dict["n_scans"],
                            n_frames=result_dict["n_frames"],
                            total_detections=result_dict["total_detections"],
                            thermal_candidate_count=result_dict["thermal_candidate_count"],
                            solar_false_alarm_count=result_dict["solar_false_alarm_count"],
                            total_frp_mw=result_dict["total_frp_mw"],
                            clusters=clusters,
                            detections=detections,
                        )
                        if "timestamp" in result_dict:
                            g_res.timestamp = result_dict["timestamp"]
                        if "metadata" in result_dict:
                            g_res.metadata = result_dict["metadata"]
                        catalog.add_granule_result(g_res)
                        total_pixels_processed += result_dict["n_scans"] * 10 * result_dict["n_frames"]
                except Exception as exc:
                    failures[gid] = str(exc)
        except TimeoutError:
            for fut, gid in future_to_task.items():
                if not fut.done():
                    fut.cancel()
                    if gid not in failures:
                        failures[gid] = "Batch execution overall timeout exceeded"

    if config.stitch_orbit_tracks:
        catalog = link_orbit_clusters(
            catalog,
            max_time_gap_sec=config.max_orbit_time_gap_sec,
            max_spatial_gap_km=config.max_orbit_spatial_gap_km,
        )

    t_elapsed = time.perf_counter() - t_start
    fail_count = len(failures)
    succ_count = num_granules - fail_count
    total_mp = total_pixels_processed / 1e6

    summary = BatchProcessingSummary(
        total_granules=num_granules,
        successful_granules=succ_count,
        failed_granules=fail_count,
        total_active_pixels=catalog.total_detections,
        total_clusters=len(catalog.all_clusters),
        total_frp_mw=catalog.total_frp_mw,
        total_runtime_sec=t_elapsed,
        granules_per_sec=(num_granules / t_elapsed) if t_elapsed > 0 else 0.0,
        megapixels_processed=total_mp,
        megapixels_per_sec=(total_mp / t_elapsed) if t_elapsed > 0 else 0.0,
        failure_log=failures,
    )

    return catalog, summary


def generate_synthetic_granule_batch(
    num_granules: int = 4,
    lines: int = 100,
    frames: int = 100,
    num_anomalies_per_granule: int = 3,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate synthetic granule tasks for multi-process batch testing."""
    rng = np.random.default_rng(seed)
    tasks = []

    for g_idx in range(num_granules):
        granule_id = f"SYNTH_MODIS_G{g_idx:03d}"
        t31 = rng.uniform(285.0, 305.0, size=(lines, frames))
        t21 = t31 + rng.normal(0, 1.0, size=(lines, frames))
        r2 = rng.uniform(0.05, 0.25, size=(lines, frames))
        lat = 30.0 + (g_idx * 0.5) + np.linspace(0, 0.4, lines)[:, None] * np.ones((1, frames))
        lon = 25.0 + np.linspace(0, 0.4, frames)[None, :] * np.ones((lines, 1))

        # Inject hot thermal anomalies
        for _ in range(num_anomalies_per_granule):
            r_center = int(rng.integers(10, lines - 10))
            c_center = int(rng.integers(10, frames - 10))
            t21[r_center - 1:r_center + 2, c_center - 1:c_center + 2] = rng.uniform(360.0, 420.0)
            r2[r_center - 1:r_center + 2, c_center - 1:c_center + 2] = 0.02  # Glint-dark

        tasks.append({
            "granule_id": granule_id,
            "t21": t21,
            "t31": t31,
            "r2": r2,
            "lat": lat,
            "lon": lon,
            "valid_mask": np.ones((lines, frames), dtype=bool),
            "timestamp": f"2026-08-12T0{g_idx}:00:00Z",
            "metadata": {"shape": (lines, frames)},
        })

    return tasks


def main() -> None:
    """CLI Entrypoint for Distributed Batch MODIS Ingestion."""
    ap = argparse.ArgumentParser(description="Distributed MODIS Multi-Granule Ingestion Pipeline")
    ap.add_argument("--synthetic-granules", type=int, default=8, help="Number of synthetic granules to test")
    ap.add_argument("--workers", type=int, default=4, help="Number of concurrent worker processes")
    ap.add_argument("--out-catalog", default="reports/modis_batch_catalog.geojson", help="Output GeoJSON catalog")
    ap.add_argument("--out-summary", default="reports/modis_batch_summary.json", help="Output batch summary JSON")
    args = ap.parse_args()

    print(f"Generating {args.synthetic_granules} synthetic granule tasks...")
    tasks = generate_synthetic_granule_batch(num_granules=args.synthetic_granules)

    config = DistributedBatchConfig(max_workers=args.workers)
    print(f"Executing distributed batch ingestion with {args.workers} workers...")
    catalog, summary = process_granules_distributed(tasks, config)

    print("\n=== Distributed Batch Processing Summary ===")
    print(f"Total Granules Processed: {summary.total_granules} ({summary.successful_granules} succeeded, {summary.failed_granules} failed)")
    print(f"Total Thermal Clusters:   {summary.total_clusters}")
    print(f"Total Active Pixels:      {summary.total_active_pixels}")
    print(f"Total Fire Power (FRP):   {summary.total_frp_mw:.2f} MW")
    print(f"Total Runtime:            {summary.total_runtime_sec:.3f} s ({summary.granules_per_sec:.2f} granules/s, {summary.megapixels_per_sec:.2f} MP/s)")

    # Export catalog & summary
    os.makedirs(os.path.dirname(os.path.abspath(args.out_catalog)), exist_ok=True)
    catalog.save_geojson(args.out_catalog)
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)

    print(f"\nWrote catalog to: {args.out_catalog}")
    print(f"Wrote summary to: {args.out_summary}")


if __name__ == "__main__":
    main()
