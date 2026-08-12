"""Unit and integration tests for Distributed Multi-Granule Batch Pipeline."""
import json
import os
import tempfile

import numpy as np

from src.modis.distributed_batch import (
    DistributedBatchConfig,
    generate_synthetic_granule_batch,
    link_orbit_clusters,
    main,
    process_granules_distributed,
)
from src.modis.multi_granule_pipeline import GranuleResult, MultiGranuleCatalog, ThermalCluster
from src.modis.mwir_detector import DetectionConfig, ThermalDetection


def test_synthetic_granule_batch_generation():
    """Verify synthetic granule generator produces NumPy array task dictionaries."""
    tasks = generate_synthetic_granule_batch(num_granules=3, lines=40, frames=40, num_anomalies_per_granule=2)
    assert len(tasks) == 3
    for t in tasks:
        assert "granule_id" in t
        assert isinstance(t["t21"], np.ndarray)
        assert t["t21"].shape == (40, 40)
        assert isinstance(t["t31"], np.ndarray)
        assert t["t31"].shape == (40, 40)
        assert isinstance(t["r2"], np.ndarray)
        assert t["r2"].shape == (40, 40)
        assert isinstance(t["lat"], np.ndarray)
        assert t["lat"].shape == (40, 40)
        assert isinstance(t["lon"], np.ndarray)
        assert t["lon"].shape == (40, 40)


def test_distributed_batch_processing():
    """Test multi-process parallel execution across synthetic granules."""
    tasks = generate_synthetic_granule_batch(num_granules=4, lines=50, frames=50, num_anomalies_per_granule=2)
    config = DistributedBatchConfig(max_workers=2, stitch_orbit_tracks=False)

    catalog, summary = process_granules_distributed(tasks, config)

    assert summary.total_granules == 4
    assert summary.successful_granules == 4
    assert summary.failed_granules == 0
    assert summary.total_active_pixels > 0
    assert summary.total_clusters > 0
    assert summary.total_frp_mw > 0
    assert summary.granules_per_sec > 0
    assert summary.megapixels_processed > 0
    assert catalog.total_granules == 4


def test_detection_records_preservation():
    """Verify that ThermalDetection pixel records are properly reconstructed on GranuleResult."""
    tasks = generate_synthetic_granule_batch(num_granules=2, lines=50, frames=50, num_anomalies_per_granule=3)
    catalog, _ = process_granules_distributed(tasks, DistributedBatchConfig(max_workers=2))

    for g in catalog.granule_results:
        assert len(g.detections) > 0
        for d in g.detections:
            assert isinstance(d, ThermalDetection)
            assert isinstance(d.pixel_row, int)
            assert isinstance(d.pixel_col, int)
            assert d.bt21 > 0.0
            assert d.bt31 > 0.0


def test_detection_config_forwarding():
    """Verify that custom DetectionConfig is forwarded to worker processes and respected."""
    tasks = generate_synthetic_granule_batch(num_granules=2, lines=50, frames=50, num_anomalies_per_granule=2)

    # Standard detection
    cat_default, _ = process_granules_distributed(tasks, DistributedBatchConfig(max_workers=1))

    # Ultra-strict threshold that suppresses detections
    strict_det_config = DetectionConfig(
        t21_abs_threshold_day=540.0,
        delta_t_min=80.0,
        context_k1=20.0,
        context_k2=20.0,
    )
    cat_strict, _ = process_granules_distributed(
        tasks,
        DistributedBatchConfig(max_workers=1, detection_config=strict_det_config),
    )

    assert cat_default.total_detections > cat_strict.total_detections


def test_distributed_batch_fault_tolerance():
    """Verify that a corrupted granule task does not crash the entire batch."""
    tasks = generate_synthetic_granule_batch(num_granules=3, lines=30, frames=30)
    # Corrupt the second task
    tasks[1]["t21"] = "invalid_non_numeric_array"

    config = DistributedBatchConfig(max_workers=2)
    catalog, summary = process_granules_distributed(tasks, config)

    assert summary.total_granules == 3
    assert summary.successful_granules == 2
    assert summary.failed_granules == 1
    assert "SYNTH_MODIS_G001" in summary.failure_log
    assert catalog.total_granules == 2


def test_link_orbit_clusters_spatial_merging_and_integer_ids():
    """Verify that adjacent thermal clusters get stitched with integer cluster_id."""
    catalog = MultiGranuleCatalog()

    # Cluster 1 in Granule A
    c1 = ThermalCluster(
        cluster_id=1, granule_id="GRANULE_A", pixel_count=1,
        peak_bt21=380.0, mean_bt31=295.0, peak_delta_bt=85.0,
        total_frp_mw=12.5, center_lat=30.00, center_lon=25.00,
        is_thermal_candidate=True, is_solar_false_alarm=False,
        pixel_coords=[(10, 10)],
    )
    g1 = GranuleResult(
        granule_id="GRANULE_A", n_scans=1, n_frames=100, total_detections=1,
        thermal_candidate_count=1, solar_false_alarm_count=0, total_frp_mw=12.5,
        clusters=[c1], detections=[],
    )

    # Cluster 2 in Granule B (adjacent: ~1 km away)
    c2 = ThermalCluster(
        cluster_id=2, granule_id="GRANULE_B", pixel_count=1,
        peak_bt21=390.0, mean_bt31=295.0, peak_delta_bt=95.0,
        total_frp_mw=18.0, center_lat=30.008, center_lon=25.00,
        is_thermal_candidate=True, is_solar_false_alarm=False,
        pixel_coords=[(0, 10)],
    )
    g2 = GranuleResult(
        granule_id="GRANULE_B", n_scans=1, n_frames=100, total_detections=1,
        thermal_candidate_count=1, solar_false_alarm_count=0, total_frp_mw=18.0,
        clusters=[c2], detections=[],
    )

    catalog.add_granule_result(g1)
    catalog.add_granule_result(g2)

    stitched = link_orbit_clusters(catalog, max_spatial_gap_km=2.0)
    assert len(stitched.all_clusters) == 1
    merged = stitched.all_clusters[0]
    assert isinstance(merged.cluster_id, int)
    assert merged.pixel_count == 2
    assert np.isclose(merged.total_frp_mw, 30.5)
    assert np.isclose(merged.peak_bt21, 390.0)


def test_link_orbit_clusters_antimeridian_wrapping():
    """Verify cluster stitching across the +/-180 antimeridian."""
    catalog = MultiGranuleCatalog()

    # Cluster 1 near +179.99 deg longitude
    c1 = ThermalCluster(
        cluster_id=1, granule_id="GRANULE_EAST", pixel_count=1,
        peak_bt21=380.0, mean_bt31=295.0, peak_delta_bt=85.0,
        total_frp_mw=10.0, center_lat=20.0, center_lon=179.99,
        is_thermal_candidate=True, is_solar_false_alarm=False,
        pixel_coords=[(10, 10)],
    )
    g1 = GranuleResult(
        granule_id="GRANULE_EAST", n_scans=1, n_frames=100, total_detections=1,
        thermal_candidate_count=1, solar_false_alarm_count=0, total_frp_mw=10.0,
        clusters=[c1], detections=[],
    )

    # Cluster 2 near -179.99 deg longitude (~2.2 km away across antimeridian)
    c2 = ThermalCluster(
        cluster_id=2, granule_id="GRANULE_WEST", pixel_count=1,
        peak_bt21=385.0, mean_bt31=295.0, peak_delta_bt=90.0,
        total_frp_mw=15.0, center_lat=20.0, center_lon=-179.99,
        is_thermal_candidate=True, is_solar_false_alarm=False,
        pixel_coords=[(0, 10)],
    )
    g2 = GranuleResult(
        granule_id="GRANULE_WEST", n_scans=1, n_frames=100, total_detections=1,
        thermal_candidate_count=1, solar_false_alarm_count=0, total_frp_mw=15.0,
        clusters=[c2], detections=[],
    )

    catalog.add_granule_result(g1)
    catalog.add_granule_result(g2)

    stitched = link_orbit_clusters(catalog, max_spatial_gap_km=5.0)
    assert len(stitched.all_clusters) == 1
    merged = stitched.all_clusters[0]
    assert isinstance(merged.cluster_id, int)
    assert merged.pixel_count == 2
    assert np.isclose(merged.total_frp_mw, 25.0)
    # Longitude should wrap around +/-180 (e.g. near 180 or -180)
    assert abs(abs(merged.center_lon) - 180.0) < 0.1


def test_link_orbit_clusters_max_time_gap_enforced():
    """Verify that clusters separated by more than max_time_gap_sec are not merged."""
    catalog = MultiGranuleCatalog()

    c1 = ThermalCluster(
        cluster_id=1, granule_id="G1", pixel_count=1,
        peak_bt21=380.0, mean_bt31=295.0, peak_delta_bt=85.0,
        total_frp_mw=10.0, center_lat=30.0, center_lon=25.0,
        is_thermal_candidate=True, is_solar_false_alarm=False,
        pixel_coords=[(10, 10)],
    )
    g1 = GranuleResult(
        granule_id="G1", n_scans=1, n_frames=100, total_detections=1,
        thermal_candidate_count=1, solar_false_alarm_count=0, total_frp_mw=10.0,
        clusters=[c1], detections=[],
    )
    g1.timestamp = "2026-08-12T00:00:00Z"

    # G2 is spatially adjacent (~1 km) but 1 hour later
    c2 = ThermalCluster(
        cluster_id=2, granule_id="G2", pixel_count=1,
        peak_bt21=380.0, mean_bt31=295.0, peak_delta_bt=85.0,
        total_frp_mw=10.0, center_lat=30.008, center_lon=25.0,
        is_thermal_candidate=True, is_solar_false_alarm=False,
        pixel_coords=[(0, 10)],
    )
    g2 = GranuleResult(
        granule_id="G2", n_scans=1, n_frames=100, total_detections=1,
        thermal_candidate_count=1, solar_false_alarm_count=0, total_frp_mw=10.0,
        clusters=[c2], detections=[],
    )
    g2.timestamp = "2026-08-12T01:00:00Z"

    catalog.add_granule_result(g1)
    catalog.add_granule_result(g2)

    stitched = link_orbit_clusters(catalog, max_time_gap_sec=300.0, max_spatial_gap_km=2.0)
    assert len(stitched.all_clusters) == 2


def test_link_orbit_clusters_secondary_granule_unmerged_preserved():
    """Verify that secondary granules retain unmerged clusters."""
    catalog = MultiGranuleCatalog()

    # Granule 1 has cluster 1
    c1 = ThermalCluster(
        cluster_id=1, granule_id="G1", pixel_count=1,
        peak_bt21=380.0, mean_bt31=295.0, peak_delta_bt=85.0,
        total_frp_mw=10.0, center_lat=30.0, center_lon=25.0,
        is_thermal_candidate=True, is_solar_false_alarm=False,
        pixel_coords=[(10, 10)],
    )
    g1 = GranuleResult(
        granule_id="G1", n_scans=1, n_frames=100, total_detections=1,
        thermal_candidate_count=1, solar_false_alarm_count=0, total_frp_mw=10.0,
        clusters=[c1], detections=[],
    )

    # Granule 2 has cluster 2 (near c1) and cluster 3 (distant)
    c2 = ThermalCluster(
        cluster_id=2, granule_id="G2", pixel_count=1,
        peak_bt21=380.0, mean_bt31=295.0, peak_delta_bt=85.0,
        total_frp_mw=10.0, center_lat=30.005, center_lon=25.0,
        is_thermal_candidate=True, is_solar_false_alarm=False,
        pixel_coords=[(0, 10)],
    )
    c3 = ThermalCluster(
        cluster_id=3, granule_id="G2", pixel_count=1,
        peak_bt21=370.0, mean_bt31=295.0, peak_delta_bt=75.0,
        total_frp_mw=8.0, center_lat=35.0, center_lon=25.0,
        is_thermal_candidate=True, is_solar_false_alarm=False,
        pixel_coords=[(50, 50)],
    )
    g2 = GranuleResult(
        granule_id="G2", n_scans=1, n_frames=100, total_detections=2,
        thermal_candidate_count=2, solar_false_alarm_count=0, total_frp_mw=18.0,
        clusters=[c2, c3], detections=[],
    )

    catalog.add_granule_result(g1)
    catalog.add_granule_result(g2)

    stitched = link_orbit_clusters(catalog, max_spatial_gap_km=2.0)
    assert len(stitched.all_clusters) == 2
    # g2 should still have c3
    assert len(stitched.granule_results[1].clusters) == 1
    assert stitched.granule_results[1].clusters[0].cluster_id == 3
    assert np.isclose(stitched.granule_results[1].total_frp_mw, 8.0)


def test_empty_batch_handling():
    """Verify empty task list produces clean zero-summary."""
    catalog, summary = process_granules_distributed([])
    assert summary.total_granules == 0
    assert summary.successful_granules == 0
    assert summary.total_clusters == 0
    assert catalog.total_granules == 0


def test_link_orbit_clusters_multi_granule_chain():
    """Verify cluster stitching across a 3-granule chain crossing the antimeridian."""
    catalog = MultiGranuleCatalog()
    c1 = ThermalCluster(1, "G1", 1, 20.0, 179.99, 380.0, 290.0, 90.0, 10.0, False, True, [(0, 0)])
    g1 = GranuleResult("G1", 1, 100, 1, 1, 0, 10.0, [c1], [])
    c2 = ThermalCluster(2, "G2", 1, 20.0, -179.99, 385.0, 290.0, 95.0, 15.0, False, True, [(0, 0)])
    g2 = GranuleResult("G2", 1, 100, 1, 1, 0, 15.0, [c2], [])
    c3 = ThermalCluster(3, "G3", 1, 20.0, -179.98, 390.0, 290.0, 100.0, 20.0, False, True, [(0, 0)])
    g3 = GranuleResult("G3", 1, 100, 1, 1, 0, 20.0, [c3], [])
    catalog.add_granule_result(g1)
    catalog.add_granule_result(g2)
    catalog.add_granule_result(g3)

    stitched = link_orbit_clusters(catalog, max_spatial_gap_km=5.0)
    assert len(stitched.all_clusters) == 1
    merged = stitched.all_clusters[0]
    assert merged.pixel_count == 3
    assert np.isclose(merged.total_frp_mw, 45.0)
    assert np.isclose(merged.peak_bt21, 390.0)


def test_detection_config_various_types_forwarding():
    """Verify worker handles DetectionConfig instances, dicts, and None in task payloads."""
    tasks = generate_synthetic_granule_batch(num_granules=3, lines=30, frames=30)
    tasks[0]["config"] = DetectionConfig(t21_abs_threshold_day=320.0)
    tasks[1]["config"] = {"t21_abs_threshold_day": 320.0}
    tasks[2]["config"] = None

    catalog, summary = process_granules_distributed(tasks, DistributedBatchConfig(max_workers=2))
    assert summary.successful_granules == 3
    assert summary.failed_granules == 0
    assert catalog.total_granules == 3


def test_cli_execution(monkeypatch):
    """Test CLI entrypoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_cat = os.path.join(tmpdir, "catalog.geojson")
        out_sum = os.path.join(tmpdir, "summary.json")
        test_args = [
            "distributed_batch.py",
            "--synthetic-granules", "2",
            "--workers", "2",
            "--out-catalog", out_cat,
            "--out-summary", out_sum,
        ]
        monkeypatch.setattr("sys.argv", test_args)
        main()

        assert os.path.exists(out_cat)
        assert os.path.exists(out_sum)
        with open(out_sum, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_granules"] == 2

