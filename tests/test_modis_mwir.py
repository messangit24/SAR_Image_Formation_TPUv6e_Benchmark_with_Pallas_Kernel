"""Unit & integration tests for MODIS MWIR thermal anomaly detection & multi-granule pipeline."""
import json
import numpy as np

from src.modis.mwir_detector import (
    LAMBDA_LWIR_31,
    LAMBDA_MWIR_21,
    DetectionConfig,
    blend_high_dynamic_range,
    calculate_frp,
    detect_thermal_anomalies,
    inv_planck,
    planck,
    solve_dozier,
)
from src.modis.multi_granule_pipeline import (
    MultiGranuleCatalog,
    process_granule_arrays,
)


def test_planck_and_inversion_consistency():
    """Verify forward and inverse Planck functions are exact inverses."""
    temps = np.array([250.0, 300.0, 350.0, 450.0, 600.0, 800.0])
    for lam in [LAMBDA_MWIR_21, LAMBDA_LWIR_31]:
        rad = planck(lam, temps)
        t_rec = inv_planck(lam, rad)
        np.testing.assert_allclose(t_rec, temps, rtol=1e-7)


def test_blend_high_dynamic_range():
    """Verify smooth handover from high-gain Band 22 to low-gain Band 21."""
    # Band 22 accurate below 329 K, saturated / NaN above
    bt22 = np.array([[300.0, 320.0], [330.0, np.nan]])
    bt21 = np.array([[301.0, 319.5], [375.0, 420.0]])

    blended = blend_high_dynamic_range(bt21, bt22, saturation_k=329.0)
    assert np.isclose(blended[0, 0], 300.0)  # Uses Band 22
    assert np.isclose(blended[0, 1], 320.0)  # Uses Band 22
    assert np.isclose(blended[1, 0], 375.0)  # Exceeds 329 K -> uses Band 21
    assert np.isclose(blended[1, 1], 420.0)  # NaN in Band 22 -> uses Band 21


def test_dozier_subpixel_inversion():
    """Verify Dozier non-linear dual-channel subpixel inversion recovers target T and area."""
    t_bg = 295.0
    true_target_t = 720.0  # 720 K hot target
    true_fraction = 0.03   # 3% fractional area

    # Synthesize forward observed radiances
    b21_bg = planck(LAMBDA_MWIR_21, t_bg)
    b21_t = planck(LAMBDA_MWIR_21, true_target_t)
    l21_obs = true_fraction * b21_t + (1.0 - true_fraction) * b21_bg

    b31_bg = planck(LAMBDA_LWIR_31, t_bg)
    b31_t = planck(LAMBDA_LWIR_31, true_target_t)
    l31_obs = true_fraction * b31_t + (1.0 - true_fraction) * b31_bg

    # Invert using Dozier solver
    p_rec, t_rec = solve_dozier(l21_obs, l31_obs, t_bg=t_bg)

    assert np.isclose(t_rec, true_target_t, atol=2.0), f"Recovered T {t_rec} != {true_target_t}"
    assert np.isclose(p_rec, true_fraction, atol=0.005), f"Recovered p {p_rec} != {true_fraction}"


def test_calculate_frp_physical_scaling():
    """Verify Fire Radiative Power (FRP) scaling and boundary behavior."""
    t_bg = 300.0
    # Zero fraction -> zero power
    assert calculate_frp(0.0, 600.0, t_bg) == 0.0

    # Target equal to background -> zero power
    assert calculate_frp(0.05, 300.0, t_bg) == 0.0

    # Hot fire: 1% of 1 km^2 (10,000 m^2) at 800 K
    p = 0.01
    t_target = 800.0
    frp = calculate_frp(p, t_target, t_bg, pixel_area_m2=1.0e6)
    assert frp > 0.0
    # Expected: 0.01 * 1e6 * 5.67037e-8 * (800^4 - 300^4) * 1e-6 ~ 227.6 MW
    expected = 0.01 * 1.0e6 * 5.670374419e-8 * (800.0**4 - 300.0**4) * 1e-6
    assert np.isclose(frp, expected, rtol=1e-5)


def test_thermal_anomaly_detection_and_discrimination():
    """Verify absolute & contextual triggers and solar glint discrimination."""
    ny, nx = 40, 40
    # Uniform background at 295 K with small noise
    np.random.seed(42)
    bt21 = 295.0 + np.random.normal(0, 0.5, (ny, nx))
    bt31 = 295.0 + np.random.normal(0, 0.5, (ny, nx))
    nir = np.full((ny, nx), 200.0)  # Background NIR level

    # 1. True thermal anomaly at (10, 10): High T21, high excess, dark NIR
    bt21[10, 10] = 380.0
    bt31[10, 10] = 302.0
    nir[10, 10] = 220.0  # Dark NIR (~1.1x bg)

    # 2. Solar glint false-alarm at (25, 25): High T21, high excess, bright NIR
    bt21[25, 25] = 375.0
    bt31[25, 25] = 304.0
    nir[25, 25] = 1800.0  # Very bright NIR (9x bg)

    config = DetectionConfig(
        t21_abs_threshold_day=360.0,
        context_window_size=15,
        nir_bg_ratio_solar=4.0,
        nir_bg_ratio_thermal=2.0,
    )

    detections, masks = detect_thermal_anomalies(bt21, bt31, nir=nir, config=config, is_day=True)

    assert len(detections) >= 2
    det_map = {(d.pixel_row, d.pixel_col): d for d in detections}

    assert (10, 10) in det_map
    d_thermal = det_map[(10, 10)]
    assert d_thermal.is_thermal_candidate is True
    assert d_thermal.is_solar_false_alarm is False
    assert d_thermal.frp_mw is not None
    assert d_thermal.frp_mw > 0.0

    assert (25, 25) in det_map
    d_solar = det_map[(25, 25)]
    assert d_solar.is_solar_false_alarm is True
    assert d_solar.is_thermal_candidate is False


def test_spatial_clustering_and_multi_granule_catalog(tmp_path):
    """Verify 8-connected spatial clustering and multi-granule GeoJSON export."""
    ny, nx = 30, 30
    bt21_g1 = np.full((ny, nx), 295.0)
    bt31_g1 = np.full((ny, nx), 295.0)
    lat_g1 = np.linspace(30.0, 31.0, ny)[:, None] * np.ones((1, nx))
    lon_g1 = np.ones((ny, 1)) * np.linspace(31.0, 32.0, nx)[None, :]

    # Insert 2 connected pixels (Cluster 1) and 1 isolated pixel (Cluster 2)
    bt21_g1[5, 5] = 390.0
    bt21_g1[5, 6] = 385.0
    bt21_g1[20, 20] = 370.0

    result1 = process_granule_arrays(
        bt21=bt21_g1,
        bt31=bt31_g1,
        granule_id="GRANULE_001",
        lat=lat_g1,
        lon=lon_g1,
    )

    assert result1.total_detections == 3
    assert len(result1.clusters) == 2

    c1 = next(c for c in result1.clusters if c.pixel_count == 2)
    assert c1.peak_bt21 == 390.0
    assert len(c1.pixel_coords) == 2

    # Second granule
    bt21_g2 = np.full((ny, nx), 295.0)
    bt31_g2 = np.full((ny, nx), 295.0)
    bt21_g2[12, 12] = 410.0
    result2 = process_granule_arrays(
        bt21=bt21_g2,
        bt31=bt31_g2,
        granule_id="GRANULE_002",
        lat=lat_g1,
        lon=lon_g1,
    )

    catalog = MultiGranuleCatalog()
    catalog.add_granule_result(result1)
    catalog.add_granule_result(result2)

    assert catalog.total_granules == 2
    assert catalog.total_detections == 4
    assert len(catalog.all_clusters) == 3

    # Export to GeoJSON
    geojson_path = tmp_path / "catalog.geojson"
    catalog.save_geojson(geojson_path)
    assert geojson_path.exists()

    with open(geojson_path) as f:
        data = json.load(f)
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 3
    assert data["metadata"]["total_granules"] == 2


def test_water_filtering():
    """Verify that water surface false alarms are filtered when water mask is applied."""
    ny, nx = 30, 30
    bt21 = np.full((ny, nx), 295.0)
    bt31 = np.full((ny, nx), 295.0)
    water_mask = np.zeros((ny, nx), dtype=bool)

    # Place a hot anomaly over water at (10, 10)
    water_mask[10, 10] = True
    bt21[10, 10] = 380.0
    bt31[10, 10] = 298.0

    # Place a valid land thermal anomaly at (20, 20)
    water_mask[20, 20] = False
    bt21[20, 20] = 380.0
    bt31[20, 20] = 298.0

    # 1. With filter_water = True (default)
    config_filter = DetectionConfig(filter_water=True)
    detections, masks = detect_thermal_anomalies(
        bt21, bt31, water_mask=water_mask, config=config_filter, is_day=True
    )
    det_coords = [(d.pixel_row, d.pixel_col) for d in detections if d.is_thermal_candidate]
    assert (20, 20) in det_coords, "Land thermal candidate must be detected"
    assert (10, 10) not in det_coords, "Water anomaly must be screened out by filter_water"
    assert masks["water_mask"][10, 10] is True or masks["water_mask"][10, 10] == 1

    # 2. With filter_water = False
    config_no_filter = DetectionConfig(filter_water=False)
    detections_all, _ = detect_thermal_anomalies(
        bt21, bt31, water_mask=water_mask, config=config_no_filter, is_day=True
    )
    det_coords_all = [(d.pixel_row, d.pixel_col) for d in detections_all]
    assert (10, 10) in det_coords_all
    assert (20, 20) in det_coords_all

