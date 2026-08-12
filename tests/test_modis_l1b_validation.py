"""Unit and integration tests for MODIS L1B Calibration Validation Suite."""
import json
import os
import tempfile

import numpy as np
import pytest

from src.modis.l1b_validator import (
    GranuleValidationReport,
    ValidationMetrics,
    compute_validation_metrics,
    evaluate_calibrated_band,
    export_report_json,
    generate_markdown_summary,
    generate_synthetic_l1a_l1b,
    main,
)


def test_synthetic_data_generation():
    """Verify that synthetic L1A/L1B generator produces well-formed arrays across bands."""
    bands = (20, 21, 31, 32)
    synth = generate_synthetic_l1a_l1b(
        num_scans=10,
        num_frames=80,
        bands=bands,
        noise_level=0.002,
        seed=123,
    )

    assert set(synth.keys()) == set(bands)
    for b in bands:
        data = synth[b]
        assert data["EV"].shape == (100, 80)
        assert data["SV"].shape == (100, 50)
        assert data["BB"].shape == (100, 50)
        assert data["Lref"].shape == (100, 80)
        assert data["valid"].shape == (100, 80)
        assert np.all(data["EV"] > 0)
        assert np.all(data["SV"] > 0)
        assert np.all(data["BB"] > 0)
        assert np.all(data["Lref"] > 0)
        assert np.sum(data["valid"]) > 0.95 * 100 * 80


def test_validation_metrics_exact_match():
    """Verify metrics when reconstructed radiance perfectly matches reference."""
    np.random.seed(42)
    ref = np.random.uniform(5.0, 15.0, size=(100, 100))
    recon = ref.copy()
    valid = np.ones_like(ref, dtype=bool)

    metrics = compute_validation_metrics(
        ref_radiance=ref,
        recon_radiance=recon,
        valid_mask=valid,
        band=31,
        wavelength_um=11.017,
        r2_threshold=0.95,
        rel_rmse_threshold=1.0,
    )

    assert metrics.num_valid_pixels == 10000
    assert np.isclose(metrics.rmse, 0.0, atol=1e-10)
    assert np.isclose(metrics.relative_rmse_pct, 0.0, atol=1e-10)
    assert np.isclose(metrics.r_squared, 1.0, atol=1e-7)
    assert np.isclose(metrics.mean_bias_error, 0.0, atol=1e-10)
    assert np.isclose(metrics.max_absolute_error, 0.0, atol=1e-10)
    assert np.isclose(metrics.bt_rmse_k, 0.0, atol=1e-7)
    assert np.isclose(metrics.bt_mbe_k, 0.0, atol=1e-7)
    assert metrics.passes_thresholds is True
    assert len(metrics.threshold_failures) == 0


def test_validation_metrics_with_bias_and_noise():
    """Verify metrics calculation under controlled perturbation."""
    np.random.seed(42)
    ref = np.random.uniform(8.0, 12.0, size=(50, 50))
    bias = 0.2
    noise = np.random.normal(0, 0.05, size=ref.shape)
    recon = ref + bias + noise
    valid = np.ones_like(ref, dtype=bool)

    metrics = compute_validation_metrics(
        ref_radiance=ref,
        recon_radiance=recon,
        valid_mask=valid,
        band=31,
        wavelength_um=11.017,
    )

    assert metrics.num_valid_pixels == 2500
    assert np.isclose(metrics.mean_bias_error, bias, atol=0.02)
    assert metrics.rmse > 0.15
    assert metrics.relative_rmse_pct > 1.0
    assert metrics.r_squared > 0.90
    assert metrics.p90_absolute_error >= metrics.p50_absolute_error
    assert metrics.p99_absolute_error >= metrics.p90_absolute_error
    assert metrics.bt_rmse_k > 0.0


def test_validation_metrics_empty_or_nan_mask():
    """Verify graceful handling when no valid pixels exist."""
    ref = np.full((20, 20), np.nan)
    recon = np.zeros((20, 20))
    valid = np.zeros((20, 20), dtype=bool)

    metrics = compute_validation_metrics(
        ref_radiance=ref,
        recon_radiance=recon,
        valid_mask=valid,
        band=31,
    )

    assert metrics.num_valid_pixels == 0
    assert np.isnan(metrics.rmse)
    assert np.isnan(metrics.r_squared)
    assert metrics.passes_thresholds is False
    assert len(metrics.threshold_failures) > 0


def test_evaluate_calibrated_band_synthetic():
    """Test full calibration & evaluation workflow on synthetic MODIS counts."""
    synth = generate_synthetic_l1a_l1b(
        num_scans=12,
        num_frames=120,
        bands=(31, 32),
        noise_level=0.003,
        seed=999,
    )

    for b in (31, 32):
        d = synth[b]
        recon_ind, metrics_ind = evaluate_calibrated_band(
            d["EV"], d["SV"], d["BB"], d["Lref"], d["valid"], b, mode="independent"
        )
        assert recon_ind.shape == d["Lref"].shape
        assert metrics_ind.r_squared > 0.95
        assert metrics_ind.relative_rmse_pct < 3.0
        assert metrics_ind.passes_thresholds is True

        # Affine recovery mode should achieve even higher R^2
        recon_aff, metrics_aff = evaluate_calibrated_band(
            d["EV"], d["SV"], d["BB"], d["Lref"], d["valid"], b, mode="affine"
        )
        assert metrics_aff.r_squared >= metrics_ind.r_squared - 0.01

        # RVS mode
        recon_rvs, metrics_rvs = evaluate_calibrated_band(
            d["EV"], d["SV"], d["BB"], d["Lref"], d["valid"], b, mode="rvs"
        )
        assert metrics_rvs.r_squared > 0.90


def test_evaluate_calibrated_band_invalid_mode():
    """Verify that unknown calibration mode raises ValueError."""
    ev = np.ones((10, 10))
    sv = np.ones((10, 50))
    bb = np.ones((10, 50))
    lref = np.ones((10, 10))
    valid = np.ones((10, 10), dtype=bool)

    with pytest.raises(ValueError, match="Unknown calibration mode"):
        evaluate_calibrated_band(ev, sv, bb, lref, valid, band=31, mode="nonexistent")


def test_report_export_json_and_markdown():
    """Test JSON export, deserialization, and Markdown summary generation."""
    m31 = ValidationMetrics(
        band=31,
        wavelength_um=11.017,
        num_valid_pixels=50000,
        ref_radiance_mean=9.245,
        recon_radiance_mean=9.250,
        rmse=0.085,
        relative_rmse_pct=0.92,
        r_squared=0.9942,
        mean_bias_error=0.005,
        max_absolute_error=0.35,
        p50_absolute_error=0.06,
        p90_absolute_error=0.15,
        p99_absolute_error=0.28,
        bt_rmse_k=0.55,
        bt_mbe_k=0.04,
        bt_max_abs_k=2.1,
        implied_lbb_mean=8.45,
        implied_lbb_std=0.02,
        implied_lbb_spread_pct=0.24,
        derived_tbb_k=289.8,
        passes_thresholds=True,
    )

    report = GranuleValidationReport(
        granule_id="TEST_GRANULE_A2026213.0710",
        mode="independent",
        band_metrics={31: m31},
        summary_mean_r2=0.9942,
        summary_mean_rel_rmse_pct=0.92,
        all_bands_passed=True,
    )

    # Test Markdown formatting
    md = generate_markdown_summary(report)
    assert "# MODIS L1B Calibration Validation Report" in md
    assert "TEST_GRANULE_A2026213.0710" in md
    assert "| **31** |" in md
    assert "✅ PASS" in md

    # Test JSON file export & reading
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "test_report.json")
        export_report_json(report, json_path)

        assert os.path.exists(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["granule_id"] == "TEST_GRANULE_A2026213.0710"
        assert data["mode"] == "independent"
        assert "31" in data["bands"]
        assert data["bands"]["31"]["r_squared"] == 0.9942


def test_cli_synthetic_execution(monkeypatch):
    """Test CLI main() function running with --synthetic flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = os.path.join(tmpdir, "out.json")
        out_md = os.path.join(tmpdir, "out.md")

        test_args = [
            "l1b_validator.py",
            "--synthetic",
            "--bands", "20", "31",
            "--mode", "independent",
            "--out-json", out_json,
            "--out-md", out_md,
            "--strict",
        ]
        monkeypatch.setattr("sys.argv", test_args)

        main()

        assert os.path.exists(out_json)
        assert os.path.exists(out_md)

        with open(out_json, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        assert report_data["granule_id"] == "SYNTHETIC_GRANULE"
        assert "20" in report_data["bands"]
        assert "31" in report_data["bands"]
        assert report_data["all_bands_passed"] is True
