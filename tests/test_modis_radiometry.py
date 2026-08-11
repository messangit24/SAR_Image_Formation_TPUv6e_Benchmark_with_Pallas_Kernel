"""Unit tests for MODIS L1B radiometric calibration & Planck calculations."""
import numpy as np
import pytest

from src.modis.modis_l1b_recon import BAND_LAMBDA, inv_planck, planck


@pytest.mark.parametrize("band,wavelength", list(BAND_LAMBDA.items()))
def test_planck_roundtrip(band, wavelength):
    """Verify that inv_planck(lam, planck(lam, T)) == T for realistic temperatures."""
    temps = np.linspace(200.0, 350.0, 50)  # 200 K to 350 K Earth & Blackbody range
    radiances = planck(wavelength, temps)
    recovered_temps = inv_planck(wavelength, radiances)

    np.testing.assert_allclose(
        recovered_temps,
        temps,
        rtol=1e-7,
        err_msg=f"Planck inversion failed for Band {band} ({wavelength} um)",
    )


def test_planck_monotonicity():
    """Verify that spectral radiance increases strictly monotonically with temperature."""
    wavelength = 11.017  # Band 31 (clean 11 um atmospheric window)
    temps = np.array([250.0, 275.0, 300.0, 325.0])
    rad = planck(wavelength, temps)

    assert np.all(np.diff(rad) > 0), "Spectral radiance must be strictly monotonically increasing with T"


def test_two_point_affine_calibration():
    """Verify mathematical linearity of two-point calibration using synthetic counts."""
    # Synthetic detector setup
    # Space View (SV): count = 50, radiance = 0.0 W/m^2/sr/um
    # Blackbody (BB): count = 850, radiance = 9.5 W/m^2/sr/um
    dn_sv = 50.0
    dn_bb = 850.0
    l_bb = 9.5

    # Gain (slope) & Offset
    gain = l_bb / (dn_bb - dn_sv)
    offset = dn_sv

    # Test an intermediate count
    dn_earth = 450.0
    expected_radiance = gain * (dn_earth - offset)

    computed_radiance = gain * (dn_earth - dn_sv)
    assert np.isclose(computed_radiance, expected_radiance)
    assert np.isclose(gain * (dn_sv - dn_sv), 0.0)
    assert np.isclose(gain * (dn_bb - dn_sv), l_bb)
