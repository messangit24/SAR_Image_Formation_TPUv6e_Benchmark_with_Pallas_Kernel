"""Unit tests for complex coherence calculation and spatial invariants."""
import numpy as np

from src.sar.coherence import complex_coherence_peak


def test_coherence_identity():
    """Verify that coherence of a signal with itself is exactly 1.0."""
    rng = np.random.default_rng(42)
    chip = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))

    gamma = complex_coherence_peak(chip, chip)
    assert np.isclose(gamma, 1.0, atol=1e-5), f"Expected coherence 1.0, got {gamma}"


def test_coherence_phase_invariance():
    """Verify that a global constant phase offset does not degrade complex coherence."""
    rng = np.random.default_rng(42)
    chip_a = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
    # Apply arbitrary phase rotation
    phase_offset = np.exp(1j * 1.234)
    chip_b = chip_a * phase_offset

    gamma = complex_coherence_peak(chip_a, chip_b)
    assert np.isclose(gamma, 1.0, atol=1e-5), f"Expected phase-invariant coherence 1.0, got {gamma}"


def test_coherence_spatial_shift_invariance():
    """Verify that small spatial shifts are compensated by peak coherence search."""
    rng = np.random.default_rng(42)
    chip_a = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
    # Apply spatial shift of 1 pixel in both dimensions
    chip_b = np.roll(chip_a, shift=(1, 1), axis=(0, 1))

    gamma = complex_coherence_peak(chip_a, chip_b, upsample=2, search=1)
    assert np.isclose(gamma, 1.0, atol=1e-5), f"Expected shift-invariant coherence ~1.0, got {gamma}"


def test_coherence_uncorrelated_signals():
    """Verify that two independent random signals produce near-zero coherence."""
    rng = np.random.default_rng(42)
    chip_a = rng.standard_normal((128, 128)) + 1j * rng.standard_normal((128, 128))
    chip_b = rng.standard_normal((128, 128)) + 1j * rng.standard_normal((128, 128))

    gamma = complex_coherence_peak(chip_a, chip_b)
    assert gamma < 0.15, f"Expected low coherence for noise, got {gamma}"


def test_coherence_cauchy_schwarz_bound():
    """Verify that complex coherence is always bounded in [0, 1]."""
    rng = np.random.default_rng(123)
    for _ in range(5):
        chip_a = rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))
        chip_b = rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))
        gamma = complex_coherence_peak(chip_a, chip_b)
        assert 0.0 <= gamma <= 1.0001, f"Coherence outside [0, 1]: {gamma}"
