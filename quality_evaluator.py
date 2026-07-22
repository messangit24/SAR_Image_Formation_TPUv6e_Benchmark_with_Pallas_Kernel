"""Quality Evaluation Metrics (Coherence gamma_mag, PSNR, dB scaling) for GCP deployment."""

import numpy as np


def clean_array(arr: np.ndarray) -> np.ndarray:
  """Replaces NaNs and Infs with zeros."""
  return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def compute_complex_coherence(ref_img: np.ndarray, test_img: np.ndarray) -> float:
  """Computes magnitude correlation coherence (gamma_mag) between two SAR images."""
  ref_c = clean_array(ref_img)
  test_c = clean_array(test_img)

  ref_mag = np.abs(ref_c)
  test_mag = np.abs(test_c)

  ref_norm = ref_mag - np.mean(ref_mag)
  test_norm = test_mag - np.mean(test_mag)

  denom = np.sqrt(np.sum(ref_norm**2) * np.sum(test_norm**2))
  if denom == 0:
    return 0.0

  return float(np.sum(ref_norm * test_norm) / denom)


def compute_psnr(ref_img: np.ndarray, test_img: np.ndarray) -> float:
  """Computes Peak Signal-to-Noise Ratio (PSNR in dB)."""
  ref_mag = np.abs(clean_array(ref_img))
  test_mag = np.abs(clean_array(test_img))

  max_val = np.max(ref_mag)
  if max_val == 0:
    return 0.0

  mse = np.mean((ref_mag - test_mag) ** 2)
  if mse == 0:
    return 100.0

  return float(20.0 * np.log10(max_val) - 10.0 * np.log10(mse))


def to_uint8_db(img_complex: np.ndarray, vmin: float = -40.0, vmax: float = 0.0) -> np.ndarray:
  """Scales complex SAR image magnitude to dB scale and converts to uint8 [0, 255]."""
  mag = np.abs(clean_array(img_complex))
  max_val = np.max(mag)
  if max_val > 0:
    mag = mag / max_val

  mag_db = 20.0 * np.log10(mag + 1e-12)
  clipped = np.clip(mag_db, vmin, vmax)
  normalized = (clipped - vmin) / (vmax - vmin)
  return (normalized * 255.0).astype(np.uint8)
