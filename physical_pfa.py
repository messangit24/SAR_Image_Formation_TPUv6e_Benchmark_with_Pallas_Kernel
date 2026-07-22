"""Polar Format Algorithm (PFA) 2D Fourier Transform Engine for GCP deployment."""

import numpy as np


def process_physical_pfa(
    signal_data: np.ndarray,
    sc0: np.ndarray,
    scss: np.ndarray,
    tx_pos: np.ndarray,
    rcv_pos: np.ndarray,
    srp_pos: np.ndarray,
    output_shape: tuple[int, int] = (512, 512),
) -> np.ndarray:
  """Walker 2-step PFA with relative polar angle grid centering."""
  num_vectors, num_samples = signal_data.shape
  c = 299792458.0

  # Compute aspect angles from trajectory vectors
  los_vecs = srp_pos - 0.5 * (tx_pos + rcv_pos)
  los_norm = los_vecs / np.linalg.norm(los_vecs, axis=-1, keepdims=True)
  polar_ang = np.arctan2(los_norm[:, 1], los_norm[:, 0])

  mid_idx = num_vectors // 2
  polar_ang_mid = polar_ang[mid_idx]
  rel_polar_ang = polar_ang - polar_ang_mid

  # Step 1: Range interpolation
  range_interp = np.zeros((num_vectors, output_shape[1]), dtype=np.complex128)
  k_r_std = np.linspace(-1.0, 1.0, output_shape[1])

  for i in range(num_vectors):
    fc = sc0[i] + 0.5 * (num_samples - 1) * scss[i]
    k_r_i = 4.0 * np.pi * (fc + (np.arange(num_samples) - num_samples / 2) * scss[i]) / c
    scale = np.cos(rel_polar_ang[i])
    k_r_scaled = k_r_i * scale
    k_r_norm = (k_r_scaled - np.mean(k_r_scaled)) / (np.max(k_r_scaled) - np.min(k_r_scaled) + 1e-12)
    range_interp[i, :] = np.interp(k_r_std, k_r_norm, signal_data[i, :], left=0, right=0)

  # Step 2: Azimuth interpolation
  az_interp = np.zeros((output_shape[0], output_shape[1]), dtype=np.complex128)
  k_az_std = np.linspace(-1.0, 1.0, output_shape[0])

  k_az_j = np.sin(rel_polar_ang)
  k_az_norm = (k_az_j - np.mean(k_az_j)) / (np.max(k_az_j) - np.min(k_az_j) + 1e-12)
  sort_idx = np.argsort(k_az_norm)
  k_az_norm_sorted = k_az_norm[sort_idx]

  for j in range(output_shape[1]):
    fp = range_interp[sort_idx, j]
    az_interp[:, j] = np.interp(k_az_std, k_az_norm_sorted, fp, left=0, right=0)

  # 2D IFFT to spatial domain
  pfa_img = np.fft.fftshift(np.fft.ifft2(az_interp, s=output_shape))
  return pfa_img
