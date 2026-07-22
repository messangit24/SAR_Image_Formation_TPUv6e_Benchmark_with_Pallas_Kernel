"""Chirp Scaling Algorithm (CSA) Range-Doppler Engine for GCP deployment."""

import numpy as np


def process_physical_csa(
    signal_data: np.ndarray,
    sc0: np.ndarray,
    scss: np.ndarray,
    output_shape: tuple[int, int] = (512, 512),
) -> np.ndarray:
  """Physical Chirp Scaling Algorithm with phase multiplier screens."""
  num_vectors, num_samples = signal_data.shape

  # Step 1: Azimuth FFT to Range-Doppler domain
  rd_signal = np.fft.fftshift(np.fft.fft(signal_data, axis=0), axes=0)

  # Step 2: Chirp scaling phase screen multiplier
  eta = np.linspace(-1.0, 1.0, num_vectors, dtype=np.float32)[:, None]
  tau = np.linspace(-1.0, 1.0, num_samples, dtype=np.float32)[None, :]
  
  phase_val = np.pi * 0.1 * (eta**2) * tau
  cs_phase = np.cos(phase_val) + 1j * np.sin(phase_val)
  del phase_val
  
  scaled_signal = rd_signal * cs_phase
  del rd_signal, cs_phase

  # Step 3: Range FFT -> 2D Frequency Domain Range Cell Migration Correction (RCMC)
  freq_signal = np.fft.fftshift(np.fft.fft(scaled_signal, axis=1), axes=1)
  del scaled_signal

  phase_val_rcmc = -np.pi * 0.05 * eta * (tau**2)
  rcmc_phase = np.cos(phase_val_rcmc) + 1j * np.sin(phase_val_rcmc)
  del phase_val_rcmc, eta, tau

  corrected_signal = freq_signal * rcmc_phase
  del freq_signal, rcmc_phase

  # Step 4: 2D IFFT to spatial domain
  csa_img = np.fft.fftshift(np.fft.ifft2(corrected_signal, s=output_shape))
  return csa_img
