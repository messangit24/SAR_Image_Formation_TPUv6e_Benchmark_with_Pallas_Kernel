---
name: sar-jax-review
description: Reviews Synthetic Aperture Radar (SAR) and JAX high-performance computing code for numerical accuracy, vectorization, and coherence.
phase: review
user-invocable: true
---

# SAR & JAX High-Performance Code Review

Perform a focused review on SAR image formation algorithms (BPA, FFBP, PFA, RMA, CSA) and JAX/XLA implementations.

## Scope
Focus on files under `src/sar/` and related numerical pipelines.

## Key Review Areas

### 1. Numerical Precision & Coherence
- **FP64 / Double Precision**: Ensure phase history arithmetic and slant-plane coordinate transformations preserve IEEE 754 64-bit precision where needed (`JAX_ENABLE_X64=True` / `float64`).
- **Coherence Verification**: Verify complex coherence algorithms maintain $>97.0\%$ ($>0.97$) coherence vs ground truth SICD products.
- **Interpolation & Kernel Accuracy**: Check that sinc/polyphase or Kaiser/Chirp kernels have sufficient oversampling to prevent aliasing.

### 2. High-Performance Vectorization & XLA Compilation
- **No Hot Python Loops**: Ensure inner loops across pulses, range bins, or sub-apertures are vectorized via `jax.vmap`, `jax.lax.scan`, or array broadcasting.
- **Dynamic Shape Pitfalls**: Ensure array dimensions are compile-time static or properly padded to avoid excessive XLA recompilations.
- **Memory Footprint**: Check large intermediate matrices for TPU HBM and host RAM scalability.

### 3. Modularity & Error Handling
- Clean module docstrings explaining the mathematical algorithms and coordinate systems.
- Graceful fallbacks to NumPy reference implementations when running without accelerator hardware.
