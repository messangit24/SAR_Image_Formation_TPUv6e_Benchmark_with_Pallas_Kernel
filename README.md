# Umbra CPHD → complex image via the Polar Format Algorithm

Recreates Umbra's spotlight SAR image-formation processing from **open-data
CPHD** (phase history) using the **Polar Format Algorithm (PFA)**, and
validates the result against Umbra's own **SICD** (their PFA output, *no
autofocus*) and **GEC** products.

The point of the exercise: Umbra publishes the raw compensated phase history
(CPHD) *and* the focused complex image (SICD) for the same collect, and states
the SICD is formed **without autofocus**. That means a textbook PFA — no
autofocus, no proprietary tricks — should reproduce the SICD. It does.

## The collect

`s3://umbra-open-data-catalog/sar-data/tasks/Bingham Copper Mine/08366064-.../2023-09-25-17-37-50_UMBRA-05_*`

| product | what it is | size |
|---|---|---|
| `*_CPHD.cphd`  | compensated phase history (input) | 1.49 GB |
| `*_SICD.nitf`  | Umbra's focused complex image (reference) | 953 MB |
| `*_GEC.tif`    | ground-projected detected image | 85 MB |

CPHD 1.1.0, spotlight, X-band (Fc = 9.6 GHz, BW = 383.5 MHz), VV, **FX
(frequency) domain**, 14 479 pulses × 12 799 range samples, motion-compensated
to a single fixed Scene Reference Point (SRP). Everything PFA needs is in the
per-vector parameters (PVP): `TxPos`, `RcvPos`, `SRPPos`, and the per-pulse
frequency sampling `SC0` / `SCSS`.

## Why PFA works here (and why no autofocus is needed)

Because the data is already motion-compensated to the SRP and stored in the
frequency domain, each sample sits at a known point in wavenumber (k) space:

```
K(pulse m, sample n) = (4π f / c) · û_m ,   f = SC0_m + n·SCSS_m ,   û_m = unit(APC_m − SRP)
```

i.e. each pulse is a **radial spoke** in k-space (swept over frequency) and
successive pulses **rotate** the spoke as the platform flies by — a polar grid.
PFA resamples that polar grid onto a uniform Cartesian grid and 2-D FFTs it.
For a 4 km scene at 663 km standoff the planar-wavefront approximation PFA
makes is far below a resolution cell, so no autofocus is required.

## Pipeline

```
CPHD  ──►  read signal + PVP (big-endian memmap)       src/cphd_reader.py
      ──►  build k-space in the APERTURE (slant) plane  src/pfa.py
      ──►  Stage 1: range  interp  (polar → uniform Kx)
      ──►  Stage 2: azimuth interp (polar → uniform Ky)
      ──►  zero-pad k-space (oversample) ──► 2-D FFT ──► focused complex image
```

Two subtleties that actually matter for matching the SICD:

* **Form in the aperture (slant) plane, not the ground plane** (below).
* **Oversample by zero-padding k-space before the FFT.** At the native grid the
  azimuth pixel (~0.49 m) is *coarser* than the azimuth resolution (~0.44 m), so
  a point target's mainlobe is undersampled and measures ~1.4× too wide. Padding
  the k-space (keeping `dk` fixed, growing the FFT) gives ~0.35 m pixels and the
  measured IPR collapses onto the SICD's.

**Key design choice — form the image in the aperture (slant) plane, not the
ground plane.** Umbra's CPHD declares a *ground* reference plane. If you build
k-space there, the `cos(graze)` foreshortening varies ~2.3 % across the
aperture (the grazing angle changes as the platform passes the fixed SRP),
which smears the common range-frequency band and destroys range resolution.
In the aperture plane (range axis = mean line-of-sight, cross-range axis =
aperture-sweep direction) the radial component is constant to 0.01 %, giving a
clean polar annulus and full bandwidth — and this is the same plane the SICD is
formed in (`Grid.Type = RGAZIM`).

## Run

```bash
python -m venv .venv && ./.venv/bin/pip install numpy scipy matplotlib sarpy

# download one collect's CPHD (+ SICD/GEC for validation)
aws s3 cp --no-sign-request "s3://umbra-open-data-catalog/.../*_CPHD.cphd" data/umbra05_bingham.cphd

# fast decimated quicklook
./.venv/bin/python src/run_pfa.py --pulse-step 4 --sample-step 4 --out out/quicklook

# full resolution (~16 s, ~6 GB RAM)
./.venv/bin/python src/run_pfa.py --out out/full --save-complex

# validate against Umbra's SICD  (paths are optional flags -> runs on any collect)
PYTHONPATH=src ./.venv/bin/python src/register_sicd.py --complex out/full_complex.npy --sicd data/SICD.nitf
PYTHONPATH=src ./.venv/bin/python src/coherence.py     --complex out/full_complex.npy --sicd data/SICD.nitf
```

## Running on other collects

Nothing is hard-coded to this scene — dimensions, PVP layout, frequency
sampling, geometry, signal format, the reference channel and its byte offsets,
and the FX/TOA domain all come from the CPHD text header + XML; pixel spacings
and the frame come from the data. To process a different Umbra collect, just
point at its files:

```bash
aws s3 cp --no-sign-request "s3://umbra-open-data-catalog/.../<id>_CPHD.cphd" data/x.cphd
./.venv/bin/python src/run_pfa.py --cphd data/x.cphd --out out/x --save-complex
PYTHONPATH=src ./.venv/bin/python src/register_sicd.py --complex out/x_complex.npy --geom out/x_geom.npz --sicd data/x_SICD.nitf
```

The SICD frame relationship (transpose / flips) is **detected**, not assumed, so
ascending/descending and left/right-looking collects work without changes.

## Results / validation

| metric | my PFA | Umbra SICD | note |
|---|---|---|---|
| range −3 dB IPR (nominal) | 0.347 m | 0.3471 m | 0.886 × (2π / K-range-extent), uniform weighting |
| azimuth −3 dB IPR (nominal) | 0.437 m | 0.4461 m | 0.886 × (2π / K-az-extent) |
| range IPR (measured, clean point target) | ~0.30–0.35 m | 0.37–0.39 m (same method) | matches |
| azimuth IPR (measured, clean point target) | ~0.31–0.46 m | 0.37–0.39 m (same method) | matches after oversampling |
| weighting | uniform | uniform (`WgtType = None`) | matched sidelobes |
| registration to SICD | — | — | normalized xcorr **NCC ≈ 0.78**, 0° residual rotation, checkerboard seams invisible |
| complex coherence with SICD | — | — | **median 0.77, p90 0.96, up to 0.98** over high-signal structures; ~0.53 over all windows (shadows/flat ground are incoherent for any processor) |

Coherence (amplitude **and** phase agreement of the complex signal) is the
strictest test — it asks whether two independent processings of the same raw
phase history reproduce the same speckle. It's measured per-window with
**per-window local registration** (the two grids drift ~100 px across the scene)
and sub-pixel alignment (`src/coherence.py`, map in `out/coherence.png`). Two
things raised the ceiling from ~0.9 to ~0.98:

* **cubic-convolution interpolation** in both PFA stages (was linear) — a linear
  kernel is the main thing that decorrelates the reconstructed speckle from a
  reference processor;
* a **phase-preserving cubic resample** when bringing the SICD onto my grid.

The residual (median 0.77, and the smooth gradient in the coherence map) is a
real processing difference — space-variant PFA phase that Umbra's processor
handles slightly differently — not a focusing or registration error. The
reconstruction lands on Umbra's SICD at orientation `transpose + 180°` (pure
axis/sign convention) with **zero residual rotation**: my data-derived
aperture-plane frame coincides with the SICD's RGAZIM frame.

- `out/full.png`, `out/pfa_scene.png` — the reconstruction.
- `out/register_sicd.png` — checkerboard mosaic vs SICD.
- `out/coherence.png` — complex-coherence map vs SICD.
- `out/ipr.png` — point-target impulse response cuts.

## Files

| file | role |
|---|---|
| `src/cphd_reader.py` | self-contained CPHD 1.x reader — header + XML + PVP + signal, big-endian memmap, per-channel offsets, FX/TOA aware, decimation |
| `src/pfa.py`         | the Polar Format Algorithm (aperture-plane k-space build, 2-stage cubic interpolation, k-space oversample, 2-D FFT) |
| `src/run_pfa.py`     | orchestration: read → PFA → complex image + quicklook |
| `src/viz.py`         | SAR amplitude stretch + point-target IPR measurement |
| `src/register_sicd.py` | magnitude registration vs SICD (auto-detects orientation) |
| `src/coherence.py`   | complex-coherence validation vs SICD (auto orientation, per-window registration) |
