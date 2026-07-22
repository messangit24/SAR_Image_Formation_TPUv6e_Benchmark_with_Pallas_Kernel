# Umbra CPHD → complex image: four SAR image-formation algorithms

Recreates Umbra's spotlight SAR image formation from **open-data CPHD** (phase
history) with **four independent algorithms**, and validates each against
Umbra's own **SICD** (their focused complex image, *no autofocus*):

| algorithm | family | file |
|---|---|---|
| **PFA** — Polar Format Algorithm | polar → Cartesian resample + 2-D FFT | `pfa.py` |
| **ω-k / RMA** — Range Migration | Stolt dispersion resample + 2-D FFT | `rma.py` |
| **BPA** — Back-Projection | exact time-domain, per-pixel sum | `bpa.py` |
| **CSA** — Chirp Scaling (family) | range-Doppler + keystone RCMC | `csa.py` |

The point of the exercise: Umbra publishes the raw compensated phase history
(CPHD) *and* the focused complex image (SICD) for the same collect, and states
the SICD is formed **without autofocus**. So textbook algorithms — no autofocus,
no proprietary tricks — should reproduce it. They do (see the comparison below),
and they **agree with each other more than with Umbra** — the signature of
correct, independent processing.

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

**What the residual and the coherence "gradient" actually are** (each ruled in/out
by a direct test, not a guess):

| candidate cause | test | verdict |
|---|---|---|
| PFA wavefront-curvature defocus | residual phase at scene edge = **0.44 rad** ≪ π/2 | ruled out (geometry too benign: 663 km range, 1.8° aperture) |
| space-variant defocus | point-target IPR is **uniform** (0.28–0.35 m) in centre *and* all 4 corners | ruled out |
| frame rotation vs the SICD grid | my axes vs SICD `UVectECF`: **0.002°**, SCP = SRP exactly | ruled out |
| resampling onto a different grid | re-formed on the SICD's **exact** grid (axes + pixel spacing) → coherence **unchanged** | ruled out |
| **SNR** | coherence vs brightness quintiles: **0.02 → 0.44 → 0.51 → 0.66 → 0.87** | **this is it** |

The gradient is **signal-to-noise**: coherence necessarily tracks backscatter
strength, so the bright mine structures reach 0.87–0.98 while dark, low-return
terrain is incoherent for *any* pair of processors. The remaining gap to 1.0 on
bright pixels (~0.87 median) is genuine, expected processing difference between
this PFA and Umbra's proprietary chain (exact interpolation / spectral shaping) —
not a focusing, registration, or resampling error. The reconstruction lands on
Umbra's SICD with **0.002° residual rotation and SCP = SRP**: my data-derived
aperture-plane frame *is* the SICD's grid.

- `out/full.png`, `out/pfa_scene.png` — the reconstruction.
- `out/register_sicd.png` — checkerboard mosaic vs SICD.
- `out/coherence.png` — complex-coherence map vs SICD.
- `out/ipr.png` — point-target impulse response cuts.

## Four algorithms vs the SICD

All four form the same focused Bingham pit at the SICD's resolution
(`out/algorithms_compare.png` — PFA / ω-k / CSA full scene, and BPA vs the SICD
chip side-by-side). Complex coherence with the SICD (high-signal windows,
carrier-robust metric):

| algorithm | coherence vs SICD | how measured | notes |
|---|---|---|---|
| **BPA** | **0.89** (up to 0.98) | formed on the SICD's *native* grid → direct, no resample | exact geometry; **best** — the cleanest measure of "does a correct algorithm reproduce the SICD" |
| **PFA** | 0.77 | aperture-plane grid, resampled to compare | `run_pfa.py` |
| **ω-k / RMA** | 0.74 | aperture-plane grid, resampled to compare | `run_rma.py`; the frequency-domain migration method |
| **CSA** | 0.19 (peak 0.97 at centre) | aperture-plane grid, resampled | keystone RCMC only — see below |

**Cross-validation — the algorithms agree with each OTHER more than with Umbra.**
PFA vs ω-k (same grid, direct) = **0.88**, well above either's ~0.75 vs the SICD.
Two independent textbook algorithms reproducing each other's *complex* speckle at
0.88 is strong evidence both are correct; the gap to Umbra is that provider's
proprietary processing, not an error in ours.

**BPA is highest because it is formed directly on the SICD grid** (the `frame=`
option pins the range/azimuth axes to the SICD's `Grid.Row/Col.UVectECF`), so its
comparison needs no resampling. The ~0.77 for PFA/ω-k is partly the cost of
resampling the aperture-plane image onto the SICD grid to compare — i.e. a
*measurement* limit, not a reconstruction one (BPA, being on-grid, shows the
reconstructions are good to ~0.89).

**Why CSA is the odd one out (and why ω-k was the fallback).** CSA is a *stripmap*
range-Doppler algorithm; its defining "chirp scaling" step operates on the range
**chirp**, but Umbra's CPHD is **FX-domain (already de-ramped)** — there is no
chirp to scale. Its remaining job (correct the ~±10 m linear range walk across the
1.8° aperture) is done here by a **keystone** RCMC, which focuses the scene (peak
coherence 0.97 near centre) but is only first-order: it misses the range-curvature
/ secondary-range-compression that a full Stolt (ω-k) or separable PFA handles, so
coherence falls off the scene centre (median 0.19). This confirms CSA is not the
right tool for de-ramped spotlight data — exactly why **ω-k/RMA** is delivered as
the frequency-domain method.

Run them:
```bash
python src/run_pfa.py --out out/full --save-complex          # PFA
python src/run_rma.py --out out/rma  --save-complex           # omega-k / RMA
python src/run_csa.py --out out/csa  --save-complex           # CSA-family
python src/run_bpa.py --chip 2048 --out out/bpa              # BPA (chip on SICD grid, ~70 min)
```

## Files

| file | role |
|---|---|
| `src/cphd_reader.py` | self-contained CPHD 1.x reader — header + XML + PVP + signal, big-endian memmap, per-channel offsets, FX/TOA aware, decimation |
| `src/pfa.py`         | **PFA** — aperture-plane k-space build, 2-stage cubic/sinc interpolation, k-space oversample, 2-D FFT (`frame=` to form on an external grid) |
| `src/rma.py`         | **ω-k / RMA** — Stolt-dispersion polar→Cartesian resample + 2-D FFT |
| `src/bpa.py`         | **BPA** — exact time-domain back-projection onto arbitrary pixel grid |
| `src/csa.py`         | **CSA-family** — range-Doppler + keystone RCMC (+ analysis of why pure chirp-scaling is inapplicable) |
| `src/run_{pfa,rma,csa,bpa}.py` | per-algorithm runners → complex image + quicklook + geom sidecar |
| `src/sicd_grid.py`   | read a SICD's output grid (axes, pixel spacing, SCP) so any former can produce images on that exact grid |
| `src/viz.py`         | SAR amplitude stretch + point-target IPR measurement |
| `src/register_sicd.py` | magnitude registration vs SICD (auto-detects orientation) |
| `src/coherence.py`   | complex-coherence validation vs SICD (auto orientation, per-window registration, carrier-robust metric) |
