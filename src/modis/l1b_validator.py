"""MODIS Level-1B (L1B) Radiometric Calibration Validation Suite.

Provides mathematical validation, error metric calculation, and comparison reports
between reconstructed L1B radiances (derived from L0/L1A counts via onboard
blackbody radiometry) and NASA operational MOD021KM Level-1B reference products.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .modis_l1b_recon import (
    BAND_LAMBDA,
    calibrate,
    inv_planck,
    planck,
    read_band,
)


@dataclass
class ValidationMetrics:
    """Validation and agreement metrics for a single spectral band."""

    band: int
    wavelength_um: float
    num_valid_pixels: int
    ref_radiance_mean: float
    recon_radiance_mean: float
    rmse: float
    relative_rmse_pct: float
    r_squared: float
    mean_bias_error: float
    max_absolute_error: float
    p50_absolute_error: float
    p90_absolute_error: float
    p99_absolute_error: float
    bt_rmse_k: float
    bt_mbe_k: float
    bt_max_abs_k: float
    implied_lbb_mean: float
    implied_lbb_std: float
    implied_lbb_spread_pct: float
    derived_tbb_k: float
    passes_thresholds: bool = True
    threshold_failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass
class GranuleValidationReport:
    """Validation report across multiple bands for a granule collect."""

    granule_id: str
    mode: str
    band_metrics: Dict[int, ValidationMetrics] = field(default_factory=dict)
    summary_mean_r2: float = 0.0
    summary_mean_rel_rmse_pct: float = 0.0
    all_bands_passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "granule_id": self.granule_id,
            "mode": self.mode,
            "summary_mean_r2": self.summary_mean_r2,
            "summary_mean_rel_rmse_pct": self.summary_mean_rel_rmse_pct,
            "all_bands_passed": self.all_bands_passed,
            "bands": {
                str(band): metrics.to_dict()
                for band, metrics in self.band_metrics.items()
            },
        }


def compute_validation_metrics(
    ref_radiance: np.ndarray,
    recon_radiance: np.ndarray,
    valid_mask: np.ndarray,
    band: int,
    wavelength_um: Optional[float] = None,
    r2_threshold: float = 0.90,
    rel_rmse_threshold: float = 3.5,
    lbb_spread_threshold: float = 2.0,
    implied_lbb_array: Optional[np.ndarray] = None,
    derived_tbb_k: float = 290.0,
) -> ValidationMetrics:
    """Compute comprehensive radiometric and brightness temperature validation metrics.

    Args:
        ref_radiance: Reference radiance array from MOD021KM (W/m^2/sr/um).
        recon_radiance: Reconstructed radiance array (W/m^2/sr/um).
        valid_mask: Boolean mask indicating valid pixels.
        band: MODIS spectral band number (e.g. 31).
        wavelength_um: Center wavelength in micrometers. Defaults to BAND_LAMBDA[band].
        r2_threshold: Minimum R^2 threshold for acceptance.
        rel_rmse_threshold: Maximum relative RMSE (%) threshold for acceptance.
        lbb_spread_threshold: Maximum relative spread (%) for implied L_BB.
        implied_lbb_array: Optional array of implied L_BB values across detector groups.
        derived_tbb_k: Derived blackbody temperature in Kelvin.

    Returns:
        ValidationMetrics dataclass with computed metrics.
    """
    lam = wavelength_um if wavelength_um is not None else BAND_LAMBDA.get(band, 11.0)

    # Filter by valid mask and finite values
    v = valid_mask & np.isfinite(recon_radiance) & np.isfinite(ref_radiance) & (ref_radiance > 0)
    num_valid = int(np.sum(v))

    if num_valid == 0:
        return ValidationMetrics(
            band=band,
            wavelength_um=lam,
            num_valid_pixels=0,
            ref_radiance_mean=0.0,
            recon_radiance_mean=0.0,
            rmse=float("nan"),
            relative_rmse_pct=float("nan"),
            r_squared=float("nan"),
            mean_bias_error=float("nan"),
            max_absolute_error=float("nan"),
            p50_absolute_error=float("nan"),
            p90_absolute_error=float("nan"),
            p99_absolute_error=float("nan"),
            bt_rmse_k=float("nan"),
            bt_mbe_k=float("nan"),
            bt_max_abs_k=float("nan"),
            implied_lbb_mean=0.0,
            implied_lbb_std=0.0,
            implied_lbb_spread_pct=0.0,
            derived_tbb_k=derived_tbb_k,
            passes_thresholds=False,
            threshold_failures=["No valid pixels found for validation"],
        )

    y_ref = ref_radiance[v]
    y_rec = recon_radiance[v]
    residuals = y_rec - y_ref
    abs_res = np.abs(residuals)

    ref_mean = float(np.mean(y_ref))
    rec_mean = float(np.mean(y_rec))

    # Radiometric error metrics
    rmse = float(np.sqrt(np.mean(residuals**2)))
    rel_rmse_pct = float(100.0 * rmse / ref_mean) if ref_mean > 0 else float("inf")
    mbe = float(np.mean(residuals))
    max_abs = float(np.max(abs_res))

    p50_abs = float(np.percentile(abs_res, 50))
    p90_abs = float(np.percentile(abs_res, 90))
    p99_abs = float(np.percentile(abs_res, 99))

    # R^2 determination
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_ref - ref_mean)**2)
    r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0

    # Brightness temperature metrics
    # Clip positive radiances before inverse planck
    min_rad = 1e-4
    bt_ref = inv_planck(lam, np.maximum(y_ref, min_rad))
    bt_rec = inv_planck(lam, np.maximum(y_rec, min_rad))
    bt_diff = bt_rec - bt_ref

    bt_rmse_k = float(np.sqrt(np.mean(bt_diff**2)))
    bt_mbe_k = float(np.mean(bt_diff))
    bt_max_abs_k = float(np.max(np.abs(bt_diff)))

    # Blackbody consistency metrics
    if implied_lbb_array is not None and len(implied_lbb_array) > 0:
        lbb_mean = float(np.mean(implied_lbb_array))
        lbb_std = float(np.std(implied_lbb_array))
        lbb_spread = float(100.0 * lbb_std / lbb_mean) if lbb_mean > 0 else 0.0
    else:
        lbb_mean = planck(lam, derived_tbb_k)
        lbb_std = 0.0
        lbb_spread = 0.0

    # Quality Gate threshold checks
    failures: List[str] = []
    if r_squared < r2_threshold:
        failures.append(f"R^2 ({r_squared:.4f}) < threshold ({r2_threshold:.4f})")
    if rel_rmse_pct > rel_rmse_threshold:
        failures.append(f"Relative RMSE ({rel_rmse_pct:.2f}%) > threshold ({rel_rmse_threshold:.2f}%)")
    if lbb_spread > lbb_spread_threshold:
        failures.append(f"L_BB spread ({lbb_spread:.2f}%) > threshold ({lbb_spread_threshold:.2f}%)")

    passes = len(failures) == 0

    return ValidationMetrics(
        band=band,
        wavelength_um=lam,
        num_valid_pixels=num_valid,
        ref_radiance_mean=ref_mean,
        recon_radiance_mean=rec_mean,
        rmse=rmse,
        relative_rmse_pct=rel_rmse_pct,
        r_squared=r_squared,
        mean_bias_error=mbe,
        max_absolute_error=max_abs,
        p50_absolute_error=p50_abs,
        p90_absolute_error=p90_abs,
        p99_absolute_error=p99_abs,
        bt_rmse_k=bt_rmse_k,
        bt_mbe_k=bt_mbe_k,
        bt_max_abs_k=bt_max_abs_k,
        implied_lbb_mean=lbb_mean,
        implied_lbb_std=lbb_std,
        implied_lbb_spread_pct=lbb_spread,
        derived_tbb_k=derived_tbb_k,
        passes_thresholds=passes,
        threshold_failures=failures,
    )


def evaluate_calibrated_band(
    ev_counts: np.ndarray,
    sv_counts: np.ndarray,
    bb_counts: np.ndarray,
    ref_radiance: np.ndarray,
    valid_mask: np.ndarray,
    band: int,
    mode: str = "independent",
    r2_threshold: float = 0.90,
    rel_rmse_threshold: float = 3.5,
) -> Tuple[np.ndarray, ValidationMetrics]:
    """Calibrate counts and evaluate validation metrics against reference radiance.

    Args:
        ev_counts: Earth view raw counts (NL, NF).
        sv_counts: Space view raw counts (NL, 50).
        bb_counts: Blackbody raw counts (NL, 50).
        ref_radiance: Reference radiance from MOD021KM.
        valid_mask: Valid pixel mask.
        band: MODIS band number.
        mode: Calibration mode: 'independent' (blind BB), 'affine' (fit), or 'rvs' (scan angle).
        r2_threshold: Minimum acceptable R^2.
        rel_rmse_threshold: Maximum acceptable relative RMSE (%).

    Returns:
        Tuple of (reconstructed_radiance_array, ValidationMetrics).
    """
    if mode not in ("independent", "affine", "rvs"):
        raise ValueError(f"Unknown calibration mode '{mode}'. Choose from 'independent', 'affine', 'rvs'.")

    cal_result = calibrate(ev_counts, sv_counts, bb_counts, ref_radiance, valid_mask, band)

    if mode == "independent":
        recon = cal_result["recon_bb"]
    elif mode == "affine":
        recon = cal_result["recon_aff"]
    elif mode == "rvs":
        recon = cal_result["recon_rvs"]
    else:
        raise ValueError(f"Unknown calibration mode '{mode}'. Choose from 'independent', 'affine', 'rvs'.")


    metrics = compute_validation_metrics(
        ref_radiance=ref_radiance,
        recon_radiance=recon,
        valid_mask=valid_mask,
        band=band,
        wavelength_um=cal_result["lam"],
        r2_threshold=r2_threshold,
        rel_rmse_threshold=rel_rmse_threshold,
        implied_lbb_array=cal_result["LBB_implied"],
        derived_tbb_k=float(cal_result["T_BB"]),
    )

    return recon, metrics


def generate_synthetic_l1a_l1b(
    num_scans: int = 10,
    num_frames: int = 100,
    bands: Tuple[int, ...] = (20, 21, 22, 29, 31, 32),
    noise_level: float = 0.005,
    temp_earth_range: Tuple[float, float] = (270.0, 310.0),
    temp_bb: float = 290.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate synthetic L1A counts and L1B reference radiance fixtures for testing.

    Args:
        num_scans: Number of 10-detector scans.
        num_frames: Number of Earth-view frames per scan line.
        bands: Tuple of MODIS thermal emissive bands to generate.
        noise_level: Proportional Gaussian noise added to detector readouts.
        temp_earth_range: (min_K, max_K) temperature range of the simulated Earth scene.
        temp_bb: Temperature in Kelvin of the onboard blackbody (nominal ~290 K).
        seed: Random seed for deterministic reproducibility.

    Returns:
        Dictionary mapping band -> dict(EV, SV, BB, Lref, valid, NL, NF).
    """
    rng = np.random.default_rng(seed)
    num_lines = num_scans * 10
    results = {}

    for b in bands:
        lam = BAND_LAMBDA.get(b, 11.0)
        l_bb_true = planck(lam, temp_bb)

        # Base detector gains and space-view offsets across the 10 detectors and 2 mirror sides
        dn_sv_base = 50.0 + rng.uniform(-2.0, 2.0, size=(num_lines, 1))
        # Gain in counts per unit radiance: ~ (1000 - 50) / l_bb
        gain_counts_per_rad = (800.0 / l_bb_true) * (1.0 + rng.uniform(-0.02, 0.02, size=(num_lines, 1)))

        # Simulate Space View (SV): count around dn_sv_base + noise (50 frames)
        sv = dn_sv_base + rng.normal(0, 1.0, size=(num_lines, 50))

        # Simulate Blackbody (BB): dn_sv_base + gain * l_bb_true + noise (50 frames)
        bb = dn_sv_base + (gain_counts_per_rad * l_bb_true) + rng.normal(0, 1.0, size=(num_lines, 50))

        # Simulate Earth Scene temperatures
        t_scene = rng.uniform(temp_earth_range[0], temp_earth_range[1], size=(num_lines, num_frames))
        l_earth_true = planck(lam, t_scene)

        # Add optional minor RVS (scan-angle curve)
        scan_angle_factor = 1.0 + 0.005 * np.cos(np.linspace(-np.pi / 2, np.pi / 2, num_frames))
        l_earth_at_sensor = l_earth_true * scan_angle_factor[None, :]

        # Raw counts with instrument noise
        ev_counts = dn_sv_base + (gain_counts_per_rad * l_earth_at_sensor)
        ev_counts += rng.normal(0, noise_level * np.mean(ev_counts), size=ev_counts.shape)

        valid = np.ones((num_lines, num_frames), dtype=bool)

        # Mark 0.5% of pixels as invalid / missing for robust testing
        invalid_mask = rng.uniform(size=(num_lines, num_frames)) < 0.005
        valid[invalid_mask] = False

        results[b] = {
            "EV": ev_counts,
            "SV": sv,
            "BB": bb,
            "Lref": l_earth_true,
            "valid": valid,
            "NL": num_lines,
            "NF": num_frames,
            "lam": lam,
            "temp_bb": temp_bb,
        }

    return results


def export_report_json(report: GranuleValidationReport, filepath: str) -> None:
    """Export validation report to JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)


def generate_markdown_summary(report: GranuleValidationReport) -> str:
    """Generate a clean Markdown summary table from a validation report."""
    lines = [
        f"# MODIS L1B Calibration Validation Report: {report.granule_id}",
        "",
        f"- **Mode**: `{report.mode}`",
        f"- **Overall Mean R²**: `{report.summary_mean_r2:.5f}`",
        f"- **Overall Mean Rel. RMSE**: `{report.summary_mean_rel_rmse_pct:.2f}%`",
        f"- **Quality Gate Passed**: `{'✅ YES' if report.all_bands_passed else '❌ NO'}`",
        "",
        "## Band-by-Band Radiometric & Thermal Residuals",
        "",
        "| Band | λ (µm) | Valid Pixels | Ref Mean (W/m²/sr/µm) | RMSE | Rel. RMSE (%) | R² | Mean Bias | ΔTb RMSE (K) | L_BB Spread (%) | Status |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for band, m in sorted(report.band_metrics.items()):
        status = "✅ PASS" if m.passes_thresholds else "❌ FAIL"
        lines.append(
            f"| **{band}** | {m.wavelength_um:.2f} | {m.num_valid_pixels:,} | {m.ref_radiance_mean:.3f} | "
            f"{m.rmse:.4f} | {m.relative_rmse_pct:.2f}% | {m.r_squared:.5f} | {m.mean_bias_error:+.4f} | "
            f"{m.bt_rmse_k:.3f} K | {m.implied_lbb_spread_pct:.2f}% | {status} |"
        )

    lines.append("")
    if not report.all_bands_passed:
        lines.append("### ⚠️ Threshold Failures")
        for band, m in sorted(report.band_metrics.items()):
            if not m.passes_thresholds:
                lines.append(f"- **Band {band}**:")
                for fail in m.threshold_failures:
                    lines.append(f"  - {fail}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """CLI Entrypoint for MODIS L1B Radiometric Calibration Validation."""
    ap = argparse.ArgumentParser(description="MODIS L1B Radiometric Calibration Validation Suite")
    ap.add_argument("--l1a", help="Path to input MOD01 L1A HDF file")
    ap.add_argument("--l1b", help="Path to input MOD021KM L1B HDF reference file")
    ap.add_argument("--bands", nargs="+", type=int, default=[20, 21, 22, 29, 31, 32], help="Bands to validate")
    ap.add_argument("--mode", choices=["independent", "affine", "rvs"], default="independent", help="Calibration mode")
    ap.add_argument("--synthetic", action="store_true", help="Run validation on synthetic test data")
    ap.add_argument("--out-json", default="reports/modis_l1b_validation.json", help="Path to output JSON report")
    ap.add_argument("--out-md", default="reports/modis_l1b_validation.md", help="Path to output Markdown report")
    ap.add_argument("--strict", action="store_true", help="Exit with non-zero status if any band fails quality gates")
    args = ap.parse_args()

    granule_id = "SYNTHETIC_GRANULE" if args.synthetic else os.path.basename(args.l1b or "UNKNOWN")
    report = GranuleValidationReport(granule_id=granule_id, mode=args.mode)

    if args.synthetic:
        print(f"Generating synthetic L1A/L1B test data for bands: {args.bands} ...")
        synth = generate_synthetic_l1a_l1b(num_scans=15, num_frames=150, bands=tuple(args.bands))
        for b in args.bands:
            d = synth[b]
            _, metrics = evaluate_calibrated_band(
                d["EV"], d["SV"], d["BB"], d["Lref"], d["valid"], b, mode=args.mode
            )
            report.band_metrics[b] = metrics
    else:
        if not args.l1a or not args.l1b:
            print("Error: --l1a and --l1b are required when not running with --synthetic.", file=sys.stderr)
            sys.exit(1)

        for b in args.bands:
            try:
                ev, sv, bb, lref, valid, _, _ = read_band(args.l1a, args.l1b, b)
                _, metrics = evaluate_calibrated_band(ev, sv, bb, lref, valid, b, mode=args.mode)
                report.band_metrics[b] = metrics
            except Exception as exc:
                print(f"Warning: Failed to validate band {b}: {exc}", file=sys.stderr)

    if report.band_metrics:
        r2_vals = [m.r_squared for m in report.band_metrics.values() if np.isfinite(m.r_squared)]
        rel_vals = [m.relative_rmse_pct for m in report.band_metrics.values() if np.isfinite(m.relative_rmse_pct)]
        report.summary_mean_r2 = float(np.mean(r2_vals)) if r2_vals else 0.0
        report.summary_mean_rel_rmse_pct = float(np.mean(rel_vals)) if rel_vals else 0.0
        report.all_bands_passed = all(m.passes_thresholds for m in report.band_metrics.values())

    md_summary = generate_markdown_summary(report)
    print("\n" + md_summary + "\n")

    export_report_json(report, args.out_json)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md_summary)

    print(f"Exported JSON report to: {args.out_json}")
    print(f"Exported Markdown report to: {args.out_md}")

    if args.strict and not report.all_bands_passed:
        sys.exit(2)


if __name__ == "__main__":
    main()
