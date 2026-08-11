"""High-performance MODIS Mid-Wave Infrared (MWIR) Thermal Anomaly Detection Engine.

Provides contextual dual-band thermal anomaly detection, solar glint vs thermal source
discrimination, land/water body filtering, high dynamic range dual-gain processing (Bands 21 & 22),
non-linear Dozier sub-pixel temperature/area inversion, and Fire Radiative Power (FRP) quantification.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy.ndimage import uniform_filter

# Planck radiation constants (wavelength form)
C1 = 1.191042953e8  # W * um^4 / (m^2 * sr) (2 * h * c^2)
C2 = 1.4387770e4    # um * K                (h * c / k_B)
STEFAN_BOLTZMANN = 5.670374419e-8  # W / (m^2 * K^4)

# Nominal MODIS band-center wavelengths (um)
LAMBDA_MWIR_21 = 3.992
LAMBDA_MWIR_22 = 3.968
LAMBDA_LWIR_31 = 11.017
LAMBDA_LWIR_32 = 12.028
LAMBDA_NIR_2 = 0.858


@dataclass(frozen=True)
class DetectionConfig:
    """Configuration parameters for contextual thermal anomaly detection."""
    t21_abs_threshold_day: float = 360.0    # Absolute detection threshold for high-T sources (K)
    t21_abs_threshold_night: float = 320.0  # Nighttime absolute threshold (K)
    delta_t_min: float = 10.0               # Minimum MWIR - LWIR excess (K)
    context_window_size: int = 15           # Context window size in pixels (odd int)
    context_k1: float = 3.0                 # Multiplier for background T21 standard deviation
    context_k2: float = 3.0                 # Multiplier for background delta_T standard deviation
    nir_bg_ratio_solar: float = 4.0         # NIR / background threshold for solar glint
    nir_bg_ratio_thermal: float = 2.0       # NIR / background threshold for true thermal source
    min_cloud_t31: float = 265.0            # Cloud screening threshold on Band 31 (K)
    min_valid_t21: float = 200.0            # Minimum valid physical temperature (K)
    max_valid_t21: float = 550.0            # Maximum valid physical temperature (K)
    saturation_k: float = 329.0             # Handover temperature from Band 22 to Band 21 (K)
    filter_water: bool = True               # Exclude water pixels from active fire triggers
    pixel_area_m2: float = 1.0e6            # 1 km x 1 km nominal nadir pixel area (m^2)


@dataclass
class ThermalDetection:
    """Individual thermal anomaly detection with radiometric characterization."""
    pixel_row: int
    pixel_col: int
    lat: float | None
    lon: float | None
    bt21: float
    bt31: float
    delta_bt: float
    nir_count: float | None
    is_solar_false_alarm: bool
    is_thermal_candidate: bool
    is_water_pixel: bool
    confidence: float
    target_temp_k: float | None = None
    target_fraction: float | None = None
    frp_mw: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def planck(lam_um: float | np.ndarray, temp_k: float | np.ndarray) -> float | np.ndarray:
    """Compute blackbody spectral radiance in W / (m^2 * sr * um)."""
    t = np.maximum(temp_k, 1e-3)
    return C1 / (lam_um**5 * (np.expm1(C2 / (lam_um * t))))


def inv_planck(lam_um: float | np.ndarray, radiance: float | np.ndarray) -> float | np.ndarray:
    """Compute brightness temperature in Kelvin from spectral radiance."""
    rad = np.maximum(radiance, 1e-9)
    return C2 / (lam_um * np.log(C1 / (rad * lam_um**5) + 1.0))


def blend_high_dynamic_range(
    bt21: np.ndarray,
    bt22: np.ndarray,
    saturation_k: float = 329.0,
) -> np.ndarray:
    """Seamlessly blend high-gain Band 22 (low noise) and low-gain Band 21 (high range).

    Band 22 has low noise (NEΔT ~0.07 K) but saturates at ~331 K. Band 21 NEΔT is ~2 K
    but saturates at ~500 K. We use Band 22 below the saturation threshold and Band 21
    at or above it.
    """
    bt22_valid = np.isfinite(bt22) & (bt22 < saturation_k)
    return np.where(bt22_valid, bt22, bt21)


def solve_dozier(
    l21: float,
    l31: float,
    t_bg: float,
    lam21: float = LAMBDA_MWIR_21,
    lam31: float = LAMBDA_LWIR_31,
    t_min: float = 350.0,
    t_max: float = 2000.0,
    tol: float = 1e-4,
    max_iter: int = 40,
) -> tuple[float, float]:
    """Solve the Dozier non-linear dual-channel system for subpixel temperature and fraction.

    Solves:
      L_21 = p * B(lam21, T_t) + (1 - p) * B(lam21, T_bg)
      L_31 = p * B(lam31, T_t) + (1 - p) * B(lam31, T_bg)

    Returns:
      (fraction p, target_temperature T_t in K). If no physical root is found,
      returns (1.0, inv_planck(lam21, l21)).
    """
    l21_bg = float(planck(lam21, t_bg))
    l31_bg = float(planck(lam31, t_bg))

    # Check that observed radiance exceeds background
    if l21 <= l21_bg or l31 <= l31_bg:
        return 0.0, float(t_bg)

    # Function f(T) = p21(T) - p31(T) = 0
    def p_diff(t_target: float) -> float:
        b21_t = float(planck(lam21, t_target))
        b31_t = float(planck(lam31, t_target))
        denom21 = b21_t - l21_bg
        denom31 = b31_t - l31_bg
        if denom21 <= 0 or denom31 <= 0:
            return 1.0
        p21 = (l21 - l21_bg) / denom21
        p31 = (l31 - l31_bg) / denom31
        return p21 - p31

    # Bisection search over [t_min, t_max]
    low = max(float(t_min), float(t_bg) + 10.0)
    high = float(t_max)
    f_low = p_diff(low)
    f_high = p_diff(high)

    # If bounds don't bracket the root, fallback to pixel-integrated estimate
    if f_low * f_high > 0:
        t_int = float(inv_planck(lam21, l21))
        return 1.0, t_int

    t_mid = (low + high) / 2.0
    for _ in range(max_iter):
        t_mid = (low + high) / 2.0
        f_mid = p_diff(t_mid)
        if abs(f_mid) < tol or (high - low) / 2.0 < 0.1:
            break
        if f_low * f_mid < 0:
            high = t_mid
            f_high = f_mid
        else:
            low = t_mid
            f_low = f_mid

    # Calculate final fraction
    b21_final = float(planck(lam21, t_mid))
    p_final = (l21 - l21_bg) / max(b21_final - l21_bg, 1e-9)
    p_clamped = float(np.clip(p_final, 1e-7, 1.0))
    return p_clamped, float(t_mid)


def calculate_frp(
    p: float,
    t_target: float,
    t_bg: float,
    pixel_area_m2: float = 1.0e6,
) -> float:
    """Compute Fire Radiative Power (FRP) in Megawatts (MW) using Stefan-Boltzmann."""
    if p <= 0 or t_target <= t_bg:
        return 0.0
    power_watts = p * pixel_area_m2 * STEFAN_BOLTZMANN * (t_target**4 - t_bg**4)
    return float(power_watts * 1e-6)  # Watts -> MW


def compute_context_statistics(
    image: np.ndarray,
    valid_mask: np.ndarray,
    window_size: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute local mean and standard deviation over an N x N moving window."""
    valid_float = valid_mask.astype(np.float64)
    img_valid = np.where(valid_mask, image, 0.0)

    # Sum and sum of squares
    count = uniform_filter(valid_float, size=window_size, mode="reflect") * (window_size**2)
    sum_val = uniform_filter(img_valid, size=window_size, mode="reflect") * (window_size**2)
    sum_sq = uniform_filter(img_valid**2, size=window_size, mode="reflect") * (window_size**2)

    count_safe = np.maximum(count, 1.0)
    mean = sum_val / count_safe
    variance = np.maximum(sum_sq / count_safe - mean**2, 0.0)
    std = np.sqrt(variance)

    # Where fewer than 20% of pixels in the window are valid, fallback to global valid stats
    min_pixels = 0.2 * (window_size**2)
    global_mean = float(np.nanmedian(image[valid_mask])) if np.any(valid_mask) else 300.0
    global_std = float(np.nanstd(image[valid_mask])) if np.any(valid_mask) else 5.0

    insufficient = count < min_pixels
    mean[insufficient] = global_mean
    std[insufficient] = max(global_std, 1.0)

    return mean, std


def detect_thermal_anomalies(
    bt21: np.ndarray,
    bt31: np.ndarray,
    nir: np.ndarray | None = None,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    water_mask: np.ndarray | None = None,
    config: DetectionConfig | None = None,
    is_day: bool = True,
) -> tuple[list[ThermalDetection], dict[str, np.ndarray]]:
    """Run full dual-band contextual MWIR thermal anomaly detection.

    Args:
      bt21: 2D array of Band 21 (3.99 um) brightness temperatures in Kelvin.
      bt31: 2D array of Band 31 (11 um) brightness temperatures in Kelvin.
      nir: Optional 2D array of Band 2 (858 nm) reflective counts or reflectance.
      lat: Optional 2D array of pixel latitudes.
      lon: Optional 2D array of pixel longitudes.
      water_mask: Optional boolean 2D array (True = water pixel).
      config: DetectionConfig parameters.
      is_day: Whether the collect is daytime (enables solar glint discrimination).

    Returns:
      (list of ThermalDetection objects, dict of intermediate boolean mask arrays)
    """
    if config is None:
        config = DetectionConfig()

    dbt = bt21 - bt31

    # Water mask resolution
    is_water = np.zeros_like(bt21, dtype=bool)
    if water_mask is not None:
        is_water = np.asarray(water_mask, dtype=bool)

    valid_data = (
        np.isfinite(bt21)
        & np.isfinite(bt31)
        & (bt21 >= config.min_valid_t21)
        & (bt21 <= config.max_valid_t21)
        & (bt31 >= config.min_cloud_t31)
    )

    if config.filter_water:
        valid_land = valid_data & (~is_water)
    else:
        valid_land = valid_data

    # Absolute trigger
    abs_thresh = config.t21_abs_threshold_day if is_day else config.t21_abs_threshold_night
    abs_hot = valid_land & (bt21 >= abs_thresh)

    # Background context window statistics
    # Exclude abnormally hot pixels from background context
    bg_candidate_mask = valid_land & (bt21 < abs_thresh) & (dbt < config.delta_t_min)
    mean_t21, std_t21 = compute_context_statistics(bt21, bg_candidate_mask, config.context_window_size)
    mean_dbt, std_dbt = compute_context_statistics(dbt, bg_candidate_mask, config.context_window_size)

    # Contextual trigger
    context_hot = (
        valid_land
        & (dbt >= mean_dbt + config.context_k2 * np.maximum(std_dbt, 1.0))
        & (dbt >= config.delta_t_min)
        & (bt21 >= mean_t21 + config.context_k1 * np.maximum(std_t21, 1.0))
    )

    candidate_mask = abs_hot | context_hot

    # Solar glint vs thermal discrimination using NIR Band 2
    solar_glint_mask = np.zeros_like(candidate_mask, dtype=bool)
    thermal_candidate_mask = candidate_mask.copy()

    if nir is not None and is_day:
        nir_valid = np.isfinite(nir) & (nir > 0)
        nir_bg_mask = valid_land & (bt21 < 305.0) & nir_valid
        nir_bg_val = float(np.nanmedian(nir[nir_bg_mask])) if np.any(nir_bg_mask) else 100.0

        solar_glint_mask = candidate_mask & nir_valid & (nir > config.nir_bg_ratio_solar * nir_bg_val)
        thermal_candidate_mask = candidate_mask & (~solar_glint_mask)

    # Build catalog of detections
    detections: list[ThermalDetection] = []
    rows, cols = np.where(candidate_mask)

    for r, c in zip(rows, cols):
        t21_val = float(bt21[r, c])
        t31_val = float(bt31[r, c])
        dbt_val = float(dbt[r, c])
        lat_val = float(lat[r, c]) if lat is not None else None
        lon_val = float(lon[r, c]) if lon is not None else None
        nir_val = float(nir[r, c]) if nir is not None else None

        is_solar = bool(solar_glint_mask[r, c])
        is_thermal = bool(thermal_candidate_mask[r, c])
        pixel_is_water = bool(is_water[r, c])

        # Confidence calculation
        t21_score = np.clip((t21_val - 310.0) / 90.0, 0.0, 1.0)
        dbt_score = np.clip((dbt_val - 5.0) / 45.0, 0.0, 1.0)
        conf = 0.5 * t21_score + 0.5 * dbt_score
        if is_solar or pixel_is_water:
            conf *= 0.1

        # Solve Dozier subpixel temperature & area
        rad21 = float(planck(LAMBDA_MWIR_21, t21_val))
        rad31 = float(planck(LAMBDA_LWIR_31, t31_val))
        p_sub, t_sub = solve_dozier(rad21, rad31, t_bg=float(mean_t21[r, c]))

        # Calculate Fire Radiative Power (MW)
        frp = calculate_frp(p_sub, t_sub, t_bg=float(mean_t21[r, c]), pixel_area_m2=config.pixel_area_m2)

        detections.append(
            ThermalDetection(
                pixel_row=int(r),
                pixel_col=int(c),
                lat=lat_val,
                lon=lon_val,
                bt21=t21_val,
                bt31=t31_val,
                delta_bt=dbt_val,
                nir_count=nir_val,
                is_solar_false_alarm=is_solar,
                is_thermal_candidate=is_thermal,
                is_water_pixel=pixel_is_water,
                confidence=float(conf),
                target_temp_k=t_sub if is_thermal else None,
                target_fraction=p_sub if is_thermal else None,
                frp_mw=frp if is_thermal else None,
            )
        )

    masks = {
        "valid": valid_data,
        "valid_land": valid_land,
        "water_mask": is_water,
        "candidate": candidate_mask,
        "solar_glint": solar_glint_mask,
        "thermal_candidate": thermal_candidate_mask,
        "mean_t21": mean_t21,
        "mean_dbt": mean_dbt,
    }

    return detections, masks
