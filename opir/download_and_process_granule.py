#!/usr/bin/env python3
"""
NOAA-20 / NOAA-21 VIIRS Granule Downloader & OPIR Processing Pipeline

Searches, downloads, and processes authentic VIIRS satellite granules directly from NOAA's public AWS S3 repository (s3://noaa-nesdis-n20-pds/).
Calculates radiometric calibration, monochromatic vs NOAA SRF Bandpass Planck inversion, WGS84 geolocation mapping, and dual-band thermal fire detection.

Usage:
    python opir/download_and_process_granule.py --date 2026-07-29 --latest
    python opir/download_and_process_granule.py --date 2026-07-29 --stamp d20260729_t0001036
    python opir/download_and_process_granule.py --list --date 2026-07-29
"""

import os
import sys
import argparse
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Physical Constants & NOAA SRF Bandpass Parameters for VIIRS Band M12 (3.7 µm)
H_PLANCK = 6.62607015e-34     # J s
C_LIGHT = 2.99792458e8        # m/s
K_BOLTZMANN = 1.380649e-23    # J/K
LAMBDA_M12_MONO = 3.700e-6    # 3.700 µm nominal central wavelength
LAMBDA_M12_SRF = 3.719521e-6  # 3.719521 µm effective SRF central wavelength
B0_SRF = 0.000081             # K
B1_SRF = 1.002780             # Scale factor

AWS_S3_BASE_URL = "https://noaa-nesdis-n20-pds.s3.amazonaws.com"

def planck_inversion(radiance_w_m2_sr_um, wavelength_m=LAMBDA_M12_MONO):
    """Computes spectral brightness temperature T (Kelvin) from spectral radiance."""
    radiance_si = radiance_w_m2_sr_um * 1e6  # W / (m^3 sr)
    c1 = 2.0 * H_PLANCK * (C_LIGHT ** 2)
    c2 = (H_PLANCK * C_LIGHT) / K_BOLTZMANN
    
    safe_rad = np.maximum(radiance_si, 1e-12)
    term = (c1 / (wavelength_m ** 5 * safe_rad)) + 1.0
    t_kelvin = c2 / (wavelength_m * np.log(term))
    return t_kelvin

def noaa_srf_planck_correction(radiance_w_m2_sr_um):
    """Applies NOAA's exact Bandpass Spectral Response Function (SRF) Planck correction."""
    t_p = planck_inversion(radiance_w_m2_sr_um, LAMBDA_M12_SRF)
    t_srf = B0_SRF + B1_SRF * t_p
    return t_srf

def fetch_s3_keys(prefix, max_keys=100):
    """Queries public AWS S3 bucket XML endpoint for key listing under prefix."""
    url = f"{AWS_S3_BASE_URL}/?prefix={prefix}&max-keys={max_keys}"
    req = urllib.request.urlopen(url)
    body = req.read()
    root = ET.fromstring(body)
    ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    keys = []
    for c in root.findall('s3:Contents', ns):
        key = c.find('s3:Key', ns).text
        keys.append(key)
    return keys

def list_granules_for_date(date_str):
    """Lists available VIIRS M12 granules for a given YYYY-MM-DD date."""
    year, month, day = date_str.split('-')
    prefix = f"VIIRS-M12-SDR/{year}/{month}/{day}/"
    print(f"Querying NOAA AWS S3 for granules under prefix: {prefix}")
    keys = fetch_s3_keys(prefix, max_keys=200)
    
    granules = []
    for k in keys:
        filename = os.path.basename(k)
        if filename.startswith("SVM12"):
            # Extract timestamp identifier e.g. d20260729_t0001036
            parts = filename.split('_')
            d_stamp = [p for p in parts if p.startswith('d20')][0]
            t_stamp = [p for p in parts if p.startswith('t0') or p.startswith('t1') or p.startswith('t2')][0]
            stamp = f"{d_stamp}_{t_stamp}"
            granules.append((stamp, filename, k))
    return granules

def download_file(s3_key, dest_path):
    """Downloads a file from NOAA AWS S3 if not already present locally."""
    if os.path.exists(dest_path):
        print(f"  [Exists] {os.path.basename(dest_path)} ({os.path.getsize(dest_path):,} bytes)")
        return
    url = f"{AWS_S3_BASE_URL}/{s3_key}"
    print(f"  [Downloading] {url} -> {dest_path}")
    urllib.request.urlretrieve(url, dest_path)
    print(f"  [Downloaded] {os.path.basename(dest_path)} ({os.path.getsize(dest_path):,} bytes)")

def process_granule(m12_path, m15_path, geo_path, output_plot_path=None):
    """Processes downloaded granule files and computes full L1B OPIR evaluation metrics."""
    print("\n" + "=" * 80)
    print("      PROCESSING NOAA VIIRS SATELLITE GRANULE")
    print("=" * 80)

    # 1. Load MWIR Band M12 (3.7 µm)
    print(f"\n[1] Loading MWIR Band M12: {os.path.basename(m12_path)}")
    with h5py.File(m12_path, 'r') as f_m12:
        sdr12 = f_m12['All_Data/VIIRS-M12-SDR_All']
        q_raw_m12 = sdr12['Radiance'][:]
        rad_factors_m12 = sdr12['RadianceFactors'][:]
        bt_counts_m12 = sdr12['BrightnessTemperature'][:]
        bt_factors_m12 = sdr12['BrightnessTemperatureFactors'][:]

    valid_mask_m12 = (q_raw_m12 < 65000) & (bt_counts_m12 < 65000)

    # Radiometric Calibration: L_lambda = Scale * q_raw + Offset
    rad_m12 = np.zeros_like(q_raw_m12, dtype=np.float32)
    rad_m12[valid_mask_m12] = q_raw_m12[valid_mask_m12] * rad_factors_m12[0] + rad_factors_m12[1]

    # NOAA Official GT Brightness Temperature (°C)
    bt_m12_official = np.zeros_like(bt_counts_m12, dtype=np.float32)
    bt_m12_official[valid_mask_m12] = bt_counts_m12[valid_mask_m12] * bt_factors_m12[0] + bt_factors_m12[1]
    t_gt_c = bt_m12_official[valid_mask_m12] - 273.15

    # Inversion Methods
    t_mono_c = planck_inversion(rad_m12[valid_mask_m12], LAMBDA_M12_MONO) - 273.15
    t_srf_c = noaa_srf_planck_correction(rad_m12[valid_mask_m12]) - 273.15

    # Similarity Evaluation
    mae_mono = np.mean(np.abs(t_mono_c - t_gt_c))
    r_mono = np.corrcoef(t_mono_c, t_gt_c)[0, 1]

    diff_srf = np.abs(t_srf_c - t_gt_c)
    mae_srf = np.mean(diff_srf)
    r_srf = np.corrcoef(t_srf_c, t_gt_c)[0, 1]
    rmse_srf = np.sqrt(np.mean((t_srf_c - t_gt_c) ** 2))
    max_ae_srf = np.max(diff_srf)

    print(f"  Valid Scene Pixels: {len(t_gt_c):,} ({np.mean(valid_mask_m12)*100:.2f}%)")
    print(f"  Radiance Range: {rad_m12[valid_mask_m12].min():.4f} to {rad_m12[valid_mask_m12].max():.4f} W/(m^2 sr µm)")
    print(f"  Temperature Range: {t_gt_c.min():.2f}°C to {t_gt_c.max():.2f}°C")
    print("\n[2] Quantitative Evaluation Results (Formed L1B vs NOAA Ground Truth):")
    print(f"  [Method A: Monochromatic Planck]  MAE: {mae_mono:.6f} °C, r: {r_mono:.10f}")
    print(f"  [Method B: NOAA SRF Bandpass]     MAE: {mae_srf:.6f} °C, r: {r_srf:.12f}, RMSE: {rmse_srf:.6f} °C, MaxAE: {max_ae_srf:.6f} °C")
    print(f"  ===> Error Improvement: {mae_mono / mae_srf:.1f}x error reduction using NOAA SRF Method!")

    # 3. Load LWIR Band M15 (10.76 µm)
    print(f"\n[3] Loading LWIR Band M15: {os.path.basename(m15_path)}")
    with h5py.File(m15_path, 'r') as f_m15:
        sdr15 = f_m15['All_Data/VIIRS-M15-SDR_All']
        bt_counts_m15 = sdr15['BrightnessTemperature'][:]
        bt_factors_m15 = sdr15['BrightnessTemperatureFactors'][:]

    valid_mask_m15 = (bt_counts_m15 < 65000)
    bt_m15_official = np.zeros_like(bt_counts_m15, dtype=np.float32)
    bt_m15_official[valid_mask_m15] = bt_counts_m15[valid_mask_m15] * bt_factors_m15[0] + bt_factors_m15[1]
    t_m15_c = np.full_like(rad_m12, np.nan, dtype=np.float32)
    t_m15_c[valid_mask_m15] = bt_m15_official[valid_mask_m15] - 273.15

    # 4. Geolocation Integration
    print(f"\n[4] Integrating Geolocation GMTCO: {os.path.basename(geo_path)}")
    with h5py.File(geo_path, 'r') as f_geo:
        geo = f_geo['All_Data/VIIRS-MOD-GEO-TC_All']
        lats = geo['Latitude'][:]
        lons = geo['Longitude'][:]
        vza = geo['SatelliteZenithAngle'][:]
        sza = geo['SolarZenithAngle'][:]
        height = geo['Height'][:]

    geo_valid = (lats > -90) & (lats < 90) & (lons > -180) & (lons < 180)
    vza_deg = vza[geo_valid].astype(np.float32)
    along_track_gsd = 0.751 / np.maximum(np.cos(np.radians(vza_deg)), 0.1)
    cross_track_gsd = 0.751 / np.maximum((np.cos(np.radians(vza_deg))) ** 2, 0.1)
    pixel_area_km2 = along_track_gsd * cross_track_gsd

    print(f"  Geodetic Bounds: Lat {lats[geo_valid].min():.4f}° to {lats[geo_valid].max():.4f}°, Lon {lons[geo_valid].min():.4f}° to {lons[geo_valid].max():.4f}°")
    print(f"  Terrain Height: {height[geo_valid].min():.1f} m to {height[geo_valid].max():.1f} m")
    print(f"  Sensor Zenith Angle (VZA): {vza_deg.min():.2f}° (Nadir) to {vza_deg.max():.2f}° (Limb)")
    print(f"  Nadir Pixel Area: {pixel_area_km2.min():.3f} km²  |  Swath Edge Area: {pixel_area_km2.max():.3f} km²")
    print(f"  Swath Bowtie Area Expansion Factor: {pixel_area_km2.max() / pixel_area_km2.min():.2f}x")

    # 5. Dual-Band Active Fire Detection (MWIR 3.7 µm - LWIR 10.76 µm > 8.0°C)
    print("\n[5] Executing Dual-Band Thermal Anomaly Fire Detection...")
    t_m12_full = np.full_like(rad_m12, np.nan, dtype=np.float32)
    t_m12_full[valid_mask_m12] = t_srf_c
    delta_t = t_m12_full - t_m15_c

    fire_mask = (delta_t > 8.0) & (t_m12_full > 10.0) & (~np.isnan(delta_t))
    fire_rows, fire_cols = np.where(fire_mask)
    num_fires = len(fire_rows)
    print(f"  Detected {num_fires} active fire hotspot pixels (ΔT > 8.0°C)")

    if num_fires > 0:
        print("  Top Fire Hotspots Detected:")
        for idx in range(min(5, num_fires)):
            r_idx, c_idx = fire_rows[idx], fire_cols[idx]
            lat_val = lats[r_idx, c_idx]
            lon_val = lons[r_idx, c_idx]
            dt_val = delta_t[r_idx, c_idx]
            t12_val = t_m12_full[r_idx, c_idx]
            print(f"    - Hotspot #{idx+1}: Position ({r_idx}, {c_idx}) -> Lat: {lat_val:.4f}°, Lon: {lon_val:.4f}°, MWIR: {t12_val:.2f}°C, ΔT: {dt_val:.2f}°C")

    # 6. Generate Plot Visualization
    if output_plot_path:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)

        # 6a. Formed MWIR Temperature Map
        im0 = axes[0, 0].imshow(np.nan_to_num(t_m12_full, nan=np.nanmin(t_srf_c)), cmap='inferno')
        axes[0, 0].set_title("Formed L1B MWIR Brightness Temp (°C)\n(NOAA SRF Bandpass Method)", fontsize=11, fontweight='bold')
        fig.colorbar(im0, ax=axes[0, 0], label="Temperature (°C)")

        # 6b. Error Map (Method A vs Method B)
        err_map = np.zeros_like(rad_m12)
        err_map[valid_mask_m12] = diff_srf
        im1 = axes[0, 1].imshow(err_map, cmap='magma', vmin=0, vmax=0.01)
        axes[0, 1].set_title("Parity Absolute Error Map (°C)\n(Our Formed L1B vs NOAA GT)", fontsize=11, fontweight='bold')
        fig.colorbar(im1, ax=axes[0, 1], label="Absolute Error (°C)")

        # 6c. Sensor Zenith Angle (VZA) Map
        im2 = axes[1, 0].imshow(vza, cmap='viridis')
        axes[1, 0].set_title("Satellite Zenith Angle (VZA) Map\n(0° Nadir -> 70° Limb Edge)", fontsize=11, fontweight='bold')
        fig.colorbar(im2, ax=axes[1, 0], label="Zenith Angle (deg)")

        # 6d. Dual-Band Thermal Anomaly Fire Detections
        dt_plot = np.nan_to_num(delta_t, nan=0.0)
        im3 = axes[1, 1].imshow(dt_plot, cmap='plasma', vmin=0, vmax=15)
        if num_fires > 0:
            axes[1, 1].scatter(fire_cols, fire_rows, color='cyan', marker='x', s=35, label=f'Fire Hotspots ({num_fires})')
            axes[1, 1].legend(loc='upper right')
        axes[1, 1].set_title("Dual-Band Thermal Fire Anomaly (ΔT MWIR - LWIR)\nCyan Marks = Detected Fire Hotspots", fontsize=11, fontweight='bold')
        fig.colorbar(im3, ax=axes[1, 1], label="ΔT (°C)")

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
        plt.savefig(output_plot_path, bbox_inches='tight')
        plt.close()
        print(f"\n[Plot Saved] {output_plot_path}")

    return {
        'pixels': len(t_gt_c),
        'mae_mono': mae_mono,
        'mae_srf': mae_srf,
        'r_srf': r_srf,
        'rmse_srf': rmse_srf,
        'num_fires': num_fires
    }

def main():
    parser = argparse.ArgumentParser(description="Download and process NOAA VIIRS satellite granules from AWS S3.")
    parser.add_argument("--date", type=str, default="2026-07-29", help="Date in YYYY-MM-DD format (default: 2026-07-29)")
    parser.add_argument("--stamp", type=str, default=None, help="Granule stamp identifier (e.g. d20260729_t0001036)")
    parser.add_argument("--latest", action="store_true", help="Process the latest available granule for the date")
    parser.add_argument("--list", action="store_true", help="List available granules for the date and exit")
    parser.add_argument("--output_dir", type=str, default=None, help="Target local directory for downloaded files")
    
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(args.output_dir, exist_ok=True)

    granules = list_granules_for_date(args.date)
    if not granules:
        print(f"No VIIRS M12 granules found for date {args.date}.")
        sys.exit(1)

    print(f"Found {len(granules)} granules for date {args.date}:")
    for idx, (stamp, filename, key) in enumerate(granules[:10]):
        print(f"  [{idx+1:02d}] Stamp: {stamp}  |  File: {filename}")
    if len(granules) > 10:
        print(f"  ... and {len(granules)-10} more granules.")

    if args.list:
        sys.exit(0)

    selected_stamp = args.stamp
    if args.latest or selected_stamp is None:
        selected_stamp = granules[0][0]
        print(f"\nSelected granule stamp: {selected_stamp}")

    # Search for matching M12, M15, and GMTCO files
    year, month, day = args.date.split('-')
    date_path = f"{year}/{month}/{day}"

    m12_key = None
    m15_key = None
    geo_key = None

    print(f"\nSearching matching S3 keys for stamp '{selected_stamp}'...")
    m12_keys = fetch_s3_keys(f"VIIRS-M12-SDR/{date_path}/", max_keys=200)
    for k in m12_keys:
        if selected_stamp in k:
            m12_key = k
            break

    m15_keys = fetch_s3_keys(f"VIIRS-M15-SDR/{date_path}/", max_keys=200)
    for k in m15_keys:
        if selected_stamp in k:
            m15_key = k
            break

    geo_keys = fetch_s3_keys(f"VIIRS-MOD-GEO-TC/{date_path}/", max_keys=200)
    for k in geo_keys:
        if selected_stamp in k:
            geo_key = k
            break

    print(f"  M12 Key: {m12_key}")
    print(f"  M15 Key: {m15_key}")
    print(f"  GEO Key: {geo_key}")

    if not m12_key or not m15_key or not geo_key:
        print("Error: Could not locate complete matching dataset (M12, M15, GMTCO) on S3.")
        sys.exit(1)

    m12_dest = os.path.join(args.output_dir, os.path.basename(m12_key))
    m15_dest = os.path.join(args.output_dir, os.path.basename(m15_key))
    geo_dest = os.path.join(args.output_dir, os.path.basename(geo_key))

    print("\nDownloading satellite granule files from NOAA AWS S3...")
    download_file(m12_key, m12_dest)
    download_file(m15_key, m15_dest)
    download_file(geo_key, geo_dest)

    plot_dest = os.path.join(args.output_dir, f"noaa_viirs_granule_{selected_stamp}.png")
    results = process_granule(m12_dest, m15_dest, geo_dest, output_plot_path=plot_dest)
    print("\nDone! Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
