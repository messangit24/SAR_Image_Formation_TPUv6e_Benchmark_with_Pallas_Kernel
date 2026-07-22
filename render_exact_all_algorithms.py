"""Standalone CLI Benchmark Runner for Cloud TPU / GPU / CPU deployment on GCP.

Processes CPHD / SICD satellite SAR data using Time-Domain Backprojection (BPA),
Polar Format (PFA), and Chirp Scaling (CSA), producing a 4-panel comparison image
and quantitative performance matrix.
"""

import argparse
import os
import sys
import time
import numpy as np
import scipy.ndimage
from PIL import Image, ImageDraw, ImageFont

import cphd_parser
import physical_csa
import physical_pfa
import quality_evaluator
import sicd_parser
import tpu_bpa


def parse_args():
  parser = argparse.ArgumentParser(description="GCP TPU/GPU Benchmark Suite for SAR Image Formation")
  parser.add_argument(
      "--sicd_path",
      type=str,
      default="/usr/local/google/home/ermlickb/tpu_sar_data/2023-07-28-01-49-09_UMBRA-05_SICD.nitf",
      help="Path to SICD NITF reference image file.",
  )
  parser.add_argument(
      "--cphd_path",
      type=str,
      default="/usr/local/google/home/ermlickb/tpu_sar_data/2023-07-28-01-49-09_UMBRA-05_CPHD.cphd",
      help="Path to CPHD phase history signal file.",
  )
  parser.add_argument(
      "--grid_size",
      type=int,
      default=512,
      help="Output spatial image resolution (N x N).",
  )
  parser.add_argument(
      "--output_image",
      type=str,
      default="exact_physical_comparison.png",
      help="Output 4-panel PNG comparison filename.",
  )
  parser.add_argument(
      "--center_r",
      type=int,
      default=None,
      help="Override row crop center (default: SCPPixel from XML).",
  )
  parser.add_argument(
      "--center_c",
      type=int,
      default=None,
      help="Override col crop center (default: SCPPixel from XML).",
  )
  return parser.parse_args()


def main():
  args = parse_args()
  grid_shape = (args.grid_size, args.grid_size)

  print("=== GCP TPU Benchmark Suite for Satellite SAR Image Formation ===")
  print(f"Loading SICD reference: {args.sicd_path}")
  sicd = sicd_parser.parse_sicd_nitf(args.sicd_path)
  ref_full = quality_evaluator.clean_array(sicd.image_array)

  print(f"Loading CPHD dataset: {args.cphd_path}")
  cphd = cphd_parser.load_cphd(args.cphd_path)

  pvp = cphd.pvp_array
  tx_pos = np.column_stack((pvp["TxPos_X"], pvp["TxPos_Y"], pvp["TxPos_Z"]))
  rcv_pos = np.column_stack((pvp["RcvPos_X"], pvp["RcvPos_Y"], pvp["RcvPos_Z"]))
  tx_pos = np.where(np.isnan(tx_pos), rcv_pos, tx_pos)
  srp_pos = np.column_stack((pvp["SRPPos_X"], pvp["SRPPos_Y"], pvp["SRPPos_Z"]))
  sc0, scss = pvp["SC0"], pvp["SCSS"]
  tropo_delay = pvp["TDTropoSRP"]

  root = sicd.xml_tree.getroot()
  grid_node = sicd_parser.find_elem(root, "Grid")
  row_node = sicd_parser.find_elem(grid_node, "Row")
  col_node = sicd_parser.find_elem(grid_node, "Col")
  row_ss = float(sicd_parser.find_elem(row_node, "SS").text)
  col_ss = float(sicd_parser.find_elem(col_node, "SS").text)

  ur = sicd_parser.find_elem(row_node, "UVectECF")
  uc = sicd_parser.find_elem(col_node, "UVectECF")
  row_vec = np.array([
      float(sicd_parser.find_elem(ur, "X").text),
      float(sicd_parser.find_elem(ur, "Y").text),
      float(sicd_parser.find_elem(ur, "Z").text),
  ])
  col_vec = np.array([
      float(sicd_parser.find_elem(uc, "X").text),
      float(sicd_parser.find_elem(uc, "Y").text),
      float(sicd_parser.find_elem(uc, "Z").text),
  ])

  signal_data = np.array(cphd.signal_memmap)
  signal_data = np.where(np.isnan(signal_data) | np.isinf(signal_data), 0.0 + 0.0j, signal_data)
  root = sicd.xml_tree.getroot()
  scp_node = sicd_parser.find_elem(root, "SCPPixel")
  default_center_r = int(float(sicd_parser.find_elem(scp_node, "Row").text))
  default_center_c = int(float(sicd_parser.find_elem(scp_node, "Col").text))
  
  center_r = args.center_r if args.center_r is not None else default_center_r
  center_c = args.center_c if args.center_c is not None else default_center_c
  print(f"Using crop center: Row={center_r}, Col={center_c}")
  ref_crop = ref_full[
      center_r - grid_shape[0] // 2 : center_r + grid_shape[0] // 2,
      center_c - grid_shape[1] // 2 : center_c + grid_shape[1] // 2,
  ]

  # Construct 3D ECEF grid for Backprojection
  row_idx = np.arange(grid_shape[0]) - grid_shape[0] // 2
  col_idx = np.arange(grid_shape[1]) - grid_shape[1] // 2
  grid_r, grid_c = np.meshgrid(row_idx, col_idx, indexing="ij")
  
  shift_r = center_r - default_center_r
  shift_c = center_c - default_center_c
  
  grid_r_rel = grid_r + shift_r
  grid_c_rel = grid_c + shift_c
  
  mean_srp = np.mean(srp_pos, axis=0)
  pixel_pts_ecef = (
      mean_srp
      + grid_r_rel[:, :, None] * row_ss * row_vec
      + grid_c_rel[:, :, None] * col_ss * col_vec
  )

  # 1. Backprojection (BPA) Execution
  print("\nExecuting Time-Domain Backprojection (BPA)...")
  t0 = time.time()
  img_bpa = tpu_bpa.process_physical_bpa_tpu(
      signal_data, tx_pos, rcv_pos, srp_pos, sc0, scss, tropo_delay, pixel_pts_ecef
  )
  bpa_lat = (time.time() - t0) * 1000.0
  bpa_clean = quality_evaluator.clean_array(img_bpa)
  
  # Search for best shift to align with reference
  print("Aligning BPA with reference...")
  ref_mag = np.abs(ref_crop)
  ref_norm = ref_mag - np.mean(ref_mag)
  bpa_mag = np.abs(bpa_clean)
  best_corr = -1.0
  best_shift = (0, 0)
  for r_shift in range(-10, 11):
      for c_shift in range(-10, 11):
          shifted = scipy.ndimage.shift(bpa_mag, (r_shift, c_shift), order=3, mode='constant', cval=0.0)
          shifted_norm = shifted - np.mean(shifted)
          denom = np.sqrt(np.sum(ref_norm**2) * np.sum(shifted_norm**2))
          corr = np.sum(ref_norm * shifted_norm) / denom if denom > 0 else 0.0
          if corr > best_corr:
              best_corr = corr
              best_shift = (r_shift, c_shift)
  
  print(f"  Best BPA shift found: {best_shift} with correlation {best_corr:.4f}")
  # Apply best shift to the complex image
  bpa_shifted_real = scipy.ndimage.shift(bpa_clean.real, best_shift, order=3, mode='constant', cval=0.0)
  bpa_shifted_imag = scipy.ndimage.shift(bpa_clean.imag, best_shift, order=3, mode='constant', cval=0.0)
  bpa_clean = bpa_shifted_real + 1j * bpa_shifted_imag
  bpa_corr = best_corr
  
  print(f"  BPA Execution Latency: {bpa_lat:.2f} ms | Shift-Aligned Magnitude Correlation: {bpa_corr:.4f}")
 
  # 2. Polar Format (PFA) Execution
  print("\nExecuting Polar Format Algorithm (PFA)...")
  t0 = time.time()
  img_pfa = physical_pfa.process_physical_pfa(
      signal_data, sc0, scss, tx_pos, rcv_pos, srp_pos, output_shape=grid_shape
  )
  pfa_lat = (time.time() - t0) * 1000.0
  pfa_clean = quality_evaluator.clean_array(img_pfa)
  pfa_corr = quality_evaluator.compute_complex_coherence(ref_crop, pfa_clean)
  print(f"  PFA Execution Latency: {pfa_lat:.2f} ms | Magnitude Correlation: {pfa_corr:.4f}")
 
  # 3. Chirp Scaling (CSA) Execution
  print("\nExecuting Chirp Scaling Algorithm (CSA)...")
  t0 = time.time()
  img_csa = physical_csa.process_physical_csa(signal_data, sc0, scss, output_shape=grid_shape)
  csa_lat = (time.time() - t0) * 1000.0
  csa_clean = quality_evaluator.clean_array(img_csa)
  csa_corr = quality_evaluator.compute_complex_coherence(ref_crop, csa_clean)
  print(f"  CSA Execution Latency: {csa_lat:.2f} ms | Magnitude Correlation: {csa_corr:.4f}")

  # Create 4-Panel Comparison Layout
  u8_gt = quality_evaluator.to_uint8_db(ref_crop)
  u8_bpa = quality_evaluator.to_uint8_db(bpa_clean)
  u8_pfa = quality_evaluator.to_uint8_db(pfa_clean)
  u8_csa = quality_evaluator.to_uint8_db(csa_clean)

  h, w = grid_shape
  canvas = Image.new("L", (w * 4, h))
  canvas.paste(Image.fromarray(u8_gt), (0, 0))
  canvas.paste(Image.fromarray(u8_bpa), (w, 0))
  canvas.paste(Image.fromarray(u8_pfa), (w * 2, 0))
  canvas.paste(Image.fromarray(u8_csa), (w * 3, 0))

  # Draw labels
  draw = ImageDraw.Draw(canvas)
  try:
      font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
  except IOError:
      font = ImageFont.load_default()

  def draw_text_with_shadow(draw, position, text, font, fill=255):
      x, y = position
      draw.text((x+2, y+2), text, fill=0, font=font)
      draw.text((x, y), text, fill=fill, font=font)

  draw_text_with_shadow(draw, (10, 10), "Ground Truth (SICD)", font)
  draw_text_with_shadow(draw, (w + 10, 10), "Backprojection (BPA)", font)
  draw_text_with_shadow(draw, (w * 2 + 10, 10), "Polar Format (PFA)", font)
  draw_text_with_shadow(draw, (w * 3 + 10, 10), "Chirp Scaling (CSA)", font)

  canvas.save(args.output_image)
  print(f"\nSaved Labeled 4-Panel Comparison Image to: {args.output_image}")


if __name__ == "__main__":
  main()
