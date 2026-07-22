"""Parser for CPHD 1.1 (Compensated Phase History Data) format for GCP deployment.

Extracts XML header metadata, PVP (Per-Vector Parameters) array, and signal memmap.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np


class CPHDData:
  """Container for parsed CPHD phase history signal array and PVP metadata."""

  def __init__(self, xml_tree: ET.ElementTree, pvp_array: np.ndarray, signal_memmap: np.ndarray):
    self.xml_tree = xml_tree
    self.pvp_array = pvp_array
    self.signal_memmap = signal_memmap
    self.num_vectors = signal_memmap.shape[0]
    self.num_samples = signal_memmap.shape[1]


def parse_cphd_header(header_str: str) -> dict[str, str]:
  """Parses CPHD 1.1 text header key-value pairs."""
  kv_pairs = {}
  for line in header_str.splitlines():
    line = line.strip()
    if line and ':=' in line:
      k, v = line.split(':=', 1)
      kv_pairs[k.strip()] = v.strip()
  return kv_pairs


def load_cphd(file_path: str) -> CPHDData:
  """Parses CPHD 1.1 file header, XML metadata block, PVP array, and signal memmap."""
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"CPHD file not found: {file_path}")

  file_size = os.path.getsize(file_path)

  with open(file_path, "rb") as f:
    header_raw = f.read(2048).decode("ascii", errors="ignore")

  header_info = parse_cphd_header(header_raw)
  xml_byte_size = int(header_info.get("XML_BLOCK_SIZE", header_info.get("XML_DATA_SIZE", 0)))

  xml_start = -1
  xml_size = 0
  with open(file_path, "rb") as f:
    f.seek(0)
    buf = f.read(min(file_size, 500000))
    start_idx = buf.find(b"<CPHD")
    if start_idx != -1:
      xml_start = start_idx
      end_idx = buf.find(b"</CPHD>")
      if end_idx != -1:
        xml_size = (end_idx + len(b"</CPHD>")) - start_idx

  if xml_start == -1 or xml_size == 0:
    raise ValueError("Failed to locate <CPHD> XML block in header.")

  with open(file_path, "rb") as f:
    f.seek(xml_start)
    xml_bytes = f.read(xml_size)

  xml_root = ET.fromstring(xml_bytes.decode("utf-8", errors="ignore"))
  xml_tree = ET.ElementTree(xml_root)

  def find_val(root, tag_name, default=0):
    for elem in root.iter():
      if elem.tag == tag_name or elem.tag.endswith("}" + tag_name):
        return elem.text
    return default

  def find_path_val(root, path_tags, default="0"):
    curr_elems = [root]
    for tag in path_tags:
      next_elems = []
      for elem in curr_elems:
        for child in elem:
          if child.tag == tag or child.tag.endswith("}" + tag):
            next_elems.append(child)
      curr_elems = next_elems
      if not curr_elems:
        return default
    return curr_elems[0].text if curr_elems else default

  num_vectors = int(find_path_val(xml_root, ["Data", "Channel", "NumVectors"], "0"))
  num_samples = int(find_path_val(xml_root, ["Data", "Channel", "NumSamples"], "0"))

  pvp_block_offset = int(header_info.get("PVP_BLOCK_BYTE_OFFSET", header_info.get("PVP_DATA_OFFSET", xml_start + xml_size)))
  signal_block_offset = int(header_info.get("SIGNAL_BLOCK_BYTE_OFFSET", header_info.get("SIGNAL_DATA_OFFSET", pvp_block_offset + 100000)))

  num_bytes_pvp = int(find_val(xml_root, "NumBytesPVP", 0))
  if num_bytes_pvp == 240:
    pvp_dtype = np.dtype({
        "names": [
            "TxTime",
            "TxPos_X", "TxPos_Y", "TxPos_Z",
            "RcvPos_X", "RcvPos_Y", "RcvPos_Z",
            "SRPPos_X", "SRPPos_Y", "SRPPos_Z",
            "SC0", "SCSS", "TDTropoSRP"
        ],
        "formats": [
            ">f8",
            ">f8", ">f8", ">f8",
            ">f8", ">f8", ">f8",
            ">f8", ">f8", ">f8",
            ">f8", ">f8", ">f8"
        ],
        "offsets": [
            0,             # TxTime (offset 0)
            16, 24, 32,    # TxPos (offset 2)
            40, 48, 56,    # RcvPos (offset 5)
            112, 120, 128,  # SRPPos (offset 14)
            208,           # SC0 (offset 26)
            216,           # SCSS (offset 27)
            224            # TDTropoSRP (offset 28)
        ],
        "itemsize": 240
    })
  else:
    pvp_dtype = np.dtype([
        ("TxPos_X", ">f8"),
        ("TxPos_Y", ">f8"),
        ("TxPos_Z", ">f8"),
        ("RcvPos_X", ">f8"),
        ("RcvPos_Y", ">f8"),
        ("RcvPos_Z", ">f8"),
        ("SRPPos_X", ">f8"),
        ("SRPPos_Y", ">f8"),
        ("SRPPos_Z", ">f8"),
        ("SC0", ">f8"),
        ("SCSS", ">f8"),
        ("TxTime", ">f8"),
        ("TOA1", ">f8"),
        ("TOA2", ">f8"),
        ("TDTropoSRP", ">f8"),
    ])

  try:
    pvp_array = np.memmap(
        file_path,
        dtype=pvp_dtype,
        mode="r",
        offset=pvp_block_offset,
        shape=(num_vectors,),
    )
  except Exception as e:
    print(f"Error loading PVP array: {e}")
    pvp_array = np.zeros(num_vectors, dtype=pvp_dtype)

  try:
    raw_iq = np.memmap(
        file_path,
        dtype=">f4",
        mode="r",
        offset=signal_block_offset,
        shape=(num_vectors, num_samples, 2),
    )
    real_part = np.nan_to_num(raw_iq[..., 0].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    imag_part = np.nan_to_num(raw_iq[..., 1].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    signal_memmap = real_part + 1j * imag_part
  except Exception as e:
    print(f"Error loading signal memmap: {e}")
    signal_memmap = np.zeros((num_vectors, num_samples), dtype=np.complex64)

  return CPHDData(xml_tree, pvp_array, signal_memmap)
