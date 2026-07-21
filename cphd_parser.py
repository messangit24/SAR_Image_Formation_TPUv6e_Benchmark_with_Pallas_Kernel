"""Parser for CPHD 1.1 (Compensated Phase History Data) format for GCP deployment.

Extracts XML header metadata, PVP (Per-Vector Parameters) array, and signal memmap.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np


def find_elem(root: ET.Element, tag_name: str) -> ET.Element:
  """Finds an element by tag name, ignoring XML namespaces."""
  for elem in root.iter():
    if elem.tag == tag_name or elem.tag.endswith('}' + tag_name):
      return elem
  return None


def build_pvp_dtype(xml_root) -> np.dtype:
  """Dynamically builds numpy dtype for PVP array based on CPHD XML definition."""
  pvp_node = find_elem(xml_root, "PVP")
  if pvp_node is None:
    raise ValueError("CPHD XML missing <PVP> definition block.")

  names = []
  formats = []
  offsets = []

  def find_val(root, tag_name, default=0):
    for elem in root.iter():
      if elem.tag == tag_name or elem.tag.endswith("}" + tag_name):
        return elem.text
    return default

  for child in pvp_node:
    name = child.tag.split("}")[-1]
    offset_elem = find_elem(child, "Offset")
    size_elem = find_elem(child, "Size")
    format_elem = find_elem(child, "Format")

    if offset_elem is None or size_elem is None:
      continue

    offset = int(offset_elem.text)
    size = int(size_elem.text)

    # Map F8 -> float64, I8 -> int64
    # If size is 3, map to _X, _Y, _Z for known vector fields
    if name in ["TxPos", "RcvPos", "SRPPos", "TxVel", "RcvVel"] and size == 3:
      for axis_idx, axis in enumerate(["X", "Y", "Z"]):
        names.append(f"{name}_{axis}")
        formats.append(">f8")
        offsets.append((offset + axis_idx) * 8)
    elif size == 1:
      names.append(name)
      fmt_text = format_elem.text if format_elem is not None else "F8"
      if "I" in fmt_text:
        formats.append(">i8")
      else:
        formats.append(">f8")
      offsets.append(offset * 8)

  num_bytes_pvp = int(find_val(xml_root, "NumBytesPVP", 0))

  return np.dtype({
      "names": names,
      "formats": formats,
      "offsets": offsets,
      "itemsize": num_bytes_pvp,
  })


class LazyComplexMemmap:
  """Lazy wrapper for large CPHD complex signal data to avoid loading 2.6GB into RAM."""
  def __init__(self, raw_iq):
    self.raw_iq = raw_iq
    self.shape = raw_iq.shape[:2]
    self.ndim = 2
    self.dtype = np.complex64

  def __getitem__(self, key):
    sub_raw = self.raw_iq[key]
    real_p = np.nan_to_num(sub_raw[..., 0].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    imag_p = np.nan_to_num(sub_raw[..., 1].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return real_p + 1j * imag_p

  def __array__(self, dtype=None, copy=None):
    real_p = np.nan_to_num(self.raw_iq[..., 0].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    imag_p = np.nan_to_num(self.raw_iq[..., 1].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    arr = real_p + 1j * imag_p
    if dtype:
      return arr.astype(dtype, copy=copy if copy is not None else True)
    return arr


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

  pvp_dtype = build_pvp_dtype(xml_root)

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
    signal_memmap = LazyComplexMemmap(raw_iq)
  except Exception as e:
    print(f"Error loading signal memmap: {e}")
    signal_memmap = np.zeros((num_vectors, num_samples), dtype=np.complex64)

  return CPHDData(xml_tree, pvp_array, signal_memmap)
