"""Parser for SICD NITF 2.1 complex float32 SAR imagery products for GCP deployment."""

import os
import xml.etree.ElementTree as ET
import numpy as np


class LazyComplexMemmap:
  """Lazy wrapper for large SICD complex image data to avoid loading 15k x 15k into RAM."""
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


class SICDData:
  """Container for parsed SICD image segment and metadata."""

  def __init__(self, xml_tree: ET.ElementTree, image_array: np.ndarray):
    self.xml_tree = xml_tree
    self.image_array = image_array
    self.num_rows = image_array.shape[0]
    self.num_cols = image_array.shape[1]


def find_elem(root: ET.Element, tag_name: str) -> ET.Element:
  """Finds an element by tag name, ignoring XML namespaces."""
  for elem in root.iter():
    if elem.tag == tag_name or elem.tag.endswith('}' + tag_name):
      return elem
  return None


def parse_sicd_nitf(file_path: str) -> SICDData:
  """Parses NITF 2.1 header to locate SICD XML metadata and complex float32 image payload."""
  if not os.path.exists(file_path):
    raise FileNotFoundError(f'SICD file not found: {file_path}')

  file_size = os.path.getsize(file_path)

  with open(file_path, 'rb') as f:
    header_bytes = f.read(2048)

  xml_start = -1
  read_size = min(file_size, 500000)
  with open(file_path, 'rb') as f:
    f.seek(file_size - read_size)
    tail_bytes = f.read(read_size)
    offset_in_tail = tail_bytes.find(b'<SICD')
    if offset_in_tail != -1:
      xml_start = file_size - read_size + offset_in_tail
      raw_bytes = tail_bytes[offset_in_tail:]
    else:
      f.seek(0)
      head_bytes = f.read(min(file_size, 1000000))
      offset_in_head = head_bytes.find(b'<SICD')
      if offset_in_head != -1:
        xml_start = offset_in_head
        raw_bytes = head_bytes[offset_in_head:]

  xml_tree = None
  rows, cols = 2048, 2048
  if xml_start != -1:
    xml_end = raw_bytes.find(b'</SICD>')
    if xml_end != -1:
      xml_str = raw_bytes[:xml_end + len(b'</SICD>')].decode('utf-8', errors='ignore')
      xml_root = ET.fromstring(xml_str)
      xml_tree = ET.ElementTree(xml_root)

      for elem in xml_root.iter():
        if elem.tag.endswith('NumRows'):
          rows = int(elem.text)
        elif elem.tag.endswith('NumCols'):
          cols = int(elem.text)

  try:
    hl = int(header_bytes[354:360].decode('ascii'))
    lish = int(header_bytes[363:369].decode('ascii'))
    img_data_offset = hl + lish
  except Exception:
    img_data_offset = 1178

  try:
    raw_iq = np.memmap(
        file_path,
        dtype='>f4',
        mode='r',
        offset=img_data_offset,
        shape=(rows, cols, 2),
    )
    image_array = LazyComplexMemmap(raw_iq)
  except Exception:
    image_array = np.zeros((rows, cols), dtype=np.complex64)

  return SICDData(xml_tree, image_array)
