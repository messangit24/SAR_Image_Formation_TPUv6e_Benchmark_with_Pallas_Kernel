"""Unit tests for MODIS L0 packet decommutation and 12-bit sample extraction."""
import numpy as np

from src.modis.modis_l0_decom import PKT, SCI, _hdr, _words


def test_words_12bit_unpack_accuracy():
    """Verify that vectorized _words unpacks 12-bit big-endian words bit-exactly."""
    # Construct a synthetic 642-byte packet
    pkt = np.zeros((1, PKT), dtype=np.uint8)

    # 3 bytes store two 12-bit words: word0 (12 bits) and word1 (12 bits)
    # Byte 0 = word0[11:4]
    # Byte 1 = (word0[3:0] << 4) | (word1[11:8])
    # Byte 2 = word1[7:0]

    # Test values: word0 = 0xABC (2748), word1 = 0x123 (291)
    byte0 = 0xAB
    byte1 = (0x0C << 4) | 0x01  # 0xC1
    byte2 = 0x23

    pkt[0, SCI] = byte0
    pkt[0, SCI + 1] = byte1
    pkt[0, SCI + 2] = byte2

    words = _words(pkt)
    assert words.shape == (1, 416)
    assert words[0, 0] == 0xABC
    assert words[0, 1] == 0x123


def test_words_boundary_values():
    """Verify 12-bit unpacking for min (0x000) and max (0xFFF) 12-bit limits."""
    pkt = np.zeros((1, PKT), dtype=np.uint8)

    # Pack 0xFFF and 0x000
    pkt[0, SCI] = 0xFF
    pkt[0, SCI + 1] = 0xF0
    pkt[0, SCI + 2] = 0x00

    words = _words(pkt)
    assert words[0, 0] == 4095  # 0xFFF
    assert words[0, 1] == 0     # 0x000


def test_hdr_bitfield_parsing():
    """Verify decoding of telemetry header fields."""
    pkt = np.zeros((1, PKT), dtype=np.uint8)

    # seq: P[2] >> 6 -> set bits 6 & 7 (e.g. 0x80 -> 2)
    pkt[0, 2] = 0x80

    # Earth view packet with earth frame count (efc) = 500
    # is_cal = 0 (P[15] MSB = 0)
    # efc = ((P[15] & 0x7f) << 4) | (P[16] >> 4)
    # 500 = 0x1F4 -> P[15] lower 7 bits = 0x1F, P[16] upper 4 bits = 0x04 (0x40)
    pkt[0, 15] = 0x1F
    pkt[0, 16] = 0x40

    hdr = _hdr(pkt)
    assert hdr["seq"][0] == 2
    assert hdr["is_cal"][0] == 0
    assert hdr["efc"][0] == 500


def test_hdr_cal_sector_decoding():
    """Verify decoding of calibration sector headers (Blackbody & Space View)."""
    pkt = np.zeros((1, PKT), dtype=np.uint8)

    # is_cal = 1 (Cal packet, MSB of P[15] set)
    # cal_type = 2 (Blackbody, bits 5..6 of P[15])
    # cal_fcnt = 25 (0x19 -> P[15] bits 0..1 = 1, P[16] bits 4..7 = 9 -> 0x90)
    pkt[0, 15] = 0x80 | (2 << 5) | 0x01
    pkt[0, 16] = 0x90

    hdr = _hdr(pkt)
    assert hdr["is_cal"][0] == 1
    assert hdr["cal_type"][0] == 2
    assert hdr["cal_fcnt"][0] == 25
