"""Bit-level packing of SRAM words into Verilog $readmemh files.

A word is a vector of fixed-width elements; element i occupies bits
[i*bits +: bits] (element 0 in the LSBs), signed values in two's complement.
Each line of a .hex file is one word, most significant hex digit first.

    activation word  L x u8                       element c   = lane c
    weight word      L*L x s8                     element c*L+k = w[c][k]
    param word       L x {shift:8, m_res:M, m_main:M, bias_res:B, bias_main:B}
                     lane k at [k*P +: P], bias_main in the lane's LSBs
"""

import numpy as np


def pack_word(values, bits):
    word = 0
    mask = (1 << bits) - 1
    for i, v in enumerate(np.asarray(values).ravel().tolist()):
        word |= (int(v) & mask) << (i * bits)
    return word


def unpack_word(word, count, bits, signed):
    out = []
    for i in range(count):
        v = (word >> (i * bits)) & ((1 << bits) - 1)
        if signed and v >= 1 << (bits - 1):
            v -= 1 << bits
        out.append(v)
    return np.array(out, dtype=np.int64)


def param_lane_fields(cfg):
    """[(row index in param_mem, bits, signed)] from LSB to MSB within a lane."""
    return [(0, cfg.bias_bits, True), (1, cfg.bias_bits, True),
            (2, cfg.mult_bits, False), (3, cfg.mult_bits, False), (4, 8, False)]


def pack_param_word(word, cfg):
    """param_mem word [5, L] -> integer."""
    lane_bits = cfg.param_lane_bits
    total, pos = 0, 0
    for k in range(cfg.lanes):
        lane, offset = 0, 0
        for row, bits, _ in param_lane_fields(cfg):
            lane |= (int(word[row, k]) & ((1 << bits) - 1)) << offset
            offset += bits
        total |= lane << pos
        pos += lane_bits
    return total


def unpack_param_word(value, cfg):
    word = np.zeros((5, cfg.lanes), dtype=np.int64)
    for k in range(cfg.lanes):
        lane = (value >> (k * cfg.param_lane_bits)) & ((1 << cfg.param_lane_bits) - 1)
        offset = 0
        for row, bits, signed in param_lane_fields(cfg):
            word[row, k] = unpack_word(lane >> offset, 1, bits, signed)[0]
            offset += bits
    return word


def write_hex(path, words, word_bits):
    digits = -(-word_bits // 4)
    with open(path, "w", newline="\n") as f:
        for w in words:
            f.write(f"{w:0{digits}x}\n")


def read_hex(path):
    with open(path) as f:
        return [int(line, 16) for line in f if line.strip() and not line.startswith("//")]
