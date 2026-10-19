"""Descriptor encoding, SRAM word packing and memory-file formats."""

import numpy as np
import pytest

from resnet8.accel import isa
from resnet8.accel.compiler import pack_activation, pack_weights, unpack_activation
from resnet8.accel.config import AccelConfig
from resnet8.accel.memfile import (
    pack_param_word, pack_word, read_hex, unpack_param_word, unpack_word, write_hex,
)


def test_descriptor_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        values = {name: int(rng.integers(0, 1 << bits)) for name, bits in isa.FIELDS}
        desc = isa.Descriptor(**values)
        word = desc.encode()
        assert word < 1 << isa.WORD_BITS
        assert isa.Descriptor.decode(word) == desc


def test_descriptor_field_overflow_raises():
    with pytest.raises(ValueError):
        isa.Descriptor(in_h=256).encode()


def test_field_slices_are_contiguous():
    slices = isa.field_slices()
    assert slices[0][2] == 0
    for (_, msb, _), (_, _, lsb) in zip(slices, slices[1:]):
        assert lsb == msb + 1
    assert slices[-1][1] < isa.WORD_BITS


@pytest.mark.parametrize("lanes", [8, 16])
def test_activation_pack_round_trip_and_layout(lanes):
    rng = np.random.default_rng(lanes)
    t = rng.integers(0, 256, size=(37, 5, 7)).astype(np.uint8)
    words = pack_activation(t, lanes)
    ct = -(-37 // lanes)
    assert words.shape == (5 * 7 * ct, lanes)
    assert np.array_equal(unpack_activation(words, t.shape, lanes), t)
    # Spot-check the documented address formula.
    for c, y, x in [(0, 0, 0), (36, 4, 6), (lanes, 2, 3)]:
        assert words[(y * 7 + x) * ct + c // lanes, c % lanes] == t[c, y, x]
    # Lanes beyond the channel count are zero.
    assert np.all(words[:, 37 % lanes:][ct - 1::ct] == 0)


def test_weight_pack_layout():
    lanes = 8
    rng = np.random.default_rng(3)
    w = rng.integers(-127, 128, size=(19, 11, 3, 3)).astype(np.int8)
    words = pack_weights(w, lanes)
    it, ot = 2, 3
    assert words.shape == (ot * 9 * it, lanes, lanes)
    for o, c, ky, kx in [(0, 0, 0, 0), (18, 10, 2, 2), (9, 8, 1, 0), (17, 3, 0, 2)]:
        addr = ((o // lanes * 3 + ky) * 3 + kx) * it + c // lanes
        assert words[addr, c % lanes, o % lanes] == w[o, c, ky, kx]
    assert not words[it - 1::it, 11 % lanes:, :].any()       # padded input-channel lanes are zero


def test_word_pack_round_trip():
    rng = np.random.default_rng(4)
    v = rng.integers(-128, 128, size=256)
    assert np.array_equal(unpack_word(pack_word(v, 8), 256, 8, signed=True), v)
    u = rng.integers(0, 256, size=16)
    assert np.array_equal(unpack_word(pack_word(u, 8), 16, 8, signed=False), u)
    assert pack_word([1, 0xFF], 8) == 0xFF01                 # element 0 in the LSBs


def test_param_word_round_trip():
    cfg = AccelConfig()
    rng = np.random.default_rng(5)
    word = np.stack([
        rng.integers(-2**31, 2**31, size=cfg.lanes), rng.integers(-2**31, 2**31, size=cfg.lanes),
        rng.integers(0, 2**16, size=cfg.lanes), rng.integers(0, 2**16, size=cfg.lanes),
        rng.integers(1, 64, size=cfg.lanes),
    ])
    value = pack_param_word(word, cfg)
    assert value < 1 << cfg.param_word_bits
    assert np.array_equal(unpack_param_word(value, cfg), word)


def test_hex_file_round_trip(tmp_path):
    words = [0, 1, (1 << 127) | 5]
    write_hex(tmp_path / "x.hex", words, 128)
    lines = (tmp_path / "x.hex").read_text().splitlines()
    assert all(len(line) == 32 for line in lines)
    assert read_hex(tmp_path / "x.hex") == words
