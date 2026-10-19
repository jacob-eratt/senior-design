"""The integer primitives against exact rational arithmetic."""

import math
from fractions import Fraction

import numpy as np
import pytest

from resnet8.accel.fixedpoint import (
    bits_for_signed, fits_signed, ppu_requantize, quantize_multipliers, rounding_shift_right, wrap,
)


def exact_round_half_up(num, den):
    return math.floor(Fraction(num, den) + Fraction(1, 2))


def test_rounding_shift_right_matches_exact_rounding():
    rng = np.random.default_rng(0)
    x = rng.integers(-2**40, 2**40, size=2000)
    shift = rng.integers(1, 40, size=2000)
    got = rounding_shift_right(x, shift)
    want = [exact_round_half_up(int(a), 2 ** int(s)) for a, s in zip(x, shift)]
    assert got.tolist() == want


def test_rounding_shift_right_ties_round_up():
    assert rounding_shift_right(np.array([1, 3, -1, -3, 2, -2]), 1).tolist() == [1, 2, 0, -1, 1, -1]


def test_rounding_shift_right_rejects_zero_shift():
    with pytest.raises(ValueError):
        rounding_shift_right(5, 0)


def test_wrap_is_twos_complement():
    assert wrap(np.array([127, 128, 255, 256, -128, -129]), 8).tolist() == [127, -128, -1, 0, -128, 127]
    big = np.array([2**31 - 1, 2**31, -2**31 - 1])
    assert wrap(big, 32).tolist() == [2**31 - 1, -2**31, 2**31 - 1]


def test_width_helpers():
    assert bits_for_signed(127) == 8 and bits_for_signed(128) == 9 and bits_for_signed(0) == 1
    assert fits_signed([-128, 127], 8) and not fits_signed([128], 8)


@pytest.mark.parametrize("mult_bits", [8, 16, 24])
def test_quantize_multipliers_precision_and_range(mult_bits):
    rng = np.random.default_rng(mult_bits)
    for _ in range(200):
        reals = rng.uniform(1e-5, 2.0, size=2) * 10.0 ** rng.integers(-3, 1)
        m, shift = quantize_multipliers(reals, mult_bits, 6)
        assert 1 <= shift <= 63
        assert np.all(m <= 2 ** mult_bits - 1)
        # Each multiplier is the nearest integer at the shared shift.
        assert np.all(np.abs(m / 2.0 ** shift - reals) <= 0.5 / 2.0 ** shift + 1e-18)
        # The largest one is normalised: shift could not have been one larger
        # (unless capped by the shift field width).
        if shift < 63:
            assert round(reals.max() * 2.0 ** (shift + 1)) > 2 ** mult_bits - 1


def test_quantize_multipliers_edge_cases():
    m, shift = quantize_multipliers([0.0, 0.0], 16, 6)
    assert m.tolist() == [0, 0] and shift == 1
    with pytest.raises(ValueError):
        quantize_multipliers([-1.0], 16, 6)
    with pytest.raises(ValueError):
        quantize_multipliers([1e6], 16, 6)          # would need shift < 1


def test_ppu_requantize_matches_exact_formula():
    rng = np.random.default_rng(1)
    n = 3000
    acc = rng.integers(-2**24, 2**24, size=n)
    bias = rng.integers(-2**20, 2**20, size=n)
    m = rng.integers(0, 2**16, size=n)
    op = rng.integers(-2**24, 2**24, size=n)
    m_res = rng.integers(0, 2**16, size=n)
    shift = rng.integers(1, 40, size=n)
    got = ppu_requantize(acc, bias, m, shift, op, m_res, acc_bits=32, out_min=0, out_max=255)
    for i in range(n):
        t = (int(acc[i]) + int(bias[i])) * int(m[i]) + int(op[i]) * int(m_res[i])
        want = min(255, max(0, exact_round_half_up(t, 2 ** int(shift[i]))))
        assert got[i] == want


def test_ppu_requantize_wraps_acc_plus_bias():
    # An accumulator overflow behaves like the RTL's 32-bit register, not like Python ints.
    got = ppu_requantize(np.array([2**31 - 1]), np.array([1]), np.array([1]), np.array([1]),
                         acc_bits=32, out_min=-2**40, out_max=2**40)
    assert got[0] == rounding_shift_right(-2**31, 1)
