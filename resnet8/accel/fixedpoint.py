"""Integer arithmetic primitives.

These functions *are* the numeric specification of the datapath: the
cycle-level model and the functional reference both call them, and the RTL
must reproduce them bit for bit. All inputs/outputs are numpy integer arrays
(int64 internally so that no intermediate silently overflows).
"""

import math

import numpy as np


def wrap(x, bits):
    """Two's-complement wrap to a signed `bits`-wide register."""
    x = np.asarray(x, dtype=np.int64)
    half = np.int64(1) << (bits - 1)
    return ((x + half) & ((half << 1) - 1)) - half


def fits_signed(x, bits):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    x = np.asarray(x)
    return bool(np.all((x >= lo) & (x <= hi)))


def rounding_shift_right(x, shift):
    """floor((x + 2**(shift-1)) / 2**shift): round half towards +infinity.

    This is the RTL's "add half an LSB, then arithmetic shift right".
    `shift` may be a scalar or broadcastable array and must be >= 1.
    """
    x = np.asarray(x, dtype=np.int64)
    shift = np.asarray(shift, dtype=np.int64)
    if np.any(shift < 1):
        raise ValueError("shift must be >= 1")
    return (x + (np.int64(1) << (shift - 1))) >> shift


def quantize_multipliers(reals, mult_bits, shift_bits):
    """Represent non-negative reals as integers sharing one right shift.

    Returns (m, shift) with m[i] / 2**shift ~= reals[i], m[i] < 2**mult_bits,
    choosing the largest shift (best precision) that keeps the largest
    multiplier in range. Sharing the shift lets the PPU add two scaled terms
    (main path and residual path) before a single rounding step.
    """
    reals = np.asarray(reals, dtype=np.float64)
    if np.any(reals < 0) or not np.all(np.isfinite(reals)):
        raise ValueError(f"multipliers must be finite and non-negative: {reals}")
    m_max = (1 << mult_bits) - 1
    shift_max = (1 << shift_bits) - 1
    largest = float(reals.max(initial=0.0))
    if largest == 0.0:
        return np.zeros(reals.shape, dtype=np.int64), 1
    # Largest shift with largest * 2**shift < m_max + 0.5, i.e. it still rounds into range.
    shift = min(shift_max, math.floor(math.log2((m_max + 0.5) / largest)))
    while shift >= 1 and round(largest * 2.0 ** shift) > m_max:
        shift -= 1
    if shift < 1:
        raise ValueError(f"multiplier {largest} too large for {mult_bits}-bit m with shift >= 1")
    m = np.round(reals * 2.0 ** shift).astype(np.int64)
    return m, shift


def ppu_requantize(acc, bias, m, shift, operand=None, m_res=None, *, acc_bits, out_min, out_max):
    """Post-processing unit arithmetic for one output vector (or tensor).

        t = wrap(acc + bias) * m  [+ operand * m_res]
        y = clamp(rounding_shift_right(t, shift), out_min, out_max)

    `operand` is the residual term: a uint8 activation (identity shortcut)
    or the shortcut convolution's wrap(acc_sc + bias_sc) (projection).
    ReLU is the lower clamp at 0, because activations have zero point 0.
    """
    t = wrap(np.asarray(acc, np.int64) + bias, acc_bits) * np.asarray(m, np.int64)
    if operand is not None:
        t = t + np.asarray(operand, np.int64) * np.asarray(m_res, np.int64)
    return np.clip(rounding_shift_right(t, shift), out_min, out_max)


def bits_for_signed(max_abs):
    """Smallest two's-complement width holding [-max_abs, max_abs]."""
    return int(max_abs).bit_length() + 1
