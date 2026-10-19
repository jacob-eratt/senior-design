"""Datapath units. Each class corresponds to one RTL module.

The arithmetic here is written out with the fixedpoint primitives rather than
calling fixedpoint.ppu_requantize, so the cycle model and the functional
reference are two separate implementations of the same specification.
"""

import numpy as np

from .. import isa
from ..fixedpoint import rounding_shift_right, wrap


class MacArray:
    """lanes x lanes uint8 x int8 multipliers + one adder tree per output lane.

    Stage MUL:  p[c][k] = act[c] * w[c][k]           (17-bit signed products)
    Stage ADD:  psum[k] = sum_c p[c][k]              (log2(lanes)-level tree)
    """

    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def multiply(act, weights):
        return act.astype(np.int32)[:, None] * weights.astype(np.int32)

    @staticmethod
    def reduce(products):
        return products.sum(axis=0, dtype=np.int64)


class Accumulators:
    """Two banks of `lanes` acc_bits registers: bank 0 main kernel, bank 1 shortcut.

    `clear` loads the incoming partial sum instead of adding (no reset cycle).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.regs = np.zeros((2, cfg.lanes), dtype=np.int64)

    def update(self, bank, psum, clear):
        value = psum if clear else self.regs[bank] + psum
        self.regs[bank] = wrap(value, self.cfg.acc_bits)

    def snapshot(self):
        return self.regs[0].copy(), self.regs[1].copy()


class PostProcessingUnit:
    """Bias, requantisation, residual add, ReLU/saturation (3 pipeline stages).

    PPU_RD   read the param word (and the identity-residual word)
    PPU_MAC  t = wrap(acc + bias) * m_main + operand * m_res      (64-bit)
             operand = residual u8           (RES_IDENTITY)
                     = wrap(acc_res + bias_res)  (RES_PROJECTION)
             S32 mode: t = wrap(acc + bias)   (bypass)
    PPU_WB   y = clamp((t + 2^(shift-1)) >> shift, 0, 255) -> activation bank
             S32 mode: write t to the logit registers
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def multiply_add(self, desc, acc_main, acc_res, params, residual):
        bias_main, bias_res, m_main, m_res, shift = params
        main = wrap(acc_main + bias_main, self.cfg.acc_bits)
        if desc.out_mode == isa.OUT_S32:
            return main, None
        t = main * m_main
        if desc.res_mode == isa.RES_IDENTITY:
            t = t + residual.astype(np.int64) * m_res
        elif desc.res_mode == isa.RES_PROJECTION:
            t = t + wrap(acc_res + bias_res, self.cfg.acc_bits) * m_res
        return t, shift

    def round_clamp(self, t, shift):
        return np.clip(rounding_shift_right(t, shift), 0, self.cfg.act_max).astype(np.uint8)


class PoolingUnit:
    """Global-average-pool engine: `lanes` running sums, then scale and round."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.sums = np.zeros(cfg.lanes, dtype=np.int64)

    def accumulate(self, act, clear):
        self.sums = act.astype(np.int64) if clear else self.sums + act

    def finish(self, total, desc):
        y = rounding_shift_right(total * desc.pool_m, desc.pool_shift)
        return np.clip(y, 0, self.cfg.act_max).astype(np.uint8)
