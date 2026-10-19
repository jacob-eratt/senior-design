"""Architectural parameters of the accelerator.

The defaults are the baseline design point described in resnet8/SPEC.md.
Every structural size in the hardware model and every field width in the
memory-image export derives from this one object.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AccelConfig:
    # Compute array: lanes x lanes multipliers. One activation word holds
    # `lanes` channels of one pixel; one weight word holds a lanes x lanes
    # (input-channel x output-channel) block of one kernel tap.
    lanes: int = 16

    # Datapath widths.
    act_bits: int = 8         # unsigned activations, zero point 0
    weight_bits: int = 8      # signed symmetric weights
    acc_bits: int = 32        # signed accumulators (wrap on overflow, like RTL)
    bias_bits: int = 32       # signed biases, added in the PPU
    mult_bits: int = 16       # unsigned requantisation multiplier
    shift_bits: int = 6       # requantisation right-shift amount

    # On-chip memories (depth in words).
    num_act_banks: int = 3
    act_bank_depth: int = 1024
    weight_depth: int = 512
    param_depth: int = 64
    instr_depth: int = 32
    out_regs: int = 64        # int32 logit registers read by the host

    # Extra pipeline registers beyond the baseline 7-stage CONV pipeline, for
    # matching an RTL that pipelines the adder tree (ADD -> ACC) or the PPU
    # multipliers (PPU_MAC -> PPU_WB) more deeply. They only add drain cycles.
    mac_extra_latency: int = 0
    ppu_extra_latency: int = 0

    clock_mhz: float = 200.0

    def __post_init__(self):
        if self.lanes < 1:
            raise ValueError("lanes must be positive")
        # The PPU's widest intermediate is (acc + bias) * m + operand * m_res;
        # the Python models evaluate it in int64.
        if self.acc_bits + self.mult_bits + 2 > 63:
            raise ValueError("acc_bits + mult_bits too wide for the int64 model")
        if self.mac_extra_latency < 0 or self.ppu_extra_latency < 0:
            raise ValueError("extra latencies must be >= 0")

    @property
    def conv_drain(self):
        """Cycles from a CONV layer's last issue to the next descriptor fetch."""
        return 7 + self.mac_extra_latency + self.ppu_extra_latency

    @property
    def gap_drain(self):
        return 3

    @property
    def act_max(self):
        return (1 << self.act_bits) - 1

    @property
    def weight_max(self):
        return (1 << (self.weight_bits - 1)) - 1

    @property
    def mult_max(self):
        return (1 << self.mult_bits) - 1

    @property
    def shift_max(self):
        return (1 << self.shift_bits) - 1

    @property
    def act_word_bits(self):
        return self.lanes * self.act_bits

    @property
    def weight_word_bits(self):
        return self.lanes * self.lanes * self.weight_bits

    @property
    def param_lane_bits(self):
        # bias_main, bias_res, m_main, m_res, shift (see compiler.pack_param_word)
        return 2 * self.bias_bits + 2 * self.mult_bits + 8

    @property
    def param_word_bits(self):
        return self.lanes * self.param_lane_bits
