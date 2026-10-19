"""Layer descriptors: the accelerator's instruction format.

The sequencer executes one descriptor per layer. A descriptor is a 128-bit
word; fields are packed LSB-first in the order of FIELDS, so the RTL decoder is
a fixed set of bit slices (see `field_slices()` / SPEC.md section "ISA").

Tensors always start at address 0 of their activation bank, so descriptors
carry bank numbers rather than activation addresses. Word addressing inside a
bank is (y * width + x) * channel_tiles + channel_tile.
"""

from dataclasses import asdict, dataclass

OP_END, OP_CONV, OP_GAP = 0, 1, 2
RES_NONE, RES_IDENTITY, RES_PROJECTION = 0, 1, 2
OUT_U8, OUT_S32 = 0, 1

WORD_BITS = 128
FIELDS = (
    ("op", 2),            # OP_END / OP_CONV / OP_GAP
    ("in_bank", 2),       # activation bank holding the layer input
    ("out_bank", 2),      # activation bank receiving the output (u8 mode)
    ("res_bank", 2),      # bank holding the shortcut source tensor
    ("res_mode", 2),      # RES_NONE / RES_IDENTITY / RES_PROJECTION
    ("out_mode", 1),      # OUT_U8 (requantise to bank) / OUT_S32 (logit registers)
    ("in_h", 8), ("in_w", 8), ("in_ct", 4),      # input height, width, channel tiles
    ("out_h", 8), ("out_w", 8), ("out_ct", 4),   # output height, width, channel tiles
    ("kernel", 2),        # square kernel size (1..3)
    ("stride", 2),
    ("pad", 2),
    ("res_w", 8),         # shortcut source width   (projection addressing)
    ("res_ct", 4),        # shortcut source channel tiles
    ("res_stride", 2),    # shortcut 1x1 conv stride
    ("w_base", 12),       # weight SRAM word address of the main kernel
    ("res_w_base", 12),   # weight SRAM word address of the 1x1 shortcut kernel
    ("p_base", 8),        # parameter SRAM word address (one word per output tile)
    ("pool_m", 16),       # GAP multiplier
    ("pool_shift", 6),    # GAP rounding shift
)
assert sum(bits for _, bits in FIELDS) <= WORD_BITS


@dataclass
class Descriptor:
    op: int = OP_END
    in_bank: int = 0
    out_bank: int = 0
    res_bank: int = 0
    res_mode: int = RES_NONE
    out_mode: int = OUT_U8
    in_h: int = 0
    in_w: int = 0
    in_ct: int = 0
    out_h: int = 0
    out_w: int = 0
    out_ct: int = 0
    kernel: int = 0
    stride: int = 0
    pad: int = 0
    res_w: int = 0
    res_ct: int = 0
    res_stride: int = 0
    w_base: int = 0
    res_w_base: int = 0
    p_base: int = 0
    pool_m: int = 0
    pool_shift: int = 0
    name: str = ""        # debug only, not encoded

    def encode(self):
        word, pos = 0, 0
        values = asdict(self)
        for field_name, bits in FIELDS:
            value = int(values[field_name])
            if not 0 <= value < (1 << bits):
                raise ValueError(f"{self.name or 'descriptor'}: {field_name}={value} does not fit in {bits} bits")
            word |= value << pos
            pos += bits
        return word

    @classmethod
    def decode(cls, word):
        values, pos = {}, 0
        for field_name, bits in FIELDS:
            values[field_name] = (word >> pos) & ((1 << bits) - 1)
            pos += bits
        return cls(**values)


def field_slices():
    """[(name, msb, lsb)] for documentation and RTL decoders."""
    out, pos = [], 0
    for name, bits in FIELDS:
        out.append((name, pos + bits - 1, pos))
        pos += bits
    return out
