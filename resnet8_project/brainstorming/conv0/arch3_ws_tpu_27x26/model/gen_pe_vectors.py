"""Cycle-accurate reference model of ws_pe and testbench vector generator.

Writes tb/vectors/pe_stim.mem, pe_exp.mem and pe_count.mem (hex, 64-bit words).

stim word: [7:0] act_in, [8] act_vld_in, [9] swap_in, [10] wld_en,
           [23:16] wld_in, [63:32] psum_in
exp word:  [7:0] act_out, [8] act_vld_out, [9] swap_out,
           [23:16] wld_out, [63:32] psum_out
exp[i] is the DUT output after the clock edge that consumed stim[i].
"""

import argparse
import random
from pathlib import Path

PSUM_W = 32


def wrap(value, bits):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


class PeModel:
    def __init__(self):
        self.act_out = 0
        self.act_vld_out = 0
        self.swap_out = 0
        self.w_next = 0
        self.w_active = 0
        self.psum_out = 0

    def step(self, act, vld, swap, wld_en, wld_in, psum_in):
        w_sel = self.w_next if swap else self.w_active
        self.psum_out = wrap(psum_in + act * w_sel, PSUM_W)
        if swap:
            self.w_active = self.w_next
        if wld_en:
            self.w_next = wld_in
        self.act_out, self.act_vld_out, self.swap_out = act, vld, swap
        return self.outputs()

    def outputs(self):
        return (self.act_out, self.act_vld_out, self.swap_out,
                self.w_next, self.psum_out)


def pack_stim(act, vld, swap, wld_en, wld_in, psum_in):
    return ((act & 0xFF) | vld << 8 | swap << 9 | wld_en << 10
            | (wld_in & 0xFF) << 16 | (psum_in & 0xFFFFFFFF) << 32)


def pack_exp(act_out, vld_out, swap_out, wld_out, psum_out):
    return ((act_out & 0xFF) | vld_out << 8 | swap_out << 9
            | (wld_out & 0xFF) << 16 | (psum_out & 0xFFFFFFFF) << 32)


def directed():
    """(act, vld, swap, wld_en, wld_in, psum_in) cycles with known answers."""
    return [
        (0, 0, 0, 1, 5, 0),            # load 5 into w_next
        (3, 1, 1, 0, 0, 10),           # swap: 10 + 3*5 = 25, w_active <- 5
        (-2, 1, 0, 0, 0, 0),           # 0 + -2*5 = -10
        (4, 1, 0, 1, -7, 100),         # load -7 while computing with 5: 120
        (1, 1, 1, 1, 9, 0),            # swap uses -7, w_next <- 9 same cycle
        (2, 1, 0, 0, 0, 0),            # w_active is -7: -14
        (0, 0, 0, 0, 0, 0),            # bubble, psum passes through
        (0, 0, 0, 1, -128, 7),         # load -128, psum pass-through
        (-128, 1, 1, 0, 0, 0),         # -128 * -128 = 16384
        (127, 1, 0, 0, 0, 0),          # 127 * -128 = -16256
        (-128, 1, 0, 0, 0, 2**31 - 16385),  # large psum: 2^31-1 exactly
        (0, 0, 0, 0, 0, -(2**31)),     # most negative psum passes through
    ]


def random_cycles(rng, count):
    cycles = []
    for _ in range(count):
        vld = int(rng.random() < 0.8)
        act = rng.randint(-128, 127) if vld else 0
        swap = int(vld and rng.random() < 0.05)
        wld_en = int(rng.random() < 0.3)
        wld_in = rng.randint(-128, 127)
        psum_in = rng.randint(-(2**28), 2**28)
        cycles.append((act, vld, swap, wld_en, wld_in, psum_in))
    return cycles


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count", type=int, default=20000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, default=root / "tb" / "vectors")
    a = p.parse_args()
    cycles = directed() + random_cycles(random.Random(a.seed), a.count)
    model = PeModel()
    stim, exp = [], []
    for c in cycles:
        stim.append(pack_stim(*c))
        exp.append(pack_exp(*model.step(*c)))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "pe_stim.mem").write_text("".join(f"{w:016x}\n" for w in stim))
    (a.out / "pe_exp.mem").write_text("".join(f"{w:016x}\n" for w in exp))
    (a.out / "pe_count.mem").write_text(f"{len(cycles):08x}\n")
    print(f"Wrote {len(cycles)} PE vectors to {a.out} (seed {a.seed})")


if __name__ == "__main__":
    main()
