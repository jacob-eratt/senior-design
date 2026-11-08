"""Cycle-accurate reference model of ws_column and testbench vector generator.

A column is ROWS ws_pe models chained top -> bottom. The generator builds
passes (one weight vector + a stream of pixel vectors), skews them onto the
rows, schedules every weight load inside its legal window, runs the cycle
model, and asserts that every valid bottom output equals bias + sum(w * x).

Writes tb/vectors/column_r<ROWS>_{stim,exp,count}.mem (hex).

Both words are 10*ROWS + 41 bits, LSB first:
stim: pixel_in[r] at [8r +: 8], pixel_valid_in at [8R + r],
      new_pass_start_in at [9R + r], weight_load_enable at [10R],
      weight_load_in at [10R+1 +: 8], partial_sum_in at [10R+9 +: 32]
exp:  pixel_out[r] at [8r +: 8], pixel_valid_out at [8R + r],
      new_pass_start_out at [9R + r], weight_load_out at [10R +: 8],
      partial_sum_out at [10R+8 +: 32], partial_sum_valid_out at [10R+40]
exp[i] is the DUT output after the clock edge that consumed stim[i].
count: word 0 = cycles, word 1 = dot products checked.

Load window rule (shared weight_load_enable): the ROWS shifts for pass k+1
must happen at edges [t0_k + ROWS - 1, t0_{k+1} - 1], where t0 is the edge at
which a pass's first pixel enters row 0. So back-to-back passes need
P >= 2*ROWS - 1 pixel slots.
"""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from gen_pe_vectors import PSUM_W, PeModel, wrap


class ColumnModel:
    def __init__(self, rows):
        self.pes = [PeModel() for _ in range(rows)]

    def step(self, pixels, valids, swaps, wld_en, wld_in, psum_in):
        # Bottom-up, so row r reads row r-1's registers from before this edge.
        for r in reversed(range(len(self.pes))):
            above = self.pes[r - 1] if r else None
            self.pes[r].step(pixels[r], valids[r], swaps[r], wld_en,
                             above.w_next if above else wld_in,
                             above.psum_out if above else psum_in)
        return self.outputs()

    def outputs(self):
        last = self.pes[-1]
        return ([pe.act_out for pe in self.pes],
                [pe.act_vld_out for pe in self.pes],
                [pe.swap_out for pe in self.pes],
                last.w_next, last.psum_out, last.act_vld_out)


@dataclass
class Pass:
    weights: list
    slots: list            # (pixel vector, bias) or None for a bubble
    gap_after: int = 0     # idle row-0 cycles before the next pass
    load_at: str = "random"  # where the load for the NEXT pass goes


def word_width(rows):
    return 10 * rows + 41


def pack_stim(rows, pix, vld, swp, wld_en, wld_in, psum_in):
    w = 0
    for r in range(rows):
        w |= (pix[r] & 0xFF) << (8 * r)
        w |= vld[r] << (8 * rows + r)
        w |= swp[r] << (9 * rows + r)
    w |= wld_en << (10 * rows)
    w |= (wld_in & 0xFF) << (10 * rows + 1)
    w |= (psum_in & 0xFFFFFFFF) << (10 * rows + 9)
    return w


def pack_exp(rows, pix, vld, swp, wld_out, psum_out, psum_vld):
    w = 0
    for r in range(rows):
        w |= (pix[r] & 0xFF) << (8 * r)
        w |= vld[r] << (8 * rows + r)
        w |= swp[r] << (9 * rows + r)
    w |= (wld_out & 0xFF) << (10 * rows)
    w |= (psum_out & 0xFFFFFFFF) << (10 * rows + 8)
    w |= psum_vld << (10 * rows + 40)
    return w


def directed_passes(rows):
    min_p = 2 * rows - 1
    ones = [1] * rows
    a_slots = [(ones, 0)]
    a_slots += [([int(i == r) for i in range(rows)], 0) for r in range(rows)]
    a_slots += [(ones, 1000), ([-1] * rows, -5)]
    a_slots += [([(r % 5) - 2 for r in range(rows)], i)
                for i in range(min_p - len(a_slots))]
    a = Pass([wrap(r + 1, 8) for r in range(rows)], a_slots, load_at="early")
    b_slots = [([-128] * rows, 0), ([127] * rows, 0), ([0] * rows, 2**31 - 1),
               ([-128] * rows, 2**31 - 1), ([127] * rows, -(2**31))]
    b_slots += [([-128 if (r + i) % 2 else 127 for r in range(rows)], 0)
                for i in range(min_p - len(b_slots))]
    b = Pass([-128] * rows, b_slots, load_at="late")
    c_slots = [([r - rows // 2 for r in range(rows)], 7)]
    for i in range(1, 64):
        c_slots.append(None if i % 9 == 4 else
                       ([((3 * r + i) % 255) - 127 for r in range(rows)], -i))
    c = Pass([127 if r % 2 else -127 for r in range(rows)], c_slots, gap_after=3)
    return [a, b, c]


def random_passes(rows, rng, count):
    def value():
        x = rng.random()
        return -128 if x < 0.05 else 127 if x < 0.1 else rng.randint(-128, 127)

    passes = []
    for _ in range(count):
        n = rng.randint(2 * rows - 1, 2 * rows + 40)
        slots = [([value() for _ in range(rows)], rng.randint(-2**20, 2**20))]
        for _ in range(n - 1):
            slots.append(None if rng.random() < 0.15 else
                         ([value() for _ in range(rows)],
                          rng.randint(-2**20, 2**20)))
        gap = 0 if rng.random() < 0.6 else rng.randint(1, 6)
        load_at = rng.choice(["early", "late", "random"])
        passes.append(Pass([value() for _ in range(rows)], slots, gap, load_at))
    return passes


def build(rows, passes, rng):
    t0s = []
    t = rows + 2
    for p in passes:
        t0s.append(t)
        t += len(p.slots) + p.gap_after
    length = t + rows + 4
    pix = [[0] * rows for _ in range(length)]
    vld = [[0] * rows for _ in range(length)]
    swp = [[0] * rows for _ in range(length)]
    wld_en = [0] * length
    wld_in = [rng.randint(-128, 127) for _ in range(length)]
    psum_in = [0] * length
    expect = {}
    for k, p in enumerate(passes):
        t0 = t0s[k]
        assert p.slots[0] is not None, "a pass must start with a real pixel"
        for i, slot in enumerate(p.slots):
            if slot is None:
                continue
            x, bias = slot
            for r in range(rows):
                pix[t0 + i + r][r] = x[r]
                vld[t0 + i + r][r] = 1
                swp[t0 + i + r][r] = int(i == 0)
            psum_in[t0 + i] = bias
            dot = sum(w * a for w, a in zip(p.weights, x))
            expect[t0 + i + rows - 1] = wrap(bias + dot, PSUM_W)
        lo = 0 if k == 0 else t0s[k - 1] + rows - 1
        hi = t0 - rows
        assert lo <= hi, f"pass {k - 1} too short to hide the next load"
        policy = "early" if k == 0 else passes[k - 1].load_at
        s = lo if policy == "early" else hi if policy == "late" else rng.randint(lo, hi)
        for j in range(rows):
            wld_en[s + j] = 1
            wld_in[s + j] = p.weights[rows - 1 - j]
    return pix, vld, swp, wld_en, wld_in, psum_in, expect


def run(rows, stim_lists, expect):
    pix, vld, swp, wld_en, wld_in, psum_in = stim_lists
    model = ColumnModel(rows)
    stim, exp = [], []
    for c in range(len(wld_en)):
        args = (pix[c], vld[c], swp[c], wld_en[c], wld_in[c], psum_in[c])
        out = model.step(*args)
        stim.append(pack_stim(rows, *args))
        exp.append(pack_exp(rows, *out))
        psum_out, psum_vld = out[4], out[5]
        if c in expect:
            assert psum_vld == 1 and psum_out == expect[c], (
                f"cycle {c}: model {psum_out} (valid {psum_vld}) != dot {expect[c]}")
        else:
            assert psum_vld == 0, f"cycle {c}: unexpected valid output"
    return stim, exp


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=int, default=27)
    p.add_argument("--passes", type=int, default=60, help="random passes")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, default=root / "tb" / "vectors")
    a = p.parse_args()
    rng = random.Random(a.seed)
    passes = directed_passes(a.rows) + random_passes(a.rows, rng, a.passes)
    *stim_lists, expect = build(a.rows, passes, rng)
    stim, exp = run(a.rows, stim_lists, expect)
    digits = (word_width(a.rows) + 3) // 4
    name = f"column_r{a.rows}"
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{name}_stim.mem").write_text("".join(f"{w:0{digits}x}\n" for w in stim))
    (a.out / f"{name}_exp.mem").write_text("".join(f"{w:0{digits}x}\n" for w in exp))
    (a.out / f"{name}_count.mem").write_text(f"{len(stim):08x}\n{len(expect):08x}\n")
    print(f"Wrote {len(stim)} cycles, {len(expect)} dot products verified against "
          f"bias + sum(w*x), {len(passes)} passes, ROWS={a.rows} to {a.out} (seed {a.seed})")


if __name__ == "__main__":
    main()
