"""Cycle-accurate reference model of ws_mxu (skews + ws_array) and vectors.

The array is COLS step-2 ColumnModels side by side; the edges are skew delay
lines. The generator streams passes (one ROWS x COLS weight tile + a stream of
pixel vectors), places each weight load in its legal window, runs the cycle
model, and asserts that every valid output equals the matrix product
y[p][c] = sum_r x[p][r] * W[r][c], at the column-skewed time
t + ROWS - 1 + c (t = edge at which pixel p was presented).

Writes tb/vectors/mxu_r<ROWS>c<COLS>_{stim,exp,count}.mem (hex), LSB first:
stim (8R + 3 + 8C bits): pixel_in[r] at [8r +: 8], pixel_valid_in at [8R],
     new_pass_start_in at [8R+1], weight_load_enable at [8R+2],
     weight_load_in[c] at [8R+3+8c +: 8]
exp (33C bits): partial_sum_out[c] at [32c +: 32],
     partial_sum_valid_out[c] at [32C + c]
exp[i] is the DUT output after the clock edge that consumed stim[i].
count: word 0 = cycles, word 1 = output values checked (pixels x COLS).
"""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from gen_column_vectors import ColumnModel
from gen_pe_vectors import PSUM_W, wrap


class SkewModel:
    """Lane i delayed by i cycles; lane 0 is a wire."""

    def __init__(self, lanes, reset_value):
        self.regs = [[reset_value] * i for i in range(lanes)]

    def view(self, xs):
        return [r[-1] if r else x for r, x in zip(self.regs, xs)]

    def shift(self, xs):
        for r, x in zip(self.regs, xs):
            if r:
                r.insert(0, x)
                r.pop()


class ArrayModel:
    def __init__(self, rows, cols):
        self.cols = [ColumnModel(rows) for _ in range(cols)]

    def step(self, pixels, valids, swaps, wld_en, wld_in):
        # Right to left, so column c reads column c-1's registers from before
        # this edge.
        for c in reversed(range(len(self.cols))):
            if c:
                left = self.cols[c - 1].pes
                p = [pe.act_out for pe in left]
                v = [pe.act_vld_out for pe in left]
                s = [pe.swap_out for pe in left]
            else:
                p, v, s = pixels, valids, swaps
            self.cols[c].step(p, v, s, wld_en[c], wld_in[c], 0)

    def outputs(self):
        return [(col.pes[-1].psum_out, col.pes[-1].act_vld_out) for col in self.cols]


class MxuModel:
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.row_skew = SkewModel(rows, (0, 0, 0))
        self.col_skew = SkewModel(cols, (0, 0))
        self.array = ArrayModel(rows, cols)

    def step(self, pixels, valid, swap, wld_en, wld_row):
        row_in = [(x, valid, swap) for x in pixels]
        col_in = [(wld_en, w) for w in wld_row]
        rows_now = self.row_skew.view(row_in)
        cols_now = self.col_skew.view(col_in)
        self.array.step([x for x, _, _ in rows_now], [v for _, v, _ in rows_now],
                        [s for _, _, s in rows_now], [e for e, _ in cols_now],
                        [w for _, w in cols_now])
        self.row_skew.shift(row_in)
        self.col_skew.shift(col_in)
        return self.array.outputs()


@dataclass
class Pass:
    weights: list          # weights[r][c]
    slots: list            # pixel vector, or None for a bubble
    gap_after: int = 0
    load_at: str = "random"  # where the load for the NEXT pass goes


def directed_passes(rows, cols):
    min_p = 2 * rows - 1
    a_w = [[wrap((r + 1) * (c + 1) - 3 * c, 8) for c in range(cols)] for r in range(rows)]
    a_slots = [[1] * rows] + [[int(i == r) for i in range(rows)] for r in range(rows)]
    a_slots += [[(r % 5) - 2 for r in range(rows)]] * max(0, min_p - len(a_slots))
    b_w = [[-128] * cols for _ in range(rows)]
    b_slots = [[-128] * rows, [127] * rows]
    b_slots += [[-128 if (r + i) % 2 else 127 for r in range(rows)]
                for i in range(min_p - len(b_slots))]
    c_w = [[127 if (r + c) % 2 else -127 for c in range(cols)] for r in range(rows)]
    c_slots = [[r - rows // 2 for r in range(rows)]]
    for i in range(1, 64):
        c_slots.append(None if i % 9 == 4 else
                       [((3 * r + i) % 255) - 127 for r in range(rows)])
    return [Pass(a_w, a_slots, load_at="early"), Pass(b_w, b_slots, load_at="late"),
            Pass(c_w, c_slots, gap_after=3)]


def random_passes(rows, cols, rng, count):
    def value():
        x = rng.random()
        return -128 if x < 0.05 else 127 if x < 0.1 else rng.randint(-128, 127)

    passes = []
    for _ in range(count):
        n = rng.randint(2 * rows - 1, 2 * rows + 40)
        slots = [[value() for _ in range(rows)]]
        slots += [None if rng.random() < 0.15 else [value() for _ in range(rows)]
                  for _ in range(n - 1)]
        weights = [[value() for _ in range(cols)] for _ in range(rows)]
        gap = 0 if rng.random() < 0.6 else rng.randint(1, 6)
        passes.append(Pass(weights, slots, gap, rng.choice(["early", "late", "random"])))
    return passes


def build(rows, cols, passes, rng):
    """Per-cycle mxu inputs and the expected outputs {(cycle, col): value}."""
    t0s = []
    t = rows + 2
    for p in passes:
        t0s.append(t)
        t += len(p.slots) + p.gap_after
    length = t + rows + cols + 4
    garbage = lambda n: [rng.randint(-128, 127) for _ in range(n)]
    pix = [garbage(rows) for _ in range(length)]   # invalid slots carry garbage
    vld = [0] * length
    swp = [0] * length
    wld_en = [0] * length
    wld_row = [garbage(cols) for _ in range(length)]
    expect = {}
    for k, p in enumerate(passes):
        t0 = t0s[k]
        assert p.slots[0] is not None, "a pass must start with a real pixel"
        for i, x in enumerate(p.slots):
            if x is None:
                continue
            pix[t0 + i], vld[t0 + i], swp[t0 + i] = list(x), 1, int(i == 0)
            for c in range(cols):
                y = sum(x[r] * p.weights[r][c] for r in range(rows))
                expect[(t0 + i + rows - 1 + c, c)] = wrap(y, PSUM_W)
        lo = 0 if k == 0 else t0s[k - 1] + rows - 1
        hi = t0 - rows
        assert lo <= hi, f"pass {k - 1} too short to hide the next load"
        policy = "early" if k == 0 else passes[k - 1].load_at
        s = lo if policy == "early" else hi if policy == "late" else rng.randint(lo, hi)
        for j in range(rows):
            wld_en[s + j] = 1
            wld_row[s + j] = list(p.weights[rows - 1 - j])
    return (pix, vld, swp, wld_en, wld_row), expect


def pack_stim(rows, cols, pixels, valid, swap, wld_en, wld_row):
    w = 0
    for r, x in enumerate(pixels):
        w |= (x & 0xFF) << (8 * r)
    w |= valid << (8 * rows) | swap << (8 * rows + 1) | wld_en << (8 * rows + 2)
    for c, x in enumerate(wld_row):
        w |= (x & 0xFF) << (8 * rows + 3 + 8 * c)
    return w


def pack_exp(cols, outs):
    w = 0
    for c, (psum, vld) in enumerate(outs):
        w |= (psum & 0xFFFFFFFF) << (32 * c) | vld << (32 * cols + c)
    return w


def run(rows, cols, stim_lists, expect):
    model = MxuModel(rows, cols)
    stim, exp = [], []
    for cyc, args in enumerate(zip(*stim_lists)):
        outs = model.step(*args)
        stim.append(pack_stim(rows, cols, *args))
        exp.append(pack_exp(cols, outs))
        for c, (psum, vld) in enumerate(outs):
            want = expect.get((cyc, c))
            if want is None:
                assert vld == 0, f"cycle {cyc} col {c}: unexpected valid output"
            else:
                assert vld == 1 and psum == want, (
                    f"cycle {cyc} col {c}: model {psum} (valid {vld}) != matmul {want}")
    return stim, exp


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=int, default=27)
    p.add_argument("--cols", type=int, default=26)
    p.add_argument("--passes", type=int, default=30, help="random passes")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, default=root / "tb" / "vectors")
    a = p.parse_args()
    rng = random.Random(a.seed)
    passes = directed_passes(a.rows, a.cols) + random_passes(a.rows, a.cols, rng, a.passes)
    stim_lists, expect = build(a.rows, a.cols, passes, rng)
    stim, exp = run(a.rows, a.cols, stim_lists, expect)
    name = f"mxu_r{a.rows}c{a.cols}"
    sd = (8 * a.rows + 3 + 8 * a.cols + 3) // 4
    ed = (33 * a.cols + 3) // 4
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{name}_stim.mem").write_text("".join(f"{w:0{sd}x}\n" for w in stim))
    (a.out / f"{name}_exp.mem").write_text("".join(f"{w:0{ed}x}\n" for w in exp))
    (a.out / f"{name}_count.mem").write_text(f"{len(stim):08x}\n{len(expect):08x}\n")
    print(f"Wrote {len(stim)} cycles, {len(expect)} outputs "
          f"({len(expect) // a.cols} pixels x {a.cols} columns) verified against X*W, "
          f"{len(passes)} passes, {a.rows}x{a.cols} to {a.out} (seed {a.seed})")


if __name__ == "__main__":
    main()
