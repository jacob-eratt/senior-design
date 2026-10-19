"""Top level of the cycle-stepped accelerator model.

Pipeline (one column per clock; a register sits between every pair of stages):

    ISSUE ─► MUL ─► ADD ─► ACC ─► PPU_RD ─► PPU_MAC ─► PPU_WB      (CONV, 7 stages)
      │                     └ emits one token per finished output pixel
      └────► POOL_ACC ─► POOL_WB                                  (GAP, 3 stages)

ISSUE    sequencer emits a micro-op; activation + weight SRAM reads issued
MUL      lanes x lanes products            (activation/weight data now valid)
ADD      per-lane adder trees -> partial sums
ACC      accumulator update; on the last op of a pixel, snapshot -> PPU
PPU_RD   param SRAM read (+ identity-residual activation read)
PPU_MAC  bias add, requant multiply, residual multiply-add
PPU_WB   round, shift, clamp, write activation bank (or logit registers)

`step()` evaluates stages from the back of the pipe to the front. Each stage
consumes its input register and fills the next one; because downstream stages
have already consumed their inputs this cycle, this reproduces the behaviour
of edge-triggered registers exactly. SRAM writes commit in `clock()`.

The ADD->ACC and PPU_MAC->PPU_WB registers are delay lines whose length is
1 + cfg.mac_extra_latency / 1 + cfg.ppu_extra_latency, so the model can match
an RTL that pipelines the adder tree or the PPU multipliers more deeply.
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .. import isa
from ..compiler import unpack_activation
from .sequencer import DONE, KIND_GAP, Sequencer
from .sram import SRAM, HardwareError
from .units import Accumulators, MacArray, PoolingUnit, PostProcessingUnit

# Activation banks are filled with this before every run, so reading a word the
# schedule never wrote corrupts the result instead of silently reading zeros.
POISON = 0xA5


class DelayLine:
    """A chain of `length` pipeline registers."""

    def __init__(self, length):
        self.regs = [None] * length

    def take(self):
        """Consumer side: the token leaving the last register this cycle."""
        tok, self.regs[-1] = self.regs[-1], None
        return tok

    def put(self, tok):
        """Producer side (after `take` in the same cycle): shift and insert."""
        self.regs[1:] = self.regs[:-1]
        self.regs[0] = tok

    def busy(self):
        return any(r is not None for r in self.regs)


@dataclass
class LayerStats:
    name: str
    op: str
    start: int = 0           # cycle of the descriptor fetch
    cycles: int = 0          # fetch-to-next-fetch
    counts: Counter = field(default_factory=Counter)


@dataclass
class RunResult:
    logits: np.ndarray
    cycles: int
    layers: list
    snapshots: dict
    trace: list
    memories: dict


class Core:
    def __init__(self, compiled, trace=False):
        self.compiled = compiled
        self.cfg = cfg = compiled.cfg
        L = cfg.lanes
        self.trace = [] if trace else None
        self.act = [
            SRAM(f"act{b}", cfg.act_bank_depth, (L,), np.uint8, trace=self.trace)
            for b in range(cfg.num_act_banks)
        ]
        self.wgt = SRAM("weight", cfg.weight_depth, (L, L), np.int8)
        self.prm = SRAM("param", cfg.param_depth, compiled.param_mem.shape[1:], np.int64)
        self.wgt.load(compiled.weight_mem)
        self.prm.load(compiled.param_mem)
        self.memories = [*self.act, self.wgt, self.prm]
        self.mac = MacArray(cfg)
        self.ppu = PostProcessingUnit(cfg)
        self.reset()

    def reset(self):
        cfg = self.cfg
        self.seq = Sequencer(cfg, self.compiled.program)
        self.accs = Accumulators(cfg)
        self.pool = PoolingUnit(cfg)
        self.out_regs = np.zeros(cfg.out_regs, dtype=np.int64)
        self.rd_q = self.mul_q = self.acc_q = self.prm_q = self.pool_q = None
        self.add_q = DelayLine(1 + cfg.mac_extra_latency)
        self.ppu_q = DelayLine(1 + cfg.ppu_extra_latency)
        self.cycle = 0
        self.layers = []
        self._on_layer_done = None
        for sram in self.memories:
            sram.reads = sram.writes = 0
        for bank in self.act:
            bank.mem[:] = POISON
        if self.trace is not None:
            self.trace.clear()

    def load_input(self, image):
        """Host preload of the input tensor into its bank (DMA, outside the timed run)."""
        self.act[self.compiled.input_bank].load(self.compiled.input_words(image))

    # -- pipeline stages, back to front ---------------------------------------

    def _stage_ppu_wb(self):
        tok = self.ppu_q.take()
        if tok is None:
            return
        op, t, shift = tok
        if op.desc.out_mode == isa.OUT_S32:
            self.out_regs[op.lane_base:op.lane_base + self.cfg.lanes] = t
            self._count(op, "logit_writes")
        else:
            self.act[op.desc.out_bank].write(op.out_addr, self.ppu.round_clamp(t, shift))
            self._count(op, "act_writes")

    def _stage_ppu_mac(self):
        tok, self.prm_q = self.prm_q, None
        if tok is None:
            self.ppu_q.put(None)
            return
        op, acc_main, acc_res, params, residual = tok
        t, shift = self.ppu.multiply_add(op.desc, acc_main, acc_res, params, residual)
        self.ppu_q.put((op, t, shift))

    def _stage_ppu_rd(self):
        tok, self.acc_q = self.acc_q, None
        if tok is None:
            return
        op, acc_main, acc_res = tok
        params = self.prm.read(op.p_addr)
        self._count(op, "param_reads")
        residual = None
        if op.desc.res_mode == isa.RES_IDENTITY:
            residual = self.act[op.desc.res_bank].read(op.out_addr)
            self._count(op, "residual_reads")
        self.prm_q = (op, acc_main, acc_res, params, residual)

    def _stage_pool_wb(self):
        tok, self.pool_q = self.pool_q, None
        if tok is None:
            return
        op, total = tok
        self.act[op.desc.out_bank].write(op.out_addr, self.pool.finish(total, op.desc))
        self._count(op, "act_writes")

    def _stage_acc(self):
        tok = self.add_q.take()
        if tok is None:
            return
        op, psum = tok
        self.accs.update(op.acc_bank, psum, op.clear)
        if op.emit:
            self.acc_q = (op, *self.accs.snapshot())

    def _stage_add(self):
        tok, self.mul_q = self.mul_q, None
        self.add_q.put(None if tok is None else (tok[0], self.mac.reduce(tok[1])))

    def _stage_mul_or_pool(self):
        tok, self.rd_q = self.rd_q, None
        if tok is None:
            return
        op, act, weights = tok
        if op.kind == KIND_GAP:
            self.pool.accumulate(act, op.clear)
            if op.emit:
                self.pool_q = (op, self.pool.sums.copy())
        else:
            self.mul_q = (op, self.mac.multiply(act, weights))

    def _stage_issue(self, busy):
        op = self.seq.tick(busy)
        if self.seq.fetched:
            self._layer_boundary()
        if op is None:
            return
        self._count(op, "issue")
        if op.src_bank is None:
            act = np.zeros(self.cfg.lanes, dtype=np.uint8)
            self._count(op, "padding_ops")
        else:
            act = self.act[op.src_bank].read(op.act_addr)
            self._count(op, "act_reads")
        weights = None
        if op.kind != KIND_GAP:
            weights = self.wgt.read(op.w_addr)
            self._count(op, "weight_reads")
        self.rd_q = (op, act, weights)

    # -- control ---------------------------------------------------------------

    def _busy(self):
        return self.add_q.busy() or self.ppu_q.busy() or any(r is not None for r in (
            self.rd_q, self.mul_q, self.acc_q, self.prm_q, self.pool_q,
        ))

    def _count(self, op, key):
        self.layers[op.layer].counts[key] += 1

    def _layer_boundary(self):
        if self.layers:
            done = self.layers[-1]
            done.cycles = self.cycle - done.start
            if self._on_layer_done:
                self._on_layer_done(len(self.layers) - 1)
        desc = self.seq.desc
        if desc.op != isa.OP_END:
            name = self.compiled.layer_names[len(self.layers)]
            self.layers.append(LayerStats(name, "gap" if desc.op == isa.OP_GAP else "conv", self.cycle))

    def step(self):
        busy = self._busy()
        self._stage_ppu_wb()
        self._stage_ppu_mac()
        self._stage_ppu_rd()
        self._stage_pool_wb()
        self._stage_acc()
        self._stage_add()
        self._stage_mul_or_pool()
        self._stage_issue(busy)
        for sram in self.memories:
            sram.clock(self.cycle)
        self.cycle += 1

    def run(self, image, snapshot=False, max_cycles=5_000_000):
        """Reset, preload `image` (uint8 [C,H,W]), run to DONE."""
        self.reset()
        self.load_input(image)
        snapshots = {}

        def capture(index):
            dst = self.compiled.layer_outputs[index]
            if dst in self.compiled.placements:
                placement = self.compiled.placements[dst]
                snapshots[dst] = unpack_activation(self.act[placement.bank].mem, placement.shape, self.cfg.lanes)
            else:
                snapshots[dst] = self.out_regs[:self.compiled.output_channels].copy()

        self._on_layer_done = capture if snapshot else None
        while self.seq.state != DONE:
            if self.cycle >= max_cycles:
                raise HardwareError(f"no DONE after {max_cycles} cycles")
            self.step()
        return RunResult(
            logits=self.out_regs[:self.compiled.output_channels].copy(),
            cycles=self.cycle,
            layers=self.layers,
            snapshots=snapshots,
            trace=self.trace,
            memories={m.name: {"reads": m.reads, "writes": m.writes} for m in self.memories},
        )
