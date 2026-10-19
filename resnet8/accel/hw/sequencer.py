"""Sequencer: instruction fetch/decode FSM + address-generation loop counters.

States
  WAIT  wait for the datapath to drain (`busy` low), then fetch + decode the
        next descriptor (1 cycle). END -> DONE; otherwise -> RUN.
  RUN   issue exactly one micro-op per cycle; after the last one -> WAIT.
  DONE  idle; `done` is asserted to the host.

Waiting for a full drain between layers is the simple, hazard-free policy: a
layer only reads banks after the previous layer's last write has committed.

CONV loop nest (outermost first), one micro-op per innermost iteration:

    for ot in out_ct:                      # output-channel tile -> param word
      for oy in out_h:
        for ox in out_w:                   # one output pixel, `lanes` channels
          for ky, kx in kernel x kernel:   # main phase, accumulator bank 0
            for it in in_ct:
          for rt in res_ct:                # projection phase, accumulator bank 1
                                           # (only if res_mode == PROJECTION)

GAP loop nest:  for ct in in_ct: for p in in_h*in_w
"""

from .. import isa

WAIT, RUN, DONE = "WAIT", "RUN", "DONE"
KIND_CONV, KIND_GAP = "conv", "gap"


class MicroOp:
    """Control token issued by the sequencer and carried down the pipeline."""

    __slots__ = (
        "kind", "layer", "desc", "src_bank", "act_addr", "w_addr", "acc_bank",
        "clear", "emit", "out_addr", "p_addr", "lane_base",
    )

    def __init__(self, **kw):
        for name in self.__slots__:
            setattr(self, name, kw.get(name))


class Sequencer:
    def __init__(self, cfg, imem):
        self.cfg = cfg
        self.imem = imem
        self.state = WAIT
        self.pc = 0
        self.layer = -1           # index of the descriptor being executed
        self.desc = None
        self.fetched = False      # pulses for one cycle on each fetch

    def tick(self, busy):
        """Advance one cycle. Returns the MicroOp issued this cycle, or None."""
        self.fetched = False
        if self.state == DONE:
            return None
        if self.state == WAIT:
            if busy:
                return None
            self.desc = isa.Descriptor.decode(self.imem[self.pc])
            self.pc += 1
            self.layer += 1
            self.fetched = True
            if self.desc.op == isa.OP_END:
                self.state = DONE
            elif self.desc.op in (isa.OP_CONV, isa.OP_GAP):
                self._reset_counters()
                self.state = RUN
            else:
                raise ValueError(f"illegal opcode {self.desc.op} at pc {self.pc - 1}")
            return None

        op = self._issue_conv() if self.desc.op == isa.OP_CONV else self._issue_gap()
        if op.emit and self._advance_pixel_done():
            self.state = WAIT
        return op

    # -- counters -----------------------------------------------------------

    def _reset_counters(self):
        self.ot = self.oy = self.ox = 0
        self.phase = 0            # 0 = main kernel, 1 = projection shortcut
        self.ky = self.kx = self.it = 0
        self.ct = self.p = 0

    def _issue_conv(self):
        d, L = self.desc, self.cfg.lanes
        common = dict(
            kind=KIND_CONV, layer=self.layer, desc=d,
            out_addr=(self.oy * d.out_w + self.ox) * d.out_ct + self.ot,
            p_addr=d.p_base + self.ot, lane_base=self.ot * L,
        )
        if self.phase == 0:
            iy = self.oy * d.stride - d.pad + self.ky
            ix = self.ox * d.stride - d.pad + self.kx
            inside = 0 <= iy < d.in_h and 0 <= ix < d.in_w
            last_tap = self.ky == d.kernel - 1 and self.kx == d.kernel - 1 and self.it == d.in_ct - 1
            op = MicroOp(
                src_bank=d.in_bank if inside else None,       # None: zero padding, read gated
                act_addr=(iy * d.in_w + ix) * d.in_ct + self.it if inside else None,
                w_addr=d.w_base + ((self.ot * d.kernel + self.ky) * d.kernel + self.kx) * d.in_ct + self.it,
                acc_bank=0, clear=self.ky == 0 and self.kx == 0 and self.it == 0,
                emit=last_tap and d.res_mode != isa.RES_PROJECTION, **common,
            )
            self._advance_main(last_tap)
        else:
            iy, ix = self.oy * d.res_stride, self.ox * d.res_stride
            op = MicroOp(
                src_bank=d.res_bank, act_addr=(iy * d.res_w + ix) * d.res_ct + self.it,
                w_addr=d.res_w_base + self.ot * d.res_ct + self.it,
                acc_bank=1, clear=self.it == 0, emit=self.it == d.res_ct - 1, **common,
            )
            self.it += 1
            if self.it == d.res_ct:
                self.it, self.phase = 0, 0
        return op

    def _advance_main(self, last_tap):
        d = self.desc
        if last_tap:
            self.ky = self.kx = self.it = 0
            if d.res_mode == isa.RES_PROJECTION:
                self.phase = 1
            return
        self.it += 1
        if self.it == d.in_ct:
            self.it = 0
            self.kx += 1
            if self.kx == d.kernel:
                self.kx = 0
                self.ky += 1

    def _issue_gap(self):
        d = self.desc
        pixels = d.in_h * d.in_w
        op = MicroOp(
            kind=KIND_GAP, layer=self.layer, desc=d, src_bank=d.in_bank,
            act_addr=self.p * d.in_ct + self.ct, clear=self.p == 0, emit=self.p == pixels - 1,
            out_addr=self.ct,
        )
        self.p += 1
        if self.p == pixels:
            self.p = 0
        return op

    def _advance_pixel_done(self):
        """Called after each emitting micro-op; steps the outer loops. True when the layer is done."""
        d = self.desc
        if d.op == isa.OP_GAP:
            self.ct += 1
            return self.ct == d.in_ct
        self.ox += 1
        if self.ox == d.out_w:
            self.ox = 0
            self.oy += 1
            if self.oy == d.out_h:
                self.oy = 0
                self.ot += 1
                return self.ot == d.out_ct
        return False
