"""Closed-form performance, memory-traffic and resource model.

Derived from the descriptors alone (no simulation), so it can be evaluated for
any design point in microseconds; tests check that it reproduces the
cycle-level model's cycle count and SRAM access counts exactly.
"""

from dataclasses import dataclass, replace

import numpy as np

from . import isa
from .compiler import compile_model
from .config import AccelConfig
from .qmodel import ConvLayer


@dataclass
class LayerPerf:
    name: str
    op: str
    issue: int            # cycles issuing micro-ops
    cycles: int           # including descriptor fetch and pipeline drain
    macs: int             # useful multiply-accumulates (network definition)
    act_reads: int
    padding_ops: int
    weight_reads: int
    param_reads: int
    residual_reads: int
    act_writes: int

    def utilization(self, lanes):
        return self.macs / (self.cycles * lanes * lanes) if self.cycles else 0.0


def _valid_taps(out_len, in_len, stride, pad, kernel):
    """Number of (output position, kernel offset) pairs that land inside the input."""
    o = np.arange(out_len)[:, None] * stride - pad + np.arange(kernel)[None, :]
    return int(np.sum((o >= 0) & (o < in_len)))


def layer_perf(desc, cfg, layer=None):
    if desc.op == isa.OP_GAP:
        issue = desc.in_ct * desc.in_h * desc.in_w
        return LayerPerf(desc.name, "gap", issue, issue + cfg.gap_drain, 0, issue, 0, 0, 0, 0, desc.in_ct)

    pixels = desc.out_h * desc.out_w
    taps = desc.kernel * desc.kernel
    res_ops = desc.res_ct if desc.res_mode == isa.RES_PROJECTION else 0
    issue = desc.out_ct * pixels * (taps * desc.in_ct + res_ops)
    valid = (_valid_taps(desc.out_h, desc.in_h, desc.stride, desc.pad, desc.kernel)
             * _valid_taps(desc.out_w, desc.in_w, desc.stride, desc.pad, desc.kernel))
    main_reads = valid * desc.in_ct * desc.out_ct
    macs = 0
    if isinstance(layer, ConvLayer):
        macs = pixels * layer.out_channels * layer.in_channels * taps
        if layer.res_mode == "projection":
            macs += pixels * layer.out_channels * layer.res_weight.shape[1]
    emitted = desc.out_ct * pixels
    return LayerPerf(
        desc.name, "conv", issue, issue + cfg.conv_drain, macs,
        act_reads=main_reads + desc.out_ct * pixels * res_ops,
        padding_ops=desc.out_ct * pixels * taps * desc.in_ct - main_reads,
        weight_reads=issue, param_reads=emitted,
        residual_reads=emitted if desc.res_mode == isa.RES_IDENTITY else 0,
        act_writes=emitted if desc.out_mode == isa.OUT_U8 else 0,
    )


def analyze(compiled, qmodel=None):
    """Per-layer LayerPerf list and total cycles (incl. the final END fetch)."""
    layers = {l.name: l for l in qmodel.layers} if qmodel else {}
    rows = [layer_perf(d, compiled.cfg, layers.get(d.name)) for d in compiled.descriptors if d.op != isa.OP_END]
    return rows, sum(r.cycles for r in rows) + 1


def resources(compiled):
    """Rough FPGA-oriented resource estimate (BRAM36 as 512 x 72)."""
    cfg = compiled.cfg

    def bram36(width, depth):
        return -(-width // 72) * -(-depth // 512)

    act_bits = cfg.num_act_banks * cfg.act_bank_depth * cfg.act_word_bits
    return {
        "mac_multipliers_8x8": cfg.lanes * cfg.lanes,
        "dsp48e2_int8_packed": cfg.lanes * cfg.lanes // 2,
        "ppu_multipliers": f"{2 * cfg.lanes} x ({cfg.acc_bits}b x {cfg.mult_bits}b)",
        "act_sram_kbit": act_bits / 1024,
        "weight_sram_kbit_used": compiled.weight_words * cfg.weight_word_bits / 1024,
        "weight_sram_kbit": cfg.weight_depth * cfg.weight_word_bits / 1024,
        "param_sram_kbit_used": compiled.param_words * cfg.param_word_bits / 1024,
        "bram36_act": cfg.num_act_banks * bram36(cfg.act_word_bits, cfg.act_bank_depth),
        "bram36_weight": bram36(cfg.weight_word_bits, cfg.weight_depth),
    }


def format_report(compiled, qmodel=None, measured=None):
    """Text table; `measured` (cycles per layer from the cycle model) is shown alongside if given."""
    cfg = compiled.cfg
    rows, total = analyze(compiled, qmodel)
    lines = [
        f"Design point: {cfg.lanes}x{cfg.lanes} MAC array, {cfg.clock_mhz:.0f} MHz",
        "",
        f"{'layer':6s} {'op':4s} {'cycles':>8s} {'MACs':>11s} {'util':>6s} {'act rd':>7s} {'pad':>6s} "
        f"{'w rd':>7s} {'res rd':>6s} {'wr':>5s}" + (f" {'sim cycles':>10s}" if measured else ""),
    ]
    for i, r in enumerate(rows):
        line = (f"{r.name:6s} {r.op:4s} {r.cycles:8d} {r.macs:11,d} {r.utilization(cfg.lanes):6.1%} "
                f"{r.act_reads:7d} {r.padding_ops:6d} {r.weight_reads:7d} {r.residual_reads:6d} {r.act_writes:5d}")
        if measured:
            line += f" {measured[i]:10d}"
        lines.append(line)
    macs = sum(r.macs for r in rows)
    latency_us = total / cfg.clock_mhz
    lines += [
        "",
        f"total cycles        {total:,d}" + (f"   (cycle model: {sum(measured) + 1:,d})" if measured else ""),
        f"latency             {latency_us:.1f} us @ {cfg.clock_mhz:.0f} MHz -> {1e6 / latency_us:,.0f} inferences/s",
    ]
    if macs:
        lines += [
            f"MACs / inference    {macs:,d}",
            f"array utilisation   {macs / (total * cfg.lanes ** 2):.1%}",
            f"effective GOPS      {2 * macs / latency_us / 1e3:.1f} (peak {2 * cfg.lanes ** 2 * cfg.clock_mhz / 1e3:.1f})",
        ]
    return "\n".join(lines)


def explore(qmodel, lanes_options=(8, 16, 32), clock_mhz=200.0):
    """Cycle count / utilisation for several array sizes (memories sized to fit).

    Fewer than 8 lanes would overflow the descriptor's channel-tile fields.
    """
    results = []
    for lanes in lanes_options:
        cfg = replace(AccelConfig(), lanes=lanes, clock_mhz=clock_mhz,
                      act_bank_depth=1 << 16, weight_depth=1 << 12, param_depth=256)
        compiled = compile_model(qmodel, cfg)
        rows, total = analyze(compiled, qmodel)
        macs = sum(r.macs for r in rows)
        results.append({
            "lanes": lanes, "multipliers": lanes * lanes, "cycles": total,
            "latency_us": total / clock_mhz, "utilization": macs / (total * lanes * lanes),
            "weight_words": compiled.weight_words,
            "act_bank_words": max(p.shape[1] * p.shape[2] * -(-p.shape[0] // lanes)
                                  for p in compiled.placements.values()),
        })
    return results
