"""Map a QModel onto the accelerator: program, memory images, and static checks.

Output (`Compiled`):
  * descriptors / program   one 128-bit descriptor per layer, END-terminated
  * weight_mem              int8  [weight_depth, lanes(c), lanes(k)]
  * param_mem               int64 [param_depth, 5, lanes]
                            rows: bias_main, bias_res, m_main, m_res, shift
  * placements              which activation bank each tensor lives in
  * widths                  worst-case datapath widths, proven per layer

Memory layouts (also in SPEC.md):
  activation word  (y*W + x)*CT + ct      lane c  = channel ct*L + c
  weight word      w_base + ((ot*K + ky)*K + kx)*IT + it
                   element [c][k] = weight[ot*L + k, it*L + c, ky, kx]
  1x1 shortcut     res_w_base + ot*RT + rt (same element order)
  param word       p_base + ot, lane k = output channel ot*L + k
"""

from dataclasses import dataclass, field

import numpy as np

from . import isa
from .config import AccelConfig
from .fixedpoint import bits_for_signed
from .qmodel import ConvLayer, PoolLayer

PARAM_ROWS = ("bias_main", "bias_res", "m_main", "m_res", "shift")


def tiles(channels, lanes):
    return -(-channels // lanes)


def pack_activation(tensor, lanes):
    """uint8 [C,H,W] -> bank words [H*W*CT, lanes]."""
    c, h, w = tensor.shape
    ct = tiles(c, lanes)
    padded = np.zeros((ct * lanes, h, w), dtype=np.uint8)
    padded[:c] = tensor
    return padded.reshape(ct, lanes, h, w).transpose(2, 3, 0, 1).reshape(h * w * ct, lanes)


def unpack_activation(words, shape, lanes):
    """bank words -> uint8 [C,H,W] (inverse of pack_activation)."""
    c, h, w = shape
    ct = tiles(c, lanes)
    data = np.asarray(words[:h * w * ct]).reshape(h, w, ct, lanes).transpose(2, 3, 0, 1)
    return data.reshape(ct * lanes, h, w)[:c].astype(np.uint8)


def pack_weights(weight, lanes):
    """int8 [O,C,K,K] -> weight words [OT*K*K*IT, lanes(c), lanes(k)]."""
    o, c, k, _ = weight.shape
    ot, it = tiles(o, lanes), tiles(c, lanes)
    padded = np.zeros((ot * lanes, it * lanes, k, k), dtype=np.int8)
    padded[:o, :c] = weight
    blocks = padded.reshape(ot, lanes, it, lanes, k, k)      # ot, k_lane, it, c_lane, ky, kx
    return blocks.transpose(0, 4, 5, 2, 3, 1).reshape(ot * k * k * it, lanes, lanes)


@dataclass
class Placement:
    bank: int
    shape: tuple


@dataclass
class Compiled:
    cfg: AccelConfig
    descriptors: list
    weight_mem: np.ndarray
    param_mem: np.ndarray
    placements: dict
    input_name: str
    output_name: str
    output_channels: int
    layer_names: list
    layer_outputs: list
    weight_words: int = 0
    param_words: int = 0
    widths: list = field(default_factory=list)

    @property
    def program(self):
        return [d.encode() for d in self.descriptors]

    def input_words(self, image):
        return pack_activation(image, self.cfg.lanes)

    @property
    def input_bank(self):
        return self.placements[self.input_name].bank


def _allocate_banks(qmodel, shapes, cfg):
    last_use = {}
    for i, layer in enumerate(qmodel.layers):
        for name in (layer.src, getattr(layer, "res_src", None)):
            if name is not None:
                last_use[name] = i
    bank_of = {qmodel.input_name: 0}
    for i, layer in enumerate(qmodel.layers):
        if isinstance(layer, ConvLayer) and layer.out_mode == "s32":
            continue
        busy = {bank_of[t] for t in bank_of if last_use.get(t, -1) >= i}
        free = [b for b in range(cfg.num_act_banks) if b not in busy]
        if not free:
            raise RuntimeError(f"{layer.name}: no free activation bank (live: {sorted(busy)})")
        bank_of[layer.dst] = free[0]
    placements = {}
    for name, bank in bank_of.items():
        c, h, w = shapes[name]
        words = h * w * tiles(c, cfg.lanes)
        if words > cfg.act_bank_depth:
            raise RuntimeError(f"tensor {name} needs {words} words > bank depth {cfg.act_bank_depth}")
        placements[name] = Placement(bank, shapes[name])
    return placements


def compile_model(qmodel, cfg=AccelConfig()):
    lanes = cfg.lanes
    shapes = qmodel.shapes()
    placements = _allocate_banks(qmodel, shapes, cfg)
    weight_mem = np.zeros((cfg.weight_depth, lanes, lanes), dtype=np.int8)
    param_mem = np.zeros((cfg.param_depth, len(PARAM_ROWS), lanes), dtype=np.int64)
    w_ptr = p_ptr = 0
    descriptors, widths, names, outputs = [], [], [], []

    def place_weights(words):
        nonlocal w_ptr
        if w_ptr + len(words) > cfg.weight_depth:
            raise RuntimeError(f"weights exceed weight SRAM depth {cfg.weight_depth}")
        weight_mem[w_ptr:w_ptr + len(words)] = words
        base, w_ptr = w_ptr, w_ptr + len(words)
        return base

    for layer in qmodel.layers:
        src = placements[layer.src]
        in_c, in_h, in_w = shapes[layer.src]
        out_c, out_h, out_w = shapes[layer.dst]
        common = dict(
            name=layer.name, in_bank=src.bank, in_h=in_h, in_w=in_w, in_ct=tiles(in_c, lanes),
            out_h=out_h, out_w=out_w, out_ct=tiles(out_c, lanes),
        )

        if isinstance(layer, PoolLayer):
            if layer.m > 0xFFFF:
                raise RuntimeError(f"{layer.name}: pool multiplier does not fit the descriptor")
            desc = isa.Descriptor(
                op=isa.OP_GAP, out_bank=placements[layer.dst].bank,
                pool_m=layer.m, pool_shift=layer.shift, **common,
            )
            pool_sum = cfg.act_max * in_h * in_w
            widths.append({"layer": layer.name, "pool_sum_bits": int(pool_sum).bit_length(),
                           "pool_product_bits": int(pool_sum * layer.m).bit_length()})
        else:
            ot = tiles(out_c, lanes)
            if p_ptr + ot > cfg.param_depth:
                raise RuntimeError("parameters exceed param SRAM depth")
            if np.any(np.abs(layer.weight.astype(np.int64)) > cfg.weight_max):
                raise RuntimeError(f"{layer.name}: weights exceed {cfg.weight_bits}-bit symmetric range")
            desc = isa.Descriptor(
                op=isa.OP_CONV, kernel=layer.kernel, stride=layer.stride, pad=layer.pad,
                w_base=place_weights(pack_weights(layer.weight, lanes)), p_base=p_ptr,
                out_mode=isa.OUT_S32 if layer.out_mode == "s32" else isa.OUT_U8,
                res_mode={"none": isa.RES_NONE, "identity": isa.RES_IDENTITY,
                          "projection": isa.RES_PROJECTION}[layer.res_mode],
                **common,
            )
            if layer.out_mode == "s32":
                if ot * lanes > cfg.out_regs:
                    raise RuntimeError(f"{layer.name}: {out_c} logits exceed {cfg.out_regs} output registers")
            else:
                desc.out_bank = placements[layer.dst].bank
            if layer.res_mode != "none":
                res = placements[layer.res_src]
                desc.res_bank = res.bank
                rc, _, rw = shapes[layer.res_src]
                desc.res_w, desc.res_ct = rw, tiles(rc, lanes)
            if layer.res_mode == "projection":
                desc.res_stride = layer.res_stride
                desc.res_w_base = place_weights(pack_weights(layer.res_weight, lanes))

            flat = np.zeros((len(PARAM_ROWS), ot * lanes), dtype=np.int64)
            flat[4] = 1                                  # unused lanes: shift 1, m 0 -> output 0
            flat[0, :out_c] = layer.bias
            flat[2, :out_c] = layer.m
            flat[4, :out_c] = layer.shift
            if layer.res_mode != "none":
                flat[3, :out_c] = layer.res_m
            if layer.res_mode == "projection":
                flat[1, :out_c] = layer.res_bias
            param_mem[p_ptr:p_ptr + ot] = flat.reshape(len(PARAM_ROWS), ot, lanes).transpose(1, 0, 2)
            p_ptr += ot

            widths.append(_conv_widths(layer, flat[:, :out_c], cfg))
        descriptors.append(desc)
        names.append(layer.name)
        outputs.append(layer.dst)

    descriptors.append(isa.Descriptor(op=isa.OP_END, name="end"))
    if len(descriptors) > cfg.instr_depth:
        raise RuntimeError("program exceeds instruction memory")
    for d in descriptors:
        d.encode()                                        # raises if any field overflows
    return Compiled(
        cfg, descriptors, weight_mem, param_mem, placements, qmodel.input_name,
        qmodel.output_name, shapes[qmodel.output_name][0], names, outputs, w_ptr, p_ptr, widths,
    )


def _conv_widths(layer, params, cfg):
    """Worst-case magnitudes over all uint8 inputs; raise if a register could overflow."""
    bias_main, bias_res, m_main, m_res, shift = params
    if np.any(np.abs(bias_main) >= 2 ** (cfg.bias_bits - 1)) or np.any(np.abs(bias_res) >= 2 ** (cfg.bias_bits - 1)):
        raise RuntimeError(f"{layer.name}: bias exceeds {cfg.bias_bits} bits")
    if np.any(m_main > cfg.mult_max) or np.any(m_res > cfg.mult_max) or np.any(shift > cfg.shift_max) or np.any(shift < 1):
        raise RuntimeError(f"{layer.name}: multiplier/shift out of range")
    acc = cfg.act_max * np.abs(layer.weight.astype(np.int64)).sum(axis=(1, 2, 3))
    main = acc + np.abs(bias_main)
    operand = np.zeros_like(main)
    if layer.res_mode == "identity":
        operand = np.full_like(main, cfg.act_max)
    elif layer.res_mode == "projection":
        operand = cfg.act_max * np.abs(layer.res_weight.astype(np.int64)).sum(axis=(1, 2, 3)) + np.abs(bias_res)
    report = {
        "layer": layer.name,
        "acc_bits": bits_for_signed(max(acc.max(), operand.max() if layer.res_mode == "projection" else 0)),
        "acc_plus_bias_bits": bits_for_signed(max(main.max(), operand.max())),
    }
    if layer.out_mode == "u8":
        report["ppu_sum_bits"] = bits_for_signed((main * m_main + operand * m_res).max())
    for key in ("acc_bits", "acc_plus_bias_bits"):
        if report[key] > cfg.acc_bits:
            raise RuntimeError(f"{layer.name}: worst case needs {report[key]}-bit accumulators > {cfg.acc_bits}")
    return report
