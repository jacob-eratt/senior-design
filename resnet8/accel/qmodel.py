"""Quantized network description: the contract between quantizer and hardware.

A QModel is a list of integer layers over named tensors. It knows nothing about
SRAM layout or scheduling; the compiler maps it onto the accelerator, and the
functional reference executes it directly. Keeping these two consumers
independent is what makes the cycle model vs. reference comparison meaningful.

Numerics (all tensors NCHW, activations uint8 with zero point 0):

    ConvLayer   acc[k]  = sum_{c,ky,kx} x[c, iy, ix] * weight[k, c, ky, kx]
                (padding positions contribute 0)
                res_mode "none":       operand = none
                res_mode "identity":   operand = res_src[k, oy, ox]          (uint8)
                res_mode "projection": operand = wrap(acc_res[k] + res_bias[k]),
                                        acc_res from a 1x1 conv of res_src
                out_mode "u8":  y = ppu_requantize(acc, bias, m, shift, operand, res_m) in [0, 255]
                out_mode "s32": y = wrap(acc + bias)                          (logits)
    PoolLayer   global average: y[c] = clamp(rshift_round(sum_{h,w} x[c,h,w] * m, shift), 0, 255)
"""

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

RES_MODES = ("none", "identity", "projection")
OUT_MODES = ("u8", "s32")


@dataclass
class ConvLayer:
    name: str
    src: str
    dst: str
    weight: np.ndarray            # int8  [OC, IC, K, K]
    bias: np.ndarray              # int32 [OC], scale = s_src * s_w[k]
    stride: int
    pad: int
    m: np.ndarray                 # int   [OC] requant multiplier (u8 mode)
    shift: np.ndarray             # int   [OC] requant shift, shared with res_m
    out_mode: str = "u8"
    res_mode: str = "none"
    res_src: str | None = None
    res_m: np.ndarray | None = None        # int [OC]
    res_weight: np.ndarray | None = None   # int8 [OC, IC_res, 1, 1] (projection)
    res_bias: np.ndarray | None = None     # int32 [OC] (projection)
    res_stride: int = 1

    @property
    def kernel(self):
        return self.weight.shape[2]

    @property
    def out_channels(self):
        return self.weight.shape[0]

    @property
    def in_channels(self):
        return self.weight.shape[1]

    def output_shape(self, in_shape):
        _, h, w = in_shape
        oh = (h + 2 * self.pad - self.kernel) // self.stride + 1
        ow = (w + 2 * self.pad - self.kernel) // self.stride + 1
        return (self.out_channels, oh, ow)

    def validate(self, shapes):
        assert self.weight.dtype == np.int8 and self.weight.ndim == 4
        assert self.weight.shape[2] == self.weight.shape[3], "square kernels only"
        assert shapes[self.src][0] == self.in_channels, f"{self.name}: channel mismatch"
        assert self.out_mode in OUT_MODES and self.res_mode in RES_MODES
        out_shape = self.output_shape(shapes[self.src])
        if self.res_mode == "identity":
            assert shapes[self.res_src] == out_shape, f"{self.name}: identity shape mismatch"
        if self.res_mode == "projection":
            c, h, w = shapes[self.res_src]
            assert self.res_weight.shape == (self.out_channels, c, 1, 1)
            assert ((h - 1) // self.res_stride + 1, (w - 1) // self.res_stride + 1) == out_shape[1:]
        if self.res_mode != "none":
            assert self.out_mode == "u8", "residual add requires requantised output"
        return out_shape


@dataclass
class PoolLayer:
    """Global average pool over the full spatial extent."""
    name: str
    src: str
    dst: str
    m: int
    shift: int

    def validate(self, shapes):
        return (shapes[self.src][0], 1, 1)


@dataclass
class QModel:
    input_name: str
    input_shape: tuple            # (C, H, W)
    layers: list
    output_name: str
    scales: dict = field(default_factory=dict)   # tensor name -> real value of one LSB

    def shapes(self):
        shapes = {self.input_name: tuple(self.input_shape)}
        for layer in self.layers:
            shapes[layer.dst] = layer.validate(shapes)
        return shapes

    def save(self, path):
        arrays, meta = {}, []
        for i, layer in enumerate(self.layers):
            entry = {"type": type(layer).__name__}
            for f in fields(layer):
                value = getattr(layer, f.name)
                if isinstance(value, np.ndarray):
                    arrays[f"{i}.{f.name}"] = value
                else:
                    entry[f.name] = value
            meta.append(entry)
        header = {
            "input_name": self.input_name, "input_shape": list(self.input_shape),
            "output_name": self.output_name, "scales": self.scales, "layers": meta,
        }
        np.savez(path, __meta__=np.array(json.dumps(header)), **arrays)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            header = json.loads(str(data["__meta__"]))
            layers = []
            for i, entry in enumerate(header["layers"]):
                kind = {"ConvLayer": ConvLayer, "PoolLayer": PoolLayer}[entry.pop("type")]
                for f in fields(kind):
                    key = f"{i}.{f.name}"
                    if key in data:
                        entry[f.name] = data[key]
                layers.append(kind(**entry))
        return cls(header["input_name"], tuple(header["input_shape"]), layers,
                   header["output_name"], header["scales"])


def random_qmodel(rng, input_shape=(3, 32, 32), widths=(16, 16, 32, 64), classes=10, calib_images=4):
    """ResNet8-topology QModel with random integer parameters.

    Each layer's multipliers are calibrated on random images so that every
    activation tensor spreads over the uint8 range (std ~40 LSB) instead of
    collapsing to zero or saturating. Used by the tests, which must not depend
    on a trained checkpoint; `input_shape`/`widths` allow odd sizes that
    exercise partial channel tiles.
    """
    from .config import AccelConfig
    from .fixedpoint import quantize_multipliers
    from .reference import conv_accumulate, run_layer

    cfg = AccelConfig()
    tensors = {"input": rng.integers(0, 256, size=(calib_images, *input_shape)).astype(np.uint8)}
    layers = []

    def add(layer):
        layers.append(layer)
        tensors[layer.dst] = run_layer(layer, tensors, cfg)

    def spread(values):
        return 40.0 / max(float(np.std(values)), 1.0) * rng.uniform(0.8, 1.2, size=values.shape[1])

    def conv(name, src, dst, oc, k, stride, res_mode="none", res_src=None, res_stride=1):
        x = tensors[src]
        weight = rng.integers(-127, 128, size=(oc, x.shape[1], k, k)).astype(np.int8)
        acc = conv_accumulate(x, weight, stride, k // 2)
        bias = (rng.normal(0.0, 0.5, size=oc) * acc.std()).astype(np.int32)
        real_main = spread(acc)
        layer = ConvLayer(name, src, dst, weight, bias, stride, k // 2,
                          np.zeros(oc, np.int64), np.zeros(oc, np.int64), res_mode=res_mode, res_src=res_src)
        real_res = None
        if res_mode == "identity":
            real_res = spread(tensors[res_src])
        elif res_mode == "projection":
            r = tensors[res_src]
            layer.res_weight = rng.integers(-127, 128, size=(oc, r.shape[1], 1, 1)).astype(np.int8)
            acc_res = conv_accumulate(r, layer.res_weight, res_stride, 0)
            layer.res_bias = (rng.normal(0.0, 0.5, size=oc) * acc_res.std()).astype(np.int32)
            layer.res_stride = res_stride
            real_res = spread(acc_res)
        if real_res is not None:
            layer.res_m = np.zeros(oc, np.int64)
            real_main, real_res = real_main * 0.7, real_res * 0.7
        for ch in range(oc):
            group = [real_main[ch]] if real_res is None else [real_main[ch], real_res[ch]]
            m, layer.shift[ch] = quantize_multipliers(group, cfg.mult_bits, cfg.shift_bits)
            layer.m[ch] = m[0]
            if real_res is not None:
                layer.res_m[ch] = m[1]
        add(layer)

    c0, c1, c2, c3 = widths
    conv("stem", "input", "x0", c0, 3, 1)
    src = "x0"
    for i, (cout, stride) in enumerate(((c1, 1), (c2, 2), (c3, 2)), start=1):
        conv(f"s{i}c1", src, f"s{i}m", cout, 3, stride)
        identity = stride == 1 and tensors[src].shape[1] == cout
        conv(f"s{i}c2", f"s{i}m", f"s{i}o", cout, 3, 1,
             res_mode="identity" if identity else "projection", res_src=src, res_stride=stride)
        src = f"s{i}o"
    _, _, h, w = tensors[src].shape
    pool_m, pool_shift = quantize_multipliers([1.0 / (h * w)], cfg.mult_bits, cfg.shift_bits)
    add(PoolLayer("pool", src, "pooled", int(pool_m[0]), pool_shift))
    fc_w = rng.integers(-127, 128, size=(classes, c3, 1, 1)).astype(np.int8)
    add(ConvLayer("fc", "pooled", "logits", fc_w, rng.integers(-5000, 5000, size=classes).astype(np.int32),
                  1, 0, np.zeros(classes, np.int64), np.ones(classes, np.int64), out_mode="s32"))
    return QModel("input", input_shape, layers, "logits")
