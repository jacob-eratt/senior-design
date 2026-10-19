"""Vectorised bit-exact functional model of the accelerator (the golden model).

Executes a QModel directly on whole batches with no knowledge of tiling, SRAM
layout or scheduling. The cycle-level model must agree with it bit for bit on
every intermediate tensor; it is also fast enough to score the full CIFAR-10
test set.

Convolutions are computed as float64 matrix products of integer-valued
operands. Every partial sum is an integer below 2**53 (asserted), so the BLAS
result is exact and the speed costs nothing in accuracy.
"""

import numpy as np

from .config import AccelConfig
from .fixedpoint import ppu_requantize, rounding_shift_right, wrap
from .qmodel import ConvLayer, PoolLayer

_EXACT_LIMIT = 2 ** 53


def conv_accumulate(x, weight, stride, pad):
    """Integer convolution x[N,C,H,W] * weight[O,C,K,K] -> int64 [N,O,OH,OW]."""
    n, c, h, w = x.shape
    o, wc, k, _ = weight.shape
    assert wc == c
    bound = c * k * k * float(np.abs(x).max(initial=0)) * float(np.abs(weight.astype(np.int64)).max(initial=0))
    assert bound < _EXACT_LIMIT, "partial sums could exceed float64's exact-integer range"
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    # Channels-last, so each kernel tap is one (pixels x C) @ (C x O) matmul.
    xp = np.pad(x.transpose(0, 2, 3, 1).astype(np.float64), ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    wf = weight.astype(np.float64).transpose(2, 3, 1, 0)          # ky, kx, C, O
    acc = np.zeros((n * oh * ow, o), dtype=np.float64)
    for ky in range(k):
        for kx in range(k):
            patch = xp[:, ky:ky + stride * (oh - 1) + 1:stride, kx:kx + stride * (ow - 1) + 1:stride]
            acc += np.ascontiguousarray(patch).reshape(-1, c) @ wf[ky, kx]
    return np.rint(acc).astype(np.int64).reshape(n, oh, ow, o).transpose(0, 3, 1, 2)


def _per_channel(v):
    return np.asarray(v, dtype=np.int64)[None, :, None, None]


def run_layer(layer, tensors, cfg):
    x = tensors[layer.src]
    if isinstance(layer, PoolLayer):
        total = x.astype(np.int64).sum(axis=(2, 3), keepdims=True)
        y = rounding_shift_right(total * layer.m, layer.shift)
        return np.clip(y, 0, cfg.act_max).astype(np.uint8)

    assert isinstance(layer, ConvLayer)
    acc = wrap(conv_accumulate(x, layer.weight, layer.stride, layer.pad), cfg.acc_bits)
    bias = _per_channel(layer.bias)
    if layer.out_mode == "s32":
        return wrap(acc + bias, cfg.acc_bits)

    operand = res_m = None
    if layer.res_mode == "identity":
        operand, res_m = tensors[layer.res_src], _per_channel(layer.res_m)
    elif layer.res_mode == "projection":
        acc_res = wrap(conv_accumulate(tensors[layer.res_src], layer.res_weight, layer.res_stride, 0), cfg.acc_bits)
        operand = wrap(acc_res + _per_channel(layer.res_bias), cfg.acc_bits)
        res_m = _per_channel(layer.res_m)
    y = ppu_requantize(
        acc, bias, _per_channel(layer.m), _per_channel(layer.shift), operand, res_m,
        acc_bits=cfg.acc_bits, out_min=0, out_max=cfg.act_max,
    )
    return y.astype(np.uint8)


def run(qmodel, images, cfg=AccelConfig(), keep_intermediates=False):
    """Run a batch of uint8 images [N,C,H,W].

    Returns int64 logits [N, classes], or (logits, {tensor name: array}) when
    keep_intermediates is set.
    """
    images = np.asarray(images)
    assert images.dtype == np.uint8 and images.shape[1:] == tuple(qmodel.input_shape)
    tensors = {qmodel.input_name: images}
    for layer in qmodel.layers:
        tensors[layer.dst] = run_layer(layer, tensors, cfg)
    logits = tensors[qmodel.output_name].reshape(len(images), -1).astype(np.int64)
    return (logits, tensors) if keep_intermediates else logits


def predict(qmodel, images, cfg=AccelConfig(), batch_size=500):
    """Logits for a large set of images, processed in memory-friendly batches."""
    return np.concatenate([
        run(qmodel, images[i:i + batch_size], cfg) for i in range(0, len(images), batch_size)
    ])
