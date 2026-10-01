"""NumPy-only, deterministic integer datapath. No TensorFlow operations here.

INT8 symmetric tensors (zero point 0), INT32 MAC/bias, positive Q31
multiplier and right shift, nearest rounding with ties away from zero.
INT64 is used ONLY for requantization products and overflow validation.
"""

import math

import numpy as np


def round_away(x):
    x = np.asarray(x, np.float64)
    return np.copysign(np.floor(np.abs(x) + 0.5), x)


def quantize(x, scale):
    if np.any(np.asarray(scale) <= 0) or not np.all(np.isfinite(scale)):
        raise ValueError("Invalid quantization scale")
    return np.clip(round_away(np.asarray(x) / scale), -128, 127).astype(np.int8)


def multiplier_shift(ratio):
    """ratio ~= multiplier / 2**shift; one rounding, no double rounding."""
    multipliers, shifts = [], []
    for value in np.asarray(ratio).reshape(-1):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Requantization ratio must be finite and positive")
        fraction, exponent = math.frexp(float(value))
        multiplier = int(math.floor(fraction * (1 << 31) + 0.5))
        if multiplier == 1 << 31:
            multiplier //= 2
            exponent += 1
        shift = 31 - exponent
        if not 1 <= shift <= 62:
            raise ValueError(f"Ratio outside supported hardware range: {value}")
        multipliers.append(multiplier)
        shifts.append(shift)
    return np.asarray(multipliers, np.int32), np.asarray(shifts, np.int32)


def requantize_wide(x, multiplier, shift):
    x = np.asarray(x)
    if x.size and (x.min() < -(1 << 31) or x.max() > (1 << 31) - 1):
        raise OverflowError("Requantizer input does not fit INT32")
    product = x.astype(np.int64) * np.asarray(multiplier, np.int64)
    shift = np.asarray(shift, np.int64)
    if np.any(shift < 1) or np.any(shift > 62):
        raise ValueError("Right shift must be 1..62")
    rounded = (np.abs(product) + (np.int64(1) << (shift - 1))) >> shift
    return np.where(product < 0, -rounded, rounded)


def saturate(x, relu=False):
    return np.clip(x, 0 if relu else -128, 127).astype(np.int8)


def same_padding(height, width, kernel, stride):
    oh, ow = (height + stride - 1) // stride, (width + stride - 1) // stride
    ph = max((oh - 1) * stride + kernel - height, 0)
    pw = max((ow - 1) * stride + kernel - width, 0)
    return (ph // 2, ph - ph // 2, pw // 2, pw - pw // 2), (oh, ow)


def check_accumulator(weights, bias):
    """Worst-case absolute sum bounds every partial sum, for any INT8 input."""
    axes = tuple(range(weights.ndim - 1))
    bound = 128 * np.abs(weights.astype(np.int64)).sum(axis=axes) + np.abs(bias.astype(np.int64))
    if np.any(bound > np.iinfo(np.int32).max):
        raise OverflowError("Layer can overflow INT32 MAC/bias accumulator")
    return int(bound.max())


def quantize_parameters(w, b, input_scale, output_scale):
    axes = tuple(range(w.ndim - 1))
    ws = np.maximum(np.max(np.abs(w.astype(np.float64)), axis=axes) / 127.0, 1e-12)
    # Near-zero BN gamma can leave a constant-output channel. Reserve half
    # the accumulator range for bias instead of dropping the channel.
    ws = np.maximum(ws, np.abs(b.astype(np.float64)) / (input_scale * ((1 << 30) - 1)))
    qw = quantize(w, ws)
    qb_wide = round_away(b.astype(np.float64) / (input_scale * ws))
    if np.any(np.abs(qb_wide) > np.iinfo(np.int32).max):
        raise OverflowError("Bias out of INT32 range")
    qb = qb_wide.astype(np.int32)
    bound = check_accumulator(qw, qb)
    m, s = multiplier_shift(input_scale * ws / output_scale)
    return {"weights": qw, "bias": qb, "weight_scales": ws,
            "multiplier": m, "shift": s, "accumulator_abs_bound": bound}


def conv_accumulate(x, weights, bias, stride=1):
    """NHWC input, HWIO kernel; zero padding implements TensorFlow SAME."""
    if x.dtype != np.int8 or weights.dtype != np.int8 or bias.dtype != np.int32:
        raise TypeError("Convolution requires INT8 input/weights and INT32 bias")
    if stride not in (1, 2) or weights.shape[0] != weights.shape[1]:
        raise ValueError("Unsupported convolution")
    n, h, w, ci = x.shape
    k, _, wci, co = weights.shape
    if ci != wci or bias.shape != (co,):
        raise ValueError("Convolution shape mismatch")
    check_accumulator(weights, bias)
    (top, bottom, left, right), (oh, ow) = same_padding(h, w, k, stride)
    padded = np.pad(x, ((0, 0), (top, bottom), (left, right), (0, 0)))
    # Patch axes are ky,kx,ci, matching flatten(HWIO). INT32 GEMM only.
    patches = np.stack([padded[:, ky:ky + oh * stride:stride, kx:kx + ow * stride:stride, :]
                        for ky in range(k) for kx in range(k)], axis=3)
    patches = patches.reshape(n * oh * ow, k * k * ci).astype(np.int32)
    result = patches @ weights.reshape(-1, co).astype(np.int32)
    return (result + bias).reshape(n, oh, ow, co)


def conv_scalar(x, weights, bias, stride=1):
    """Independent slow Python-integer oracle for small convolution tests."""
    n, h, w, ci = x.shape
    k, _, _, co = weights.shape
    (top, _, left, _), (oh, ow) = same_padding(h, w, k, stride)
    out = np.empty((n, oh, ow, co), np.int32)
    for b in range(n):
        for y in range(oh):
            for xx in range(ow):
                for oc in range(co):
                    acc = int(bias[oc])
                    for ky in range(k):
                        iy = y * stride + ky - top
                        for kx in range(k):
                            ix = xx * stride + kx - left
                            if 0 <= iy < h and 0 <= ix < w:
                                for ic in range(ci):
                                    acc += int(x[b, iy, ix, ic]) * int(weights[ky, kx, ic, oc])
                    if not -(1 << 31) <= acc < (1 << 31):
                        raise OverflowError("Oracle INT32 overflow")
                    out[b, y, xx, oc] = acc
    return out


class IntegerModel:
    def __init__(self, layers, scales, metadata=None):
        self.layers = layers
        self.scales = scales
        self.metadata = metadata or {}

    def run(self, x, already_quantized=False):
        x = np.asarray(x)
        if x.ndim != 4 or x.shape[1:] != (32, 32, 3):
            raise ValueError("Expected NHWC [N,32,32,3]")
        if already_quantized:
            if x.dtype != np.int8:
                raise TypeError("Quantized input must be INT8")
            q = x
        else:
            if not np.all(np.isfinite(x)) or x.min() < 0 or x.max() > 255:
                raise ValueError("Expected RGB pixels in [0,255]")
            q = quantize(x, self.scales["input"])
        values = {"input": q}
        for layer in self.layers:
            name, op = layer["name"], layer["op"]
            source = values[layer["inputs"][0]]
            if op in ("conv", "dense"):
                if op == "conv":
                    acc = conv_accumulate(source, layer["weights"], layer["bias"], layer["stride"])
                else:
                    check_accumulator(layer["weights"], layer["bias"])
                    acc = source.reshape(len(source), -1).astype(np.int32) @ layer["weights"].astype(np.int32)
                    acc += layer["bias"]
                values[name + "_acc"] = acc
                values[name] = saturate(requantize_wide(acc, layer["multiplier"], layer["shift"]), layer["relu"])
            elif op == "add":
                # Align both branches before addition; do NOT clip either branch
                # before adding. Keep a wide temporary through cancellation.
                a = requantize_wide(source, layer["multiplier"][0], layer["shift"][0])
                b = requantize_wide(values[layer["inputs"][1]], layer["multiplier"][1], layer["shift"][1])
                total = a + b
                if total.min() < -(1 << 31) or total.max() >= (1 << 31):
                    raise OverflowError("Residual aligned sum overflow")
                values[name + "_aligned_sum"] = total.astype(np.int32)
                values[name] = saturate(total)
                # ReLU uses wide sum, then clips; the signed pre-ReLU dump is diagnostic.
                values[layer["relu_name"]] = saturate(total, True)
            elif op == "avgpool":
                acc = source.sum(axis=(1, 2), keepdims=True, dtype=np.int32)
                values[name + "_acc"] = acc
                values[name] = saturate(requantize_wide(acc, layer["multiplier"], layer["shift"]))
            else:
                raise ValueError(f"Unknown operation {op}")
        return values

    def probabilities(self, tensors):
        # Presentation only: no softmax is needed to select the hardware argmax.
        logits = tensors["logits"].astype(np.float64) * self.scales["logits"]
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)


def calibrate(reference, images, batch_size=32):
    """Max-absolute per-tensor calibration, using training data only."""
    maxima = {}
    for start in range(0, len(images), batch_size):
        values = reference.run(images[start:start + batch_size])
        for name, a in values.items():
            maxima[name] = max(maxima.get(name, 0.0), float(np.max(np.abs(a))))
    scales = {n: max(v / 127.0, 1e-12) for n, v in maxima.items()}
    scales["input"] = 255.0 / 127.0
    for i in (1, 2, 3):
        # One numeric scale for the signed add diagnostic and following ReLU.
        scales[f"block{i}_relu"] = scales[f"block{i}_add"]
    # Integer pool averages in the same quantization domain, with exact /64.
    scales["avgpool"] = scales["block3_relu"]
    return scales


def build_integer(reference, scales):
    from .fp32 import CONVS
    folded = reference.folded_weights()
    layers = []

    def weighted(name, source, output, op, stride, relu):
        w, b = folded[name]
        return {"name": output, "op": op, "inputs": [source], "stride": stride, "relu": relu,
                **quantize_parameters(w, b, scales[source], scales[output])}

    for name, source, _, _, _, stride, _, relu in CONVS:
        layers.append(weighted(name, source, name, "conv", stride, relu))
        if name == "block1_conv2" or name.endswith("shortcut"):
            i = int(name[5])
            inputs = [f"block{i}_conv2", "conv0" if i == 1 else name]
            output = f"block{i}_add"
            m, s = multiplier_shift([scales[n] / scales[output] for n in inputs])
            layers.append({"name": output, "op": "add", "inputs": inputs,
                           "relu_name": f"block{i}_relu", "multiplier": m, "shift": s, "relu": True})
    m, s = multiplier_shift([1 / 64])
    layers.append({"name": "avgpool", "op": "avgpool", "inputs": ["block3_relu"],
                   "multiplier": m, "shift": s, "relu": False})
    layers.append(weighted("dense", "avgpool", "logits", "dense", 1, False))
    return IntegerModel(layers, scales)
