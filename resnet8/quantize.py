"""Post-training quantisation of the float ResNet8 into an accelerator QModel.

    python -m resnet8.quantize                      # writes resnet8/resnet8_int8.npz

Scheme (what the hardware implements, see SPEC.md "Numerics"):
  * BatchNorm folded into the preceding convolution.
  * Activations: uint8, zero point 0, one scale per tensor. The input image is
    the raw pixel (scale 1/255). Every other tensor is post-ReLU, so no
    activation ever needs a zero point or a sign bit.
  * Weights: int8 symmetric in [-127, 127]; per output channel for
    convolutions, per tensor for the classifier (so argmax over the raw int32
    logits is valid without dequantising).
  * Biases: int32 at scale s_in * s_w[k].
  * Requantisation: real multipliers M = s_in * s_w[k] / s_out become
    (m, shift) pairs; a residual layer's two multipliers share one shift.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .accel.config import AccelConfig
from .accel.fixedpoint import quantize_multipliers
from .accel.qmodel import ConvLayer, PoolLayer, QModel
from .data import load_cifar10, to_float
from .model import ResNet8, load_resnet8

PACKAGE_DIR = Path(__file__).resolve().parent
TENSORS = ("x0", "s1m", "s1o", "s2m", "s2o", "s3m", "s3o")


def fold_bn(conv_bn):
    """ConvBN -> (weight [O,I,K,K], bias [O]) as float64 numpy arrays."""
    bn = conv_bn.bn
    gain = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().double()
    weight = conv_bn.conv.weight.detach().double() * gain[:, None, None, None]
    bias = bn.bias.detach().double() - bn.running_mean.double() * gain
    return weight.numpy(), bias.numpy()


def folded_layers(model: ResNet8):
    """Float parameters of every accelerator layer after BN folding."""
    params = {"stem": fold_bn(model.stem)}
    for i, stack in enumerate((model.stack1, model.stack2, model.stack3), start=1):
        params[f"s{i}c1"] = fold_bn(stack.conv1)
        params[f"s{i}c2"] = fold_bn(stack.conv2)
        if isinstance(stack.shortcut, torch.nn.Conv2d):
            params[f"s{i}sc"] = (stack.shortcut.weight.detach().double().numpy(),
                                 stack.shortcut.bias.detach().double().numpy())
    params["fc"] = (model.fc.weight.detach().double().numpy()[:, :, None, None],
                    model.fc.bias.detach().double().numpy())
    return params


@torch.inference_mode()
def folded_forward(model, x):
    """Float forward pass through the folded graph, returning every named tensor."""
    p = {k: (torch.from_numpy(w).float(), torch.from_numpy(b).float()) for k, (w, b) in folded_layers(model).items()}
    conv = torch.nn.functional.conv2d
    t = {"input": x}
    t["x0"] = torch.relu(conv(x, *p["stem"], padding=1))
    src = "x0"
    for i, stack in enumerate((model.stack1, model.stack2, model.stack3), start=1):
        stride = stack.conv1.conv.stride[0]
        t[f"s{i}m"] = torch.relu(conv(t[src], *p[f"s{i}c1"], stride=stride, padding=1))
        y = conv(t[f"s{i}m"], *p[f"s{i}c2"], padding=1)
        shortcut = conv(t[src], *p[f"s{i}sc"], stride=stride) if f"s{i}sc" in p else t[src]
        t[f"s{i}o"] = torch.relu(y + shortcut)
        src = f"s{i}o"
    t["pooled"] = t[src].mean(dim=(2, 3), keepdim=True)
    t["logits"] = conv(t["pooled"], *p["fc"]).flatten(1)
    return t


def choose_scale(values, method):
    """Scale for a non-negative tensor quantised to uint8 [0, 255]."""
    values = np.asarray(values, dtype=np.float64).ravel()
    top = float(values.max())
    if top <= 0:
        return 1.0
    if method == "max":
        return top / 255.0
    if method.startswith("p"):                         # e.g. p99.99
        return max(float(np.percentile(values, float(method[1:]))), 1e-12) / 255.0
    if method == "mse":                                # clip value minimising quantisation MSE
        sample = values if len(values) <= 400_000 else np.random.default_rng(0).choice(values, 400_000, replace=False)
        best = None
        for clip in top * np.linspace(0.2, 1.0, 81):
            scale = clip / 255.0
            err = np.mean((np.clip(np.round(sample / scale), 0, 255) * scale - sample) ** 2)
            if best is None or err < best[0]:
                best = (err, scale)
        return best[1]
    raise ValueError(f"unknown calibration method {method}")


def calibrate(model, images_u8, method, batch_size=500):
    collected = {name: [] for name in TENSORS}
    for start in range(0, len(images_u8), batch_size):
        t = folded_forward(model, to_float(images_u8[start:start + batch_size]))
        for name in TENSORS:
            collected[name].append(t[name].numpy().ravel())
    scales = {name: choose_scale(np.concatenate(v), method) for name, v in collected.items()}
    scales["input"] = 1.0 / 255.0
    scales["pooled"] = scales["s3o"]       # the pooling unit preserves scale
    return scales


def quantize_weight(weight, per_channel, cfg):
    qmax = cfg.weight_max
    if per_channel:
        scale = np.abs(weight).reshape(len(weight), -1).max(axis=1) / qmax
    else:
        scale = np.full(len(weight), np.abs(weight).max() / qmax)
    scale = np.where(scale > 0, scale, 1.0)
    q = np.clip(np.round(weight / scale[:, None, None, None]), -qmax, qmax).astype(np.int8)
    return q, scale


def quantize_bias(bias, scale):
    q = np.round(bias / scale)
    if np.any(np.abs(q) >= 2 ** 31):
        raise OverflowError("bias does not fit in int32")
    return q.astype(np.int32)


def build_qmodel(model, scales, cfg=AccelConfig()):
    params = folded_layers(model)
    layers = []

    def multipliers(layer, real_main, real_res=None):
        oc = len(real_main)
        layer.m, layer.shift = np.zeros(oc, np.int64), np.zeros(oc, np.int64)
        if real_res is not None:
            layer.res_m = np.zeros(oc, np.int64)
        for k in range(oc):
            group = [real_main[k]] if real_res is None else [real_main[k], real_res[k]]
            m, layer.shift[k] = quantize_multipliers(group, cfg.mult_bits, cfg.shift_bits)
            layer.m[k] = m[0]
            if real_res is not None:
                layer.res_m[k] = m[1]
        return layer

    def conv(name, src, dst, stride, **res):
        w, b = params[name]
        qw, sw = quantize_weight(w, True, cfg)
        layer = ConvLayer(name, src, dst, qw, quantize_bias(b, scales[src] * sw), stride, w.shape[2] // 2,
                          None, None, **res)
        return layer, scales[src] * sw / scales[dst]

    layer, real = conv("stem", "input", "x0", 1)
    layers.append(multipliers(layer, real))
    src = "x0"
    for i, stack in enumerate((model.stack1, model.stack2, model.stack3), start=1):
        stride = stack.conv1.conv.stride[0]
        layer, real = conv(f"s{i}c1", src, f"s{i}m", stride)
        layers.append(multipliers(layer, real))
        dst = f"s{i}o"
        if f"s{i}sc" in params:
            layer, real = conv(f"s{i}c2", f"s{i}m", dst, 1, res_mode="projection", res_src=src, res_stride=stride)
            wsc, bsc = params[f"s{i}sc"]
            layer.res_weight, ssc = quantize_weight(wsc, True, cfg)
            layer.res_bias = quantize_bias(bsc, scales[src] * ssc)
            real_res = scales[src] * ssc / scales[dst]
        else:
            layer, real = conv(f"s{i}c2", f"s{i}m", dst, 1, res_mode="identity", res_src=src)
            real_res = np.full(len(real), scales[src] / scales[dst])
        layers.append(multipliers(layer, real, real_res))
        src = dst

    pool_m, pool_shift = quantize_multipliers([1.0 / 64.0], cfg.mult_bits, cfg.shift_bits)
    layers.append(PoolLayer("pool", src, "pooled", int(pool_m[0]), pool_shift))

    w, b = params["fc"]
    qw, sw = quantize_weight(w, False, cfg)
    oc = len(b)
    layers.append(ConvLayer("fc", "pooled", "logits", qw, quantize_bias(b, scales["pooled"] * sw), 1, 0,
                            np.zeros(oc, np.int64), np.ones(oc, np.int64), out_mode="s32"))
    scales = dict(scales, logits=float(scales["pooled"] * sw[0]))
    return QModel("input", (3, 32, 32), layers, "logits", {k: float(v) for k, v in scales.items()})


def quantize(model, calib_images, method="mse", cfg=AccelConfig()):
    return build_qmodel(model, calibrate(model, calib_images, method), cfg)


def main():
    from .evaluate import compare

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=PACKAGE_DIR / "resnet8.pth")
    parser.add_argument("--output", type=Path, default=PACKAGE_DIR / "resnet8_int8.npz")
    parser.add_argument("--calib-images", type=int, default=2000)
    parser.add_argument("--method", default="mse", help="mse | max | p<percentile>, e.g. p99.99")
    args = parser.parse_args()

    model = load_resnet8(args.checkpoint)
    train_x, _ = load_cifar10(train=True)
    calib = train_x[np.random.default_rng(0).choice(len(train_x), args.calib_images, replace=False)]

    with torch.inference_mode():
        x = to_float(calib[:256])
        assert torch.allclose(folded_forward(model, x)["logits"], model(x), atol=1e-3), "BN folding mismatch"

    qmodel = quantize(model, calib, args.method)
    qmodel.save(args.output)
    print(f"Wrote {args.output}")
    for name in ("input", *TENSORS, "pooled", "logits"):
        print(f"  scale[{name:6s}] = {qmodel.scales[name]:.6g}")
    compare(model, qmodel)


if __name__ == "__main__":
    main()
