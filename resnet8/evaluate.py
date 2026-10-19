"""Accuracy of the bit-exact integer model vs. the float model on CIFAR-10 test.

    python -m resnet8.evaluate [--qmodel resnet8/resnet8_int8.npz]

The integer numbers come from accel.reference, which the cycle-level model is
verified against bit for bit, so they are the accuracy the hardware delivers.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .accel import reference
from .accel.qmodel import QModel
from .data import load_cifar10, to_float
from .model import load_resnet8

PACKAGE_DIR = Path(__file__).resolve().parent


def sqnr_db(ref, approx):
    noise = np.mean((ref - approx) ** 2)
    return float("inf") if noise == 0 else 10 * np.log10(np.mean(ref ** 2) / noise)


def compare(model, qmodel, images=None, labels=None, diag_images=500):
    from .quantize import folded_forward

    if images is None:
        images, labels = load_cifar10(train=False)
    with torch.inference_mode():
        float_logits = np.concatenate([
            model(to_float(images[i:i + 1000])).numpy() for i in range(0, len(images), 1000)
        ])
    int_logits = reference.predict(qmodel, images)
    float_pred, int_pred = float_logits.argmax(1), int_logits.argmax(1)
    deq = int_logits * qmodel.scales["logits"]

    print(f"\nCIFAR-10 test ({len(labels)} images)")
    print(f"  float accuracy          {np.mean(float_pred == labels):.2%}")
    print(f"  int8 accelerator        {np.mean(int_pred == labels):.2%}")
    print(f"  prediction agreement    {np.mean(float_pred == int_pred):.2%}")
    print(f"  logit SQNR              {sqnr_db(float_logits, deq):.1f} dB")

    # Where does quantisation error enter? Compare every tensor on a subset.
    sub = images[:diag_images]
    with torch.inference_mode():
        float_t = folded_forward(model, to_float(sub))
    _, int_t = reference.run(qmodel, sub, keep_intermediates=True)
    print("\n  tensor   scale       SQNR dB   saturated  zero")
    for name, value in int_t.items():
        if name == "input":
            continue
        f = float_t[name].numpy().reshape(value.shape)
        q = value.astype(np.float64) * qmodel.scales[name]
        sat = np.mean(value == 255) if value.dtype == np.uint8 else 0.0
        print(f"  {name:7s}  {qmodel.scales[name]:.3e}  {sqnr_db(f, q):8.1f}   {sat:8.3%}  {np.mean(value == 0):6.1%}")
    return {
        "float_accuracy": float(np.mean(float_pred == labels)),
        "int_accuracy": float(np.mean(int_pred == labels)),
        "agreement": float(np.mean(float_pred == int_pred)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=PACKAGE_DIR / "resnet8.pth")
    parser.add_argument("--qmodel", type=Path, default=PACKAGE_DIR / "resnet8_int8.npz")
    args = parser.parse_args()
    compare(load_resnet8(args.checkpoint), QModel.load(args.qmodel))


if __name__ == "__main__":
    main()
