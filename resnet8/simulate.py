"""Run CIFAR-10 images through the cycle-level accelerator model.

    python -m resnet8.simulate --images 4

For each image the cycle model's output of *every* layer is compared against
the functional reference (bit-exact required), then the measured cycle counts
are shown next to the closed-form performance model.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .accel import perf, reference
from .accel.compiler import compile_model
from .accel.config import AccelConfig
from .accel.hw import Core
from .accel.qmodel import QModel
from .data import CLASSES, load_cifar10

PACKAGE_DIR = Path(__file__).resolve().parent


def check_image(core, qmodel, image):
    """Run one image; return (RunResult, list of mismatching tensor names)."""
    result = core.run(image, snapshot=True)
    _, expected = reference.run(qmodel, image[None], core.cfg, keep_intermediates=True)
    bad = [name for name, got in result.snapshots.items()
           if not np.array_equal(got, expected[name][0].reshape(got.shape))]
    return result, bad


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qmodel", type=Path, default=PACKAGE_DIR / "resnet8_int8.npz")
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument("--start", type=int, default=0, help="first CIFAR-10 test index")
    parser.add_argument("--clock-mhz", type=float, default=200.0)
    args = parser.parse_args()

    qmodel = QModel.load(args.qmodel)
    compiled = compile_model(qmodel, AccelConfig(clock_mhz=args.clock_mhz))
    images, labels = load_cifar10(train=False)
    core = Core(compiled)

    print("Program:")
    for d in compiled.descriptors[:-1]:
        res = {0: "", 1: f" + identity(bank {d.res_bank})", 2: f" + 1x1 proj(bank {d.res_bank})"}[d.res_mode]
        dst = "logit regs" if d.out_mode == 1 else f"bank {d.out_bank}"
        print(f"  {d.name:6s} bank {d.in_bank} {d.in_h:2d}x{d.in_w:2d}x{d.in_ct * compiled.cfg.lanes:<3d}"
              f" -> {dst:10s} {d.out_h:2d}x{d.out_w:2d}{res}")
    print(f"Weight SRAM {compiled.weight_words}/{compiled.cfg.weight_depth} words, "
          f"param SRAM {compiled.param_words}/{compiled.cfg.param_depth} words\n")

    failures, measured = 0, None
    for index in range(args.start, args.start + args.images):
        start = time.perf_counter()
        result, bad = check_image(core, qmodel, images[index])
        elapsed = time.perf_counter() - start
        pred = int(np.argmax(result.logits))
        status = "bit-exact (all layers)" if not bad else f"MISMATCH in {bad}"
        failures += bool(bad)
        print(f"image {index:5d}: label {CLASSES[labels[index]]:10s} predicted {CLASSES[pred]:10s} "
              f"{result.cycles:,d} cycles  {status}  [{elapsed:.1f}s wall]")
        measured = [layer.cycles for layer in result.layers]

    print()
    print(perf.format_report(compiled, qmodel, measured))
    print("\nResources:")
    for key, value in perf.resources(compiled).items():
        print(f"  {key:24s} {value}")
    if failures:
        print(f"\n{failures} image(s) FAILED bit-exact comparison")
        sys.exit(1)


if __name__ == "__main__":
    main()
