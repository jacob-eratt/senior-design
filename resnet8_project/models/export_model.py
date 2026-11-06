"""Fold BatchNorm, calibrate INT8, export binary and simulator memory images."""

import argparse
from pathlib import Path

from resnet8.common import ARTIFACTS, WEIGHTS, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", nargs="?", type=Path, default=WEIGHTS)
    parser.add_argument("--calibration-count", type=int, default=512)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "export")
    args = parser.parse_args()
    if not 1 <= args.calibration_count <= 50000:
        parser.error("--calibration-count must be 1..50000")
    from resnet8.fp32 import Reference
    from resnet8.data import cifar10
    from resnet8.integer import calibrate, build_integer
    from resnet8.export import export_model
    reference = Reference(args.weights)
    (train, _), _ = cifar10(download=args.download)
    scales = calibrate(reference, train[:args.calibration_count])
    model = build_integer(reference, scales)
    model.metadata = {"weights_sha256": sha256(args.weights), "calibration_split": "CIFAR-10 training",
                      "calibration_indices": list(range(args.calibration_count)), "rounding": "nearest, ties away from zero"}
    export_model(model, args.output)
    print(f"Exported {len(model.layers)} operations to {args.output}")


if __name__ == "__main__":
    main()
