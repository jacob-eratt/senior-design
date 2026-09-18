#!/usr/bin/env python3
"""Convert the trained MNIST PyTorch model into an hls4ml HLS project.

By default this script only generates synthesizable HLS C++ in ``hls4ml_prj``.
Compilation, numerical validation, and vendor HLS synthesis are opt-in because
they require additional local toolchain support.
"""

from __future__ import annotations

import argparse
import platform
import shutil
from pathlib import Path

import hls4ml
import numpy as np
import torch

from model import MNISTModel


REPO_ROOT = Path(__file__).resolve().parent
INPUT_SHAPE = (1, 28, 28)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert mnist_model.pth to an hls4ml HLS C++ project."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "mnist_model.pth",
        help="PyTorch state_dict checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "hls4ml_prj",
        help="Generated HLS project directory (default: %(default)s)",
    )
    parser.add_argument(
        "--project-name",
        default="mnist_cnn",
        help="HLS top-level project/function name (default: %(default)s)",
    )
    parser.add_argument(
        "--backend",
        default="Vitis",
        help="hls4ml backend, such as Vitis or Vivado (default: %(default)s)",
    )

    target = parser.add_mutually_exclusive_group()
    target.add_argument("--part", help="Exact FPGA part number")
    target.add_argument("--board", help="hls4ml board name")

    parser.add_argument(
        "--clock-period",
        type=float,
        default=5.0,
        help="Target clock period in ns (default: %(default)s)",
    )
    parser.add_argument(
        "--precision",
        default="ap_fixed<16,6>",
        help="Default fixed-point precision (default: %(default)s)",
    )
    parser.add_argument(
        "--reuse-factor",
        type=positive_int,
        default=1,
        help="Multiplier reuse factor; larger values save resources (default: %(default)s)",
    )
    parser.add_argument(
        "--strategy",
        choices=("Latency", "Resource"),
        default="Latency",
        help="HLS implementation strategy (default: %(default)s)",
    )
    parser.add_argument(
        "--io-type",
        choices=("io_parallel", "io_stream"),
        default="io_parallel",
        help="hls4ml interface/dataflow style (default: %(default)s)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the generated C++ model for software simulation",
    )
    parser.add_argument(
        "--validate-samples",
        type=positive_int,
        metavar="N",
        help="Compare PyTorch and HLS predictions on N MNIST test images",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run vendor HLS C simulation and synthesis after conversion",
    )
    parser.add_argument(
        "--cosim",
        action="store_true",
        help="Run RTL co-simulation as part of --build",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export synthesized IP as part of --build",
    )

    args = parser.parse_args()

    if args.clock_period <= 0:
        parser.error("--clock-period must be greater than zero")
    if (args.cosim or args.export) and not args.build:
        parser.error("--cosim and --export require --build")
    if args.strategy == "Resource" and args.reuse_factor == 1:
        parser.error("Resource strategy requires --reuse-factor greater than 1")

    return args


def load_model(checkpoint: Path) -> MNISTModel:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = MNISTModel()
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def create_hls_model(model: MNISTModel, args: argparse.Namespace):
    # hls4ml uses channels-last tensors internally. In io_parallel mode it can
    # insert the external NCHW -> NHWC transpose. io_stream cannot implement
    # that transpose, so its caller must provide channel-last input instead.
    channels_last_conversion = (
        "full" if args.io_type == "io_parallel" else "internal"
    )

    hls_config = hls4ml.utils.config_from_pytorch_model(
        model,
        input_shape=INPUT_SHAPE,
        granularity="name",
        backend=args.backend,
        default_precision=args.precision,
        default_reuse_factor=args.reuse_factor,
        channels_last_conversion=channels_last_conversion,
        transpose_outputs=False,
    )
    hls_config["Model"]["Strategy"] = args.strategy

    conversion_options = {
        "model": model,
        "hls_config": hls_config,
        "output_dir": str(args.output_dir),
        "project_name": args.project_name,
        "backend": args.backend,
        "io_type": args.io_type,
        "clock_period": args.clock_period,
    }
    if args.part:
        conversion_options["part"] = args.part
    if args.board:
        conversion_options["board"] = args.board

    return hls4ml.converters.convert_from_pytorch_model(**conversion_options)


def validate_model(
    pytorch_model: MNISTModel,
    hls_model,
    sample_count: int,
    io_type: str,
) -> None:
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset = datasets.MNIST(
        root=REPO_ROOT / "data",
        train=False,
        download=True,
        transform=transform,
    )
    sample_count = min(sample_count, len(dataset))
    loader = DataLoader(
        Subset(dataset, range(sample_count)),
        batch_size=sample_count,
        shuffle=False,
    )
    images, labels = next(iter(loader))

    with torch.inference_mode():
        pytorch_logits = pytorch_model(images).numpy()

    hls_input = images.numpy().astype(np.float32)
    if io_type == "io_stream":
        hls_input = np.transpose(hls_input, (0, 2, 3, 1))
    hls_input = np.ascontiguousarray(hls_input)
    hls_logits = np.asarray(hls_model.predict(hls_input)).reshape(sample_count, 10)

    labels_numpy = labels.numpy()
    pytorch_classes = np.argmax(pytorch_logits, axis=1)
    hls_classes = np.argmax(hls_logits, axis=1)

    print("\nValidation results")
    print(f"  Samples:              {sample_count}")
    print(f"  PyTorch accuracy:     {np.mean(pytorch_classes == labels_numpy):.2%}")
    print(f"  HLS accuracy:         {np.mean(hls_classes == labels_numpy):.2%}")
    print(f"  Prediction agreement: {np.mean(hls_classes == pytorch_classes):.2%}")
    print(f"  Mean absolute error:  {np.mean(np.abs(hls_logits - pytorch_logits)):.6f}")
    print(f"  Max absolute error:   {np.max(np.abs(hls_logits - pytorch_logits)):.6f}")


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint)

    print(f"Creating {args.backend} hls4ml model...")
    hls_model = create_hls_model(model, args)

    print(f"Writing HLS C++ project: {args.output_dir}")
    hls_model.write()

    needs_compilation = args.compile or args.validate_samples is not None
    if needs_compilation:
        print("Compiling generated C++ model...")
        try:
            hls_model.compile()
        except Exception as exc:
            if platform.system() == "Darwin":
                raise RuntimeError(
                    "The HLS project was generated successfully, but hls4ml's "
                    "fixed-point headers did not compile with the macOS toolchain. "
                    "Run --compile/--validate-samples on a supported Linux host."
                ) from exc
            raise

    if args.validate_samples is not None:
        validate_model(model, hls_model, args.validate_samples, args.io_type)

    if args.build:
        required_tool = {
            "vitis": "vitis_hls",
            "vivado": "vivado_hls",
        }.get(args.backend.lower())
        if required_tool and shutil.which(required_tool) is None:
            raise RuntimeError(
                f"--build requires {required_tool} to be installed and available on PATH"
            )
        print("Running vendor HLS build...")
        hls_model.build(
            csim=True,
            synth=True,
            cosim=args.cosim,
            export=args.export,
        )

    print("\nConversion complete.")
    print(f"Generated project: {args.output_dir}")
    print(f"Top-level C++:     {args.output_dir / 'firmware' / f'{args.project_name}.cpp'}")
    if not (args.part or args.board):
        print("Note: no --part or --board was supplied; the backend default target was used.")


if __name__ == "__main__":
    main()
