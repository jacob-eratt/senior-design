"""Recheck existing golden files and exports without regenerating expected data."""

import argparse
import json
from pathlib import Path

import numpy as np

from resnet8.common import ARTIFACTS, WEIGHTS, image_input, sha256


def read_tensors(directory):
    manifest = json.loads((directory / "tensors.json").read_text(encoding="utf-8"))
    tensors = {}
    for name, info in manifest.items():
        path = directory / f"{name}.npy"
        if sha256(path) != info["sha256"]:
            raise ValueError(f"Golden checksum mismatch: {path}")
        a = np.load(path, allow_pickle=False)
        if list(a.shape) != info["shape"] or a.dtype.str != info["dtype"]:
            raise ValueError(f"Golden shape/dtype mismatch: {path}")
        tensors[name] = a
    return tensors


def check_mem(path, expected):
    bits = expected.dtype.itemsize * 8
    words = np.asarray([int(line, 16) for line in path.read_text(encoding="ascii").splitlines()], np.uint64)
    actual = words.astype(f"u{bits // 8}").view(f"i{bits // 8}").reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected, err_msg=str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    from resnet8.fp32 import Reference
    from resnet8.export import load_export
    from resnet8.integer import conv_scalar
    reference = Reference()
    integer = load_export(args.artifacts / "export")
    golden = json.loads((args.artifacts / "golden" / "manifest.json").read_text(encoding="utf-8"))
    if golden["weights_sha256"] != sha256(WEIGHTS):
        raise ValueError("Golden weight identity mismatch")
    if golden["export_manifest_sha256"] != sha256(args.artifacts / "export" / "manifest.json"):
        raise ValueError("Golden/export identity mismatch")
    for record in golden["records"]:
        directory = args.artifacts / "golden" / f"{record['test_index']:05d}_{record['label']}"
        pixels = image_input(directory / "input.png")
        fp = read_tensors(directory / "fp32")
        np.testing.assert_array_equal(pixels, fp.pop("input_uint8"))
        actual_fp = reference.run(pixels)
        if fp.keys() != actual_fp.keys():
            raise ValueError("FP32 tensor set changed")
        for name in fp:
            np.testing.assert_allclose(actual_fp[name], fp[name], atol=3e-4, rtol=3e-4, err_msg=name)
        expected_int = read_tensors(directory / "integer")
        actual_int = integer.run(pixels)
        if expected_int.keys() != actual_int.keys():
            raise ValueError("Integer tensor set changed")
        for name in actual_int:
            np.testing.assert_array_equal(actual_int[name], expected_int[name], err_msg=name)
    for layer in integer.layers:
        name = layer["name"]
        base = args.artifacts / "export"
        qp = np.stack([layer["multiplier"], layer["shift"]], axis=-1)
        check_mem(base / f"{name}_quant.mem", qp)
        if "weights" in layer:
            w = layer["weights"]
            packed = w.transpose(3, 0, 1, 2) if layer["op"] == "conv" else w.T
            check_mem(base / f"{name}_weights.mem", packed)
            check_mem(base / f"{name}_bias.mem", layer["bias"])
    fixture = read_tensors(args.artifacts / "conv_fixture")
    np.testing.assert_array_equal(conv_scalar(fixture["input"], fixture["weights_hwio"], fixture["bias"]), fixture["expected_acc"])
    print(f"PASS: {len(golden['records'])} saved FP32/integer golden sets, binary/.mem parity, starter scalar oracle")


if __name__ == "__main__":
    main()
