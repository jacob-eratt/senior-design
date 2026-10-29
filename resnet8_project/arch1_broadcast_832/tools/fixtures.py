"""Portable .mem fixture loader; independent software reference stays separate."""

import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures"


def load_int8_fixture(directory=DEFAULT_FIXTURE):
    """Load original exported quantization constants and independent INT8 golden."""
    result = load_fixture(directory)
    for key, name, bits, count in (
        ("quant", "export/conv0_quant.mem", 32, 52),
        ("expected_int8", "layer_vectors/conv0/expected_output.mem", 8, 26624),
    ):
        data = (Path(directory) / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != result["identity"]["files"][name]["sha256"]:
            raise ValueError(f"Fixture checksum mismatch: {name}")
        words = data.decode("ascii").split()
        if len(words) != count or any(len(w) != bits//4 for w in words):
            raise ValueError(f"Bad memory dimensions: {name}")
        values = np.array([int(w, 16) for w in words], dtype=np.int64)
        values = np.where(values >= 1 << (bits-1), values-(1 << bits), values)
        result[key] = values.astype(np.int8 if bits == 8 else np.int32)
    result["multiplier"] = result["quant"].reshape(26, 2)[:, 0]
    result["shift"] = result["quant"].reshape(26, 2)[:, 1]
    return result


def load_fixture(directory=DEFAULT_FIXTURE):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    def read(name, count, bits):
        path = directory / name
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest["files"][name]["sha256"]:
            raise ValueError(f"Fixture checksum mismatch: {name}")
        words = data.decode("ascii").split()
        if len(words) != count or any(len(w) != bits//4 for w in words):
            raise ValueError(f"Bad memory dimensions: {name}")
        values = np.array([int(w, 16) for w in words], dtype=np.int64)
        values = np.where(values >= 1 << (bits-1), values - (1 << bits), values)
        return values.astype(np.int8 if bits == 8 else np.int32)

    return {"image": read("layer_vectors/conv0/input.mem", 3072, 8),
            "weights": read("export/conv0_weights.mem", 702, 8),
            "bias": read("export/conv0_bias.mem", 26, 32),
            "expected": read("layer_vectors/conv0/expected_acc.mem", 26624, 32),
            "identity": manifest}
