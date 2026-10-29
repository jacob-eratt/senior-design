"""Version 1 little-endian memory image and a binary-only parameter loader."""

import json
from pathlib import Path

import numpy as np

from .common import sha256, write_json
from .integer import IntegerModel, same_padding

FIELDS = ["op", "output_id", "input0_id", "input1_id", "height", "width", "cin", "cout",
          "kernel", "stride", "out_height", "out_width", "pad_top", "pad_bottom", "pad_left", "pad_right",
          "relu", "weights_offset", "weights_count", "bias_offset", "bias_count", "quant_offset", "quant_count",
          "relu_output_id", "reserved24", "reserved25", "reserved26", "reserved27", "reserved28", "reserved29",
          "reserved30", "reserved31"]
OPS = {"conv": 1, "add": 2, "avgpool": 3, "dense": 4}


def write_mem(path, array):
    """One two's-complement hex word per line, in flattened C order."""
    a = np.asarray(array)
    bits = a.dtype.itemsize * 8
    mask = (1 << bits) - 1
    Path(path).write_text("".join(f"{int(v) & mask:0{bits // 4}x}\n" for v in a.flat), encoding="ascii")


def export_model(model, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    buffers = {"weights.bin": bytearray(), "biases.bin": bytearray(), "quant_params.bin": bytearray()}
    ids = {"input": 0}
    shapes = {"input": [32, 32, 3]}
    records, layer_info = [], []

    def tensor_id(name):
        if name not in ids:
            ids[name] = len(ids)
        return ids[name]

    for layer in model.layers:
        name, op = layer["name"], layer["op"]
        inputs = layer["inputs"]
        h, w, ci = shapes[inputs[0]]
        co, k, stride = ci, 1, 1
        pad = (0, 0, 0, 0)
        oh, ow = h, w
        wo = wc = bo = bc = 0
        if op in ("conv", "dense"):
            weights = layer["weights"]
            co = weights.shape[-1]
            if op == "conv":
                k, stride = weights.shape[0], layer["stride"]
                pad, (oh, ow) = same_padding(h, w, k, stride)
                packed = weights.transpose(3, 0, 1, 2).copy()  # OHWI
            else:
                oh = ow = 1
                packed = weights.T.copy()  # OI
            wo, wc = len(buffers["weights.bin"]), packed.size
            bo, bc = len(buffers["biases.bin"]), layer["bias"].size
            buffers["weights.bin"].extend(packed.astype("i1").tobytes())
            buffers["biases.bin"].extend(layer["bias"].astype("<i4").tobytes())
            write_mem(directory / f"{name}_weights.mem", packed)
            write_mem(directory / f"{name}_bias.mem", layer["bias"])
        elif op == "avgpool":
            k, stride, oh, ow = h, h, 1, 1
        qp = np.stack([layer["multiplier"], layer["shift"]], axis=-1).astype("<i4")
        qo, qc = len(buffers["quant_params.bin"]), len(qp)
        buffers["quant_params.bin"].extend(qp.tobytes())
        write_mem(directory / f"{name}_quant.mem", qp)
        output_id = tensor_id(name)
        relu_id = tensor_id(layer["relu_name"]) if op == "add" else -1
        record = [OPS[op], output_id, ids[inputs[0]], ids[inputs[1]] if len(inputs) > 1 else -1,
                  h, w, ci, co, k, stride, oh, ow, *pad, int(layer["relu"]), wo, wc, bo, bc, qo, qc, relu_id]
        record += [0] * (len(FIELDS) - len(record))
        records.append(record)
        shapes[name] = [oh, ow, co]
        if op == "add":
            shapes[layer["relu_name"]] = shapes[name]
        info = {"name": name, "op": op, "config": dict(zip(FIELDS, record)),
                "input_scales": [model.scales[n] for n in inputs], "output_scale": model.scales[name]}
        if "weight_scales" in layer:
            info["weight_scales"] = layer["weight_scales"].tolist()
            info["accumulator_abs_bound"] = layer["accumulator_abs_bound"]
        layer_info.append(info)
    buffers["layer_config.bin"] = np.asarray(records, dtype="<i4").tobytes()
    for name, data in buffers.items():
        (directory / name).write_bytes(data)
    write_mem(directory / "layer_config.mem", np.asarray(records, dtype=np.int32))
    manifest = {"format": "resnet8-int8-v1", "endianness": "little", "config_fields": FIELDS,
                "config_record_bytes": len(FIELDS) * 4, "tensor_ids": ids, "tensor_shapes_hwc": shapes,
                "scales": model.scales, "zero_point": 0, "weight_layout": "OHWI; dense OI",
                "activation_layout": "NHWC, C contiguous", "layers": layer_info,
                "metadata": model.metadata,
                "files": {name: {"bytes": len(data), "sha256": sha256(directory / name)}
                          for name, data in buffers.items()}}
    write_json(directory / "manifest.json", manifest)
    return manifest


def load_export(directory):
    """Reconstruct arithmetic parameters from the binary streams, not .mem/NPZ."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format"] != "resnet8-int8-v1" or manifest["config_fields"] != FIELDS:
        raise ValueError("Unsupported memory format")
    raw = {}
    for name, info in manifest["files"].items():
        if sha256(directory / name) != info["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
        raw[name] = (directory / name).read_bytes()
        if len(raw[name]) != info["bytes"]:
            raise ValueError(f"Size mismatch: {name}")
    names = {i: n for n, i in manifest["tensor_ids"].items()}
    records = np.frombuffer(raw["layer_config.bin"], "<i4").reshape(-1, len(FIELDS))
    layers = []
    for row in records:
        c = dict(zip(FIELDS, map(int, row)))
        op = {v: k for k, v in OPS.items()}[c["op"]]
        name = names[c["output_id"]]
        inputs = [names[c["input0_id"]]]
        if c["input1_id"] >= 0:
            inputs.append(names[c["input1_id"]])
        qp = np.frombuffer(raw["quant_params.bin"], "<i4", count=c["quant_count"] * 2,
                           offset=c["quant_offset"]).reshape(-1, 2)
        layer = {"name": name, "op": op, "inputs": inputs, "multiplier": qp[:, 0].copy(),
                 "shift": qp[:, 1].copy(), "relu": bool(c["relu"])}
        if op in ("conv", "dense"):
            weights = np.frombuffer(raw["weights.bin"], "i1", count=c["weights_count"], offset=c["weights_offset"])
            if op == "conv":
                weights = weights.reshape(c["cout"], c["kernel"], c["kernel"], c["cin"]).transpose(1, 2, 3, 0)
                expected, dims = same_padding(c["height"], c["width"], c["kernel"], c["stride"])
                if expected != tuple(c[n] for n in ("pad_top", "pad_bottom", "pad_left", "pad_right")) or dims != (c["out_height"], c["out_width"]):
                    raise ValueError("Inconsistent SAME padding configuration")
            else:
                weights = weights.reshape(c["cout"], c["cin"]).T
            layer["weights"] = weights.copy()
            layer["bias"] = np.frombuffer(raw["biases.bin"], "<i4", count=c["bias_count"], offset=c["bias_offset"]).copy()
            layer["stride"] = c["stride"]
        elif op == "add":
            layer["relu_name"] = names[c["relu_output_id"]]
        layers.append(layer)
    return IntegerModel(layers, manifest["scales"], manifest["metadata"])
