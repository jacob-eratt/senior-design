"""Export RTL verification vectors from the cycle-level model.

    python -m resnet8.export_vectors --image-index 0 --out resnet8/vectors

Writes ($readmemh format, one word per line, see accel/memfile.py):
  program.hex            descriptors, 128-bit
  weights.hex            weight SRAM image, L*L*8 bits per word
  params.hex             param SRAM image
  input_bank<b>.hex      input image as activation words, preloaded into bank b
  expected/NN_<layer>.hex  bank contents of each layer's output tensor
  expected/logits.txt    int32 logits (decimal), one per line
  write_trace.csv        every SRAM write: cycle, memory, address, data (with --trace)
  manifest.json          configuration, widths, memory map, field slices, cycle counts
"""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .accel import isa, perf
from .accel.compiler import compile_model, pack_activation
from .accel.config import AccelConfig
from .accel.hw import Core
from .accel.memfile import pack_param_word, pack_word, write_hex
from .accel.qmodel import QModel
from .data import load_cifar10
from .simulate import check_image

PACKAGE_DIR = Path(__file__).resolve().parent


def export(qmodel, image, out_dir, label=None, trace=False, cfg=AccelConfig()):
    out_dir = Path(out_dir)
    (out_dir / "expected").mkdir(parents=True, exist_ok=True)
    compiled = compile_model(qmodel, cfg)
    L = cfg.lanes
    core = Core(compiled, trace=trace)
    result, bad = check_image(core, qmodel, image)
    if bad:
        raise RuntimeError(f"cycle model disagrees with reference on {bad}; not exporting")

    write_hex(out_dir / "program.hex", compiled.program, isa.WORD_BITS)
    write_hex(out_dir / "weights.hex",
              [pack_word(w, cfg.weight_bits) for w in compiled.weight_mem[:compiled.weight_words]],
              cfg.weight_word_bits)
    write_hex(out_dir / "params.hex",
              [pack_param_word(w, cfg) for w in compiled.param_mem[:compiled.param_words]],
              cfg.param_word_bits)
    write_hex(out_dir / f"input_bank{compiled.input_bank}.hex",
              [pack_word(w, cfg.act_bits) for w in compiled.input_words(image)], cfg.act_word_bits)

    files = {}
    for i, (name, dst) in enumerate(zip(compiled.layer_names, compiled.layer_outputs)):
        got = result.snapshots[dst]
        if dst in compiled.placements:
            path = out_dir / "expected" / f"{i:02d}_{name}.hex"
            write_hex(path, [pack_word(w, cfg.act_bits) for w in pack_activation(got, L)], cfg.act_word_bits)
            files[name] = {"file": path.name, "bank": compiled.placements[dst].bank,
                           "words": len(pack_activation(got, L)), "shape_chw": list(got.shape)}
    (out_dir / "expected" / "logits.txt").write_text("\n".join(str(int(v)) for v in result.logits) + "\n")

    if trace:
        with open(out_dir / "write_trace.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cycle", "memory", "address", "data_hex"])
            for cycle, mem, addr, data in result.trace:
                writer.writerow([cycle, mem, addr, f"{pack_word(data, cfg.act_bits):0{cfg.act_word_bits // 4}x}"])

    rows, total = perf.analyze(compiled, qmodel)
    manifest = {
        "config": asdict(cfg),
        "word_bits": {"act": cfg.act_word_bits, "weight": cfg.weight_word_bits,
                      "param": cfg.param_word_bits, "descriptor": isa.WORD_BITS},
        "descriptor_fields": [{"name": n, "msb": m, "lsb": l} for n, m, l in isa.field_slices()],
        "descriptors": [{k: v for k, v in asdict(d).items()} for d in compiled.descriptors],
        "input_bank": compiled.input_bank,
        "layer_outputs": files,
        "weight_words": compiled.weight_words,
        "param_words": compiled.param_words,
        "datapath_widths": compiled.widths,
        "expected": {
            "logits": [int(v) for v in result.logits],
            "predicted_class": int(np.argmax(result.logits)),
            "label": None if label is None else int(label),
            "total_cycles": result.cycles,
            "layer_cycles": {layer.name: layer.cycles for layer in result.layers},
        },
        "analytic_total_cycles": total,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qmodel", type=Path, default=PACKAGE_DIR / "resnet8_int8.npz")
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--out", type=Path, default=PACKAGE_DIR / "vectors")
    parser.add_argument("--trace", action="store_true", help="also write the cycle-stamped SRAM write trace")
    args = parser.parse_args()

    images, labels = load_cifar10(train=False)
    manifest = export(QModel.load(args.qmodel), images[args.image_index], args.out,
                      labels[args.image_index], args.trace)
    exp = manifest["expected"]
    print(f"Wrote vectors for test image {args.image_index} to {args.out}")
    print(f"  expected class {exp['predicted_class']} (label {exp['label']}), {exp['total_cycles']:,d} cycles")


if __name__ == "__main__":
    main()
