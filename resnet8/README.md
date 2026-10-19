# ResNet8 accelerator: Python model of the RTL

This directory holds a bit-accurate, cycle-level Python model of an int8
inference accelerator for **ResNet8**, the MLPerf Tiny image-classification
benchmark (CIFAR-10). It covers the "design spec" stage of
[`projectflow.md`](../projectflow.md): the RTL, however it is produced, must
match this model, and the model produces the vectors to check it against.

- **[SPEC.md](SPEC.md)** is the microarchitecture specification: block
  diagram, numerics, memory layouts, instruction format, pipeline, performance
  and verification flow.
- **`accel/`** holds the model, which depends only on numpy.

## Quick start

From the repository root:

```powershell
python -m venv .venv; .venv\Scripts\activate
pip install -r resnet8/requirements.txt  # CPU torch: add --index-url https://download.pytorch.org/whl/cpu
python -m resnet8.data                   # CIFAR-10 -> data/cifar10_*.npz (HF mirror, ~1 min)

python -m resnet8.train --epochs 40      # float model -> resnet8/resnet8.pth (~1 h on a laptop CPU; checkpoint included)
python -m resnet8.quantize               # int8 model  -> resnet8/resnet8_int8.npz, prints float vs int accuracy
python -m resnet8.evaluate               # accuracy of the bit-exact integer model on all 10k test images
python -m resnet8.simulate --images 4    # cycle-level run, every layer checked bit-exact, perf report
python -m resnet8.export_vectors --image-index 0 --trace   # $readmemh vectors for an RTL testbench
python -m pytest resnet8/tests           # ~30 s, needs no dataset or checkpoint
```

## How the pieces fit

```
model.py ──train.py──► resnet8.pth ──quantize.py──► resnet8_int8.npz   (QModel: integer layers)
                                                         │
                         ┌───────────────────────────────┼─────────────────────────────┐
                         ▼                               ▼                             ▼
             accel/reference.py               accel/compiler.py                  accel/perf.py
             vectorized golden model    descriptors + SRAM images + checks     closed-form cycles,
             (accuracy on 10k images)                    │                     traffic, resources
                         │                               ▼                             │
                         │                      accel/hw/core.py                       │
                         │              cycle-stepped model: sequencer, SRAMs,         │
                         │              MAC array, accumulators, PPU, pool unit        │
                         └───────── must agree bit for bit ──┘└── must agree exactly ──┘
                                                         │
                                               export_vectors.py → RTL testbench
```

The reference and the cycle model are **independent implementations** of the
same integer specification.
- The reference runs whole tensors directly from the `QModel`.
- The cycle model only sees what the hardware sees: the packed SRAM images and
  the descriptor program. It fetches and decodes descriptors, generates
  addresses with loop counters, reads SRAM words through ports with real
  latency and port limits, and pushes tokens through pipeline registers.

When they agree on every layer's output, the packing, addressing, scheduling
and arithmetic are all checked.

| File | Role |
|---|---|
| `model.py`, `train.py`, `data.py` | float ResNet8 (77,946 params), training, uint8 CIFAR-10 cache |
| `quantize.py`, `evaluate.py` | BN folding, calibration, int8 PTQ → `QModel`; float vs int accuracy |
| `accel/config.py` | `AccelConfig`: array size, bit widths, memory depths, pipeline latencies, clock |
| `accel/fixedpoint.py` | wrap, rounding shift, multiplier quantization, PPU arithmetic (the numeric spec) |
| `accel/qmodel.py` | the integer network description; `random_qmodel` for tests |
| `accel/reference.py` | golden model |
| `accel/isa.py` | 128-bit layer descriptor format |
| `accel/compiler.py` | bank allocation, weight/param packing, worst-case width proofs |
| `accel/hw/` | `sram.py`, `sequencer.py`, `units.py` (MAC array, accumulators, PPU, pool), `core.py` (pipeline) |
| `accel/perf.py` | analytic performance, resource estimate, design-space exploration |
| `accel/memfile.py` | `$readmemh` packing |
| `simulate.py`, `export_vectors.py` | CLIs |

## Changing the design

`AccelConfig` drives everything. For example, `AccelConfig(lanes=8,
act_bank_depth=2048, weight_depth=2048)` gives an 8×8 design point, and
`mac_extra_latency=2` models a 3-stage adder tree. The compiler re-packs the
memories, the cycle model rebuilds itself, and the tests show that results are
still exact. Use `perf.explore(qmodel)` to sweep array sizes in microseconds.
