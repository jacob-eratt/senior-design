# Portable Conv0 golden fixture

This small, checked-in subset of the generated artifacts lets teammates run
the real 26-channel Conv0 accumulator comparison without downloading CIFAR-10
or installing Python/TensorFlow. It uses CIFAR-10 test image index 3 (airplane)
and the frozen `pretrainedResnet26.h5` export. Dataset attribution:
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), Alex Krizhevsky,
Vinod Nair, and Geoffrey Hinton.

| File | Format |
|---|---|
| `layer_vectors/conv0/input.mem` | 3,072 signed INT8 HWC input values |
| `export/conv0_weights.mem` | 702 signed INT8 OHWI weights |
| `export/conv0_bias.mem` | 26 signed INT32 folded/quantized biases |
| `layer_vectors/conv0/expected_acc.mem` | 26,624 signed INT32 HWC results |

The directory intentionally mirrors the generated artifact layout. Set the
testbench's `ARTIFACT_DIR` to this `fixtures/` directory. The serial/416 Vivado
setup scripts automatically fall back here when full generated artifacts are
absent. Existing local artifacts take precedence; avoid mixing files from
different exports. Provenance and file hashes are in `manifest.json`.

This is a single-image, raw-accumulator fixture. It does not replace the full
ten-image software golden set or verify requantization/ReLU. Neither reported
serial nor 416-MAC simulation enabled this fixture yet; that hardware check
remains a required next step.
