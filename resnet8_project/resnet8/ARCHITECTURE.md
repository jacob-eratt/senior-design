# Phase 0: ResNet-8 architecture freeze, revision 1

The authoritative function is `resnet_v1_eembc(conv_filters=26)` in
`upstream/keras_model.py`, from MLCommons Tiny revision
`4addd0fa08d216e20637637874e084895f289da4`.

Source SHA-256: `52a7564de25828abba7aa4aa785f1355513ad86079bad1ef9f26167073791f45`.

Selected checkpoint: `upstream/pretrainedResnet26.h5`, originally named
`pretrainedResnet.h5` at revision `eb78d0ebaf2c812ce13668f017a22171a38cd051`.

Checkpoint SHA-256: `7e469415849ea0be2d5a6cb988a0840e2ffc4184ee1bdff3442a2f1ae4774fdf`.

## Exact graph

Batch dimension is omitted below. Every convolution includes a bias, uses
TensorFlow `SAME` padding, and cross-correlation (no kernel reversal). All tensors
use channels-last layout. There is **no max pooling**.

| Operation | Kernel / stride | Cin → Cout | Output H×W×C | Following operations |
|---|---|---|---|---|
| Input | — | — | 32×32×3 | RGB float32, unnormalized pixels [0,255] |
| Conv0 | 3×3 / 1 | 3 → 26 | 32×32×26 | BN, ReLU |
| Block1 Conv1 | 3×3 / 1 | 26 → 26 | 32×32×26 | BN, ReLU |
| Block1 Conv2 | 3×3 / 1 | 26 → 26 | 32×32×26 | BN |
| Block1 Add | — | 26 | 32×32×26 | Add Conv0 identity, ReLU |
| Block2 Conv1 | 3×3 / 2 | 26 → 52 | 16×16×52 | BN, ReLU |
| Block2 Conv2 | 3×3 / 1 | 52 → 52 | 16×16×52 | BN |
| Block2 Shortcut | 1×1 / 2 | 26 → 52 | 16×16×52 | No BN, no activation |
| Block2 Add | — | 52 | 16×16×52 | Add main and shortcut, ReLU |
| Block3 Conv1 | 3×3 / 2 | 52 → 104 | 8×8×104 | BN, ReLU |
| Block3 Conv2 | 3×3 / 1 | 104 → 104 | 8×8×104 | BN |
| Block3 Shortcut | 1×1 / 2 | 52 → 104 | 8×8×104 | No BN, no activation |
| Block3 Add | — | 104 | 8×8×104 | Add main and shortcut, ReLU |
| Average pool | 8×8 / 8, VALID | 104 | 1×1×104 | Flatten to 104 |
| Dense | — | 104 → 10 | 10 | Bias, then softmax |

Each downsampling shortcut takes the preceding block's **post-add ReLU** output,
the same input as its main path. Seven BatchNorm layers use epsilon 0.001 and
their saved moving mean/variance. Inference always uses `training=False`.
There are nine physical convolution operations (seven on the main path, two
projections), three residual adds, and one Dense layer. The name “ResNet-8” does
not mean eight physical convolution operations.

Total Keras parameters including BatchNorm moving statistics: **205,306**.
Class order: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## Freeze policy

Do not change channels, kernel sizes, strides, graph connections, nonlinearities,
preprocessing, class order, or checkpoint during RTL development. The source and
checkpoint hashes are runtime guards. BatchNorm folding is a validated inference
transformation, not a topology change. Quantization introduces a separately
versioned numerical approximation; its scales, rounding and layout are frozen
with an export manifest before RTL work begins.

Any necessary change requires updating this specification and the relevant hash
guards, regenerating all golden vectors and exports, and rerunning software and
any existing RTL regressions. Do not silently switch to the incompatible
16-channel checkpoint currently shipped at the upstream master revision.
