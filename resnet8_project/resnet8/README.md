# Frozen ResNet-8 software reference

This implements Phases 0–3 and supplies software fixtures and an interface contract
for Phases 4–5. RTL development is underway: the parallel convolution passed the
user-reported synthetic simulation but exceeds the FPGA resources. The single-MAC
and 416-MAC Conv0 designs also passed reported synthetic tests; their pretrained
fixture checks and implementation evidence remain pending. See [TEAM_OVERVIEW.md](TEAM_OVERVIEW.md) for the
full design flow, current evidence, and remaining goals. Existing MNIST work is separate.
See [measured results](RESULTS.md): FP32 89.2%, INT8 87.9% on 1,000 held-out images.

## Setup and reproduce

From `resnet8_project/`, with Python 3.12 (run `cd resnet8_project` from the
repository root first). The existing Python environment stays at the repository
root, so its installed launchers continue to work:

```powershell
py -3.12 -m venv ..\.venv-resnet
..\.venv-resnet\Scripts\python.exe -m pip install -r requirements-resnet.txt
..\.venv-resnet\Scripts\python.exe -m unittest discover -s tests -v
..\.venv-resnet\Scripts\python.exe prepare_resnet.py --download
..\.venv-resnet\Scripts\python.exe verify_resnet.py
```

The last command verifies the checkpoint, downloads the checksum-checked official
CIFAR-10 binary archive if missing, compares the original H5 with the instrumented
graph, checks every folded FP32 stage, calibrates on the first **512 training**
images, and evaluates on the first **1,000 test** images. It also creates the ten
golden images, weights, all intermediates, binary exports, and convolution fixtures.
It takes several minutes on a CPU. Defaults are deterministic; no training or
random calibration selection occurs. `--eval-count 10000` evaluates the full test
set. `--calibration-count` changes the numerical model and requires regenerated
exports and golden vectors. Test data is never used to choose calibration scales.
`verify_resnet.py` rechecks the existing saved golden files, binary exports and
`.mem` files without regenerating expectations or downloading data.

Run the supplied cat image (CIFAR-10 test index 0):

```powershell
..\.venv-resnet\Scripts\python.exe run_resnet.py resnet8/artifacts/golden/00000_cat/input.png
..\.venv-resnet\Scripts\python.exe run_resnet.py resnet8/artifacts/golden/00000_cat/input.png --integer
```

With the environment activated, `python run_resnet.py cat.png` works. Both modes
print all ten class scores, the prediction, and the directory containing tensor
dumps. Input must be a 32×32 image. It is converted to RGB; other dimensions are
rejected rather than silently introducing an unspecified resizing operation.
FP32 input is **float32 RGB in [0,255], NHWC**, with no mean subtraction or `/255`.
Integer class scores are a host-side softmax of dequantized integer logits;
hardware needs only argmax, with the lowest class index winning ties.

Standalone export:

```powershell
..\.venv-resnet\Scripts\python.exe export_model.py resnet8/upstream/pretrainedResnet26.h5
```

The exporter rejects a different checkpoint hash. Retraining or replacing the
checkpoint requires a deliberate revision of the frozen model and revalidation.

## Official checkpoint mismatch, resolved

The [pinned upstream function](https://github.com/mlcommons/tiny/blob/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/image_classification/keras_model.py)
defaults to `resnet_v1_eembc(conv_filters=26)`. However, that revision's
`trained_models/pretrainedResnet.h5` contains **16/32/64** channels. It is retained
under its original name for provenance, and is **not used**.

The selected official [historical pretrainedResnet.h5](https://github.com/mlcommons/tiny/blob/eb78d0ebaf2c812ce13668f017a22171a38cd051/benchmark/training/image_classification/trained_models/pretrainedResnet.h5)
contains **26/52/104** channels and is saved locally as `upstream/pretrainedResnet26.h5`.
It is a pretrained checkpoint, not padded, remapped, retrained, or randomly initialized.
Its entire weight list and predictions are compared against a direct H5 load.

All downloaded file URLs, revisions, sizes, and SHA-256 hashes are in
[upstream/provenance.json](upstream/provenance.json). Vendored source is unmodified
and covered by [the upstream Apache-2.0 license](upstream/LICENSE.md).
`train.py`, `test.py`, and the upstream README are provenance references, not our
entry points. The upstream calibration-index file is retained for provenance;
our calibration selection is the explicitly recorded first 512 training images.

## Architecture freeze

See [ARCHITECTURE.md](ARCHITECTURE.md) for exact layer semantics and the change gate.
`resnet8/fp32.py` calls the actual vendored function and adds an observation model
over its tensors. It checks source/checkpoint hashes, convolution dimensions,
strides, biases, BatchNorm epsilon, pooling, activations, and residual connectivity.
It does not rewrite the original graph to get intermediate outputs.

Tensor names are intentionally explicit:

| Name | Meaning |
|---|---|
| `input_uint8` | Original pixels, saved beside FP32 golden tensors |
| `input` | Model input: FP32 [0,255] or quantized INT8 |
| `conv0`, `blockN_conv1` | Conv + BatchNorm + ReLU output |
| `blockN_conv2` | Conv + BatchNorm output, before residual add |
| `blockN_shortcut` | Projection Conv output; no BatchNorm or ReLU |
| `*_raw`, `*_bn` | Additional FP32 pre-BN convolution and post-BN outputs |
| `blockN_add` | Signed residual sum, before ReLU (INT8 diagnostic in integer mode) |
| `blockN_add_aligned_sum` | INT32 sum after branch scale alignment, before clipping |
| `blockN_relu` | Post-add ReLU output; the tensor consumed by the next stage |
| `avgpool` | NHWC `[N,1,1,104]` mean over the 8×8 map |
| `logits` | Dense output **before softmax**, `[N,10]` |
| `probabilities` | Original FP32 Dense softmax output |
| `*_acc` | Integer MAC + bias, or pooling sum, before requantization |

All `.npy` files preserve the batch dimension and include shape/dtype/SHA-256 in
`tensors.json`. Kernels in the folded `.npy` weight dump use Keras HWIO; hardware
exports use OHWI. Do not interchange them without transposing.

## Generated artifacts

`resnet8/artifacts/` is ignored by Git and can be regenerated. The downloaded source
and selected H5 are small enough to keep with the project. Generated content:

| Directory/file | Contents |
|---|---|
| `architecture.json` | Full Keras graph config, revisions, tensor shapes |
| `weights/original/` | Original Conv, BN, Dense parameters, including moving statistics |
| `weights/folded/` | Folded FP32 kernels and biases |
| `golden/` | Ten PNGs, labels, predictions, all FP32 and integer tensors |
| `export/` | Binary streams, per-layer `.mem`, manifest, quantization scales |
| `conv_fixture/` | Cin=3, Cout=1, 32×32, 3×3 stride-1 starter convolution |
| `layer_vectors/` | Input, INT32 accumulator, INT8 output for all nine real convolutions |
| `verification.json` | Accuracy, folding errors, binary round-trip and scalar-oracle results |
| `evaluation_predictions.json` | Ground truth and both predictions for each evaluated image |

Golden images are the first test image of each class, in class order. Their fixed
indices are `[3,6,25,0,22,12,4,13,1,11]`. Selection never filters by correctness.
The SHA-256 of the selected H5 and exported manifest is recorded with them.

## Integer reference and acceptance

[HARDWARE_CONTRACT.md](HARDWARE_CONTRACT.md) defines the complete numerical and
binary contracts. This is a **custom symmetric INT8 reference**, not a claim of
bit-exact equivalence to the MLPerf TFLite model. BatchNorm is folded, and all
convolution, dense, residual, pooling and requantization operations in
`resnet8/integer.py` use integer arithmetic. Float is used offline for calibration,
parameter generation, image quantization, error measurement, and displayed softmax.
`run_resnet.py --integer` needs NumPy and Pillow; it does not import TensorFlow.

The software accuracy gate is FP32 accuracy ≥80% and INT8 degradation ≤3 percentage
points on the declared evaluation set. This is a project smoke gate, not MLPerf
certification or evidence of full-test-set accuracy when only 1,000 are evaluated.
Both accuracy and intermediate numeric errors appear in the report. Reaching an
INT8 endpoint in the report is not necessarily clipping; exact endpoint values
also count. Full evaluation and a chosen fixed-point architecture remain necessary
before making deployment claims.

The integer tests exercise boundary arithmetic independently of TensorFlow:
negative half ties, clipping, per-channel requantization, overflow rejection,
SAME padding on odd/even images, 1×1/3×3 kernels, strides 1/2, and all input-channel
counts. A scalar Python-integer oracle checks the convolution implementation.
Binary round-trip checks compare **every integer golden tensor**, not just class
predictions. The starter real convolution also matches the scalar oracle exactly.
