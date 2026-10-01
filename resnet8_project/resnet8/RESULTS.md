# Verified software results — 2026-09-30

Environment: Windows CPU, Python 3.12.2, TensorFlow 2.16.1, Keras 3.3.3,
NumPy 1.26.4. Complete installed-package record: [environment-verified.txt](environment-verified.txt).

The selected official 26-channel H5 checkpoint is identified in
[ARCHITECTURE.md](ARCHITECTURE.md). The upstream master checkpoint's 16-channel
mismatch was detected by loading it; it was not used for these results.

| Check | Observed result |
|---|---|
| Original loaded H5 vs instrumented model | All weights identical; output comparison passed |
| BatchNorm-folded FP32 vs original, all folded stages | Passed; largest absolute error about 2.20e-5 |
| Deterministic golden set | 10 images, first occurrence of each true class |
| Calibration | First 512 CIFAR-10 training images |
| Evaluation | First 1,000 held-out CIFAR-10 test images |
| FP32 accuracy | 892/1,000 = **89.2%** |
| INT8 accuracy | 879/1,000 = **87.9%** |
| Accuracy loss | **1.3 percentage points** |
| FP32/INT8 top-1 agreement | 949/1,000 = **94.9%** |
| Generated binary export reload | Every golden integer tensor bit-exact |
| Replay of saved golden files | Passed without regenerating expectations |
| Binary vs simulator `.mem` parameters | Exact equality |
| Starter 32×32, 3→1, 3×3 convolution | Bit-exact against independent scalar Python-integer oracle |
| Arithmetic unit tests | **9 passed**, including near-zero BN kernel/nonzero bias regression |
| CLI cat image, test index 0 | Both references predict cat |

The accuracy gate (FP32 ≥80%, INT8 drop ≤3 points) passed on this declared subset.
These are not measurements on the full 10,000-image test set or a hardware device.

The cat score was approximately 0.99998522 in FP32 and 0.99997524 after integer
inference and host-side softmax. Other images need not match these confidence values.

Exported parameter image sizes:

| File | Bytes |
|---|---:|
| `weights.bin` | 203,190 |
| `biases.bin` | 2,224 |
| `quant_params.bin` | 4,504 |
| `layer_config.bin` | 1,792 |

These counts exclude activation buffers, `.mem` text, manifests and any future
hardware alignment/banking overhead. The largest statically bounded accumulator
is 1,073,741,823; all layers fit signed INT32. Requantization requires a signed
64-bit temporary. The largest exported right shift is 57.

Machine-readable recorded results, errors and parameter hashes are in
[verification_baseline.json](verification_baseline.json). The fixed image labels,
both predictions, probabilities and source/export identities are in
[golden_baseline.json](golden_baseline.json). Generated arrays are in
`artifacts/golden/`; they are intentionally ignored by Git and reproducible with
`prepare_resnet.py`. `verify_resnet.py` checks those saved arrays and exports.

At the time of this software baseline, no RTL had been written or measured.
Subsequent RTL work and the user-reported Vivado results are documented in
[TEAM_OVERVIEW.md](TEAM_OVERVIEW.md). The new shared Conv0 engine and full-network
hardware acceptance gates remain open.
