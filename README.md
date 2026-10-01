# UT Austin ECE Senior Design

## ResNet-8 convolution milestone

Python references, pretrained weights, export tools, convolution RTL, testbenches,
and a portable golden fixture are grouped in [resnet8_project/](resnet8_project/README.md).
The 416-MAC Conv0 engine passed the reported synthetic simulation; its synthesis
report and pretrained golden-fixture comparison remain pending. The immediate
milestone is one verified, implementable convolution, followed by transfer
overhead and full-network integration. See the
[team progress update](resnet8_project/PROGRESS_UPDATE.md) and
[architecture overview](resnet8_project/resnet8/TEAM_OVERVIEW.md).

The earlier MNIST/ResNet50 roadmap below is retained as project history.

## AI-assisted hardware design: Python to FPGA

This project develops and evaluates a repeatable workflow from a Python neural-network model to a specified architecture, verified synthesizable RTL, and an FPGA implementation. The Fall 2026 goal is to deploy complete RTL MNIST inference on the **AMD Kria KV260**, then scale the reusable architecture toward **complete quantized ResNet50 inference by the end of the semester**.

The engineering process is itself a major deliverable: explicit specifications, independent verification, reproducible builds, and measured evidence of how AI assistance affects development effort and implementation quality. Monday meetings review working evidence. ResNet50 work begins only after MNIST is deployed, validated, and benchmarked on KV260.

## Project guide

- [Fall 2026 roadmap and milestone acceptance gates](docs/ROADMAP.md)
- [Python-to-FPGA project flow](projectflow.md)
- [MNIST profiling instructions and measurement limits](PROFILING.md)
- [September kickoff summary](docs/meeting-notes/2026-09-kickoff.md)
- [Initial implementation issue backlog](docs/ISSUE_BACKLOG.md)

The project-flow document records the broader methodology and exploratory questions. This roadmap selects KV260 as the target and defines the current fall plan; implementation choices still require specification and validation.

## Current repository

The `schedule` branch contains the PyTorch MNIST model, saved checkpoint, training/evaluation scripts, and inference profiling tools. Quantized reference infrastructure, custom RTL, and board deployment are planned deliverables, not established results. The separate `hls4l` branch contains an existing HLS experiment; its C++ validation results do not establish custom RTL or KV260 correctness.

| File | Purpose |
|---|---|
| [model.py](model.py) | Shared `MNISTModel` definition |
| [mnist_model.pth](mnist_model.pth) | Existing saved model checkpoint |
| [train_mnist.py](train_mnist.py), [mnist.py](mnist.py) | Existing MNIST training/application scripts |
| [benchmark.py](benchmark.py) | Repeated-batch model inference timing |
| [profile_mnist.py](profile_mnist.py) | Full test-set accuracy, baseline timing, layer/operator profiling |

## Existing MNIST architecture

The following graph comes directly from `model.py`. Shapes omit the batch dimension and use channel-first order.

| Stage | Operation | Output shape |
|---|---|---|
| Input | Grayscale image | `1 x 28 x 28` |
| Features 1 | 3x3 convolution, 1 → 32 channels, padding 1; ReLU | `32 x 28 x 28` |
| Pool 1 | 2x2 max-pooling, stride 2 | `32 x 14 x 14` |
| Features 2 | 3x3 convolution, 32 → 64 channels, padding 1; ReLU | `64 x 14 x 14` |
| Pool 2 | 2x2 max-pooling, stride 2 | `64 x 7 x 7` |
| Classifier 1 | Flatten; linear 3136 → 128; ReLU | `128` |
| Training regularization | Dropout, probability 0.25; inactive in evaluation | `128` |
| Classifier 2 | Linear 128 → 10 | `10` logits |

Convolutions use stride 1 and include biases; both linear layers also include biases. There is no softmax in the model. Evaluation selects the largest logit. The profiling and benchmark scripts convert images to tensors and normalize using mean `0.1307` and standard deviation `0.3081`. Hardware arithmetic, tensor layout, rounding, saturation, and weight export must be specified explicitly before implementing the quantized equivalent.

## Run the existing baseline

Use a Python environment with PyTorch and torchvision installed, then run from the repository root:

```powershell
python profile_mnist.py --device cpu --download
```

This evaluates all 10,000 test images and writes `summary.json`, `layers.csv`, `operators.txt`, and `trace.json` under `profiling_results/`. No retraining is required. See [PROFILING.md](PROFILING.md) for shorter runs, device selection, and report interpretation. Baseline timing excludes preprocessing, transfers, and prediction selection; instrumented layer timings include profiling overhead. CPU/GPU measurements do not directly predict FPGA performance.

## Planned organization

Existing Python files and checkpoint stay at their current paths. A later, separately reviewed migration can introduce the following structure after updating imports, commands, and checkpoint paths together:

```text
docs/                 Roadmap, specifications, meeting notes, results
models/               Python models, training, quantized references
rtl/                  Reusable datapaths, controllers, top levels
verification/         Testbenches, vector generators, regressions
fpga/                 KV260 integration, constraints, build scripts
scripts/              Export, profiling, and automation entry points
```

Generated datasets, reports, weights, vectors, and FPGA artifacts need documented storage and provenance policies before that migration. ResNet50 will use configurable, time-multiplexed compute and memory resources across layers rather than 50 independent physical hardware layers.
