# Profiling MNIST inference

Use the Python environment with PyTorch and torchvision installed. From this folder:

```powershell
python profile_mnist.py --device cpu --download
```

This loads the existing checkpoint, evaluates all 10,000 test images, warms up the model, measures a separate uninstrumented baseline, and profiles every leaf layer. No retraining is needed. `--download` allows the initial MNIST download; omit it once the dataset is present.

For an NVIDIA GPU, run a separate comparison:

```powershell
python profile_mnist.py --device cuda --output-dir profiling_results/cuda
```

CPU and CUDA profiling are supported. Automatic selection prefers CUDA, otherwise CPU. For CPU experiments you can set `--threads 1`; keep the thread count consistent when comparing results. Use different output directories to retain multiple runs; the four report files are replaced on each run.

## Reports

- `summary.json`: full test accuracy, model-only mean batch time and throughput, hardware/software details, settings, and checkpoint hash.
- `layers.csv`: layers ranked by measured time, tensor shapes (including batch dimension), weight storage, output size, and multiply-accumulate counts per image.
- `operators.txt`: operator breakdown grouped by input shape, including memory statistics.
- `trace.json`: Chrome-format timeline for a compatible trace viewer.

The baseline repeats the same batch already on the device. It excludes loading, preprocessing, host/device transfers, and prediction selection. Its mean is amortized over many calls, not a distribution of individually synchronized request latencies. Batch size defaults to 1; use `--batch-size` to explore throughput separately.

Layer timings include profiler/hook overhead and should be used to locate bottlenecks, not substituted for the uninstrumented baseline. CPU totals include child operators; CUDA totals measure attributed device work. Percentages are fractions of summed layer time, not fractions of baseline wall time. Do not add nested operator totals to layer totals. Output byte counts are tensor sizes, not measured peak memory or allocation costs (flatten can share storage). MAC counts cover convolution and linear arithmetic, excluding biases, activation functions, and pooling. Dropout is inactive during evaluation.

## Choosing an FPGA workload

Start with the highest-time layers in `layers.csv`, then compare their MAC counts, weight storage, and intermediate tensor sizes. The second convolution has the most MACs in this model, but the measured ranking depends on the host and backend. CPU/GPU timing does not predict FPGA timing directly.

Before writing RTL, choose the FPGA and numerical precision, estimate transfer costs for the proposed hardware boundary, and validate any quantization against the reported test accuracy. Save reference inputs and layer outputs as the next step for hardware verification.

For a shorter timing pass (accuracy still uses the entire test set):

```powershell
python profile_mnist.py --device cpu --threads 1 --runs 20 --warmup 5 --profile-runs 5 --download
```
