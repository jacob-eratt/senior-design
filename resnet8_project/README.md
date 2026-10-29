# ResNet-8 project

Hardware acceleration of INT8 ResNet-8 (CIFAR-10) on the AMD KV260. The shared
software golden model sits at the top; each hardware architecture has its own
folder with an `ARCHITECTURE.md`.

```text
resnet8_project/
  resnet8/                    Shared golden model (all architectures verify against it)
    fp32.py, integer.py         FP32 TensorFlow reference, bit-exact INT8/INT32 reference
    export.py, common.py, data.py
    HARDWARE_CONTRACT.md        Arithmetic, rounding, layouts, binary export format
    README.md, RESULTS.md       Setup and verified software results
    upstream/, artifacts/       Official weights/sources; generated vectors (git-ignored)
  run_resnet.py               Classify an image, dump tensors
  prepare_resnet.py           Generate golden vectors
  export_model.py             Fold BatchNorm, quantize, export weights/bias/quant/config
  verify_resnet.py            Recheck golden files and exports
  benchmark_conv0.py          CPU-only Conv0 timing baseline
  fixtures/                   Portable Conv0 golden memory images (used by RTL testbench)
  tests/                      Integer reference and UART-capture tests
  uarch_study/                Step-by-step microarchitecture discussion and decision log

  arch1_broadcast_832/        EXISTING: Conv0-only, 832 broadcast MACs, 100 MHz
    ARCHITECTURE.md             How it works (diagrams in diagrams/)
    CURRENT_STATUS.md           Measured simulation/synthesis/routing status
    README.md                   RTL, AXI DMA interface, register map
    KV260_BUILD_AND_BENCHMARK.md, REVIEW_AND_VALIDATION.md
    src/ sim/ scripts/ tools/ cpu/   RTL, testbench, Vivado Tcl, Python helpers, A53 benchmark
    docs/                       AXI/INT8 design study, technical evidence map
    reports/                    Archived Vivado/XSim reports, meeting analysis, CPU baseline

  arch2_os_systolic_26x32/    PROPOSED: output-stationary systolic conv engine, all layers
    ARCHITECTURE.md
  arch3_ws_tpu_27x26/         ALTERNATIVE: weight-stationary (TPU-style) conv engine
    ARCHITECTURE.md
```

Comparison of the three: [uarch_study/03b_performance_comparison.md](uarch_study/03b_performance_comparison.md).

Run from this directory:

```powershell
cd resnet8_project
..\.venv-resnet\Scripts\python.exe run_resnet.py resnet8/artifacts/golden/00000_cat/input.png
..\.venv-resnet\Scripts\python.exe verify_resnet.py
..\.venv-resnet\Scripts\python.exe -m unittest discover -s tests -v
```

The `.venv-resnet/` environment stays at the repository root. Generated
`resnet8/artifacts/` content is ignored by Git.

Earlier work retained from `main` in `rtl/`, `reports/`, and the historical onboarding notes includes the
RTL experiments (single-pixel, 1024-pixel, serial and 416-MAC Conv0). The Python
cycle model `cycle_sim/` and its sweep scripts remain recoverable from `origin/resnet8-conv0`.
