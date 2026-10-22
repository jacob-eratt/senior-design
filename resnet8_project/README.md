# ResNet-8 project

All ResNet-8 software, starter convolution RTL, tests, documentation, pretrained
assets, and generated golden vectors are grouped here.

Start with the [15-minute meeting walkthrough and task cards](onboarding/README.md).

See the [team architecture and development overview](resnet8/TEAM_OVERVIEW.md)
for the complete flow, current verification status, and remaining goals.
For cycle-by-cycle Python architecture experiments, start with
[cycle_sim/README.md](cycle_sim/README.md) and `simulate_conv0.py`.
The architecture study and recommendation (detailed file retained locally)
compares MAC counts, output widths, buffering, transport rates, and DSP/LUT choices.
The [latency-first INT8 and AXI study](cycle_sim/AXI_INT8_STUDY.md) models
requantization/ReLU, AXI4-Stream data handshakes and AXI4-Lite controls.
For a shareable status message, see [PROGRESS_UPDATE.md](PROGRESS_UPDATE.md).
The [KV260 RTL and AXI DMA project](rtl/kv260_conv0/README.md) now contains the
832-MAC candidate, INT8 output pipeline, Vivado build scripts, testbench, and
standalone A53 benchmark. Follow its
[board build and benchmark guide](rtl/kv260_conv0/KV260_BUILD_AND_BENCHMARK.md).
The new RTL passed simulation and integrated routing/timing; bitstream/XSA export
succeeded October 19. Vitis compilation, board correctness, and measured CPU/FPGA
latency remain pending. See [current status](CURRENT_STATUS.md).

```text
resnet8_project/
  run_resnet.py                 Image classification and tensor dumps
  prepare_resnet.py             Golden-vector generation and evaluation
  export_model.py               Quantization and hardware memory export
  verify_resnet.py              Saved-vector and export verification
  benchmark_conv0.py            CPU-only Conv0 timing with resident operands
  simulate_conv0.py             Clock-stepped RTL model, host timing, and sweeps
  explore_conv0.py              Reproduce the expanded architecture study
  simulate_axi_conv0.py         INT8 output, AXIS/AXI-Lite traces, latency/resource sweep
  cycle_sim/                   Core, transfer model, configs, resource accounting
  fixtures/                    Small portable Conv0 golden memory images
  reports/                     CPU measurements and reported RTL results
  requirements-resnet.txt       Pinned Python dependencies
  tests/                       Integer arithmetic tests
  rtl/
    mac_int8.v                 Signed INT8 MAC with INT32 accumulator
    pixel.v                    27 parallel MACs for one output pixel
    conv_1024_pixels.v          1024 parallel pixels, one output channel (area experiment)
    conv0_serial_dsp.v          One DSP MAC reused over full Conv0
    conv0_416_dsp.v             416 MAC lanes with local memories and streamed output
    tb_conv0_416_dsp.sv          Self-checking 416-lane Conv0 testbench
    CONV0_416.md                Architecture, ports, and Vivado setup for 416 lanes
    RESOURCE_REUSE.md           Resource tradeoff and memory protocol
    tb_conv_1024.sv             Self-checking convolution testbench
    tb_conv0_serial_dsp.sv      Shared Conv0 testbench with memory models
    SERIAL_SIMULATION.md       New Conv0 testbench setup and coverage
    SIMULATION.md              Vivado simulation setup and test coverage
  resnet8/
    fp32.py                    Instrumented TensorFlow reference
    integer.py                 Integer hardware reference
    export.py                  Binary and .mem format implementation
    common.py, data.py          Shared utilities and CIFAR-10 loading
    README.md                  Detailed setup and usage
    ARCHITECTURE.md             Frozen architecture
    HARDWARE_CONTRACT.md        Arithmetic, memory layout, future RTL gates
    RESULTS.md                 Measured verification results
    *_baseline.json            Recorded predictions and verification evidence
    environment-verified.txt   Installed-package record
    upstream/                  Official sources, weights, license, provenance
    artifacts/                 Generated vectors, exports, images, and dataset
```

Run from this directory:

```powershell
cd resnet8_project
..\.venv-resnet\Scripts\python.exe run_resnet.py resnet8/artifacts/golden/00000_cat/input.png
..\.venv-resnet\Scripts\python.exe verify_resnet.py
..\.venv-resnet\Scripts\python.exe -m unittest discover -s tests -v
```

The existing `.venv-resnet/` stays at the repository root because Python virtual
environments contain location-dependent launchers. Repository-wide Git settings
also remain at the root. Generated `artifacts/` content is ignored by Git.

[Detailed instructions](resnet8/README.md) · [Results](resnet8/RESULTS.md) ·
[Architecture](resnet8/ARCHITECTURE.md) · [Hardware contract](resnet8/HARDWARE_CONTRACT.md)

For the parallel RTL, follow [Vivado simulation instructions](rtl/SIMULATION.md).
For the reusable Conv0 engine, see [RESOURCE_REUSE.md](rtl/RESOURCE_REUSE.md).
The user-reported parallel-core simulation passed its synthetic tests; its Python
fixture was skipped, and synthesis exceeded device resources. The new shared
Conv0 engine has a [dedicated testbench](rtl/SERIAL_SIMULATION.md); the user-reported
run passed three synthetic frames (79,872 comparisons) with the Python fixture
skipped. Synthesis results for that engine remain pending.

The [416-MAC Conv0 engine and testbench](rtl/CONV0_416.md) passed the user-reported
five-frame synthetic run (133,120 comparisons). The Python fixture was skipped;
synthesis was later found to use 832 DSPs and 16 BRAM36 tiles, with standalone
I/O exceeding the device limit. Implemented timing remains pending. The Vivado setup scripts
can use the checked-in [portable fixture](fixtures/README.md) on a fresh clone.
See [recorded measurements](reports/README.md) for evidence and CPU timing scope.
