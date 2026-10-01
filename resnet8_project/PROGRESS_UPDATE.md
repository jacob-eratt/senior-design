# ResNet-8 team update — October 1, 2026

Our immediate milestone is **one correct, implementable Conv0 accelerator**.
We will measure its resources and timing before adding host/FPGA transfer
overhead and expanding to the complete network.

**Software is ready.** We froze the MLPerf Tiny ResNet-8 architecture with
26/52/104 channels and the matching pretrained checkpoint. We have FP32 and
INT8 software references, intermediate tensor dumps, ten fixed golden images,
BatchNorm folding, and hardware-format weight/bias/quantization exports.
On the declared 1,000-image test subset, accuracy was 89.2% FP32 and 87.9% INT8.

**We have explored three RTL architectures.** The original 27,648-MAC design
computed one output channel at all 1,024 positions simultaneously, but used
about 2,490% of the K26's LUT capacity and no DSPs. A one-MAC version reused
hardware to compute all 26 Conv0 channels and passed 79,872 synthetic output
comparisons. The new version uses 416 MAC lanes arranged as 16 spatial
positions x 26 output channels, local operand memories, and four INT32 outputs
per accepted clock.

**The 416-MAC simulation passed five synthetic frames and 133,120 comparisons.**
Coverage includes signed arithmetic, bias, padding, tile boundaries, reset,
output stalls, and restart without reloading. It took 8,583 clocks for an
unstalled test frame, including six deliberate final-output stalls. The
underlying 8,577-cycle schedule corresponds to about 85.8 microseconds at an
assumed 100 MHz, excluding preload. Actual frequency and resource fit are not
known until synthesis and implementation. These numbers cover 718,848 MAC
operations for complete Conv0, not the whole network.

**A local CPU reference measurement is now available.** On the i7-13700H,
the current NumPy INT8/INT32 Conv0 reference took a median 1.26 ms across 500
warm calls. This is correctness-oriented software, not an optimized INT8
CPU baseline. We are not yet claiming an end-to-end FPGA speedup.

**Next steps:** review the 416-MAC synthesis report; run the included pretrained
Conv0 golden fixture (the reported RTL runs skipped it); verify implementation
timing; and add/check requantization and ReLU so the layer produces its final
INT8 output. Then measure loading, CPU–FPGA transfers, and control overhead.
After that, generalize convolution and add residual operations, average pooling,
and dense classification for full ResNet inference.

Everything is grouped under `resnet8_project/`: Python entry points at the top,
the model and architecture documents in `resnet8/`, Verilog/testbenches and
Vivado setup scripts in `rtl/`, a portable golden fixture in `fixtures/`, and
recorded measurements in `reports/`. Large generated datasets, build products,
and virtual environments are excluded from Git. Start with
[TEAM_OVERVIEW.md](resnet8/TEAM_OVERVIEW.md) and [CONV0_416.md](rtl/CONV0_416.md).
