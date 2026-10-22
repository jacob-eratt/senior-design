> **Publication status, October 21:** simulation, integrated routing/timing, and bitstream/XSA export have completed. Vitis compilation and board execution remain pending. Historical unrun statements below are superseded by [current status](CURRENT_STATUS.md).

# ResNet-8 team update — October 19, 2026

The selected 832-MAC Conv0 candidate now has source RTL, an AXI DMA/PS project
script, a standalone A53 benchmark, and deployment instructions. It includes
INT8 requantization and ReLU. Quantization constants are fixed at build time;
the incoming quantization packet verifies them. Image, weights and bias remain
runtime-loadable. **The new source has not been HDL-compiled/simulated,
synthesized, routed, Vitis-compiled or run on the board.**

For the October 20 meeting, start with the
Ericsson briefing (detailed file retained locally),
[latency/network analysis](reports/meeting_2026_10_20/ANALYSIS.md), and
[source review and validation gates](rtl/kv260_conv0/REVIEW_AND_VALIDATION.md).
The latest testbench adds raw INT32 checking, deliberate FIFO backpressure,
malformed packets and reset recovery; these new HDL checks have not run.
The UART summarizer now rejects incomplete/failed benchmark captures.
Fourteen additional ready-to-run fixture cases cover the ten archived images
and four synthetic arithmetic stress cases. Python independently matched the
archived raw/INT8 tensors and validated generated hashes; these are not HDL runs.

All 32 Python tests passed on October 19. The layout checker again matched
26,624 raw sums and 26,624 INT8 outputs, exercised 100,000 FIFO banking steps,
and checked C fixture hashes. These are Python checks, not proof of RTL behavior.

Export-derived full-network arithmetic totals 32,562,192 convolution/dense MAC
terms, of which Conv0 is 2.21%. New analytic latency budgets expose per-packet
DMA setup and CPU overhead sensitivity; they are assumptions, not measurements.
The earlier 19.47 us remains a Python model result. The next gate is the new
testbench PASS, followed by measured DSP mapping, integrated fit/routed timing,
and board golden comparisons and end-to-end latency.

The updates below are historical checkpoints and describe status on their dates.

# ResNet-8 team update — October 16, 2026

## New latency first INT8 Conv0 result

We extended the cycle level Python model so Conv0 now sends its trained INT8
activation values after the exported per-channel requantization, rounding,
ReLU and saturation. The model has AXI4 Stream style input/output handshakes
and AXI4 Lite start/status registers. Its per-clock trace records AXI signals,
core tap/address activity, local memory write bytes, and FIFO/backpressure.

The 540 main cold-frame architecture/transport combinations and 24 clock and
parameter-residency cases checked 15,015,936 raw INT32 values and the same
number of INT8 outputs against the fixture; every comparison passed. The
latency first candidate is 832 MAC lanes, 104 INT8 outputs per core beat, a
256-bit stream, and 16-byte-per-clock local loader. Under the selected 100 MHz,
2,000 MB/s, 2 us DMA setup assumptions it predicts 19.47 us end-to-end. A
slightly slower 832-MAC/52-output fallback predicts 20.63 us and places the
requantizer in DSPs to reduce its LUT estimate.

The DSP/LUT and banked-memory resource figures remain estimates. The fastest
candidate is close to the estimated LUT and BRAM limits. No RTL was generated,
no Vivado synthesis or HDL simulation was run, and neither 100 MHz closure nor
the assumed host bandwidth has been demonstrated. The next hardware step is
to implement/synthesize its INT8 multiplier plus LUT accumulator, requantizer,
and byte-banked buffers, then replace the host/link assumptions with measured
KV260 transfer rates. Details, reproducible commands, and traces are in
[AXI_INT8_STUDY.md](cycle_sim/AXI_INT8_STUDY.md).

The earlier October 1 update follows as a historical checkpoint.

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
