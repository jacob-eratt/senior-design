# Reusing compute hardware for Conv0

The synthesized `conv_1024_pixels` experiment computes all 1,024 positions for
one output channel at once, using 27,648 multipliers. For the K26 target it
produced 2,916,423 LUTs, 850,272 registers, and 57,596 top-level I/O bits. The
report showed zero DSPs. That structure is useful as a parallel arithmetic
experiment; it is not an implementable KV260 architecture.

`conv0_serial_dsp.v` takes the area-oriented next step: it computes the full
32×32×26 Conv0 output tensor while reusing one signed 8×8 multiplier and one
INT32 accumulator. One 3×3 RGB dot product has 27 terms. The controller reads
those terms, accumulates one output value, then moves to the next pixel and
output channel. It performs 718,848 multiply-accumulate steps per input image.
This trades throughput for much lower hardware use.

## Memory interface

The module does not put the entire image or feature tensor on top-level ports.
An external BRAM controller, testbench memory, or later DMA bridge serves the
request/ready/valid ports. Image addresses are byte indexes in HWC order; filter
addresses are signed INT8 byte indexes in OHWI order; bias addresses are INT32
channel indexes. One read is outstanding at a time. Hold each request and address
until `ready`; return its data with a `valid` pulse. The module tolerates the
response on the request-acceptance edge or later. The output HWC word index and
INT32 accumulator remain valid until `out_ready` accepts them.

Use the exported, BN-folded Conv0 arrays: 3072 input bytes, 702 filter bytes
(26×3×3×3), and 26 signed INT32 biases. One `start` computes all 26 output
channels; output addresses run from 0 through 26623. The output is the raw
convolution-plus-bias accumulator. The separate integer reference next performs
per-channel requantization, rounding, INT8 saturation, and ReLU. Those operations
must be added before this result can feed the next network layer.

The `USE_DSP="yes"` attribute requests DSP mapping for the multiplier. Confirm
the resulting synthesis utilization report shows DSP use; do not infer success
from the attribute alone. A K26 has 1,248 DSP slices. ReLU and max-pool use
compare/select logic and do not need multipliers, so DSPs are most useful for
convolution and dense multiply-accumulate work.

This is one reusable Conv0 engine, not the whole ResNet accelerator. A complete
system still needs activation/weight/bias storage, layer control, requantization
and ReLU, residual buffering/adds, pooling, dense classification, and a practical
host/FPGA interface. Later convolution layers have different channel counts and
strides; extend the engine/configuration only after this version's output is
compared bit-for-bit with the integer reference.

## Adding this to the Vivado project

Add `conv0_serial_dsp.v` under **Design Sources**. For a separate resource test,
select top **`conv0_serial_dsp`**, then run synthesis and inspect `DSPs`, `CLB
LUTs`, and `CLB Registers` in `report_utilization`. It needs a small memory model
for behavioral simulation. The old top `conv_1024_pixels` remains available for
the previous parallel experiment.

The user-reported serial-core simulation passed three synthetic frames and
79,872 comparisons; its Python fixture was skipped. No synthesis report for
this core has been supplied. The synthesis report cited above belongs to the
parallel design, not this reused implementation.
