# Testbench for the shared Conv0 engine

Design source: [conv0_serial_dsp.v](conv0_serial_dsp.v).
Simulation source: [tb_conv0_serial_dsp.sv](tb_conv0_serial_dsp.sv).
Simulation top: **tb_conv0_serial_dsp**.

The user-reported Vivado run passed three synthetic frames and 79,872 comparisons;
the Python fixture was skipped. It checks raw INT32 convolution-plus-bias results, not
requantization or ReLU.

## Add the files manually

1. Add `conv0_serial_dsp.v` under **Design Sources**. Use the current file from
   this folder if your Vivado project contains an older imported copy.
2. Add `tb_conv0_serial_dsp.sv` under **Simulation Sources**. Its file type must
   be **SystemVerilog**.
3. Set **tb_conv0_serial_dsp** as the simulation top. The design/synthesis top
   is **conv0_serial_dsp**, not the testbench.
4. Run behavioral simulation and use `run all` in the simulation console.

The old `mac_int8.v`, `pixel.v`, `conv_1024_pixels.v`, and `tb_conv_1024.sv`
are not dependencies of this testbench. Do not use the old simulation setup
script: it selects the old top.

## Automatic setup in the Vivado project console

With your project open and any previous simulation closed, run:

```tcl
source {C:/Users/Ibrah/Desktop/ECE_364D/senior-design/resnet8_project/rtl/setup_xsim_serial.tcl}
launch_simulation -simset sim_1 -mode behavioral
```

The setup adds missing sources by reference, selects the new simulation top,
sets the artifact directory for the Python fixture, and chooses
`xsim_serial_no_waves.tcl`. The latter executes `run all` without automatically
loading waveforms. The setup does not change the synthesis top or start a run
until you enter `launch_simulation`.

By default the script requires the full-Conv0 Python fixture. It falls back to
the checked-in `fixtures/` directory when generated artifacts are absent. For
synthetic tests only, set this before sourcing it:

```tcl
set resnet_serial_use_fixture 0
source {C:/Users/Ibrah/Desktop/ECE_364D/senior-design/resnet8_project/rtl/setup_xsim_serial.tcl}
launch_simulation -simset sim_1 -mode behavioral
```

Set `resnet_serial_use_fixture` back to `1` to restore the golden comparison.
If you add files manually instead, the testbench defaults to synthetic tests;
set its string parameter `ARTIFACT_DIR` to the absolute `resnet8/artifacts`
directory to enable the fixture. A simulator plusarg `ARTIFACT_DIR=...` also
overrides the parameter. Use forward slashes in paths.

## What the testbench checks

- Synchronous reset and reset priority over start.
- Reset during outstanding bias, image, and weight reads. The memory model
  flushes its pending response when reset is asserted.
- Reset while an output is stalled, followed by a fresh frame.
- Start pulses during busy do not restart the current calculation.
- Same-edge responses, one-cycle responses, and mixed zero-to-three-cycle
  responses with seeded request stalls.
- Request/address stability until accepted, valid address ranges, and at most
  one outstanding memory transaction across the engine.
- Output data/address stability under backpressure, including a directed
  six-cycle stall on the final output.
- Every output address appears exactly once: 26 channels, 1,024 values each.
- All accepted results equal an independent scalar, signed 64-bit reference.
- `done` pulses only after the last output is accepted, `busy` clears then,
  and the engine stays idle afterward.
- Unknown outputs, missing fixture words, and frame/global timeouts fail the run.

The three full synthetic frames use:

1. Zero inputs and large positive/negative INT32 biases to check bias loading.
2. Signed INT8 boundary patterns (-128, 127, -1, 1, 0) to check products/padding.
3. A deterministic random image and weights with memory/output stalls.

Each full frame compares **26,624 outputs**, addressed in HWC layout. The
scoreboard uses `out_addr`; it does not assume outputs arrive in ascending
address order. Stimulus changes on falling edges, transfers are sampled on
rising edges, and registered controls are checked after updates settle.

The optional fourth frame loads these files relative to `ARTIFACT_DIR`:

```text
layer_vectors/conv0/input.mem         3072 INT8 words
export/conv0_weights.mem               702 INT8 words
export/conv0_bias.mem                   26 INT32 words
layer_vectors/conv0/expected_acc.mem  26624 INT32 words
```

It checks the supplied Python result against the scalar reference before
comparing RTL outputs. This fixture covers all 26 output channels; the older
`conv_fixture/` contains only one output channel and is not suitable here.

## Progress and completion

A complete frame needs millions of clock cycles because the same datapath is
reused. The clock period is 10 ns. A default 1,000 ns or 10,000 ns simulation
duration is too short; use **run all**. Wall-clock runtime is not measured.

The testbench prints progress every 4,096 accepted outputs. A successful run
must end with a line shaped like:

```text
PASS: tb_conv0_serial_dsp frames=4 comparisons=106496 cycles=... python_fixture=1
```

Without the Python fixture, the expected totals are three complete frames,
79,872 comparisons, and `python_fixture=0`, with an explicit skip message.
These are expected success messages, not recorded simulation results.
Any mismatch invokes `$fatal` with the case, address, and expected/actual value.

The default deterministic seed is `2608cafe`. Parameter `SEED` or plusarg
`SEED=<hex>` changes it. `MAX_FRAME_CYCLES` defaults to 25 million; the global
watchdog is two seconds of simulated time, not two seconds of wall-clock time.
The testbench does not establish actual DSP mapping, synthesis utilization,
timing closure, or whole-network correctness.
