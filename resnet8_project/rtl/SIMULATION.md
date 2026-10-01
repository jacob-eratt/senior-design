# Vivado simulation of the 1024-pixel convolution

Testbench: `tb_conv_1024.sv`. Simulation top: **`tb_conv_1024`**.
This is a SystemVerilog testbench for the three Verilog design modules. It checks
the raw INT32 accumulators; these modules do not implement requantization/ReLU.
The user-reported Vivado run passed 904,229 synthetic comparisons. Its Python
fixture was skipped. This old parallel implementation exceeded device capacity
in synthesis; see the newer [416-MAC design](CONV0_416.md).

## Add the files in Vivado

1. Under **Design Sources**, include `mac_int8.v`, `pixel.v`, and
   `conv_1024_pixels.v` (your file names may be different).
2. Use **Add Sources → Add or create simulation sources** to add
   `tb_conv_1024.sv`. Keep its file type **SystemVerilog**.
3. In **Simulation Sources**, right-click `tb_conv_1024` and choose **Set as Top**.
4. Choose **Run Simulation → Run Behavioral Simulation**.
5. Click **Run All**, or type `run all` in the simulation Tcl console. A default
   1000 ns run is too short for this suite. Wait for the final PASS or FAIL.

Names on the `module` declarations, rather than filenames, control connections.
The default declarations expected are `mac_int8`, `pixel`, `conv_1024_pixels`.
If your top declaration is `module conv_1024`, change the line near the start of
the testbench to:

```verilog
`define CONV_DUT conv_1024
```

If you also changed the MAC or pixel module declarations, their instantiation
names in the parent modules must match. A filename containing spaces such as
`macint 8.v` does not change the module name inside it.

## What is checked

Every checked edge compares **all 1,024 outputs** using four-state comparisons,
so X/Z outputs are errors. The reference computes scalar convolution from separate
logical image/kernel arrays using signed 64-bit arithmetic. It does not read DUT
internal signals or reuse the DUT's window generation.

- Synchronous reset, including assertion before the edge and held reset.
- Priority: reset over load over enable.
- Positive, negative and zero bias; bias counted once per output.
- Bias-port changes ignored during normal accumulation/hold.
- Disabled hold while image, weights and bias all change.
- All 25 combinations of uniform input/weight values from `{-128,-1,0,1,127}`.
- Explicit corner/edge/interior expected values to check zero padding.
- Each of the 27 kernel/channel lanes activated independently.
- Distinct row/column/channel patterns for input packing and output ordering.
- Impulses at all four corners, four edge midpoints, and center, in each channel.
- Signed INT32 biases near both limits, without intentional overflow.
- Consecutive enabled edges accumulate; reloading bias starts a new result.
- 64 repeatable random trials by default, including changing-data accumulation
  and stalls.
- Optional independent Python golden-vector comparison.
- Final reset/hold, unknown-output detection, and a simulated-time watchdog.

Inputs change on falling edges. Checks occur before the next rising edge and
1 ns after it, allowing nonblocking register updates and the combinational tree
to settle. This is a **behavioral** testbench, not a post-route timing test.
It does not test output behavior outside the documented INT32 range contract.

## Expected console output

The suite prints progress groups followed by one final result:

```text
PASS: all 1024 pixels matched. cycles=... comparisons=... random_cases=64 python_fixture=0
```

`python_fixture=0` means the optional file-based comparison was not enabled;
`python_fixture=1` means it ran. A mismatch reports test name, time, row, column,
expected value, actual value, and hex output. Only the first 20 mismatches are
printed, but every mismatch is counted and causes a final `$fatal` failure.
No final PASS means the run is not yet successful, even if there are no errors
in the first 1000 ns.

## Run the real Python-generated fixture

This project already contains the relevant files under
`resnet8_project/resnet8/artifacts/conv_fixture/`:

- `input.mem`: 3,072 INT8 image values.
- `weights.mem`: 27 INT8 weights.
- `bias.mem`: one INT32 bias.
- `expected_acc.mem`: 1,024 INT32 reference convolution outputs.

Use **`expected_acc.mem`**, not `expected_output.mem`: the latter contains the
later INT8 requantized/ReLU result, which this RTL does not yet produce.

The easiest setup is to edit the testbench parameter at the top:

```systemverilog
parameter string FIXTURE_DIR = "C:/Users/Ibrah/Desktop/ECE_364D/senior-design/resnet8_project/resnet8/artifacts/conv_fixture";
```

Use your actual absolute path and forward slashes. Do not include a trailing
filename. Absolute paths avoid dependence on Vivado's simulation run directory.
The bench checks file existence, missing/unknown words, and agreement between the
SV reference, the Python reference, and the DUT. Missing requested fixtures fail
the run rather than silently skipping it.

Alternatively, XSim accepts plusargs with `-testplusarg`, as documented in
[AMD UG900](https://docs.amd.com/r/2020.2-English/ug900-vivado-logic-simulation/xsim-Executable-Options).
In the project Tcl console, **before launching simulation**, an example is:

```tcl
set_property xsim.simulate.xsim.more_options {-testplusarg RANDOM_CASES=100 -testplusarg SEED=2608cafe -testplusarg FIXTURE_DIR=C:/your/path/conv_fixture} [get_filesets sim_1]
```

This replaces that property; preserve any other existing options you need.
For paths with spaces, editing the string parameter is simpler. `RANDOM_CASES`
supports 0..1000 and `SEED` is a nonzero 32-bit hexadecimal seed. Restart/relaunch
simulation after changing parameters or launch options.

## Useful waveforms

### If simulation appears stuck at `add_wave`

The default waveform setup can try to display the testbench's large arrays and
buses. Recursive additions can include the 27,648 MAC instances as well. This
is a possible display/setup bottleneck, not evidence of an arithmetic failure.
AMD documents that wide HDL objects can slow the waveform viewer in
[UG900](https://docs.amd.com/r/2024.1-English/ug900-vivado-logic-simulation/Wave-Window).

Cancel/close the current simulation first. If Vivado is completely unresponsive,
restart it and reopen the saved project. In the **project Tcl console**, run:

```tcl
source {C:/Users/Ibrah/Desktop/ECE_364D/senior-design/resnet8_project/rtl/setup_xsim_no_waves.tcl}
launch_simulation -simset sim_1 -mode behavioral
```

Adjust the path if the project is on another computer. The setup script selects
the testbench top, disables all-signal logging, and replaces the generated startup
Tcl with `xsim_no_waves.tcl`. That script runs the complete testbench without
adding waveforms. All 1,024-output comparisons remain enabled. Watch the console
for progress groups and the final PASS/FAIL. The scripts have not been run in
Vivado here.

These settings use the documented
[XSim custom startup script option](https://docs.amd.com/r/2020.2-English/ug900-vivado-logic-simulation/Vivado-Simulator-Simulation-Options).
If a previously saved large `.wcfg` still opens automatically, remove that wave
configuration from Simulation Sources (keep the file on disk) before relaunching.
Do not add the entire DUT hierarchy to waves for this first run.

To return to Vivado's generated startup script later, close simulation and run:

```tcl
set_property xsim.simulate.custom_tcl {} [get_filesets sim_1]
```

### Small waveform selection

Add `clk`, `rst`, `load_bias`, `enable`, `bias`, `test_name`, `out_top_left`,
`out_top_right`, `out_center`, `out_bottom_left`, and `out_bottom_right`.
Display signed values with signed decimal radix. The testbench checks the entire
output bus even when you display only these five pixels.

The design expands to 27,648 MAC instances. Elaboration can take time; avoid
recursively logging the entire hierarchy unless you need it for debugging.
