# Build, program, and benchmark Conv0 on the KV260

> **Status correction, October 21, 2026:** simulation, synthesis, routing/timing,
> and managed bitstream/XSA export have completed. Statements below saying these
> have not run describe the original guide's preparation date. Vitis compilation
> and board execution remain pending. Current artifacts are in repository-root
> `.vivado/kv260_20261019_040420/`, rather than the example `build/` paths below.
> Start at section 5 for deployment of this baseline; see the
> [meeting onboarding](../../onboarding/README.md) for evidence and assignments.

This guide takes the source in this folder to a volatile JTAG-loaded bitstream and a standalone Cortex-A53 benchmark. It assumes Vivado/Vitis 2024.2, matching KV260 board files, the KV260 power supply, and a USB JTAG/UART connection. No FPGA build or board test has been performed for this new project yet.

The first target is one correct INT8 Conv0, then its **end-to-end latency**. A successful result is all 26,624 returned bytes matching the saved golden feature map, followed by repeatable cold/warm timing. This is not a full ResNet classifier or a Linux application.

## 1. Check the files and golden inputs

From the repository root in PowerShell:

```powershell
.\.venv-resnet\Scripts\python.exe resnet8_project/rtl/kv260_conv0/tools/export_fixture.py
.\.venv-resnet\Scripts\python.exe resnet8_project/rtl/kv260_conv0/tools/check_layout.py
```

These use the existing Python environment and saved fixtures; they do not install tools or run HDL. Generated `cpu/fixture_data.h` supplies the input, weights, bias, quantization, and expected output. `src/conv0_quant_pkg.sv` is generated from the same checksum-verified export. Keep these files and the bitstream together when sharing results.

The default image is one existing CIFAR-10 golden sample. The C harness takes no command-line image: its inputs are compiled into the ELF. To benchmark another image, generate its quantized HWC input and expected Conv0 output with the existing Python reference, update the fixture/export path deliberately, regenerate the header, and rebuild the application. Changing quantization requires rebuilding hardware too.

## 2. Run the provided testbench yourself

Open a **new Vivado session**. In its Tcl console:

```tcl
cd C:/Users/Ibrah/Desktop/ECE_364D/senior-design/resnet8_project/rtl/kv260_conv0
source scripts/simulate.tcl
```

This creates an isolated simulation project, sets the fixture plusarg, and runs without automatically adding waveforms. It checks cold and warm frames, every raw INT32 accumulator and output byte, TLAST/TKEEP, output stability during stalls, independent AXI-Lite AW/W arrival, response backpressure, invalid START, and rejection of wrong quantization constants. The warm frame includes a 400-clock output stall and requires FIFO backpressure to reach the requantizer. Sparse/empty KEEP, partial nonfinal input beats, early TLAST and reset recovery are also checked. Expect two frame PASS lines and the final `PASS tb_conv0_axi` only if it succeeds. This expanded source has not been compiled or run.

Do not treat the Python checks as replacing this test. A simulator/compiler error is a real integration issue to resolve before generating the board image. Existing `tb_conv0_416_dsp` does not exercise this new design. The new testbench tests the custom accelerator ports directly; DMA/DDR integration is exercised by the board harness.

## 3. Create the complete Vivado design

In a fresh Vivado Tcl session, from this folder:

```tcl
source scripts/create_project.tcl
```

If multiple board versions are installed, inspect `get_board_parts *kv260*`, then run from a Vivado command prompt with the exact board identifier:

```text
vivado -mode batch -source scripts/create_project.tcl -tclargs <installed-KV260-board-part>
```

The script deliberately requires the KV260 board preset for PS DDR/MIO configuration. Install the correct AMD board files through Vivado's board repository support if it is absent; do not substitute a generic Zynq board preset. It creates `build/conv0_kv260.xpr` and refuses to overwrite an existing project. Reopen that project for subsequent work; use a separate fresh build directory if regenerating the block design.

The block design is:

```text
A53 -> M_AXI_HPM0_FPD -> control SmartConnect -> accelerator S_AXI
                                            -> AXI DMA S_AXI_LITE

DDR <- S_AXI_HP0_FPD <- memory SmartConnect <- DMA M_AXI_MM2S / M_AXI_S2MM
DMA M_AXIS_MM2S -> accelerator S_AXIS
accelerator M_AXIS -> DMA S_AXIS_S2MM
```

The accelerator, control, DMA, and stream clocks are 100 MHz. DMA memory ports are 256 bits. The PS HP0 port is 128 bits at 200 MHz, and the memory SmartConnect performs width/clock conversion. This gives a nominal 3.2 GB/s bus capacity on each of those interfaces before protocol and DDR overhead; it does not promise sustained bandwidth. AMD documents that DMA stream width cannot exceed its memory-map width, and that SmartConnect inserts clock conversion between its connected domains. [AXI DMA width settings](https://docs.amd.com/r/en-US/pg021_axi_dma/Field-Descriptions), [SmartConnect clocking](https://docs.amd.com/r/en-US/pg247-smartconnect/Clocking).

DMA uses direct/simple mode, no scatter-gather, 64-bit addresses, 16-beat maximum bursts, and aligned buffers with DRE disabled. The CPU harness uses 64-byte-aligned buffers. The image and parameters are separate MM2S packets; all output is one S2MM packet.

Review `validate_bd_design`, the Address Editor, and generated clocks before synthesis. The MMIO addresses must be accelerator `0xA0000000`, DMA `0xA0010000`. These are mirrored in `cpu/benchmark.c`. The PS pins are dedicated PS interfaces; this design does not export image tensors as tens of thousands of package pins.

## 4. Synthesize, implement, and export

With the created project open:

```tcl
source scripts/build.tcl
```

Or from a Vivado command prompt in this folder:

```text
vivado -mode batch -source scripts/build.tcl
```

The intended outputs are:

```text
build/conv0_kv260.bit
build/conv0_kv260.xsa
build/reports/utilization_synth.rpt
build/reports/utilization_route.rpt
build/reports/timing_route.rpt
build/reports/drc.rpt
```

The script stops if synthesis/implementation fails or the worst reported setup/hold path is negative. Also inspect unconstrained-path and clock warnings, CDC reports, and DRCs yourself; positive reported slack alone does not establish complete timing coverage. PS-generated clock constraints must show 100 MHz for the accelerator and 200 MHz at HP0. Do not hide real failures with false paths or an incorrect clock declaration.

Check hierarchical synthesis counts: the MAC multipliers should account for roughly 832 DSPs, with accumulation and requantization in LUTs. If the MAC hierarchy alone uses roughly 1,664 DSPs, the intended split inference failed. If LUTs/BRAMs exceed the K26 capacity, implementation will fail. Resource estimates from the Python study are not a substitute for these reports. Preserve the reports and exact source revision with every benchmark.

## 5. Set up the Cortex-A53 application in Vitis

Use Vitis Unified IDE 2024.2 with a new workspace, preferably `vitis_workspace/` in this folder.

1. Create a **Platform Component** from `build/conv0_kv260.xsa`. Select a standalone domain targeting `psu_cortexa53_0`. Generate/build the platform and its BSP. Retain the generated PS initialization/boot support appropriate to the platform.
2. Create an **Application Component**, select that platform and standalone A53 domain, and choose an empty C application.
3. Add `cpu/benchmark.c` and `cpu/fixture_data.h` to its source/include paths. Ensure there is only one `main()`; remove the template main if necessary. The JSON manifest is provenance, not a C build input.
4. Build in Release mode with optimization (for example `-O2`). Use the platform BSP's `xaxidma`, cache, MMIO, and timer APIs. No Linux/PYNQ libraries are involved.
5. Generate/review the linker script: code, static fixture arrays, `.bss`, heap, and stack should reside in valid **PS DDR**, with sufficient space. Do not place the large buffers in small on-chip memory. Keep the DMA buffers in low DDR for straightforward first bring-up.
6. Set the BSP stdout to the board's PS UART and open the matching USB serial port, normally 115200 baud, 8-N-1. Save its output to a text file.

The source supports Vitis SDT lookup by DMA base address and classic BSP lookup through `XPAR_AXIDMA_0_DEVICE_ID`. If your BSP uses a different classic macro, inspect its generated `xparameters.h` and update that one lookup. Do not guess the instance address. The hardware identity registers catch mismatched accelerator versions once MMIO is accessible.

AMD's [Vitis embedded development guide](https://docs.amd.com/r/2024.2-English/ug1400-vitis-embedded) covers XSA platform creation and standalone debugging. The DMA API sequence and cache-maintenance pattern follow AMD's [2024.2 simple polling example](https://github.com/Xilinx/embeddedsw/blob/xilinx_v2024.2/XilinxProcessorIPLib/drivers/axidma/examples/xaxidma_example_simple_poll.c).

## 6. Program the board and run the application

Use a dedicated standalone debug session: the application owns the A53, DMA, and this PL design. Do not run it concurrently with Linux using those resources.

1. Connect KV260 power and the USB JTAG/UART port to the host. Confirm the cable appears in Vivado Hardware Manager and the serial ports appear in the OS.
2. Prepare the board for standalone JTAG debugging. The KV260 normally uses QSPI boot firmware; use AMD's **Kria-specific JTAG boot procedure** for your board revision if the normal boot chain interferes. Do not follow a ZCU102 DIP-switch recipe. See [AMD AR 000036959](https://adaptivesupport.amd.com/s/article/000036959?language=en_US) and the [KV260 user guide](https://docs.amd.com/r/en-US/ug1089-kv260-starter-kit).
3. In Vitis, create a hardware debug/run configuration for this application on **Cortex-A53 #0**. Use the matching platform/XSA and bitstream. Enable the supported target reset, FPGA programming, and PS initialization steps (generated `psu_init.tcl`/platform initialization). DDR, clocks, resets, and PS-to-PL interfaces must be initialized before loading the ELF into DDR.
4. Launch the configuration, let it load the ELF, then Resume/Continue if it stops at `main`. Watch the UART capture for the hardware banner, FRAME records, and the final PASS or failure details.

For programming only, Vivado Hardware Manager can open the target and **Program Device** with `build/conv0_kv260.bit`. That configures PL; it does **not** initialize DDR, load the A53 ELF, or execute the benchmark. Use the Vitis launch above for the complete test.

JTAG configuration is volatile and is lost on power cycle. This workflow does not modify QSPI firmware or install an SD-card OS. After a successful benchmark, a separate boot-image integration task can package the application for unattended boot. Do not assume copying this `.bit` or a generic `BOOT.BIN` to microSD replaces the KV260's QSPI-led boot flow; see AMD's [starter-kit firmware description](https://docs.amd.com/r/en-US/ds986-kv260-starter-kit/Starter-Kit-Firmware-and-Software).

## 7. What the CPU does and what the timing includes

For a cold frame, the CPU resets the accelerator readiness flags, copies/flushed payloads into the aligned transmit buffer, submits four MM2S transfers, waits for each DMA transfer and local load completion, flushes the receive buffer, arms S2MM for 26,624 bytes, and writes START. It polls completion, invalidates the receive cache, and ends the timer. It then compares every byte to the Python golden output and checks both guard regions around the receive buffer.

For a warm frame, PREPARE retains weights/bias/quantization readiness, so only the 3,072-byte image is transmitted. The output still transfers all 26,624 bytes. DMA setup, cache maintenance, control transactions, polling, and local copies are included in the measured transaction. Golden comparison and UART printing are excluded. Thus this measures one sequential application request, not maximum overlapped throughput or pure FPGA compute time.

The program runs 20 cold and 200 warm frames and stops at the first failure. Each successful record is:

```text
FRAME,mode,index,total_ns,run_cycles,raw_cycles,first_output_cycle,input_stalls,output_stalls,load_cycles
```

The first line printed with those names is a header, not a measurement. `total_ns` comes from the A53 timer using the BSP's `COUNTS_PER_SECOND`. FPGA counters are at 100 MHz (10 ns/cycle). `run_cycles` spans START to final stream acceptance, including stalls; it excludes preload and the remaining DMA-to-DDR completion. `raw_cycles` is the core-done observation point, not pure multiply time. `load_cycles` counts loader chunks, not CPU transfer time. The program prints `PASS` only after all golden and guard checks pass.

From this folder, summarize a saved UART capture:

```powershell
..\..\..\.venv-resnet\Scripts\python.exe tools/summarize_uart.py UART_CAPTURE.txt --clock-mhz 100
```

Report minimum, median, p95, and mean for both cold and warm requests, plus actual routed frequency/resources and output stalls. Preserve the full capture, including startup and PASS/FAIL. The summarizer rejects failed/truncated captures, duplicate or out-of-order frames, unexpected loader counts and inconsistent counters. It requires 20 cold and 200 warm frames and the matching final PASS. If deliberately changing the C frame counts, pass matching `--cold-frames` and `--warm-frames` options. Do not combine multiple runs in one capture.

The Python study predicted **19.47 microseconds** for its selected cold-frame assumptions (2 GB/s links and modeled setup/completion delays). This source adds actual AXI control, individual simple-DMA submissions, clock/width conversion, software copies/cache maintenance, and fixed quantization specialization. It is **not yet cycle-correlated** to the model. Use the measured counters and end-to-end timing to calibrate it; do not report 19.47 us as achieved FPGA latency.

## 8. Diagnosing first-run failures

| Symptom | First checks |
|---|---|
| No UART banner | A53 actually running; ELF loaded into initialized DDR; BSP stdout and serial port/baud |
| MMIO access hangs/aborts | Matching bitstream/XSA; PL clocks/resets; address map; PS initialization; no competing OS |
| Hardware identity failure | Accelerator VERSION/MACS/OUTPUT_LANES; stale bitstream or wrong base address |
| DMA configuration fails | BSP DMA instance/address; simple mode; aligned buffers |
| MM2S/loader timeout | Input stream connected; LOAD_KIND; TLAST/TKEEP; payload length; status/error bits |
| S2MM timeout | Arm S2MM before START; length 26624; output TLAST; HP0 DDR map and clock |
| Golden mismatch | HWC/OHWI ordering, little-endian bias/quant bytes, fixture/header/package agreement, cache maintenance |
| Correct results but long latency | CPU setup/cache costs, DMA polling, HP bandwidth, output stalls, timing clock assumptions |

The failure print includes accelerator status/error plus both DMA status registers. Consult [AXI DMA PG021](https://docs.amd.com/r/en-US/pg021_axi_dma/Direct-Register-Mode-Simple-DMA) to decode DMA errors. If aborting an active transfer, reset both DMA channels and the accelerator before retrying; a soft reset of just the accelerator does not cancel a DMA transfer.

Once this path is correct, the next goals are to measure actual utilization/timing, correlate board counters with the Python cycle model, add more independent images and randomized arithmetic tests, and decide whether the narrower requantization fallback wins on the real system. Linux integration would need a matching device tree, DMA-safe buffers, and a kernel DMA-engine path; the standalone ELF is not a Linux executable.
