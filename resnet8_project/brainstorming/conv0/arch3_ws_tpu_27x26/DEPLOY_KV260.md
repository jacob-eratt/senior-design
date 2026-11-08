# Deploying the 27×26 TPU tile on the KV260

Goal of this milestone: the A53 sends a weight tile and input vectors over DMA,
the PL computes, and the A53 checks every INT32 result against a CPU-computed
golden and prints latency. Background on the interfaces: [INTERFACING.md](INTERFACING.md).

## 1. What you need

| Item | Notes |
|---|---|
| Vivado 2024.2 + Vitis 2024.2 (Unified IDE) | Same version for both, so the XSA and BSP match |
| KV260 board files | `get_board_parts *kv260*` must return `xilinx.com:kv260_som:part0:*` (Vivado → Tools → Vivado Store → Boards) |
| Cable drivers | Installed with Vivado (Windows: `data/xicom/cable_drivers/nt64` if Hardware Manager sees no target) |
| Micro-USB cable to the KV260 J4 port | JTAG + four USB-UART ports on one cable |
| Serial terminal | PuTTY / Tera Term / Vitis serial monitor, 115200 8-N-1 |
| KV260 12 V supply | |

## 2. The flow

```text
RTL (this folder) ──create_soc_project.tcl──► block design: Zynq PS + AXI DMA + ws_axi_bd
                    ──build_soc.tcl──────────► synth → impl → timing gate → tpu_kv260.bit
                                                └► tpu_kv260.xsa  (hardware handoff: bit + PS config + address map)
Vitis: platform component from tpu_kv260.xsa (standalone, psu_cortexa53_0)
       application component + cpu/tpu_test.c ──► tpu_test.elf
JTAG:  reset → psu_init (DDR/clocks) → program PL with the .bit → load ELF on A53 #0 → UART output
```

The **XSA** is the only file Vitis needs from Vivado. It contains the bitstream,
the PS configuration (`psu_init`, which brings up DDR, clocks and the PS–PL
ports) and the address map (`0xA0000000` accelerator, `0xA0010000` DMA). It is
not a program; the ELF is.

System built by the script:

| Block | Configuration |
|---|---|
| `zynq_ultra_ps_e` | KV260 preset; `M_AXI_HPM0_FPD` (control), `S_AXI_HP0_FPD` 128-bit (DDR), `pl_clk0` = 100 MHz, `pl_clk1` = 200 MHz |
| `axi_dma` | Simple mode (no SG), MM2S + S2MM, 256-bit stream and memory ports, no DRE (buffers must be 32-byte aligned) |
| `ctrl` SmartConnect | HPM0 → accelerator AXI-Lite + DMA AXI-Lite |
| `memory` SmartConnect | DMA MM2S/S2MM → HP0 (100 → 200 MHz crossing) |
| `accelerator` (`ws_axi_bd` → `ws_axi_top`) | 100 MHz, `S_AXI` @ `0xA0000000`, `S_AXIS` ← DMA MM2S, `M_AXIS` → DMA S2MM |

## 3. Vivado (Tcl console)

With your existing `kv260_2026_tpuarch/project_1` open:

```tcl
set ws_soc_use_open_project 1
source <repo>/resnet8_project/brainstorming/conv0/arch3_ws_tpu_27x26/scripts/create_soc_project.tcl
```

Or start a clean project instead: `close_project`, then source the same script
without setting `ws_soc_use_open_project`.

Look for `TPU_SOC_CREATED`. Open the block design and check the Address Editor
(accelerator `0xA0000000`, DMA `0xA0010000`, DMA masters can reach `DDR_LOW`). Then:

```tcl
source <repo>/resnet8_project/brainstorming/conv0/arch3_ws_tpu_27x26/scripts/build_soc.tcl
```

This synthesizes, implements, refuses to write a bitstream if setup or hold
slack is negative, then writes:

- `build/tpu_kv260.bit`
- `build/tpu_kv260.xsa`
- `build/soc_reports/` (utilization, timing, clocks, CDC, DRC)

Look for `TPU_SOC_BUILT`. Review `timing_route.rpt` and `utilization_route_hier.rpt`.

## 4. Vitis 2024.2 (Unified IDE)

1. **File → New Component → Platform.** Hardware design: `build/tpu_kv260.xsa`.
   OS `standalone`, processor `psu_cortexa53_0`. Keep "Generate boot artifacts" on. Build the platform.
2. In the platform's BSP settings, confirm stdin/stdout use `psu_uart_1` (the KV260 debug UART).
3. **File → New Component → Application**, template **Empty Application (C)**,
   on that platform, domain `standalone_psu_cortexa53_0`.
4. Add [`cpu/tpu_test.c`](cpu/tpu_test.c) to the application's `src/` and build it to get `tpu_test.elf`.
   - The program uses `XAxiDma_LookupConfig(DMA_BASE)` under the default SDT flow.
   - If the build reports a missing lookup symbol, check `xparameters.h` for the DMA base or device ID.

## 5. Put the KV260 into JTAG boot mode

The KV260 normally boots its QSPI firmware and then Ubuntu, which then owns the PS.
For standalone JTAG runs, switch the boot mode to JTAG and reset. Power the board,
wait a few seconds, then in **Vitis → Vitis → XSCT/XSDB Console**:

```tcl
connect
targets -set -nocase -filter {name =~ "*PSU*"}
mwr 0xffca0010 0x0
mwr 0xff5e0200 0x0100
rst -system
```

Check these addresses against [AMD AR 000036959](https://adaptivesupport.amd.com/s/article/000036959?language=en_US)
for your board revision. This is volatile: a power cycle returns to QSPI boot.
Nothing is written to flash.

## 6. Run

1. Open the UART. Of the four USB-serial ports, PS UART1 is usually the second one, at 115200 8-N-1.
2. In the application's **Run/Debug settings** (launch configuration), target A53 #0 and enable:
   - Reset entire system
   - Program device, with the bitstream from the platform
   - Run psu_init
   - PL powerup / remove isolation
3. Run. Expected UART output:

```text
TPU 27x26 KV260 test: PL_HZ=100000000, timer_HZ=...
ID_VERSION=57530001 OK
RUN,name,vectors,total_ns,elapsed_cycles,compute_cycles
RUN,pattern,1,...
RUN,pattern,64,...
...
PASS: 332072 INT32 results checked
```

`total_ns` covers arming S2MM, START and the results landing in DDR. It excludes
loading. `elapsed_cycles` and `compute_cycles` are PL counters at 100 MHz (10 ns per cycle).

### What the test does

| Step | Content |
|---|---|
| Identity | `ID_VERSION` = `0x57530001`, `MAX_VECTORS` = 64 |
| Pattern | Same weight and input formulas as `tb_ws_axi_top`; N = 1, 64, 3, 2 with warm weight reuse |
| Extremes | All weights −128; inputs −128, 127 and 0 (max-magnitude sums) |
| Random | Random weight tile, 200 runs of random inputs (mostly N = 64, some random N) |
| Warm repeat | 10 × N = 64 back-to-back |

Every run checks every INT32, the `OUTPUT_BYTES` register, and 64-byte guard
bands on each side of the DMA buffer.

## 7. Troubleshooting

| Symptom | Check |
|---|---|
| No UART text | Wrong COM port; JTAG boot-mode step skipped (Linux still running); psu_init not enabled |
| CPU exception / abort on a register access | The register block returns SLVERR for rejected writes (START while not ready, config while busy) and for unmapped offsets. The test only writes when STATUS shows idle. |
| `FAIL,hardware identity` | Bitstream not programmed, wrong XSA, or address map differs from `0xA0000000` |
| `FAIL,DMA wait`, `mm2s_sr`/`s2mm_sr` bit 4–6 set | DMA error (decode/slave): DDR address not mapped for the DMA masters in Address Editor |
| `FAIL,loader ready`, `error` bit 0/1 | KEEP/LAST mismatch: packet length differs from `LOAD_KIND`/`VECTOR_COUNT` |
| `error` bit 2/3 | Engine capture overflow / internal timeout |
| `Mismatch,...` | Compare against `tb_ws_axi_top`; first suspect cache maintenance, then packing order (weights row-major r × c, inputs vector-major) |

## 8. Status

| Item | State |
|---|---|
| `create_soc_project.tcl` (open-project mode added), `build_soc.tcl` | Written; not yet run in Vivado |
| `cpu/tpu_test.c` | Compiles warning-free with `-Wall -Wextra -Wpedantic` against mock Xilinx headers. Passes a host model of the register/packet contract; an injected wrong result is caught. Not yet built in Vitis or run on the board. |
| Integrated timing at 100 MHz | First build (2026-11-06) failed: WNS −1.714 ns, 4,609 endpoints, all on `u_output` → result read (`index%26`, `index/26` dividers, 27 logic levels). Fixed by passing vector/column counters directly (`result_read_t`) and using row/column counters in the loader. `tb_ws_axi_top` passes again (1,820 comparisons). Rebuild met timing: WNS +0.184 ns (100 MHz), +1.484 ns (200 MHz), WHS +0.010 ns. Utilization: 33.3k LUT (28%), 85.6k FF (37%), 702 DSP (56%), 9 BRAM. DRC warnings only. `tpu_kv260.xsa` exported. |
