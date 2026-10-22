# Current ResNet-8 project status and near-term schedule

Updated: **October 19, 2026, after routed timing reports**. This page supersedes
pre-Vivado status statements in the older meeting brief and progress documents.
Routing and timing checks completed; managed bitstream/XSA export succeeded October 19 at 13:31 (export log time). Text reconciled October 21. See [meeting onboarding](onboarding/README.md).

## What we are building

An INT8 accelerator for the frozen ResNet-8 on AMD KV260. The immediate hardware
milestone is **Conv0**, not complete image classification. Our performance metric
is CPU request-to-result latency, including copies, cache maintenance, setup,
computation and transfers. Full-network inference currently exists in software.

## What is proven now

| Item | Current evidence |
|---|---|
| Software references, quantization, exports and golden data | Existing software workflow; 32 Python tests passed before Vivado work |
| New 832-MAC RTL arithmetic/protocol | XSim suite passed baseline plus 14 cases, each cold/warm: 30 frames, 798,720 raw INT32 and 798,720 final INT8 comparisons |
| Independent image coverage | Ten archived CIFAR-10 images; four additional synthetic cases exercise zero/signed inputs and positive/negative overflow; baseline repeats one image |
| IP Integrator wrapper | Additional baseline cold/warm XSim run passed through the plain-Verilog wrapper |
| No-stall simulation counters | Raw completion: 1,217 cycles; first output: 35; last output: 1,239; cold loader: 256 chunks; warm loader: 192 |
| Integrated synthesis, including PS/DMA/interconnect | 97,232 LUTs, 57,078 FFs, 832 DSPs, 8 BRAM36 + 131 BRAM18 primitives |
| DSP mapping | One DSP per MAC confirmed in isolated MAC and full core synthesis |
| Routed utilization | 95,688 LUTs (81.70%), 56,684 FFs (24.20%), 832 DSPs; 8 BRAM36 + 131 BRAM18 |
| Routed timing | PASS: worst setup slack +0.365 ns, hold +0.010 ns; zero unconstrained internal endpoints |
| Clock-domain crossings | 733 reported endpoints classified safe; zero unsafe/unknown |
| DRC | No errors or critical warnings in the reported rule summary; 2,548 warnings retained for review |
| Physical placement margin | 14,306 / 14,640 CLBs occupied (97.72%); do not infer abundant placement headroom from LUT percentage |
| Bitstream/XSA | Managed export succeeded; artifacts under repository-root `.vivado/kv260_20261019_040420/` |
| Vitis compilation and board execution | Not done |

At nominal 100 MHz, 1,239 cycles is 12.39 us **from START to final output
acceptance in simulation**. It excludes preload and CPU/DMA overhead. The PS
preset resolves the requested clocks to approximately 99.999001/199.998001 MHz;
use generated clock reports for the final conversion. **19.47 us is still a
Python-model estimate, not a measured board latency.**

## Decisions and repairs made during real tool validation

1. Fixed Tcl handling of a property value beginning with `-testplusarg`.
2. Close XSim before inspecting its buffered log; a shell exit alone is not PASS.
3. Added a plain-Verilog wrapper because IP Integrator rejected the SystemVerilog
   module-reference top. The wrapper adds no datapath logic.
4. Disabled unused PS master ports enabled by the board preset.
5. Removed hard-coded clock frequency metadata from the wrapper so it follows
   the actual PS clock, rather than relabeling that clock to suppress an error.
6. Use isolated short-path builds under `.vivado/` to reduce Windows path-length
   risk. Existing Vivado projects were preserved.
7. Corrected the UART checker: cold load chunks are 192 + 44 + 7 + 13 = **256**,
   not 257. Its two acceptance/rejection tests passed after the correction.
8. XSA export needs a managed implementation bitstream step. Resume that step
   from the existing routed checkpoint, rather than rerun synthesis/placement.

The compute/requantization/FIFO datapath has not been redesigned during these
repairs. The original candidate has met routed timing. Preserve this baseline
for board bring-up before making an area/latency optimization.

## Schedule: concrete gates, not promised completion dates

| When / gate | Deliverable | Status / dependency |
|---|---|---|
| Now, October 21 | Prepare deployment and archive reports | Routed timing and managed export complete; Vitis/board remain pending |
| October 20 Ericsson meeting | Present simulation evidence, actual synthesis resources, architecture decisions and open risks | Evidence available; update routing status before meeting |
| Next | Build standalone A53 application in Vitis | Requires exported XSA, available Vitis/BSP and review of retained warnings |
| Next board session | Initialize PS/DDR, program PL, run golden byte/guard checks | Requires physical KV260/JTAG/UART access and matching ELF/XSA/bitstream |
| After board correctness | Capture 20 cold + 200 warm frames; report median/p95 and calibrate model | Hardware latency is unknown until this passes |
| After measured Conv0 baseline | More board images; evaluate reuse for later convolutions and residual buffers | Do not equate successful Conv0 with a full-network accelerator |

The older semester schedule (detailed file retained locally) targets a synthesis-backed
kernel and DMA loopback in **October 19–25**, and a real board arithmetic result
in **November 2–8**. We have synthesis evidence; **board loopback and arithmetic
execution have not been demonstrated in this work**. That document still uses
older MNIST/generic-kernel examples. Its December 18 demo is explicitly a
placeholder, not a confirmed class deadline. These broad team commitments need
reconciliation with the current ResNet-8 scope; no new university deadlines or
teammate assignments have been assumed here.

## Reading order

1. This page: current claims, evidence, gaps and schedule.
2. [Accelerator README](rtl/kv260_conv0/README.md): architecture, arithmetic,
   specialization, layouts and register map. Its older unrun-status paragraphs
   are superseded by this snapshot.
3. [Deployment guide](rtl/kv260_conv0/KV260_BUILD_AND_BENCHMARK.md): remaining
   Vivado/Vitis/board path and exact timing boundaries; older verification status
   is superseded here.
4. [Network/latency analysis](reports/meeting_2026_10_20/ANALYSIS.md): why transfers
   matter and what extending beyond Conv0 requires. Numbers are analytic assumptions.
5. Ericsson briefing (detailed file retained locally): meeting narrative
   and likely questions. Its evidence table predates the new Vivado results;
   use this page for current status until it is refreshed.

## Exact current raw evidence

- Integrated synthesis resources (detailed file retained locally)
- Active implementation log (detailed file retained locally)
- [Routed timing](reports/vivado_2026_10_19/integrated/timing_route.rpt)
- [Routed utilization](reports/vivado_2026_10_19/integrated/utilization_route.rpt)
- [CDC report](reports/vivado_2026_10_19/integrated/cdc.rpt)
- [DRC warnings](reports/vivado_2026_10_19/integrated/drc.rpt)
- Full suite/tool log (detailed file retained locally)
- Wrapper simulation log and detailed counters (detailed file retained locally)

Generated `.vivado/` and `build/` paths are local evidence, not portable source.
Selected final reports are archived in [reports/vivado_2026_10_19](reports/vivado_2026_10_19/README.md).
