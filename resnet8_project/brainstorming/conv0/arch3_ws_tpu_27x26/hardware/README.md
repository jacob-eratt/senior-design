# KV260 TPU hardware handoff

These are the existing local November 6, 2026 integrated-build artifacts, published
with their reports. They were not rebuilt when this bundle was prepared.

| File | Use |
|---|---|
| [tpu_kv260.xsa](tpu_kv260.xsa) | Import into Vitis 2024.2 to create the standalone Cortex-A53 platform. Contains hardware metadata, PS initialization files and the bitstream. |
| [tpu_kv260.bit](tpu_kv260.bit) | Program the PL using the matching platform/debug flow or Hardware Manager. Programming PL alone does not initialize DDR or run the CPU. |
| [SHA256SUMS](SHA256SUMS) | Verify downloaded artifacts and reports, using paths relative to this architecture directory. |
| [reports/](reports/) | Original routed utilization, timing, clock, CDC, DRC and timing-coverage reports. |

The XSA archive was opened and its embedded bitstream was verified byte-for-byte
against the standalone `.bit` file. No ELF was available in the TPU build/project
directories inspected. Build [cpu/tpu_test.c](../cpu/tpu_test.c) against the XSA's
standalone A53 platform; see [DEPLOY_KV260.md](../DEPLOY_KV260.md).

## What this hardware implements

- Device: `xck26-sfvc784-2LV-c`, Vivado 2024.2.
- Integrated top: `system_wrapper`, with PS, DMA, SmartConnect and the TPU wrapper.
- Accelerator/DMA at 100 MHz; PS HP0 side at 200 MHz.
- Accelerator control: `0xA0000000`; DMA control: `0xA0010000`.
- One 27 x 26 weight tile; 1..64 vectors; raw signed INT32 outputs.
- No full Conv0 patch generation/bias/requantization/ReLU or full ResNet execution.

## Evidence from the saved routed reports

| Metric | Result |
|---|---:|
| DSP | 702 / 1,248 (56.25%) |
| LUT | 33,339 / 117,120 (28.47%) |
| Flip-flops | 85,605 / 234,240 (36.55%) |
| BRAM36 equivalents | 9 / 144 (6.25%) |
| Occupied CLBs | 13,919 / 14,640 (95.08%) |
| Worst setup slack | +0.184 ns |
| Worst hold slack | +0.010 ns |

The timing report states all user-specified constraints are met. Occupied CLBs
remain high despite modest LUT utilization; this is not evidence of unlimited
placement headroom. The CDC report lists 733 safe endpoints and zero unsafe or
unknown endpoints, with crossing warnings. DRC warnings are preserved in the
report, including DSP pipeline and RAM collision advisories; no warning review
or board-validation claim is implied by publishing the files.

The report provenance is the original local build, not a newly verified
source-to-bitstream reproduction. The historical 1,820-comparison test pass also does not establish
board correctness. Vitis compilation and board results must be recorded separately.

To reproduce the hardware from source, use `scripts/create_soc_project.tcl` and
`scripts/build_soc.tcl` from the architecture directory. Generated `.xpr` projects,
IP caches and checkpoints are excluded because those scripts recreate them.
