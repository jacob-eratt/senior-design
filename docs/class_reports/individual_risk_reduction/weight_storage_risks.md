# Weight Storage Risks: All On-FPGA vs. Streamed From CPU

Context: [uarch_study/04_on_chip_memory.md](../../../resnet8_project/brainstorming/uarch_study/04_on_chip_memory.md) (keep all weights resident) and [Conv0Spec_ChipAgents.md](../../../resnet8_project/arch_specs/Conv0Spec_ChipAgents.md) (reload per START).

## Risks of keeping all weights on the FPGA

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Lost on reconfiguration or power-cycle** | The PL has no non-volatile weight storage, so any PL reset or bitstream reload empties the weight RAM. The host must reload every time. | Reload at boot from DDR via DMA and gate START on a loaded flag (as the Conv0 spec already does). Or initialize the URAM from the bitstream. Verify the loaded contents with a checksum before the first frame. |
| **Silent corruption** | A single-event upset in weight memory changes every output that uses that weight, with no visible error. | Enable URAM ECC (SECDED) and add a periodic CRC scrub of the weight RAM against the DDR master copy. |
| **Fixed capacity** | The design is sized for ResNet-8. A larger model will not fit. | Keep the current headroom (about 9% URAM). Add a fallback path that tiles layers from DDR when the model exceeds the RAM. |
| **Read-port and timing pressure** | The 26 B/cycle weight read is tied to URAM cascade and fmax, so the memory can limit the clock. | Add pipeline registers on the read path, and check URAM timing at 200 MHz early (before the array is built on top of it). |
| **Sequencer complexity** | Layer switching depends on the descriptor table and address math for each layer. A bad address reads the wrong weights without an error. | Put a layer ID and checksum in each descriptor and test layer switching directly. |

## Risks of streaming weights from the CPU for each layer

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Stalls on every layer** | The array waits for weights before each layer. Total latency grows by the transfer time for every layer, and the CPU must schedule each one. | Use a second (ping-pong) weight buffer so the next layer's weights arrive while the current layer computes. |
| **CPU in the loop** | Per-layer interrupts or polling add latency and jitter. A busy or preempted CPU stalls the pipeline. | Chain DMA descriptors for the whole network so the CPU only starts a frame. Run bare-metal, as the benchmark already does. |
| **Memory contention** | Weight traffic shares the DDR controller and SmartConnect with image and output traffic. Bandwidth drops for both. | Set AXI QoS priorities so weight and image transfers do not starve each other. Keep weights in DDR and stream them over the PS HP port without copying them through the CPU. |
| **Cache coherency** | Weight data written by the CPU may not be visible to the DMA without cache maintenance. Stale weights give wrong results without any error. | Flush the CPU cache after writing weight buffers and invalidate after reads. Compare outputs against the golden vectors. |
| **Partial or failed transfers** | A dropped or short transfer goes unnoticed unless the loader checks it. | Use the per-kind loaded flags and length checks from the Conv0 spec (REQ-LD-01 to REQ-LD-03), plus a per-layer checksum. |

## Comparison

Streaming costs latency and adds risk on every frame, but it needs very little on-chip memory and no reload logic. Keeping weights resident removes per-frame latency and CPU involvement, but it depends on a reliable boot load and on ECC or scrubbing to protect the data. For ResNet-8 the total weight traffic is small (on the order of 200 KB per frame), so the main issue is latency and reliability, not bandwidth. A common compromise is to keep weights resident and use streaming only as a fallback for layers that do not fit.
