# Step 01 – Problem definition and first-principles bounds

Before judging any microarchitecture we need to agree on **what we are
optimizing** and **what physics limits it**. A design can only be "optimal"
relative to a goal.

## 1. Concept

Any accelerator's time is bounded below by the slowest of three things:

```text
time >= max( compute_work / compute_rate,
             bytes_moved_offchip / link_bandwidth,
             fixed_overheads )          # DMA setup, cache maintenance, polling
```

- If **compute** is the max, more/faster MACs help.
- If **data movement** is the max, more MACs do nothing; reduce bytes or overlap.
- If **overheads** dominate, hardware changes barely matter; fix software/protocol.

The architecture should be sized so that no resource is paid for but idle.

## 2. Current design in one paragraph

832 MACs (32 output columns × 26 output channels), 100 MHz, computes Conv0
only, one image row at a time: 1 init + 27 MAC + 2 drain + 8 emit = 38
cycles/row, 32 rows. Loading, computing and output are **not overlapped**
(emit stalls the MACs; compute waits for all loads). Measured in simulation:
1,239 cycles START→last output. Routed: 832 DSPs, 81.7% LUTs, **97.7% CLBs**.
See [CURRENT_STATUS.md](../arch1_broadcast_832/CURRENT_STATUS.md).

## 3. Numbers

### 3.1 Workload (derived from the export manifest, [ANALYSIS.md](../arch1_broadcast_832/reports/meeting_2026_10_20/ANALYSIS.md))

| Item | Conv0 | Full ResNet-8 |
|---|---:|---:|
| MAC terms | 718,848 | 32,562,192 |
| Share of network MACs | 2.21% | 100% |
| Weights + bias + quant bytes | 1,014 | 209,918 |
| Input bytes | 3,072 | 3,072 |
| Output bytes | 26,624 (INT8 feature map) | 10 (INT8 logits) |
| Largest single layers | — | block1_conv1/conv2: 6.23M MACs each |

Key observation: **Conv0 has a terrible compute-to-I/O ratio** (27 MACs per
output byte and its output must leave the chip). The full network, if
activations stay on chip, has ~10,600 MACs per off-chip byte.

### 3.2 Device budget (K26 SOM on KV260)

| Resource | Capacity | Logical bytes |
|---|---:|---:|
| DSP48E2 | 1,248 | — |
| LUT / FF | 117,120 / 234,240 | — |
| BRAM36 | 144 | 663,552 (648 KiB) |
| URAM (288 Kb) | 64 | 2,359,296 (2.25 MiB) |

Full-network parameters (~210 KB) plus a three-buffer activation plan
(~80 KB, ANALYSIS.md) ≈ **290 KB logical**: everything fits on chip even
before using URAM. Physical banking/port replication will increase this; still
derived, not synthesized. The current design uses **no URAM**.

### 3.3 Conv0 bounds (derived)

| Bound | Arithmetic | Cycles @100 MHz | µs |
|---|---|---:|---:|
| Compute floor, 832 MACs, 100% busy | 718,848 / 832 | 864 | 8.64 |
| Current core (no stalls) | 32 × 38 | 1,216 | 12.16 |
| Output stream floor, 256-bit AXIS (32 B/clk) | 26,624 / 32 | 832 | 8.32 |
| Cold input load, current 16 B/clk loader | 4,086 B → 256 chunks | 256 | 2.56 |
| Cold input load at full 32 B/clk AXIS | 4,086 / 32 | 128 | 1.28 |

The 256-bit/100 MHz stream equals the nominal PS HP0 port (128-bit × 200 MHz =
3.2 GB/s). Sustained DDR/DMA throughput is **assumed**, not measured.

What this says:

1. **MAC utilization today is 864/1,216 = 71%.** The 11 non-MAC cycles per row
   (init, drain, emit) cost 352 cycles. Overlapping emit with the next row's
   compute would recover most of them.
2. **For Conv0 alone, ~864 MACs at 100 MHz is the most that can ever help**,
   because the 8.32 µs output stream becomes the limit. 832 MACs is close to
   balanced *only if* compute and output overlap, which they currently do not.
   At 200 MHz, ~432 MACs would reach the same bound.
3. Current hardware path ≈ 2.56 (load) + 12.39 (compute/emit) ≈ **15 µs**
   before CPU/DMA software overhead. A fully overlapped design is bounded near
   ≈ 9–10 µs. Software overhead is **unmeasured** and may dominate both.

### 3.4 Full-network bounds (derived, 100% MAC utilization assumed)

| Configuration | MACs | Clock | Compute floor |
|---|---:|---:|---:|
| Current array reused for all layers | 832 | 100 MHz | 391 µs |
| Same array, faster clock | 832 | 200 MHz | 196 µs |
| One MAC per DSP, all DSPs | 1,248 | 200 MHz | 130 µs |
| Two INT8 MACs per DSP (shared-operand packing, AMD WP486) | 2,496 | 200 MHz | 65 µs |

Real utilization will be lower: channel counts are 26/52/104 (= 2·13·2ᵏ), so
arrays must tile by 13/26/52/104 to avoid idle lanes. INT8 DSP packing is an
**idea to verify in step 05**, not a measured result.

### 3.5 The strategic tension

The current design spends **67% of DSPs and 98% of CLBs on 2.2% of the
network**, on a layer whose latency is bounded by the output link rather than
by MACs. That is reasonable for a "Conv0 demo", and the wrong shape for a
full-network accelerator, where the work is compute-bound and the hard parts
are on-chip buffering, layer reconfiguration and residual handling.

## 4. Alternatives (framing only; detailed in later steps)

| Option | Goal it serves | Sketch |
|---|---|---|
| A. Polish current design | Fastest Conv0 demo | Overlap emit/compute and load/compute, 32 B/clk loader |
| B. Reusable layer engine, CPU orchestrates | Run every conv layer on FPGA, activations via DDR | One programmable conv engine + layer config; ~442 KB DDR traffic/inference |
| C. Full on-chip network | Lowest full-inference latency | Weights + activations resident in BRAM/URAM; image in, 10 logits out |
| D. Layer pipeline (one engine per layer) | Throughput (images/s) | Dataflow chain like FINN/hls4ml; highest area |

## 5. Open questions for the team

1. **Goal:** Conv0 demo (A), reusable engine (B), full on-chip inference (C),
   or throughput pipeline (D)? Which does the capstone/Ericsson evaluation reward?
2. **Metric:** single-image latency, images/second, or energy/area efficiency?
3. **Schedule:** when must a board result exist? Do we keep the current design
   only as the DMA/board bring-up vehicle while redesigning in parallel?
4. **Baseline:** we have only a dev-machine NumPy Conv0 time (1.26 ms median,
   [reports/README.md](../arch1_broadcast_832/reports/README.md)), not an A53 measurement of Conv0
   or the full network. Do we need one for a speedup claim?
5. **Clock:** stay at 100 MHz (met with +0.365 ns slack) or target 200+ MHz?
6. **Quantization:** keep compiled constants, or make them runtime-loadable
   (required for any multi-layer engine)?
7. **Verification capacity:** how much RTL complexity can the team realistically
   understand and verify in the remaining time?

## 6. Decisions

| ID | Decision |
|---|---|
| D-000 | Accelerate full ResNet-8 inference in hardware (option B/C), not Conv0 alone. On-chip vs DDR activations still open (step 03/04). |
