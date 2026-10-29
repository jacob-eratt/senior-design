# Microarchitecture study: understand, question, redesign

Purpose: rebuild team understanding of the accelerator from first principles,
one step at a time, and use that understanding to decide whether the current
832-MAC Conv0 design ([arch1_broadcast_832](../arch1_broadcast_832/README.md)) is the
right base or whether a different microarchitecture is better.

Ground rules:

- Every claim is either **derived** (show the arithmetic), **measured** (link the
  report), or **assumed** (label it). Model numbers are not measurements.
- Each step ends with open questions and, once agreed, a logged decision.
- The current design is the baseline to beat, not the answer to defend.

## Step template

Each step file uses the same sections:

1. **Concept** – the idea in plain terms.
2. **Current design** – what the existing RTL does, with file references.
3. **Numbers** – derived bounds and measured evidence.
4. **Alternatives** – other ways to do it, with trade-offs.
5. **Open questions** – what the team must answer.
6. **Decisions** – filled in after discussion.

## Steps

| # | Topic | Status |
|---|---|---|
| 01 | [Problem definition and first-principles bounds](01_problem_and_bounds.md) | Goal decided (D-000) |
| 02 | [A convolution as a loop nest: the vocabulary of every dataflow](02_conv_loop_nest.md) | Draft |
| 03 | [Systolic array mapping: 26×32 vs 26×16](03_systolic_array_mapping.md) | Draft, column count open |
| 03b | [Performance/resource comparison: current vs OS 26×32 vs WS 27×26](03b_performance_comparison.md) | Draft |
| 04 | [Where everything lives on chip (buffers, layout, feeder rate, budget)](04_on_chip_memory.md) | Draft |
| 05 | The MAC: DSP48E2, INT8 packing, pipelining, clock frequency | Not started |
| 06 | Accumulate, emit overlap, requantization and output path | Not started |
| 07 | Host interface: DMA, AXI, software overhead, what latency really means | Not started |
| 08 | Generalizing to all ResNet-8 layers (stride 2, 1×1, residual, pool, dense) | Not started |
| 09 | Candidate microarchitectures, comparison and verification plan | Not started |

## Decision log

| ID | Date | Decision | Rationale | Step |
|---|---|---|---|---|
| D-000 | 2026 | Goal: hardware accelerator for full ResNet-8 inference, not a Conv0-only demo | Conv0 is 2.2% of MACs and output-link bound | 01 |
| D-001 | 2026 | General conv engine on a systolic array, 26 rows (output channels) × 16 or 32 columns (output pixels) | OC is always a multiple of 26; replaces hardcoded Conv0 core | 03 |
| D-002 | 2026 | Activations stay on chip: buffers A/B/C (26,624 B each) + IN; residual add fused into conv2 | Needed by next layer and residual; ~80 KB fits easily | 04 |
| D-003 | 2026 | Target 200 MHz | 26×16 @ 200 MHz matches 26×32 @ 100 MHz with half the DSPs | 03 |
| D-004 | 2026 | Two accumulators per PE, back-to-back tiles | Hides drain and tile skew (~20–25% otherwise) | 03 |
