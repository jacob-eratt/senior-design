# September 2026 kickoff — project direction

This public summary captures project direction and planned actions. It is not a verbatim transcript and does not attribute statements or commitments to individual attendees or sponsors.

## Shared direction

- Build an end-to-end Python → architecture → RTL → verification → FPGA flow, with AMD Kria KV260 as the hardware target.
- Treat the process and methodology as major project outcomes alongside working hardware. Preserve specifications, decisions, reproducible commands, and evidence so another engineer can repeat the flow.
- Make verification first-class: establish an independent numerical reference, deterministic test vectors, automated comparisons, and explicit acceptance gates before accepting RTL.
- Start with a small but complete MNIST flow using the existing model and checkpoint. Validate every layer and deploy the integrated network before expanding scope.
- Scale the reusable architecture toward complete ResNet50 inference. Reuse configurable compute resources over time, with explicit scheduling and memory management.
- Measure hardware quality and engineering productivity. Record resources, timing, latency, throughput, and transfer overhead together with human effort, AI-assisted iterations, review/debug work, and reproducibility.

## Immediate follow-up

1. Freeze MNIST graph, arithmetic, tensor layout, and interfaces.
2. Establish quantized Python references and layer-by-layer export infrastructure.
3. Develop RTL and independent verification together; begin synthesis and KV260 bring-up early.
4. Maintain weekly evidence-based reviews against the [Fall roadmap](../ROADMAP.md).

Existing Python files remain in place. The [README](../../README.md) documents a future directory layout; migration is a separate task. The [project flow](../../projectflow.md) and [profiling guide](../../PROFILING.md) remain the baseline methodology references.
