# Fall 2026 project roadmap

**UT Austin ECE Senior Design · Planning baseline: September 28, 2026**

Prepared for project reviews with Ericsson and ChipAgents. Dates below are engineering targets, not claims of completed work or sponsor commitments. The first release is complete MNIST inference on AMD Kria KV260. The second extends the reusable RTL architecture to complete quantized ResNet50 inference by the end of the semester.

## Scope and architecture strategy

Follow the [project flow](../projectflow.md): Python reference → architecture specification → RTL → independent verification → FPGA implementation → measurement. Preserve the existing Python baseline and [profiling methodology](../PROFILING.md).

ResNet50 must use a **configurable, time-multiplexed accelerator**, reusing convolution/MAC engines across layers and tiles. It will not physically instantiate 50 independent hardware layers. Freeze the exact model variant, checkpoint, preprocessing, quantization, and supported operator graph before claiming coverage. Plan for configurable 1x1/3x3 convolution, strides, projection shortcuts, residual addition, activation, global average pooling, and the final classifier; fold inference batch normalization only after checking equivalence.

Layer descriptors, tensor lifetimes, tiled buffers, and external-memory transfers must support the complete network. Full inference means every layer executes through synthesizable RTL under the declared memory/interface contract; host-side loading and control are allowed, but unimplemented neural-network operators must not silently fall back to Python. The December target requires end-to-end simulation evidence and synthesis results. Full ResNet50 deployment on KV260 is a follow-on objective unless integration capacity permits; MNIST board deployment remains an explicit fall gate.

## Milestones

| Target | Milestone | Reviewable acceptance evidence |
|---|---|---|
| Oct 2 | MNIST architecture/specification frozen | Reviewed graph matching `model.py`; preprocessing, tensor order, signed widths, scales, accumulator bounds, rounding, saturation, interfaces, and accuracy-loss budget documented |
| Oct 9 | Quantized Python golden model and test-vector infrastructure | Reproducible checkpoint/weight/input manifests; layer outputs and arithmetic edge vectors; float-versus-quantized accuracy on all 10,000 test images against the frozen budget |
| Oct 18 | Complete MNIST RTL simulation | All layers and controller integrated; exact integer comparison on a frozen multi-image regression plus directed arithmetic/reset/handshake tests; zero unexplained mismatches |
| Oct 25 | MNIST running on KV260 | Reproducible build and host runbook; board outputs match golden vectors across repeated jobs; resource, timing, and latency evidence archived |
| Nov 1 | ResNet50 accelerator architecture frozen | Pinned graph and operator coverage; reusable compute configuration, scheduler, residual storage, bandwidth/resource estimates, and simulation-runtime budget reviewed |
| Nov 15 | Complete ResNet50 bottleneck block verified | Identity and projection/downsampling cases match integer reference at intermediate tensors and block outputs; representative block synthesis report |
| Nov 22 | Multiple blocks, scheduler, memory infrastructure working | Chained blocks including a stage transition pass with backpressure; descriptor sequencing and buffer lifetimes checked; measured cycle/traffic counts |
| Dec 4 | Complete quantized ResNet50 inference through synthesizable RTL | Full pinned graph produces exact reference logits for a declared frozen input suite; all operators covered; synthesis completes without unsupported constructs |
| Dec 7 | Final fall freeze | Regressions archived, synthesis results and limitations reported, MNIST board reproduction complete, documentation and workflow measurements frozen |

## Week-by-week execution

Roles below identify workstreams; assign named owners at the weekly review. Software/reference, RTL, verification, and board integration should progress concurrently against shared contracts.

| Week | Engineering tasks | Weekly review output |
|---|---|---|
| Sep 28–Oct 4 | Reproduce checkpoint accuracy and profile; freeze MNIST arithmetic and interfaces by Oct 2; inventory KV260/tool access; start board control/transfer smoke test; establish effort logs | Approved specification, baseline manifest, board bring-up status |
| Oct 5–11 | Build integer golden model and deterministic exports by Oct 9; implement convolution, activation/pooling, and dense units against unit tests; synthesize the compute engine early; continue control/transfer loopback | Export schema, corner-case vectors, unit regressions, early resource estimate |
| Oct 12–18 | Integrate all MNIST layers and controller; debug first divergent tensors; automate regression; synthesize full top level; integrate host/control interface in parallel | Complete MNIST simulation gate and integration build |
| Oct 19–25 | Close implementation timing; deploy bitstream; compare repeated board jobs against Python; measure compute and end-to-end latency separately; capture reproducible runbook | MNIST board demonstration and evidence package |
| Oct 26–Nov 1 | Pin ResNet50 variant/weights/data; profile graph and storage; calibrate quantization; define residual scales and BN folding; select tiling, descriptors, memory layout, and supported configurations | ResNet50 architecture freeze and complete operator checklist |
| Nov 2–8 | Extend convolution configuration to required shapes/strides; implement residual add/requantization and projection path; validate folded reference; develop block-level vectors; start global pooling/classifier units | Verified operators and first bottleneck integration; synthesis feedback |
| Nov 9–15 | Verify identity and downsampling bottlenecks; stress shortcut lifetimes and partial tiles; synthesize block; build scheduler and memory transport in parallel | Complete bottleneck gate and scheduler prototype |
| Nov 16–22 | Chain blocks across stage boundaries; integrate descriptors, external-memory model, and buffers; test stalls; finish stem, pooling, and classifier coverage; benchmark simulation runtime | Multi-block regression and full-network integration candidate |
| Nov 23–29 | Run full-network smoke inference early; debug tensor checkpoints; integrate remaining configuration coverage; schedule long regressions around holiday availability | First full-graph result, remaining defects, synthesis readiness |
| Nov 30–Dec 6 | Meet Dec 4 full-inference gate; run frozen regression; complete synthesis and resource/timing analysis; consolidate workflow metrics and rehearse documentation | Complete RTL inference evidence and release candidate |
| Dec 7 | Freeze reviewed artifacts and reports; record unresolved limits and next steps; package sponsor-facing demonstration and handoff | Final fall release and reproducibility index |

## Verification and measurement

Keep the oracle independent of generated RTL. Require exact integer agreement at unit, layer, block, and network boundaries; classification agreement alone is insufficient. Test extremes, rounding ties, saturation, padding, reset, back-to-back jobs, and stalls. Separate quantization accuracy loss from RTL implementation errors. Fix numerical acceptance budgets before evaluating the final held-out dataset.

Archive source revision, checkpoint/export hashes, seeds, tool versions, commands, and failures with each gate. Report LUT/FF/DSP/memory utilization, constrained clock and achieved timing, cycles, throughput, transfer costs, and end-to-end latency. Label synthesis estimates separately from implemented or board-measured results; report power only with a stated measurement method.

Track active engineering time, elapsed/tool-wait time, AI tool/version and task context, review/debug effort, iteration counts, first-pass acceptance, and reproducibility. Compare workflows on the same specification and tests; retain unsuccessful attempts and avoid general productivity claims from a small sample.

## Dependencies and schedule control

This is an aggressive integration schedule. Board/tool access and transfer tests begin immediately, synthesis begins at the first compute engine, and verification develops alongside RTL. MNIST interface stability and reusable arithmetic are prerequisites for the ResNet phase. Dataset/checkpoint availability, memory bandwidth, quantization accuracy, and simulation runtime must be resolved at the Nov 1 freeze.

Review progress weekly against evidence rather than code volume. If a gate slips, first reduce lane count, throughput ambition, or nonessential optimization while preserving complete graph coverage and correctness. Full-network simulation may use a small declared input suite, with broader operator/block regression and separately reported Python accuracy. Never report partial-network coverage as complete inference. Any scope or date revision must be recorded explicitly with its impact and recovery plan.

See the [initial issue backlog](ISSUE_BACKLOG.md) for MNIST work packages and dependencies, and the [kickoff summary](meeting-notes/2026-09-kickoff.md) for methodology priorities.
