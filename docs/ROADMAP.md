# Fall 2026 project roadmap

**UT Austin ECE Senior Design · Planning baseline: September 28, 2026**

Prepared for project reviews with Ericsson and ChipAgents. Dates below are engineering targets, not claims of completed work or sponsor commitments. The first release is complete MNIST inference on AMD Kria KV260. The second extends the reusable RTL architecture to complete quantized ResNet50 inference by the end of the semester.

## Scope and architecture strategy

Follow the [project flow](../projectflow.md): Python reference → architecture specification → RTL → independent verification → FPGA implementation → measurement. Preserve the existing Python baseline and [profiling methodology](../PROFILING.md).

ResNet50 must use a **configurable, time-multiplexed accelerator**, reusing convolution/MAC engines across layers and tiles. It will not physically instantiate 50 independent hardware layers. Freeze the exact model variant, checkpoint, preprocessing, quantization, and supported operator graph before claiming coverage. Plan for configurable 1x1/3x3 convolution, strides, projection shortcuts, residual addition, activation, global average pooling, and the final classifier; fold inference batch normalization only after checking equivalence.

Layer descriptors, tensor lifetimes, tiled buffers, and external-memory transfers must support the complete network. Full inference means every layer executes through synthesizable RTL under the declared memory/interface contract; host-side loading and control are allowed, but unimplemented neural-network operators must not silently fall back to Python. The December target requires end-to-end simulation evidence and synthesis results. Full ResNet50 deployment on KV260 is a follow-on objective unless integration capacity permits; MNIST board deployment remains an explicit fall gate.

## Monday review milestones

Each date is a Monday evidence review of the preceding week's work. September 28 is the planning kickoff. **Do not begin the ResNet50 workstream until MNIST is deployed, validated, and benchmarked on KV260 (M12 and M13 accepted).** The earliest planned transition is October 26. If MNIST needs additional time, shift ResNet work and explicitly reassess the December target; do not overlap the phases to conceal a missed gate.

| Target | Milestone | Reviewable acceptance evidence |
|---|---|---|
| Oct 5 | MNIST specification and board control smoke test | Reviewed graph/arithmetic/interface contract and accuracy budget; CPU reads/writes a known FPGA register; tool/board configuration recorded. Confirm provisional precision at the quantization gate. |
| Oct 12 | Quantized reference, vectors, and DMA loopback | Reproducible exports and edge vectors; float/quantized accuracy on all 10,000 test images meets the budget; repeated memory-to-stream-to-memory transfers return exact bytes; first compute engine synthesis reviewed |
| Oct 19 | Complete MNIST RTL simulation and board integration candidate | All layers/controller pass exact multi-image regression and reset/handshake tests; full synthesis and first implementation reports reviewed; host transport ready |
| Oct 26 | MNIST deployed, validated, and benchmarked; phase-transition decision | Repeated board jobs match integer logits; timing/resource reports and compute/transfer/end-to-end measurements archived; board CPU baseline and reproducible runbook completed. Accept M12/M13 before starting ResNet50. |
| Nov 2 | ResNet50 accelerator architecture frozen | Pinned graph, reference and operator coverage; reusable compute configuration, scheduler, residual storage, bandwidth/resource estimates, and simulation-runtime budget reviewed |
| Nov 9 | ResNet50 operator integration checkpoint | Configurable convolutions, residual arithmetic, and reference vectors pass unit checks; bottleneck integration and synthesis risks reviewed |
| Nov 16 | Complete ResNet50 bottleneck block verified | Identity and projection/downsampling cases match integer reference at intermediate tensors and block outputs; representative block synthesis report |
| Nov 23 | Multiple blocks, scheduler, memory infrastructure working | Chained blocks including a stage transition pass with backpressure; descriptor sequencing and buffer lifetimes checked; measured cycle/traffic counts |
| Nov 30 | Full-network integration readiness | Every required operator has tests; full graph executes in a smoke run or remaining blockers are explicitly listed; final long-running regressions and synthesis underway |
| Dec 7 | Complete quantized ResNet50 RTL inference target and final fall freeze | Full pinned graph matches reference logits on the declared suite; synthesis and regressions archived; documentation and workflow measurements frozen. Aim to finish inference runs by Dec 4 to reserve the weekend for packaging. |

## Week-by-week execution

Work is assigned at each Monday meeting and demonstrated at the next. MNIST reference, RTL, verification, and board integration progress concurrently against shared contracts; ResNet50 starts only after the MNIST phase gate. Dates after October 26 are conditional on that gate passing.

| Week | Engineering tasks | Weekly review output |
|---|---|---|
| Sep 28–Oct 4 | Reproduce checkpoint accuracy and profile; draft MNIST arithmetic and interfaces; inventory KV260/tool access; demonstrate CPU register access; establish effort logs | Oct 5: specification and board control review |
| Oct 5–11 | Build integer golden model and deterministic exports; implement convolution, activation/pooling, and dense units against unit tests; synthesize the compute engine; complete repeated DMA loopback | Oct 12: reference, vectors, loopback, and resource review |
| Oct 12–18 | Integrate all MNIST layers and controller; debug first divergent tensors; automate regression; synthesize full top level and attempt placement/routing; integrate host/control interface in parallel | Oct 19: complete MNIST simulation and integration review |
| Oct 19–25 | Close implementation timing; deploy bitstream; compare repeated board jobs against Python; measure compute, transfers, and end-to-end latency; benchmark board CPU; reproduce runbook | Oct 26: accept MNIST and authorize ResNet phase, or continue MNIST |
| Oct 26–Nov 1 | Only after MNIST acceptance: pin ResNet50 variant/weights/data; profile graph and storage; calibrate quantization; define residual scales and BN folding; select tiling, descriptors, memory layout, and supported configurations | Nov 2: architecture freeze and complete operator checklist |
| Nov 2–8 | Extend convolution configuration to required shapes/strides; implement residual add/requantization and projection path; validate folded reference; develop block-level vectors; start global pooling/classifier units | Nov 9: operator integration and synthesis review |
| Nov 9–15 | Verify identity and downsampling bottlenecks; stress shortcut lifetimes and partial tiles; synthesize block; build scheduler and memory transport in parallel | Nov 16: complete bottleneck gate and scheduler prototype |
| Nov 16–22 | Chain blocks across stage boundaries; integrate descriptors, external-memory model, and buffers; test stalls; finish stem, pooling, and classifier coverage; benchmark simulation runtime | Nov 23: multi-block regression and integration review |
| Nov 23–29 | Run full-network smoke inference early; debug tensor checkpoints; integrate remaining configuration coverage; schedule long regressions around holiday availability | Nov 30: full-graph readiness, remaining defects, synthesis status |
| Nov 30–Dec 6 | Aim for complete inference by Dec 4; run frozen regression; complete synthesis and resource/timing analysis; consolidate workflow metrics and rehearse documentation | Dec 7: complete RTL inference evidence and final review |
| Dec 7 | Freeze reviewed artifacts and reports; record unresolved limits and next steps; package sponsor-facing demonstration and handoff | Final fall release and reproducibility index |

## Seven-person execution model

Assign one primary owner to each workstream and a second reviewer for each interface. These are proposed responsibilities, not seven isolated implementations.

| Owner | Primary responsibility |
|---|---|
| 1 | Numerical architecture, quantized Python reference, accuracy and exports |
| 2 | Compute architecture, convolution/dense datapaths, AI-assisted implementation |
| 3 | Activation/pooling, local buffers, top-level sequencing and integration |
| 4 | Independent verification, assertions, regression and mismatch diagnosis |
| 5 | FPGA platform, AXI/DMA integration, clocks/resets and implementation timing |
| 6 | CPU software, DMA buffers/cache handling, host control, board benchmarking |
| 7 | System architecture/contracts, build reproducibility, integration coordination and workflow measurements |

Seven people make concurrent work possible, but they do not remove dependencies or multiply integration speed by seven. As a capacity example, 10 focused hours per person per week gives 70 team-hours before meetings, learning, and integration overhead; actual availability must be recorded before treating dates as commitments. October 26 is an aggressive MNIST target. If the October 12 reference/transport gate fails, revise the forecast immediately. A one- or two-week MNIST overrun directly reduces the already short ResNet phase.

AI is intended to perform substantial RTL drafting, test scaffolding, and implementation iteration from human-owned specifications. Humans own architecture, arithmetic, interfaces, independent acceptance tests, physical implementation review, and board sign-off. Every AI-produced block must pass the same gates as handwritten RTL. Internet references and reusable IP can reduce implementation work; record source, version, license, modifications, and validation. Use established platform/DMA IP where suitable rather than making a new DMA controller a project deliverable.

## CPU, memory, and FPGA integration

Treat transport as a first-class subsystem from week one. The initial candidate is CPU control through AXI-Lite, bulk transfers between DDR and accelerator streams through AMD AXI DMA, and on-chip buffers around the compute engine. Freeze the actual software environment, address map, transfer format, buffer ownership/cache policy, packet boundaries, resets, completion and timeout behavior in the MNIST specification. Begin with register access, then exact-byte loopback, then a compute kernel, then complete inference.

The current model contains 421,642 parameters, including biases. Uniform 8-bit storage would occupy about 412 KiB; uniform 16-bit storage about 824 KiB, excluding scales, wider biases/accumulators, activations, and memory packing overhead. The K26 has 144 x 36 Kib block RAM (648 KiB raw) and 64 x 288 Kib UltraRAM (2,304 KiB raw). These are capacity bounds, not proof of a feasible mapping: ports, widths, banking, buffering, and other IP consume resources. Validate actual inference in synthesis before choosing resident weights versus tiled DDR loading. See the [AMD K26 programmable-logic specification](https://docs.amd.com/r/en-US/ds987-k26-som/Programmable-Logic).

DMA is a data-movement choice, not a consequence solely of whether weights fit in BRAM. Even resident weights require an input/output and initialization path. Small MNIST transfers could use CPU-driven access, but a proven DMA path is useful infrastructure for the subsequent larger network. AMD's [AXI DMA overview](https://docs.amd.com/r/en-US/pg021_axi_dma/Core-Overview) describes memory-to-stream and stream-to-memory transfer channels. Benchmark setup/transfer overhead rather than assuming DMA always lowers latency for tiny inputs.

Reusable assets include the CPU driver/control contract, DMA transport, buffer utilities, MAC/requantization logic, build scripts, and regression framework. ResNet50 still requires new operator configurations, residual tensor lifetimes, scheduling and memory planning; existing MNIST blocks must demonstrate those configurations before being counted as reusable.

## Verification and measurement

Keep the oracle independent of generated RTL. Require exact integer agreement at unit, layer, block, and network boundaries; classification agreement alone is insufficient. Test extremes, rounding ties, saturation, padding, reset, back-to-back jobs, and stalls. Separate quantization accuracy loss from RTL implementation errors. Fix numerical acceptance budgets before evaluating the final held-out dataset.

Archive source revision, checkpoint/export hashes, seeds, tool versions, commands, and failures with each gate. Report LUT/FF/DSP/memory utilization, constrained clock and achieved timing, cycles, throughput, transfer costs, and end-to-end latency. Label synthesis estimates separately from implemented or board-measured results; report power only with a stated measurement method.

Track active engineering time, elapsed/tool-wait time, AI tool/version and task context, review/debug effort, iteration counts, first-pass acceptance, and reproducibility. Compare workflows on the same specification and tests; retain unsuccessful attempts and avoid general productivity claims from a small sample.

## Dependencies and schedule control

This is an aggressive integration schedule. Board/tool access and transfer tests begin immediately, synthesis begins at the first compute engine, and verification develops alongside RTL. Accepted MNIST deployment and benchmarking are prerequisites for the ResNet phase. Dataset/checkpoint availability, memory bandwidth, quantization accuracy, and simulation runtime must be reviewed at the conditional Nov 2 architecture freeze. Complete ResNet50 RTL inference remains the semester target, with lower confidence than the MNIST gate; do not describe it as guaranteed based on team size or AI access.

Review progress weekly against evidence rather than code volume. If a gate slips, first reduce lane count, throughput ambition, or nonessential optimization while preserving complete graph coverage and correctness. Full-network simulation may use a small declared input suite, with broader operator/block regression and separately reported Python accuracy. Never report partial-network coverage as complete inference. Any scope or date revision must be recorded explicitly with its impact and recovery plan.

See the [initial issue backlog](ISSUE_BACKLOG.md) for MNIST work packages and dependencies, and the [kickoff summary](meeting-notes/2026-09-kickoff.md) for methodology priorities.
