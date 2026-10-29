# Project history

A running record of how this project has evolved: what we tried, what we kept,
what we dropped, and why. Add a dated entry at the bottom of the timeline
whenever direction changes or a milestone lands. Dates are 2026 unless noted.
Branch/commit references point to where the evidence lives.

## How we operate

- **Weekly Monday reviews** of working evidence (see [docs/ROADMAP.md](docs/ROADMAP.md)).
  Partner check-ins with Ericsson (e.g. the October 20 meeting).
- **Verification first.** A frozen software golden model and bit-exact test
  vectors come before RTL. RTL is accepted only when it matches the vectors
  element for element.
- **Evidence labels.** Every number is tagged as measured (simulation, Vivado
  report, board), derived (arithmetic), or assumed (estimate). Model numbers are
  never presented as measurements.
- **AI-assisted development is part of the research question**: how AI
  assistance affects the effort and quality of accelerator development
  ([RISKS_AND_VERIFICATION.md](RISKS_AND_VERIFICATION.md)). Much of the code and
  docs were generated with AI agents. Lesson learned (late October): generated
  architecture that the team does not understand is a liability. Since then,
  architecture decisions go through the step-by-step
  [uarch_study](resnet8_project/uarch_study/README.md) with an explicit decision log.
- **Branches:** `main` (merged milestones), `schedule` (planning, now the
  reorganized tree), `resnet8-conv0` (Conv0 RTL publication), `hls4l` (hls4ml
  experiment), `python-hardware-models` (AI-generated Python accelerator model).

## Timeline

### Phase 0 – MNIST baseline (Sept 17–18)

- Uploaded a PyTorch MNIST CNN (2 conv + 2 FC, 28×28 grayscale) with training,
  application and checkpoint (`mnist_model.pth`). PyTorch accuracy 99.11% on the
  10,000-image test set.
- Added inference profiling (`profile_mnist.py`, layer/operator breakdown) to
  find where time goes before designing hardware (`fbca208`).
- **hls4ml experiment** (`hls4l` branch, `e862f59`, `d3d5033`): converted the
  MNIST model with hls4ml 1.3.0. C++ simulation reached 99.06% accuracy with
  99.95% prediction agreement against PyTorch. This was HLS C-sim only, not RTL
  or board results. It showed that a push-button HLS flow works, but the team
  chose custom RTL so the architecture would be our own.

### Phase 1 – Planning and scope (Sept 24–28)

- Kickoff direction ([docs/meeting-notes/2026-09-kickoff.md](docs/meeting-notes/2026-09-kickoff.md)):
  build a reproducible Python → architecture → RTL → verification → KV260
  flow. Process and methodology count as deliverables.
- Planning docs written: [RESNET50_ROADMAP.md](RESNET50_ROADMAP.md),
  [RTL_ARCHITECTURE_AND_DATAFLOW.md](RTL_ARCHITECTURE_AND_DATAFLOW.md),
  [ML_AND_KV260_EXPLAINED.md](ML_AND_KV260_EXPLAINED.md),
  [SEMESTER_SCHEDULE.md](SEMESTER_SCHEDULE.md), [projectflow.md](projectflow.md).
- Plan of record (`62af0e2`–`8307b4e`): deploy MNIST on KV260 by October 12, then
  scale toward ResNet-50 with time-multiplexed compute. A TPU-like array was
  named as a later capstone direction.

### Phase 2 – Pivot to ResNet-8 and quantization (Sept 28 – Oct 1)

- Target changed from MNIST/ResNet-50 to **MLPerf Tiny ResNet-8**
  (`resnet_v1_eembc(conv_filters=26)`, CIFAR-10). The MNIST board deployment was
  never completed. ResNet-8 is a real benchmark network, and its ~210 KB of
  parameters fit entirely on chip.
- Built the frozen software reference ([resnet8_project/resnet8](resnet8_project/resnet8/README.md)):
  - FP32 TensorFlow model instrumented to dump every tensor. The upstream
    master checkpoint turned out to be 16-channel, so the official 26-channel
    H5 was selected.
  - **BatchNorm folding** into conv weights and bias (max error ~2.2e-5).
  - **INT8 quantization**: symmetric, per-channel weights, calibrated on the
    first 512 training images. INT32 bias and accumulators. Requantization by
    integer `(M, R)` multiply and round-shift with an exactly specified rounding
    rule. Residual adds requantize both branches before summing.
  - Results on 1,000 test images: FP32 89.2%, INT8 87.9% (−1.3 pt), 94.9% top-1
    agreement ([RESULTS.md](resnet8_project/resnet8/RESULTS.md)).
  - Binary export (weights 203,190 B, bias 2,224 B, quant 4,504 B, 14 layer
    configs) and a [hardware contract](resnet8_project/resnet8/HARDWARE_CONTRACT.md)
    that fixes arithmetic, layouts, padding and file formats.
- First Conv0 RTL experiments (merged to `main` via PR #2, `5c210d4`):
  - **Fully parallel 27,648-MAC design** (1,024 pixels × 27 taps, one output
    channel): simulation passed, but synthesis needed 2.9 M LUTs (2,490% of the
    device). Lesson: full spatial unrolling is impossible on the K26.
  - **Serial 1-DSP engine** for all of Conv0: correct (79,872 comparisons) but
    far too slow.
  - **416-MAC shared-DSP engine**: 8,577 cycles per frame, passed 133,120
    comparisons. Synthesis showed 832 DSPs and 16 BRAM36, and standalone I/O
    exceeded the device pins.

### Phase 3 – Conv0 design-space exploration (Oct 15–16)

- Built a clock-stepped **Python cycle model** (`cycle_sim/`, removed in the
  cleanup but kept on `origin/resnet8-conv0`) with host DMA timing. 396 runs
  swept MAC counts, output widths, buffering and transfer rates.
- The INT8/AXI latency study switched output from INT32 to INT8 (4× less
  egress) and modeled AXI-Stream/AXI-Lite handshakes. It selected **832 MACs,
  104 outputs/beat, 16 B/clk loader, LUT requant**: 19.47 µs modeled cold frame
  ([AXI_INT8_STUDY.md](resnet8_project/arch1_broadcast_832/docs/AXI_INT8_STUDY.md)).
  Fallbacks: 416 MACs (34 µs) and 208 MACs (65 µs).

### Phase 4 – 832-MAC Conv0 on KV260 (Oct 19–22)

- Implemented the 832-MAC broadcast Conv0 core with AXI DMA, an INT8 requant
  pipeline with compiled quantization constants, and an A53 benchmark program
  ([arch1_broadcast_832](resnet8_project/arch1_broadcast_832/README.md)).
- XSim: 30 frames, 798,720 raw INT32 + 798,720 INT8 comparisons passed.
  1,239 cycles START → last output (12.39 µs @ 100 MHz).
- Routed at 100 MHz (+0.365 ns slack): 832 DSP, 81.7% LUT, **97.7% CLB**.
  Bitstream/XSA exported. Vitis build and board measurement are still pending
  ([CURRENT_STATUS.md](resnet8_project/arch1_broadcast_832/CURRENT_STATUS.md)).
- Parallel AI-generated Python accelerator model with an ISA, compiler and
  sequencer (`python-hardware-models` branch, `34d855a`). It has not been
  integrated.
- October 20 Ericsson meeting: presented simulation/synthesis evidence and the
  workload analysis ([reports/meeting_2026_10_20](resnet8_project/arch1_broadcast_832/reports/meeting_2026_10_20/ANALYSIS.md)).
  Onboarding docs and diagrams were published on `resnet8-conv0`.

### Phase 5 – Understanding and re-architecting (late October)

- Recognized that the 832-MAC design was largely AI-generated and not fully
  understood. It also accelerates only Conv0 (2.2% of network MACs), its
  latency is bound by output DMA, and it leaves no room on the device for
  other layers.
- Started the step-by-step [uarch_study](resnet8_project/uarch_study/README.md):
  first-principles bounds, the conv loop nest, systolic mapping, on-chip memory
  and a three-way comparison.
- Decisions so far: D-000 accelerate the full network; D-001 an
  output-stationary systolic array with 26 output-channel rows; D-002 keep
  activations on chip (A/B/C buffers, fused residual add); D-003 target 200 MHz;
  D-004 two accumulators per PE.
- Compared current vs OS 26×32 vs WS (TPU-style) 27×26
  ([03b](resnet8_project/uarch_study/03b_performance_comparison.md)): full
  inference ≈ 16–64 ms (current, CPU-bound) vs ≈ 201 µs vs ≈ 246 µs at
  200 MHz (derived).
- **Repository reorganization:**
  - Removed the MNIST code, superseded RTL experiments, the Vivado scratch
    project, the cycle model and stale status docs.
  - Split hardware into `arch1_broadcast_832/` (existing),
    `arch2_os_systolic_26x32/` and `arch3_ws_tpu_27x26/`, each with an
    `ARCHITECTURE.md`.
  - Removed material remains in git history (`main`, `hls4l`, `resnet8-conv0`).

## Things we tried and dropped

| Idea | Outcome | Why dropped |
|---|---|---|
| hls4ml for MNIST | 99.06% in C-sim | Want custom, understood RTL architecture |
| MNIST on KV260 first | Not deployed | Pivoted to a real benchmark network (ResNet-8) |
| ResNet-50 | Planning only | Too large for the semester; ResNet-8 fits on chip |
| Fully parallel Conv0 (27,648 MACs) | Sim pass, 2,490% LUTs | Does not fit |
| Serial 1-DSP Conv0 | Correct | Too slow |
| 416-MAC Conv0 | Sim pass, synth 832 DSP | Superseded by INT8/AXI study's 832-MAC choice |
| INT32 output transfer | Modeled | 4× more egress than INT8 |
| Compiled-in quant constants (arch1) | Saves LUTs | Cannot generalize to other layers |
| Conv0-only acceleration | Works in sim/route | ~1.02× full-network speedup ceiling (Amdahl) |

## Open items

- Board bring-up of arch1 (Vitis build, measured latency, CPU baseline on A53).
- Choose between arch2 and arch3 and finish uarch study steps 05–09 (PE/DSP
  mapping, output path, host interface, full-network control, verification).
- Python model of the chosen engine's feeder and sequencer before writing RTL.
