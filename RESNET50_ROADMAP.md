# Custom RTL on KV260: MNIST to ResNet-50

Prepared September 24, 2026. This is a proposed engineering plan, not a report of hardware already working.

## Start here

**MNIST is the hello world of learning machine learning. A custom FPGA implementation of MNIST is already a hardware/software system project. ResNet-50 is a major increase in arithmetic, memory traffic, and verification work.**

**Updated scope from the team:** seven people, limited weekly availability, a year-long class, and a semester goal of documenting how AI affects accelerator design. A later phase aims at a TPU-like architecture. The earlier full-ResNet-50-by-semester-end idea is retained as a stretch, not the recommended commitment.

The recommended semester deliverable is a reproducible design-and-verification flow, a measured comparison of development with and without AI, and at least one useful custom RTL accelerator deployed on KV260. Use MNIST to validate the flow and ResNet-50 to select realistic future workloads. Start with correctness and modest parallelism. Optimize after the experimental method and hardware work.

Your decision is to write your own RTL. No HLS or generated HLS hardware is required by this plan. AMD infrastructure IP such as DMA, interconnect, clocking, and memory primitives is assumed acceptable. The CNN computation is your RTL. Confirm that interpretation with the instructor before treating it as the grading contract.

## Reading order and why these files exist

| File | What it answers | Why a separate document helps |
|---|---|---|
| [ML_AND_KV260_EXPLAINED.md](ML_AND_KV260_EXPLAINED.md) | What a network computes, quantization, why ResNet is harder, what is inside the board | Gives everyone the same vocabulary before discussing RTL |
| [RTL_ARCHITECTURE_AND_DATAFLOW.md](RTL_ARCHITECTURE_AND_DATAFLOW.md) | Where every kind of data lives, how it moves, what hardware we build | Prevents arithmetic code from being designed without a workable memory/interface plan |
| [SEMESTER_SCHEDULE.md](SEMESTER_SCHEDULE.md) | Seven roles, weekly tasks, hours, dependencies, acceptance gates, and fallback decisions | Makes progress measurable and exposes an unrealistic deadline early |
| [RISKS_AND_VERIFICATION.md](RISKS_AND_VERIFICATION.md) | Likely failures, verification, and a fair AI-impact experiment | Separates numerical, protocol, physical implementation, and research-method problems |

These are Markdown because the team can review them in Git, link them from issues, and revise the engineering contract as evidence arrives. The architecture diagrams use Mermaid; the accompanying prose explains the same flow if the renderer does not support it.

## What the eventual “build ResNet-50” goal means

Use a pretrained checkpoint. Training ResNet-50 from scratch is a different project and is outside this schedule. Target one fixed model variant, batch size 1, and 224 x 224 RGB inputs. The host can resize and normalize images, export/load weights, choose operations, launch transfers, and display predictions.

The eventual full hardware goal executes convolutions, dense arithmetic, residual additions, activation functions, pooling, and runtime quantization conversions in your RTL. Batch normalization is folded into weights offline. The CPU may schedule every layer; full-network hardware arithmetic does not require an autonomous FPGA scheduler. The semester's minimum hardware milestone can be one verified kernel integrated into a mixed CPU/FPGA application.

| Outcome | What the demo actually proves |
|---|---|
| Bootcamp DPU ResNet | The board runs a vendor accelerator; useful bring-up baseline, not your custom accelerator |
| Custom MNIST | Your arithmetic, controls, and host integration work for the small model |
| Partial ResNet acceleration | Selected layers/blocks execute in your RTL; remaining arithmetic executes on CPU and is disclosed |
| Full ResNet arithmetic in RTL | All inference operators execute in your custom datapath, with host orchestration allowed |
| Optimized ResNet accelerator | Full inference plus a measured performance/resource target; a further goal |

The fallback outcomes are planning options, not permission to redefine the final requirement. Get the instructor's agreement on what earns credit.

## Time budget: realistic estimates, including heavy AI use

These are my engineering estimates for students learning this stack, not published productivity measurements or guarantees. They assume working hardware access, a usable Vivado host, frequent mentoring, pretrained models, reuse of standard AXI/DMA IP, and disciplined verification. They include learning, debugging, and documentation.

| Cumulative engineering milestone from the current state | Estimated team person-hours | Arithmetic-only calendar at 35 team hours/week |
|---|---:|---:|
| Reproducible software references and reliable board data transfer | 60-110 | 2-4 weeks |
| Custom MNIST working and verified on the board | 150-250 | 5-8 weeks |
| ResNet bottleneck and projection block working on hardware | 300-450 | 9-13 weeks |
| Full ResNet-50 arithmetic on custom RTL, modest performance | 550-850 | 16-25 weeks |
| Robust optimization for a demanding throughput target | 850-1,300+ | 25-38+ weeks |

These ranges overlap and are not additive. They describe an engineering path, not the extra work of controlled AI comparisons. Calendar divisions are optimistic capacity estimates: learning, interfaces, shared board access, and final integration create serial work, so seven people do not give seven times the throughput of one person.

**Planning recommendation for this semester: budget 300-450 team hours for a documented, evaluated flow and a bounded hardware demonstration.** Full custom ResNet-50 deserves a separate roughly 650-850-hour engineering budget including contingency; it should not consume the time needed to answer the actual AI-impact research question.

The detailed calendar assumes seven students x five hours/week, September 28 through December 20, with an illustrative December 18 demo. That is 420 available hours: 310 hours of planned tasks and 110 hours of contingency. Only team size and the broad goals are confirmed. Individual hours, the real demo date, and the start of the later TPU phase are not yet known. Use the schedule relatively if your dates differ.

AI is already included in these estimates. Do not divide them by an assumed 5x or 10x speedup. AI can accelerate code and test generation; it cannot remove memory bandwidth limits, hardware build time, missing data, or the need to understand a failed handshake.

## What the repository currently establishes

- model.py defines two convolutions and two fully connected layers for MNIST.
- train_mnist.py uses normalization `(pixel / 255 - 0.1307) / 0.3081`.
- `mnist_model.pth` is present. Exporting that checkpoint provides our own trained weights.
- [README.md](README.md) records 99.11% PyTorch accuracy and 99.06% HLS C++ accuracy. These are recorded prior results, not rerun measurements in this review.
- `hls4ml_prj/` and `convert_hls.py` are legacy artifacts from the abandoned HLS direction. Their simulation results do not validate new RTL. They are left intact as historical work.
- No deployable custom bitstream or hardware validation results were found in the inspected files.
- [PYNQ 201 MNIST](../PYNQ_Bootcamp/bootcamp_sessions/PYNQ%20201%20-%20MNIST/PYNQ%20201%20-%20MNIST.ipynb) supplies a useful dataset/evaluation example. [PYNQ 101](../PYNQ_Bootcamp/bootcamp_sessions/PYNQ%20101%20-%20Intro%20to%20PYNQ/PYNQ101_intro_to_pynq.ipynb) includes a precompiled ResNet demo. Their DPU execution calls must be replaced for our hardware.

## Decisions to settle in the first meeting

1. Actual demo date, individual hours, and access to the board/build machine.
2. Whether all network arithmetic must execute in custom RTL, and what fallback demonstration is acceptable.
3. Whether there is a hard FPS/latency target. “Runs correctly” and “30 FPS” are very different scopes.
4. Which AI-assisted development methods from [projectflow.md](projectflow.md) to compare. Start with two conditions on bounded matched tasks; three independently developed full accelerators will exceed this plan.
5. A fixed pretrained ResNet-50 variant/checkpoint and access to labeled evaluation data.
6. The definition of acceptable quantization accuracy loss.

## Immediate first milestone

Within two weeks, freeze the first arithmetic/interface specification, establish software reference outputs for a tiny kernel, and demonstrate board register access or loopback. By week four, complete reliable DMA loopback and the first synthesis-backed kernel. Load pretrained ResNet-50 in software early enough to inform reusable interfaces, without making full ResNet support a prerequisite for the semester experiment.
