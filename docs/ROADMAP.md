# Project Roadmap: Fall 2026 and Spring 2027

**UT Austin ECE Senior Design**

**Weekly progress meetings: Mondays**

Our first objective is to deploy the existing MNIST model on the AMD Kria KV260, verify its results, and benchmark its performance. With the Python baseline and much of the initial setup already available, this week's work will focus on defining the hardware architecture and beginning implementation.

The schedule targets an initial MNIST deployment by October 12, with the following week reserved for remaining integration work and benchmarking. ResNet50 development will begin once MNIST is working reliably and its results have been documented.

This semester establishes the implementation, verification, and measurement foundation for next semester's capstone. The capstone architecture and performance goals will be selected using the fall results and discussions with the project partners.

## MNIST implementation

Each Monday meeting will review the preceding week's work.

| Monday meeting | Planned progress for review |
|---|---|
| **October 5** | Review the MNIST hardware architecture, numerical formats, memory organization, and module interfaces. Review initial compute-block development and CPU-to-FPGA communication. Create the architecture and dataflow of our system. |
| **October 12** | Begin integration of the various RTL compute blocks, controller, and DMA path together. Create software harness for CPU-FPGA interactions. Review synthesis and implementation results. |
| **October 19** | Resolve remaining correctness or integration issues, repeat board measurements, and document deployment. Review accuracy, latency, throughput, resource usage, and development effort before beginning ResNet50. |

The numerical reference, RTL, verification, and board interface will be developed in parallel. Data-transfer testing will proceed independently of the compute blocks so that transport problems can be identified before full-system integration. MNIST will be considered complete when repeated FPGA inference agrees with the quantized reference, deployment can be reproduced, and measurements distinguish compute time from transfer and software overhead.

## ResNet50 development

These dates assume MNIST is complete by October 19. Any remaining MNIST work will take priority, and the subsequent dates will be adjusted accordingly.

| Monday meeting | Planned progress for review |
|---|---|
| **October 26** | Review the selected ResNet50 model, quantization approach, and accelerator architecture. Define the required changes to the compute engine, memory layout, layer sequencing, and residual-data handling. |
| **November 2** | Review additional operator support and initial bottleneck integration. Verify the required convolution configurations and residual arithmetic against the Python reference. |
| **November 9** | Demonstrate a complete bottleneck block, including identity and projection shortcuts. Review correctness, buffer usage, and synthesis results. |
| **November 16** | Demonstrate multiple connected blocks, including a transition between network stages. Verify scheduling, weight loading, and intermediate-data movement. |
| **November 23** | Review full-network integration, including the input layers, bottleneck stages, global average pooling, and final classifier. Begin complete inference tests and identify remaining discrepancies. |
| **November 30** | Target complete quantized ResNet50 inference through synthesizable RTL. Review agreement with the reference model and synthesis results. |
| **December 7** | Complete final regression, documentation, and reporting. Present hardware results, remaining limitations, and findings from the AI-assisted development process. |

ResNet50 will use a configurable accelerator that reuses compute resources across layers. DMA, host control, arithmetic blocks, and verification infrastructure developed for MNIST will be retained where suitable.



## Spring 2027 capstone direction

The proposed spring direction is a **domain-specific inference accelerator with a TPU-inspired compute engine**. A TPU-style design is one form of domain-specific architecture: it organizes computation and memory around neural-network operations. For this project, the scope would be a small programmable tensor accelerator on FPGA, with a systolic multiply-accumulate array, local buffers, DMA, supporting activation and residual operations, and a command interface for sequencing work. The [original TPU architecture paper](https://arxiv.org/abs/1704.04760) provides background on the relationship between matrix computation and software-managed memory.

The spring work will build on the fall implementation rather than require a new platform. We expect to retain the Python references, numerical contracts, host interface, transfer infrastructure, and regression framework. The final compute organization will depend on what the fall measurements show. If memory traffic limits performance, improving data reuse and scheduling may be more useful than increasing the number of arithmetic units.

The following phases are a broad plan, with detailed dates and performance targets to be agreed after the fall review.

| Period | Proposed focus | Expected outcome |
|---|---|---|
| **January** | Review fall results with the project partners, select the workload and architecture, and define comparison criteria. Complete any prerequisite board integration. | Agreed capstone scope, numerical and interface specifications, and a reproducible baseline. |
| **February** | Implement and verify the selected compute engine, local memory organization, and command sequencing. Integrate them with the existing host and DMA infrastructure. | A working accelerator that executes representative layers and blocks on the FPGA. |
| **March** | Integrate complete inference and evaluate compute utilization, memory traffic, and software overhead. If the first workload is stable, add a second supported network using the same hardware. | Correct end-to-end board execution and measurements showing the main performance limits. |
| **April through final review** | Evaluate a focused architectural improvement, complete regression and benchmarking, and prepare the demonstration and handoff. | Reproducible hardware results, an architectural comparison, and a documented assessment of the AI-assisted workflow. |

The principal goal is complete, verified inference on KV260 with documented performance. A further goal is to run two supported networks through the same bitstream by changing weights and execution commands. Architectural experiments could compare array sizes, buffering strategies, or dataflows; the team will select a bounded comparison once a reliable baseline exists. Throughput, latency, resource, and accuracy targets will be set from that baseline rather than assumed in advance.

AI tools are intended to support substantial RTL implementation and iteration. The team will own the architecture, specifications, independent acceptance tests, and review of synthesis and board results. The extent of architectural exploration will depend on progress and the tool support available through ChipAgents. Alongside hardware measurements, we will record engineering time, debugging effort, verification iterations, and incomplete attempts so the final report can assess the development process as well as the accelerator.

For implementation tasks and acceptance criteria, see the [issue backlog](ISSUE_BACKLOG.md). The [project flow](../projectflow.md), [profiling guide](../PROFILING.md), and [kickoff summary](meeting-notes/2026-09-kickoff.md) provide supporting context.
