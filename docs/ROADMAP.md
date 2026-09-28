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
| **October 5** | Review the MNIST hardware architecture, numerical formats, memory organization, and module interfaces. Establish the quantized Python reference and verification-data format. Review initial compute-block development and CPU-to-FPGA communication. |
| **October 12** | Target complete MNIST inference on the KV260. Integrate the compute blocks, controller, and DMA path; compare FPGA outputs with the Python reference and collect initial performance measurements. Review synthesis and implementation results. |
| **October 19** | Resolve remaining correctness or integration issues, repeat board measurements, and document deployment. Review accuracy, latency, throughput, resource usage, and development effort before beginning ResNet50. |

The numerical reference, RTL, verification, and board interface will be developed in parallel. Data-transfer testing will proceed independently of the compute blocks so that transport problems can be identified before full-system integration.

MNIST will be considered complete when repeated FPGA inference agrees with the quantized reference, deployment can be reproduced, and measurements distinguish compute time from transfer and software overhead.

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

The ResNet50 target is complete inference verified in RTL simulation, supported by synthesis results. Full ResNet50 deployment on KV260 will depend on integration progress and is not a firm commitment in this schedule.

The earlier dates leave the final week for verification and reporting. October 12 is an ambitious initial MNIST deployment target; October 19 remains the planned completion date before moving to ResNet50.

### ResNet model and implementation approach

ResNet is a family of convolutional networks built around residual connections. A block adds the result of its convolution layers to a shortcut from the block's input. When the dimensions change, a projection on the shortcut makes the two tensors compatible. For hardware, this means retaining the shortcut data until the main computation finishes and ensuring that both operands use compatible numerical scales. The [original ResNet paper](https://arxiv.org/abs/1512.03385) describes the network and its motivation.

Our proposed reference is pretrained **TorchVision ResNet50**, with the software version, weight checkpoint, and preprocessing pinned before implementation. TorchVision uses the ResNet v1.5 variant, which places bottleneck downsampling in the 3x3 convolution. This detail must agree between the Python reference and RTL. See the [TorchVision ResNet50 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html).

ResNet50 uses bottleneck blocks with a 1x1 convolution, a 3x3 convolution, and a final 1x1 convolution, followed by residual addition and activation. The complete graph also includes the input convolution/pooling stages, global average pooling, and a final classifier. Inference batch normalization can be folded into convolution parameters after checking the transformed reference. We will begin by evaluating INT8 weights and activations; accumulator widths, rounding, saturation, and residual rescaling will be selected from numerical analysis and accuracy measurements.

The implementation will progress from one bottleneck to a projection/downsampling block, multiple connected blocks, and finally the complete network. Compute resources will be reused across layers and tiles. External memory will hold weights and tensors as needed, with on-chip buffers used to reduce repeated transfers. The memory schedule and preservation of residual tensors are as important as the arithmetic datapath.

ResNet50 remains the planned workload because it exercises the compute configurations, memory management, and scheduling needed for the capstone. ResNet18 is a possible fallback if completing a smaller network becomes the better use of the remaining time. It uses simpler two-convolution blocks and has approximately 11.7 million parameters, compared with 25.6 million for ResNet50. Such a change would be discussed and recorded explicitly; ResNet18 is not an additional required milestone. See the [ResNet18 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) and [reference implementation](https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html).

## Spring 2027 capstone direction

The proposed spring direction is a **domain-specific inference accelerator with a TPU-inspired compute engine**. A TPU-style design is one form of domain-specific architecture: it organizes computation and memory around neural-network operations. For this project, the scope would be a small programmable tensor accelerator on FPGA, with a systolic multiply-accumulate array, local buffers, DMA, supporting activation and residual operations, and a command interface for sequencing work. The [original TPU architecture paper](https://arxiv.org/abs/1704.04760) provides background on the relationship between matrix computation and software-managed memory.

The spring work should build on the fall implementation rather than require a new platform. We expect to retain the Python references, numerical contracts, host interface, transfer infrastructure, and regression framework. The final compute organization will depend on what the fall measurements show. If memory traffic limits performance, improving data reuse and scheduling may be more useful than increasing the number of arithmetic units.

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
