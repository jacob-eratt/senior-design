# Fall 2026 Project Schedule

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

For implementation tasks and acceptance criteria, see the [issue backlog](ISSUE_BACKLOG.md). The [project flow](../projectflow.md), [profiling guide](../PROFILING.md), and [kickoff summary](meeting-notes/2026-09-kickoff.md) provide supporting context.
