### Project Roadmap: Fall 2026 and Spring 2027
UT Austin ECE Senior Design
## Project Goal

The goal of the project is to develop and evaluate an **AI-assisted hardware design flow** that takes machine-learning workloads from software models through RTL implementation, verification, FPGA deployment, and performance evaluation.

**MNIST** is intended to serve as the project’s initial learning and integration workload. Its purpose is to give the team a manageable environment for learning the complete acceleration flow from a Python model to verified RTL running on the KV260. Through this phase, we expect to build familiarity with ML inference concepts, hardware/software partitioning, FPGA communication, verification methodology, and the AI-assisted design tools used throughout the project. The goal is to establish a repeatable workflow and reusable infrastructure that can then be applied to larger and more realistic workloads such as **ResNet50**. The MNIST phase gives us a controlled environment to understand the tool, while **ResNet50** gives us a more realistic test of whether that assistance continues to scale.

## Fall 2026 Timeline

| Phase | Start | End | High-Level Goal |
|---|---|---|---|
| **MNIST Architecture + RTL** | Sep. 28 | Oct. 11 | Finalize the hardware/software partition, define the RTL architecture, and begin implementing the primary compute blocks. |
| **MNIST Integration + FPGA Deployment** | Oct. 12 | Oct. 25 | Integrate the RTL accelerator, CPU/FPGA communication, verification environment, and deploy the design on the KV260. |
| **MNIST Benchmarking + Flow Review** | Oct. 19 | Oct. 25 | Measure correctness, latency, throughput, FPGA resources, transfer overhead, and engineering effort. |
| **ResNet50 Architecture + Initial RTL** | Oct. 26 | Nov. 8 | Analyze ResNet50, define the reusable accelerator architecture, and begin implementing the functionality required for the larger workload. |
| **ResNet50 Block-Level Integration** | Nov. 9 | Nov. 22 | Implement and verify convolution configurations, residual paths, buffering, and representative bottleneck blocks. |
| **ResNet50 Full Integration** | Nov. 23 | Dec. 4 | Connect the full inference flow and validate the complete model against the software reference. |
| **Fall Evaluation + Documentation** | Dec. 4 | Dec. 7 | Freeze the fall implementation, collect results, and summarize findings from the AI-assisted design process. |

### Chip Agents Involvement

Our current flow uses manual RTL development, Claude Code, and HLS-based experimentation. As we move into the more complex ResNet50 phase, access to ChipAgents would allow us to evaluate it alongside these existing approaches and compare where each flow is most effective. In particular, we are interested in evaluating ChipAgents for:
- RTL generation and iteration,
- verification and testbench development,
- debugging and root-cause analysis,
- design review,
- and interpreting synthesis or implementation results.

The spring scope will be driven by the results of the fall work. Once MNIST and ResNet50 have been implemented, verified, and benchmarked, we should have a clearer understanding of the main architectural and workflow bottlenecks. From there, the project may focus on further optimizing ResNet50, refining the design flow, or applying the methodology to a different inference architecture. The intent is for the spring direction to follow from measured results rather than be fixed in advance.

## Spring 2027 Timeline

| Phase | Start | End | High-Level Goal |
|---|---|---|---|
| **Flow Review + Capstone Architecture Selection** | Jan. 2027 | Jan. 2027 | Review fall results and decide which parts of the flow and accelerator architecture should be retained or improved. |
| **Architecture Refinement + Implementation** | Feb. 2027 | Feb. 2027 | Develop a more general domain-specific inference accelerator using lessons from MNIST and ResNet50. |
| **Full-System Integration + Optimization** | Mar. 2027 | Mar. 2027 | Run complete workloads and iterate on dataflow, buffering, scheduling, and compute organization. |
| **Final Evaluation + Demonstration** | Apr. 2027 | Final Review | Complete benchmarking, document the methodology, and prepare the final demonstration. |

## Overall Progression

```text
Sep–Oct
MNIST
Build and validate the complete flow
        ↓
Oct–Dec
ResNet50
Test whether the flow scales
        ↓
January
Review results and refine methodology
        ↓
Feb–Mar
Develop and iterate on a more general accelerator
        ↓
April
Final hardware + workflow evaluation
```

The outcome is not only the individual accelerators, but a repeatable and evaluated workflow for moving from a high-level ML workload to verified FPGA hardware.
