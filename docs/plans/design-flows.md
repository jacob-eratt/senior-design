# AI in Hardware Design — Design Flow Comparison

## Purpose

This project evaluates how effective AI is in hardware design by comparing three design flows on the same target:

1. **Manual (LLM-assisted)**: the engineer drives each stage, using an LLM as an assistant.
2. **Claude Code**
3. **ChipAgents**

The Claude Code and ChipAgents flows both take the manual flow's documentation/spec as their input, and the two flows run side by side so they can be compared stage by stage.

## Design Flow Diagram

**Legend (Claude Code and ChipAgents flows)**

| Shape | Meaning |
|---|---|
| Circle | Process |
| Rectangle | Input / output artifact |
| Square | Comparison |

```mermaid
%%{init: {"flowchart": {"useMaxWidth": false, "nodeSpacing": 90, "rankSpacing": 110, "padding": 25, "diagramPadding": 30}, "themeVariables": {"fontSize": "26px"}}}%%
flowchart LR
    subgraph MAN["Manual Flow (LLM-Assisted)"]
        M1[Brainstorming] --> M2[Documentation / Spec]
        M2 --> M3[RTL Design]
        M3 --> M4[Optimization]
        M4 --> M5[Interfacing]
        M5 --> M6[Verification]
        M6 --> M7[FPGA Implementation]
        M7 --> M8[Performance Measurements]
    end

    subgraph CC["Claude Code Flow"]
        CC1((Generate<br/>Arch Spec)) --> CC2[Arch Spec<br/>Claude Code]
        CC3((Generate<br/>RTL)) --> CC4[RTL Code<br/>Claude Code]
        CC3 --> CC5[Synthesis Reports<br/>Claude Code]
        CC6((Verification)) --> CC7[Verified Implementation<br/>Claude Code]
    end

    subgraph CA["ChipAgents Flow"]
        CA1((Generate<br/>Arch Spec)) --> CA2[Arch Spec<br/>ChipAgents]
        CA3((Generate<br/>RTL)) --> CA4[RTL Code<br/>ChipAgents]
        CA3 --> CA5[Synthesis Reports<br/>ChipAgents]
        CA6((Verification)) --> CA7[Verified Implementation<br/>ChipAgents]
    end

    CMP1[Compare<br/>Arch<br/>Specs]:::compare
    SEL1[Selected<br/>Arch Spec]
    CMP2[Compare<br/>RTL &<br/>Synth]:::compare
    SEL2[Selected<br/>RTL Design]
    CMP3[Compare<br/>Verified<br/>Impls]:::compare
    SEL3[Selected<br/>Implementation]
    FPGA1((FPGA Implementation &<br/>Performance Analysis))

    M2 --> CC1
    M2 --> CA1
    CC2 --> CMP1
    CA2 --> CMP1
    CMP1 --> SEL1
    SEL1 --> CC3
    SEL1 --> CA3
    CC4 --> CMP2
    CC5 --> CMP2
    CA4 --> CMP2
    CA5 --> CMP2
    CMP2 --> SEL2
    SEL2 --> CC6
    SEL2 --> CA6
    CC7 --> CMP3
    CA7 --> CMP3
    CMP3 --> SEL3
    SEL3 --> FPGA1

    classDef compare stroke-width:3px,font-weight:bold
```

## Flow Descriptions

### Manual Flow (LLM-Assisted)

The manual flow is the baseline. Its goal is to show that the hardware accelerator design can be implemented on the FPGA with reasonable functional accuracy, performance, and resource utilization.

Engineers drive every stage of this flow, using AI in the form of LLMs plus partial use of various coding agents. The design is refined through hands-on learning and repeated iteration, so this flow is expected to take the longest of the three.

Setting a baseline for functional expectations here makes the Claude Code and ChipAgents evaluations easier to run and more informative to interpret.

### Claude Code Flow

This flow uses the Claude Code coding agent to generate three main artifacts:

1. An architectural spec
2. RTL design code
3. Verification outputs

It builds on the manual flow in two ways. The coding agent has complete context of the project files, and it runs on more sophisticated models than the ones used in the manual flow.

A custom skill suite for hardware accelerator design also supplements this flow. The skills give the agent design-specific guidance it would not otherwise have, so it is better equipped for accelerator work.

### ChipAgents Flow

This flow uses the agentic tool for custom hardware design from ChipAgents.ai. The tool includes functionality built specifically for hardware design and verification.

Each of these features will be evaluated for how effective it is, and the results will be compared against the manual and Claude Code flows. The goal is to measure how much benefit a dedicated hardware design tool like this adds to the hardware design process.
