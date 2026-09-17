# FPGA Design Flow Workflow

```mermaid
flowchart TB
    subgraph COL1[" "]
        direction TB
        A["Benchmark Program<br/>(Application/Domain)"]
        QA["What benchmark application/domain?<br/>General or specific?<br/>How broad or narrow?"]
        A --- QA
    end

    subgraph COL2[" "]
        direction TB
        B["Design Spec<br/>(Microarchitecture Design)"]
        QB["Potentially how many design specs<br/>to solve same problem?"]
        B --- QB
    end

    subgraph COL3[" "]
        direction TB
        V["RTL Implementation + Verification"]
        V1["Manual (w/ use of LLMs)"]
        V2["ChipAgents.ai tools"]
        V3["Claude Code (or other agentic tool)"]
        QV["Is ChipAgents.ai single-flow or multi-flow?<br/>How many DV flows to experiment?<br/>Verilog, SystemVerilog, SystemC, HLS<br/>(or variable) design/RTL generation?<br/>What verification methodologies<br/>(functional, testbench, UVM, formal)?"]
        V --> V1
        V --> V2
        V --> V3
        V1 --- QV
        V2 --- QV
        V3 --- QV
    end

    subgraph COL4[" "]
        direction TB
        C["FPGA Implementation"]
        QC["Physical or cloud FPGA?<br/>What specific FPGA model to use?"]
        C --- QC
    end

    subgraph COL5[" "]
        direction TB
        D["Performance Measurement/Metrics"]
        QD["What performance metrics to measure?<br/>What design flow metrics<br/>(time to complete, # of corrections, etc.)<br/>to measure?"]
        D --- QD
    end

    A --> B
    B --> V
    V1 --> C
    V2 --> C
    V3 --> C
    C --> D

    classDef constant fill:#E1F5EE,stroke:#0F6E56,color:#04342C;
    classDef variable fill:#FAEEDA,stroke:#854F0B,color:#412402;
    classDef question fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A,font-style:italic;
    classDef invisible fill:none,stroke:none;

    class A,C,D constant;
    class V,V1,V2,V3 variable;
    class QA,QB,QV,QC,QD question;
    class COL1,COL2,COL3,COL4,COL5 invisible;
```

## 1. Benchmark program (constant)

The ML application or domain the benchmark program represents. This anchors the whole workflow — it's the real-world task the design is meant to solve.

**Open questions:**
- What benchmark application/domain?
- How general or specific?
- How broad or narrow?
- How many programs to test/optimize?

## 2. Design spec (potentially variable)

A spec derived from our microarchitectural design for optimizing the chosen benchmark that the RTL implementation must match. There may be more than one valid spec for the same underlying problem.

**Open questions:**
- Potentially, how many design specs to solve the same problem?
- What format should the specs be in?

## 3. RTL implementation + verification (variable)

The core experimental stage. The design spec can be implemented and verified through several different flows:
- **Manual** (with use of LLMs)
- **ChipAgents.ai tools**
- **Claude Code** (or other agentic tool)

Each flow converges back to a common FPGA implementation stage, allowing the flows to be compared against one another.

**Open questions:**
- Is ChipAgents.ai a single-flow or multi-flow tool?
- How many DV flows to experiment with? (Currently 3)
- Verilog, SystemVerilog, SystemC, HLS (or variable) design/RTL generation?
- What verification methodologies (functional, testbench, UVM, formal)?
- What IDE, verification, synthesis tools to use for each flow?

## 4. FPGA implementation (constant)

Taking the verified RTL and implementing it on an FPGA, held constant across the different RTL flows so results are comparable.

**Open questions:**
- Should implementation be on physical or cloud FPGA?
- What specific FPGA model to use?

## 5. Performance measurement/metrics (constant)

Measuring the outcome of the FPGA implementation to evaluate and compare the different upstream RTL flows.

**Open questions:**
- What performance metrics to measure (runtime, throughput, area, power, etc.)?
- What design flow metrics (time to complete, number of corrections, etc.) to measure?
