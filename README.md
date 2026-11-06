# UT Austin ECE Senior Design – ResNet-8 accelerator on KV260

Goal: a hardware accelerator for full INT8 ResNet-8 (MLPerf Tiny
`resnet_v1_eembc(conv_filters=26)`, CIFAR-10) on the AMD Kria KV260.

All ResNet-8 work lives in [resnet8_project/](resnet8_project/).

## Directory structure

```
senior-design/
├── README.md
├── Ericsson Design Proposal — final.pdf   Sponsor design proposal (Oct 2026)
├── docs/
│   ├── plans/                 Project history, project flow, semester schedule
│   ├── meeting-notes/         Meeting notes (2026-09-kickoff.md)
│   ├── class_reports/         (empty)
│   └── presentation/          (empty)
├── junk/                      Retired or superseded material (MNIST-era docs, old exports, scratch)
└── resnet8_project/
    ├── models/                Python model: FP32/INT8 training, BN folding, export, verification, benchmarks
    ├── design/rtl/conv0/      Current conv0 RTL (ws_* modules: array, column, MXU, PE, skew, package)
    ├── brainstorming/
    │   ├── conv0/
    │   │   ├── arch1_broadcast_832/       Conv0-only 832-MAC broadcast design: RTL, sim, Vivado scripts, tools, CPU benchmark, reports
    │   │   ├── arch2_os_systolic_26x32/   Proposed output-stationary 26×32 systolic conv engine (ARCHITECTURE.md)
    │   │   └── arch3_ws_tpu_27x26/        Alternative weight-stationary (TPU-style) 27×26 engine: rtl/, ARCHITECTURE.md
    │   └── uarch_study/       Step-by-step microarchitecture study (01–04) and decision log
    ├── ai_flow_eval/          (empty) AI-assisted workflow evaluation
    ├── arch_specs/            (empty)
    ├── info/                  (empty)
    ├── software/              (empty) Host software
    └── verification/
        └── testbenches/       (empty)
```

Key entry points:

- [Models (golden model and export)](resnet8_project/models/)
- [arch1 broadcast design](resnet8_project/brainstorming/conv0/arch1_broadcast_832/README.md) and [architecture](resnet8_project/brainstorming/conv0/arch1_broadcast_832/ARCHITECTURE.md)
- [arch2 systolic architecture](resnet8_project/brainstorming/conv0/arch2_os_systolic_26x32/ARCHITECTURE.md)
- [arch3 TPU architecture](resnet8_project/brainstorming/conv0/arch3_ws_tpu_27x26/ARCHITECTURE.md)
- [uarch study](resnet8_project/brainstorming/uarch_study/README.md)

## Project guide

- [Project history: how we got here, what we tried](docs/plans/PROJECT_HISTORY.md)
- [Python-to-FPGA project flow](docs/plans/projectflow.md)
- [Semester schedule](docs/plans/SEMESTER_SCHEDULE.md)
- [September kickoff summary](docs/meeting-notes/2026-09-kickoff.md)

Earlier planning documents now live in [junk/](junk/). They predate the ResNet-8
focus and still mention the retired MNIST baseline. The MNIST code was removed;
it remains in git history.
