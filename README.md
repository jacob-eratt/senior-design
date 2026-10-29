# UT Austin ECE Senior Design – ResNet-8 accelerator on KV260

Goal: a hardware accelerator for full INT8 ResNet-8 (MLPerf Tiny
`resnet_v1_eembc(conv_filters=26)`, CIFAR-10) on the AMD Kria KV260.

All ResNet-8 work lives in [resnet8_project](resnet8_project/README.md):

| Folder | Contents |
|---|---|
| [resnet8/](resnet8_project/resnet8/README.md) | Frozen FP32/INT8 software golden model, BatchNorm folding, export, [hardware contract](resnet8_project/resnet8/HARDWARE_CONTRACT.md) |
| [arch1_broadcast_832/](resnet8_project/arch1_broadcast_832/ARCHITECTURE.md) | Existing Conv0-only 832-MAC broadcast design: RTL, testbench, Vivado scripts, reports |
| [arch2_os_systolic_26x32/](resnet8_project/arch2_os_systolic_26x32/ARCHITECTURE.md) | Proposed output-stationary 26×32 systolic conv engine (all layers) |
| [arch3_ws_tpu_27x26/](resnet8_project/arch3_ws_tpu_27x26/ARCHITECTURE.md) | Alternative weight-stationary (TPU-style) 27×26 conv engine |
| [uarch_study/](resnet8_project/uarch_study/README.md) | Step-by-step microarchitecture study and decision log |

## Project guide

- [Project history: how we got here, what we tried](PROJECT_HISTORY.md)
- [Fall 2026 roadmap and milestone acceptance gates](docs/ROADMAP.md)
- [Python-to-FPGA project flow](projectflow.md)
- [September kickoff summary](docs/meeting-notes/2026-09-kickoff.md)
- [Initial implementation issue backlog](docs/ISSUE_BACKLOG.md)

Earlier planning documents at the repository root
([RESNET50_ROADMAP.md](RESNET50_ROADMAP.md),
[RTL_ARCHITECTURE_AND_DATAFLOW.md](RTL_ARCHITECTURE_AND_DATAFLOW.md),
[ML_AND_KV260_EXPLAINED.md](ML_AND_KV260_EXPLAINED.md),
[RISKS_AND_VERIFICATION.md](RISKS_AND_VERIFICATION.md),
[SEMESTER_SCHEDULE.md](SEMESTER_SCHEDULE.md)) predate the ResNet-8 focus and
still mention the retired MNIST baseline. The MNIST code was removed; it remains
in git history.
