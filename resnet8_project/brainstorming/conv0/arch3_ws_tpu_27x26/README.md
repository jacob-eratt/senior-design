# TPU AXI interface test branch

This folder contains the buffered CPU/FPGA interface for the 27 x 26 INT8 matrix
unit. It accepts one weight tile and 1..64 input vectors and returns raw INT32
dot products. It is not yet a complete Conv0/ResNet inference pipeline.

This branch follows main's reorganized layout. The interface package is under
`resnet8_project/brainstorming/conv0/arch3_ws_tpu_27x26/`; the separate core-only
copy under `resnet8_project/design/rtl/conv0/` is unchanged. Build this folder's
sources together, without mixing duplicate core modules from those two paths.

| Included | Purpose |
|---|---|
| `rtl/` | Existing matrix core plus AXI register block, loader, tile engine, output packer, structs and block-design adapter |
| `tb/` | Self-checking PE, column, matrix and AXI testbenches |
| `model/` | Python cycle references, vector generators and packet/schedule checks |
| `scripts/` | Simulation, source registration, PS/DMA project creation and implementation/XSA export |
| `cpu/tpu_test.c` | Standalone A53 DMA test with CPU-computed golden results |
| [INTERFACING.md](INTERFACING.md) | Bus, memory, transaction and deployment explanation |
| [DEPLOY_KV260.md](DEPLOY_KV260.md) | Vitis/JTAG deployment instructions |
| [docs/04_axi_wrapper.md](docs/04_axi_wrapper.md), [docs/05_axi_submodules.md](docs/05_axi_submodules.md) | Register/packet contract and source walkthrough |

The [hardware handoff](hardware/README.md) includes the existing integrated-build
bitstream, XSA, six routed reports and checksums. Use the XSA to build the CPU
application in Vitis. Generated Vivado projects, caches, checkpoints and memory
vector dumps remain excluded; scripts recreate them. No compiled CPU ELF is supplied.

## Start here

From this folder, with Python available:

```powershell
python model/check_axi_tile_contract.py
```

With Vivado 2024.2 on PATH, the AXI testbench is self-contained:

```powershell
vivado -mode batch -source scripts/sim.tcl -tclargs tb_ws_axi_top
```

The separate PE/column/matrix tests require their matching `model/gen_*_vectors.py`
generator first. They are optional lower-level checks, not prerequisites for
generating the AXI test's data.

For system integration, open Vivado with no project open, change its Tcl working
directory to this folder, then run:

```tcl
source scripts/create_soc_project.tcl
```

This creates a new PS/DMA system project. Review it before explicitly running
`source scripts/build_soc.tcl`, which resets synthesis and performs synthesis,
implementation, timing checks and bitstream/XSA generation. Keep generated output
local. In the guides, replace `<repo>` with your clone's absolute path.

## Verification scope

A historical wrapper simulation passed 1,820 INT32 comparisons and AXI checks on
November 5. Local addressing/testbench changes followed that run. Publication of
this branch does not claim those newer revisions have passed HDL simulation,
Vitis compilation or board execution. Saved integrated reports and hardware
artifacts are now included under `hardware/`, with positive routed timing slack;
that evidence is distinct from a fresh build of this published commit. Run the current tests
and record results against this branch's commit before deployment. The Python
contract checker is useful preparation but does not execute the RTL.
