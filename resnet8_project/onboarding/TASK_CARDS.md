# Working-session task cards

Use after the [walkthrough](MEETING_WALKTHROUGH.md). Name owners and reviewers. Preserve the deployed candidate while experiments proceed separately. Each group produces an artifact or a reproducible blocker.

## A. Deployment — three people

Driver: ______ Reviewer: ______ Recorder: ______

**Goal:** build and, if setup permits, execute the existing standalone A53 benchmark.

Inputs: [benchmark.c](../rtl/kv260_conv0/cpu/benchmark.c), [fixture header](../rtl/kv260_conv0/cpu/fixture_data.h), [deployment guide](../rtl/kv260_conv0/KV260_BUILD_AND_BENCHMARK.md), matching XSA/bitstream under repository-root `.vivado/kv260_20261019_040420/`.

1. Confirm Vitis access/version, hardware artifacts, KV260, JTAG and UART. After ten minutes on a missing prerequisite, record and assign the blocker.
2. Create the standalone Cortex-A53 platform/application and build the ELF. Check BSP DMA lookup, addresses, DDR linker placement, stdout, and compiler settings.
3. If ready, initialize PS/DDR, program PL, execute the application, and save full UART output. Follow the guide's Kria-specific procedure; no QSPI changes are part of this session.
4. Diagnose the first failure from its exact message/registers. Rotate the driver halfway through. Reviewer checks settings; recorder captures reproducible steps.

**Deliverable:** ELF/build log and launch instructions, ideally a golden-output PASS. Otherwise: exact failing step, error, attempted fixes, next owner. Board unavailability does not prevent the application build or configuration review.

## B. Chip-agent experiment — two people

Operator: ______ Independent checker: ______ Tool/version: ______

**Goal:** evaluate one bounded hardware contribution with recorded human effort and independent verification.

Confirm the exact agent product and access; none is assumed installed. If access is unavailable after ten minutes, prepare the specification and golden cases instead.

Suggested task: an experimental residual-add primitive, isolated from the working accelerator.

1. Read residual-add behavior in [integer.py](../resnet8/integer.py). Freeze widths, branch scales, rounding, overflow/saturation, activation and valid/ready behavior before prompting.
2. Prepare independent golden cases. If scale conversion exceeds the session scope, explicitly restrict the primitive to already aligned branch values and label it incomplete network support.
3. Save prompts, generated RTL/testbench, revisions, tool/version, human changes, elapsed work and repair time.
4. Independently review arithmetic and protocol. Python checks establish expected values; generated RTL/testbenches alone do not prove HDL correctness. Record compilation/simulation pending unless actually performed using agreed tools.

**Deliverable:** specification, candidate, independent cases, findings and precise verification status. Use a separate experiment folder/branch; do not integrate into deployment RTL during the meeting.

One exploratory attempt does not establish a general productivity benefit. A controlled comparison needs matched tasks and repeated trials. See the [experiment log template](../../AI_EXPERIMENT_LOG_TEMPLATE.csv).

## C. Verification and onboarding — two people

New-reader tester: ______ Evidence reviewer: ______

**Goal:** make the workflow understandable and reproducible beyond the original laptop.

1. Follow the onboarding without verbal help; record missing prerequisites and unclear instructions.
2. Check resource/timing claims using the [technical reference](TECHNICAL_REFERENCE.md). Review retained DRC warning categories and flag unresolved concerns.
3. On a configured checkout, run from repository root:

   ```powershell
   .\.venv-resnet\Scripts\python.exe resnet8_project/rtl/kv260_conv0/tools/check_layout.py
   ```

4. Draft or implement a read-only preflight: Python/tool availability, required files, fixture consistency, artifact presence and actionable failure messages. Artifact presence does not prove execution.
5. Prepare a portable evidence handoff: selected reports/logs, fixture identifiers and matching artifact/source provenance. Current hashes alone do not establish which source revision produced an older build.

**Deliverable:** corrected instructions, readiness checklist/report or working preflight script, and evidence issues. Without tools, documentation and report review still proceed independently.

## Cross-review and closeout

At minute 90 each group demonstrates its artifact; another group reviews it. Reserve the final ten minutes for this table.

| Group | Artifact path | Passed/pending/blocked | Next action | Owner | Check-in |
|---|---|---|---|---|---|
| Deployment | | | | | |
| Agent experiment | | | | | |
| Verification | | | | | |

The critical path is Vitis build → board correctness → measured latency. The other groups work independently while preserving the known baseline.
