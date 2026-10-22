# Concise publication inventory — October 21, 2026

Destination: existing `resnet8-conv0` branch, not `main`. Compared with `5c210d4`. The publication was reduced from 133 staged files to the file list below; unrelated local work is preserved.

## Included, with rationale

### Exclude generated projects and preserve report/fixture bytes.

- `.gitattributes`
- `.vivado/.gitignore`

### Current onboarding, status, navigation or numeric contract needed by teammates.

- `AI_EXPERIMENT_LOG_TEMPLATE.csv`
- `README.md`
- `resnet8_project/CURRENT_STATUS.md`
- `resnet8_project/PROGRESS_UPDATE.md`
- `resnet8_project/PUBLICATION_INVENTORY.md`
- `resnet8_project/README.md`
- `resnet8_project/onboarding/MEETING_WALKTHROUGH.md`
- `resnet8_project/onboarding/README.md`
- `resnet8_project/onboarding/TASK_CARDS.md`
- `resnet8_project/onboarding/TECHNICAL_REFERENCE.md`
- `resnet8_project/resnet8/HARDWARE_CONTRACT.md`
- `resnet8_project/resnet8/TEAM_OVERVIEW.md`
- `resnet8_project/rtl/CONV0_416.md`

### Executable arithmetic/scheduling models and regression checks; retain source rather than bulky sweep outputs.

- `resnet8_project/cycle_sim/AXI_INT8_STUDY.md`
- `resnet8_project/cycle_sim/README.md`
- `resnet8_project/cycle_sim/__init__.py`
- `resnet8_project/cycle_sim/axi.py`
- `resnet8_project/cycle_sim/axi_resources.py`
- `resnet8_project/cycle_sim/axi_system.py`
- `resnet8_project/cycle_sim/configs/208_balanced.json`
- `resnet8_project/cycle_sim/configs/416_example_host.json`
- `resnet8_project/cycle_sim/configs/416_final_stall.json`
- `resnet8_project/cycle_sim/configs/416_ideal.json`
- `resnet8_project/cycle_sim/configs/conv0_int8_latency.json`
- `resnet8_project/cycle_sim/configs/conv0_int8_latency_fallback_52.json`
- `resnet8_project/cycle_sim/core.py`
- `resnet8_project/cycle_sim/fixtures.py`
- `resnet8_project/cycle_sim/kv260.json`
- `resnet8_project/cycle_sim/resources.py`
- `resnet8_project/cycle_sim/system.py`
- `resnet8_project/cycle_sim/trace.py`
- `resnet8_project/explore_conv0.py`
- `resnet8_project/simulate_axi_conv0.py`
- `resnet8_project/simulate_conv0.py`
- `resnet8_project/tests/test_architecture_study.py`
- `resnet8_project/tests/test_axi_conv0.py`
- `resnet8_project/tests/test_cycle_sim.py`
- `resnet8_project/tests/test_uart_capture.py`

### Small golden INT8 data and manifest required for reproducible checks.

- `resnet8_project/fixtures/export/conv0_quant.mem`
- `resnet8_project/fixtures/layer_vectors/conv0/expected_output.mem`
- `resnet8_project/fixtures/manifest.json`

### Compact final evidence or essential architecture analysis; no per-run logs or component archives.

- `resnet8_project/reports/README.md`
- `resnet8_project/reports/conv0_416_dsp_utilization_synth.rpt`
- `resnet8_project/reports/meeting_2026_10_20/ANALYSIS.md`
- `resnet8_project/reports/vivado_2026_10_19/README.md`
- `resnet8_project/reports/vivado_2026_10_19/integrated/cdc.rpt`
- `resnet8_project/reports/vivado_2026_10_19/integrated/drc.rpt`
- `resnet8_project/reports/vivado_2026_10_19/integrated/timing_route.rpt`
- `resnet8_project/reports/vivado_2026_10_19/integrated/utilization_route.rpt`
- `resnet8_project/reports/vivado_2026_10_19/summary.json`

### Current accelerator RTL, testbench, build/deployment scripts and CPU harness; required to reproduce and deploy Conv0.

- `resnet8_project/rtl/kv260_conv0/.gitignore`
- `resnet8_project/rtl/kv260_conv0/KV260_BUILD_AND_BENCHMARK.md`
- `resnet8_project/rtl/kv260_conv0/README.md`
- `resnet8_project/rtl/kv260_conv0/REVIEW_AND_VALIDATION.md`
- `resnet8_project/rtl/kv260_conv0/cpu/benchmark.c`
- `resnet8_project/rtl/kv260_conv0/cpu/fixture_data.h`
- `resnet8_project/rtl/kv260_conv0/cpu/fixture_manifest.json`
- `resnet8_project/rtl/kv260_conv0/scripts/build.tcl`
- `resnet8_project/rtl/kv260_conv0/scripts/create_project.tcl`
- `resnet8_project/rtl/kv260_conv0/scripts/resume_export.tcl`
- `resnet8_project/rtl/kv260_conv0/scripts/run_integrated.tcl`
- `resnet8_project/rtl/kv260_conv0/scripts/run_no_waves.tcl`
- `resnet8_project/rtl/kv260_conv0/scripts/simulate.tcl`
- `resnet8_project/rtl/kv260_conv0/scripts/synth_components.tcl`
- `resnet8_project/rtl/kv260_conv0/scripts/validate_and_synth.tcl`
- `resnet8_project/rtl/kv260_conv0/sim/tb_conv0_axi.sv`
- `resnet8_project/rtl/kv260_conv0/src/conv0_832_core.sv`
- `resnet8_project/rtl/kv260_conv0/src/conv0_axi_bd.v`
- `resnet8_project/rtl/kv260_conv0/src/conv0_axi_top.sv`
- `resnet8_project/rtl/kv260_conv0/src/conv0_quant_pkg.sv`
- `resnet8_project/rtl/kv260_conv0/src/conv0_requant.sv`
- `resnet8_project/rtl/kv260_conv0/src/conv0_result_fifo.sv`
- `resnet8_project/rtl/kv260_conv0/src/mac_int8_mixed.sv`
- `resnet8_project/rtl/kv260_conv0/tools/archive_vivado_results.py`
- `resnet8_project/rtl/kv260_conv0/tools/check_layout.py`
- `resnet8_project/rtl/kv260_conv0/tools/export_fixture.py`
- `resnet8_project/rtl/kv260_conv0/tools/meeting_analysis.py`
- `resnet8_project/rtl/kv260_conv0/tools/prepare_validation_vectors.py`
- `resnet8_project/rtl/kv260_conv0/tools/summarize_uart.py`

## Kept local

- Detailed component reports, per-image simulation logs, verbose Vivado logs and synthesis intermediates: final reports and a summary are sufficient for this handoff.
- Sweep CSV/JSON outputs, older meeting brief, detailed historical cycle-study notes and semester schedule: useful research history but unnecessary for the deployment entry point. Executable model source is retained.
- Full generated Vivado projects, caches, waveforms, datasets, environment and full tensor dumps: regenerable or machine-specific.
- Bitstream/XSA/ELF binaries: share as a separate matching deployment bundle, not source history. Current bit/XSA hashes remain in the evidence summary.
- Unrelated local MNIST edits and broad ResNet50 planning documents: outside this Conv0 publication.
- Redundant per-file CSV and separate validation document: merged into this inventory.

## Checks

32 Python unit tests passed on the publication source. Layout checks matched 26,624 raw and INT8 outputs and C payload hashes. After trimming, staged report hashes were verified and links to omitted files were removed; a generated-sweep link was converted to a local-path description. No new HDL, Vitis or board run was performed. Raw tool formatting is retained to preserve evidence bytes.
