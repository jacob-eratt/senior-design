# Vivado/XSim evidence, October 19, 2026

Archived October 21 from completed local runs, without rerunning HDL or Vivado.

- Full direct-top suite: 30 frames, 798,720 raw INT32 and 798,720 INT8 comparisons.
- Separate wrapper baseline: two frames; see `tool_logs/sim_initial.log`.
- Routed resources: 95,688 LUTs, 56,684 FFs, 832 DSPs, 8 RAMB36 + 131 RAMB18; 97.72% occupied CLBs.
- Routed setup/hold slack: +0.365/+0.010 ns. CDC reports 733 safe endpoints. DRC retains 2,548 warnings; no errors/critical warnings in its summary.
- Managed bitstream/XSA export succeeded; hashes are in `summary.json`. Binaries are not committed.
- Vitis compilation, board correctness, and measured board latency remain pending.

`summary.json` fingerprints source at archive time, not the exact original build revision. The original full suite used the direct top; the wrapper run is separate. Reports and logs retain original tool timestamps and local path strings. `integrated.log` includes the initial export failure; `export.log` records successful recovery. This preserves the actual sequence rather than hiding failures.

Component reports are isolated synthesis results and should not be summed as exact integrated utilization. The vectorless power report is not measured board power. No board speedup claim is established by these reports.
