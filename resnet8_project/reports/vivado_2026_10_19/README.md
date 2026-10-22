# Compact implementation evidence — October 19, 2026

Only four original final reports are published: routed utilization, timing, CDC and DRC in `integrated/`. Their hashes and the generated bitstream/XSA hashes are in `summary.json`.

Results: 95,688 LUTs, 56,684 FFs, 832 DSPs, 8 RAMB36 + 131 RAMB18; 97.72% occupied CLBs. Setup/hold slack +0.365/+0.010 ns. CDC reports 733 safe endpoints. DRC retains 2,548 warnings, no errors/critical warnings in its summary.

Historical simulation summary: 30 direct-top frames, 798,720 raw INT32 and 798,720 INT8 comparisons; a separate two-frame wrapper baseline passed. Full per-case and tool logs remain local, along with component synthesis reports and the successful managed-export log. They are not included in this compact publication. No new HDL run occurred for publication.

Source hashes are archive-time fingerprints, not proof of original build revision. Bitstream/XSA export succeeded; Vitis compilation and board execution remain pending. Binaries remain local and should be distributed as a separate matching deployment bundle when needed.
