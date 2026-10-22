# Source review and next validation gates — October 19, 2026

**Historical pre-run review below.** The subsequent XSim suite, component synthesis,
integrated synthesis and routed timing checks have now completed. Read
[CURRENT_STATUS.md](../../CURRENT_STATUS.md) for current evidence and schedule;
the unrun/pending statements below describe the earlier preparation stage.

This review inspected all six RTL sources, the testbench, integration/build Tcl, benchmark C, fixture/layout tools, UART summarizer, and the existing model/evidence documents. It is a static review, not HDL compilation or simulation. No confirmed datapath defect was found in this review; that is not a correctness proof. No datapath RTL was changed.

## Findings and changes

| Finding | Action / remaining evidence |
|---|---|
| INT8-only checking can hide raw accumulator errors behind ReLU/saturation | Testbench now compares every accepted 104-lane raw group against `expected_acc.mem`, in addition to final bytes |
| Random ready stalls may never fill the FIFO | Warm frame holds output not-ready for 400 clocks, then uses pseudorandom stalls; requires observation of requantizer backpressure |
| Malformed packet branches were not exercised | Added sparse/empty KEEP, partial nonfinal beat, early TLAST, START rejection and reset recovery tests |
| Fixture presence check sampled only endpoints | Added full-array known-value check for both golden arrays; simulation still requires Python fixture hash verification separately |
| UART summarizer could report partial runs | Requires exact final PASS totals, expected cold/warm counts, ordered indices, valid counter relationships, expected load counts and no failure text; rejects concatenated runs |
| Team documents lagged delivered source | Added current status and meeting links while retaining historical results |
| Hardware contract said overflow is an error, while RTL wraps | Clarified exported-network software safety bounds versus hardware behavior on arbitrary operands |

By default, the testbench still uses the same saved image twice, not two independent accuracy samples. It expects 53,248 raw comparisons and 53,248 final-byte comparisons across cold/warm frames. Those are intended checks, not an achieved HDL result.

## Additional ready-to-run vector cases

`tools/prepare_validation_vectors.py` prepared **14** alternative fixture directories
under `build/validation_vectors/`: the ten existing saved CIFAR-10 images and four
synthetic cases (zero input, seeded signed input, positive overflow and negative
overflow). An independent padded-window matrix calculation matched every archived
image's raw and INT8 tensors; archived source hashes and all generated fixture
hashes were checked. Overflow cases deliberately replace weights and bias while
retaining the same quantization constants. They test hardware wrapping, not model
accuracy. Original fixtures, CPU header and quantization package were preserved.

Generate again only into a **fresh** directory (the script refuses overwrite):

```powershell
# Repository root; the default output already exists after this preparation.
.\.venv-resnet\Scripts\python.exe resnet8_project/rtl/kv260_conv0/tools/prepare_validation_vectors.py --out resnet8_project/rtl/kv260_conv0/build/validation_vectors_new
```

Optional user-run HDL example, from `rtl/kv260_conv0`:

```text
vivado -mode batch -source scripts/simulate.tcl -tclargs build/validation_vectors/00006_automobile build/sim_automobile
vivado -mode batch -source scripts/simulate.tcl -tclargs build/validation_vectors/synthetic_wrap_positive build/sim_wrap_positive
```

Use a new simulation directory for every run. `suite.json` lists all cases and
their manifest hashes. Each case runs the same cold/warm protocol sequence.
The modified simulation launcher and all cases remain **unrun in HDL**. Checking
the generated fixture hashes in Python does not cause the HDL testbench itself
to verify hashes. No dataset was downloaded or recalibrated.

## Datapath review notes

- The core issues synchronous image/weight reads, registers products on the next enabled edge, then accumulates on the following edge. Two drain edges precede EMIT. Each beat selects four pixels in HWC order. The raw comparison is intended to catch any simulator interpretation or enable alignment mistake here.
- Requantization uses a shared advance enable: if the last stage is blocked, data and valid stages freeze together. Channels repeat every 26 lanes; 104 outputs equals four complete pixels, so the same fixed constant ordering applies to every beat.
- FIFO pointers count 64-bit words. Each push adds 13 words, each pop removes four. Capacity is 256 words; push requires count <=243. The 16-bank write deltas are distinct, and reads precede writes at a handshake. The existing Python ring check passes wraparound/concurrent-operation cases but does not establish inferred RAM behavior.
- AXI-Lite buffers AW/W independently and holds responses. PREPARE clears datapath/counters and retains parameter readiness; SOFT_RESET invalidates all kinds. DMA must also be reset when aborting a real active transfer.
- The CPU arms S2MM before START, flushes/invalidate buffers, includes copies/cache/setup/polling in total timing, and excludes golden checks/UART. The guard initialization is also inside the timer. Full Vitis/BSP compilation is still needed.
- Integrated Tcl uses a separate 200 MHz HP0 clock and a 100 MHz accelerator/DMA domain. CDC/reset handling and exact generated IP properties remain tool-validation tasks.

## Run-ready synthesis experiments (not executed)

From this folder in a Vivado command prompt, use a **new process** for each:

```text
vivado -mode batch -source scripts/synth_components.tcl -tclargs mac_int8_mixed
vivado -mode batch -source scripts/synth_components.tcl -tclargs conv0_requant
vivado -mode batch -source scripts/synth_components.tcl -tclargs conv0_result_fifo
vivado -mode batch -source scripts/synth_components.tcl -tclargs conv0_832_core
```

The script performs out-of-context synthesis only, creates a 10 ns clock, preserves utilization/timing/checkpoint outputs under `build/components/<top>/`, and refuses to overwrite that directory. It neither opens nor modifies the existing integrated project. Source is untested in Vivado.

| Component | Question to resolve | Expected intent, not guaranteed result |
|---|---|---|
| One MAC | Did split inference prevent two-DSP mapping? | One DSP for multiplication; fabric accumulator |
| Requantizer | Does constant arithmetic fit and pipeline adequately? | 104 lanes, no DSPs; actual LUT/FF cost unknown |
| FIFO | How does banking/muxing map? | 2 KB distributed storage; no DSPs |
| Core | Does memory replication pack and route plausibly? | Approximately 832 multiplier DSPs; actual BRAM18/BRAM36 packing must be reported |

Out-of-context tops have many unconstrained I/O paths and omit the DMA/PS infrastructure. Their resource totals cannot simply establish integrated fit, and synthesis timing cannot establish routed timing. Do not sum the standalone MAC count on top of the core count. Inspect primitives and hierarchy rather than interpreting BRAM logical capacity as occupied physical tiles.

## Coverage still to add after the first HDL compile/run

- Reset/abort while computation or stalled output is active, with explicit scoreboard cancellation.
- Full missing/late TLAST and overlength streams, writes while busy, unsupported/misaligned register accesses, all WSTRB patterns.
- Execute the prepared independent-image, signed-input and overflow fixtures; add dedicated requantization boundary/tie vectors. Current Python checks do not prove RTL realization.
- Exact stage/counter trace comparison to the concrete RTL, not an assumption that Python model conventions match.
- Vendor DMA/DDR integration through the board benchmark (the testbench drives accelerator ports directly).

## Python verification performed

On October 19, all **32** existing/new unittest cases passed, including two UART capture tests (one complete capture and twelve invalid-capture subcases). The layout checker matched **26,624 raw + 26,624 INT8** golden values, exercised **100,000 ring steps**, and verified generated C payload hashes. The workload/latency generator produced 14 operation records and 96 analytic scenarios from the existing export manifest. Separately, all ten archived image tensors matched the independent Conv0 oracle (266,240 raw and 266,240 INT8 values); all 14 prepared fixture cases passed hash/dimension validation.

```powershell
# From repository root
.\.venv-resnet\Scripts\python.exe resnet8_project/rtl/kv260_conv0/tools/check_layout.py
.\.venv-resnet\Scripts\python.exe resnet8_project/rtl/kv260_conv0/tools/meeting_analysis.py
# From resnet8_project
..\.venv-resnet\Scripts\python.exe -m unittest discover -s tests -v
```

No new HDL compilation/simulation, synthesis, place-and-route, Vitis build, bitstream, or board execution was performed. The expanded HDL and new Tcl remain unverified source.
