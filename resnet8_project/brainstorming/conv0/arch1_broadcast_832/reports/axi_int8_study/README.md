# INT8 AXI style study data

Read [the latency first model report](../../docs/AXI_INT8_STUDY.md) first.

- `timing.csv`: 540 cold frame cases: 20 MAC/output geometries, 3 stream bus
  widths, 3 local loader rates, and 3 link rates.
- `resource_scenarios.csv`: 3,240 high/low estimated resource scenarios for
  180 geometry/interface configurations, including DSP versus fabric
  requantization and different DSP/fabric MAC mixes.
- `summary.json`: sweep counts, hashes, elapsed model run time, and fixture
  comparison totals.

The AXI timing values, clocks, DMA service rates, and memory organization are
models. `passes_high_cost_scenario` compares an assumed upper resource estimate
plus reserve with the K26 published capacity. It does not indicate synthesized
or timing closed RTL. The original 416-MAC synthesis values are the only measured
resource baseline and are identified separately in the report.

Recreate the full modeled run from `resnet8_project/`:

```powershell
# simulate_axi_conv0.py was removed during cleanup; recover it from origin/resnet8-conv0.
```

The 564 full frame simulations checked the INT32 accumulator and INT8 result
against the saved reference. These are repeated runs on one CIFAR-10 image;
they do not represent 564 separate samples or system measurements.
