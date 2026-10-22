"""Reproducible Conv0 architecture study. Python only; no HDL tools.

Runs full clock/data simulations for geometry/transport combinations. Hardware
mapping alternatives are separate resource sensitivity calculations: they do
not pretend to change arithmetic latency or prove the configured clock meets
timing. Unknown LUT costs are deliberately ranges, not synthesis measurements.
"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from cycle_sim import Architecture, Conv0Core, HostConfig, run_frame
from cycle_sim.fixtures import load_fixture
from cycle_sim.resources import BOARD

ROOT = Path(__file__).resolve().parent
MAPPINGS = ("observed_two_dsp", "dsp_multiply_fabric_accumulate", "half_dsp_half_fabric",
            "quarter_dsp_three_quarters_fabric", "all_fabric")


def mapping_screen(arch, host, mapping, budget_fraction=0.8):
    """Unmeasured mappings are sensitivity scenarios, NOT inferred netlists.

    Base logic scaling is an explicit rough proxy from the whole baseline LUT
    count, with 0.75..1.5 variation. Fabric MAC range 64..160 LUTs includes an
    8x8 signed multiplier and INT32 accumulator; fabric accumulator alone uses
    32..48 LUTs. These are analyst-selected scenario ranges, not vendor specs.
    Includes all-fabric 48 FF/lane conservatively on top of baseline-scaled FF;
    this may double-count FF already present in the baseline.
    """
    n = arch.macs
    if mapping == "observed_two_dsp":
        dsp, fabric, acc_only = 2*n, 0, 0
    elif mapping == "dsp_multiply_fabric_accumulate":
        dsp, fabric, acc_only = n, 0, n
    elif mapping == "half_dsp_half_fabric":
        fabric, acc_only = n//2, 0
        dsp = 2*(n-fabric)
    elif mapping == "quarter_dsp_three_quarters_fabric":
        fabric, acc_only = 3*n//4, 0
        dsp = 2*(n-fabric)
    elif mapping == "all_fabric":
        dsp, fabric, acc_only = 0, n, 0
    else:
        raise ValueError(mapping)
    baseline = BOARD["baseline"]["resources"]
    base_lut = max(1000, baseline["lut"]*n/416)
    lut_low = math.ceil(base_lut*0.75 + fabric*64 + acc_only*32)
    lut_high = math.ceil(base_lut*1.5 + fabric*160 + acc_only*48)
    ff_high = math.ceil(max(1000, baseline["ff"]*n/416)*1.5 + fabric*48 + acc_only*32)
    measured = arch.spatial_lanes == 16 and arch.output_lanes == 4 and mapping == "observed_two_dsp"
    if measured:
        lut_low = lut_high = baseline["lut"]
        ff_high = baseline["ff"]
    # Assumed full-width simple-dual-port FIFO in <=72-bit, 512-deep BRAM36 banks.
    # Small FIFOs could instead use LUTRAM/registers; this is a BRAM proposal.
    fifo_bram = math.ceil(arch.output_lanes*32/72)*math.ceil(host.output_fifo_beats/512)
    bram = arch.spatial_lanes + fifo_bram
    cap = BOARD["capacity"]
    hard_screen = dsp <= cap["dsp"] and bram <= cap["bram36"]
    scenario_screen = (dsp <= budget_fraction*cap["dsp"] and bram <= budget_fraction*cap["bram36"]
                       and lut_high <= budget_fraction*cap["lut"] and ff_high <= budget_fraction*cap["ff"])
    return {"mapping": mapping, "dsp": dsp, "fabric_mac_lanes": fabric,
            "fabric_accumulator_only_lanes": acc_only, "lut_scenario_low": lut_low,
            "lut_scenario_high": lut_high, "ff_scenario_high": ff_high,
            "core_bram36": arch.spatial_lanes, "fifo_bram36_proposed": fifo_bram,
            "bram36_with_proposed_fifo": bram, "measured_compute_core": measured,
            "dsp_bram_device_screen": hard_screen, "conservative_scenario_screen": scenario_screen,
            "budget_fraction": budget_fraction, "physical_fit_verified": False,
            "mapping_timing_verified": False}


def scenarios():
    yield "ideal", HostConfig()
    for bandwidth in (250, 500, 1000, 2000):
        yield f"link_{bandwidth}", HostConfig(h2d_mbps=bandwidth, d2h_mbps=bandwidth,
                                            h2d_setup_us=5, launch_us=1,
                                            d2h_setup_us=5, completion_us=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "resnet8/artifacts/cycle_sim/study")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fixture = load_fixture()
    rows = []
    begun = time.perf_counter()
    for label, host_base in scenarios():
        for p in (1, 2, 4, 8, 16, 32):
            for out in (2, 4, 8, 16, 26):
                if (p*26) % out:
                    continue
                for fifo in (1, 16, 64):
                    arch = Architecture(p, out)
                    host = HostConfig(**{**asdict(host_base), "output_fifo_beats": fifo})
                    result = run_frame(Conv0Core(arch), fixture, host)
                    result.update(scenario=label, spatial_lanes=p, output_lanes=out,
                                  fifo_beats=fifo,
                                  mapping_scenarios=[mapping_screen(arch, host, m) for m in MAPPINGS])
                    rows.append(result)
                    if len(rows) % 24 == 0:
                        print(f"Completed {len(rows)} clock/data runs; {label}; all outputs match", flush=True)
    # Clock sensitivity for plausible finalists and the existing reference.
    sensitivity = []
    for p, out in ((2, 26), (4, 4), (8, 8), (16, 4), (16, 16), (16, 26)):
        for clock in (50, 100, 150):
            for bandwidth in (500, 1000):
                arch = Architecture(p, out)
                host = HostConfig(clock_mhz=clock, h2d_mbps=bandwidth, d2h_mbps=bandwidth,
                                  h2d_setup_us=5, launch_us=1, d2h_setup_us=5,
                                  completion_us=1, output_fifo_beats=16)
                result = run_frame(Conv0Core(arch), fixture, host)
                result.update(spatial_lanes=p, output_lanes=out, bandwidth_mbps=bandwidth)
                sensitivity.append(result)
    payload = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_python_seconds": time.perf_counter()-begun,
               "clock_data_runs": len(rows)+len(sensitivity),
               "output_values_checked": sum(r["output_values_checked"] for r in rows+sensitivity),
               "mapping_screen_count": len(rows)*len(MAPPINGS),
               "resource_policy": "Screen at 80% of device DSP/BRAM/LUT/FF capacity, leaving 20% as an analyst-selected reserve; not a user requirement or physical-fit proof.",
               "lut_cost_status": "Fabric resource ranges are unmeasured sensitivity assumptions; no HDL tools were run.",
               "fixture_identity": fixture["identity"], "board": BOARD,
               "runs": rows, "clock_sensitivity": sensitivity}
    (args.output/"study.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    fields = ["scenario", "mac_lanes", "spatial_lanes", "output_lanes", "fifo_beats", "core_cycles",
              "core_us", "total_cycles", "total_us", "core_stall_cycles", "fifo_peak_bytes", "outputs_match"]
    with (args.output/"timing.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)
    with (args.output/"mapping_scenarios.csv").open("w", newline="", encoding="utf-8") as handle:
        mfields = ["mac_lanes", "output_lanes", "fifo_beats", *rows[0]["mapping_scenarios"][0]]
        writer = csv.DictWriter(handle, fieldnames=mfields)
        writer.writeheader()
        for r in rows:
            if r["scenario"] == "ideal":
                for mapping in r["mapping_scenarios"]:
                    writer.writerow({"mac_lanes": r["mac_lanes"], "output_lanes": r["output_lanes"],
                                     "fifo_beats": r["fifo_beats"], **mapping})
    for label, _ in scenarios():
        eligible = [r for r in rows if r["scenario"] == label and
                    r["mapping_scenarios"][0]["conservative_scenario_screen"]]
        best = min(r["total_us"] for r in eligible)
        print(f"\n{label}: fastest modeled host latency with current mapping = {best:.2f} us")
        # Narrow-bus recommendation: smallest MAC count within 1% of the fastest
        # candidate with 128/256-bit output. This is a declared selection rule.
        bounded = [r for r in eligible if r["output_lanes"] in (4, 8)]
        bounded_best = min(r["total_us"] for r in bounded)
        near = [r for r in bounded if r["total_us"] <= bounded_best*1.01]
        chosen = min(near, key=lambda r: (r["mac_lanes"], r["output_lanes"], r["fifo_beats"]))
        print(f"  <=1% of best 128/256-bit design: {chosen['mac_lanes']} MACs, "
              f"{chosen['output_lanes']} outputs, FIFO {chosen['fifo_beats']}, "
              f"{chosen['total_us']:.2f} us")
    print(f"\nCompleted {payload['clock_data_runs']} runs; checked {payload['output_values_checked']:,} outputs.")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
