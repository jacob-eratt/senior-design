"""Latency-first Conv0 INT8 + AXI-style cycle simulation. Python/NumPy only."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from cycle_sim import Architecture
from cycle_sim.axi_system import AxiConfig, run_axi_frame
from cycle_sim.axi_resources import all_mappings, screen
from cycle_sim.fixtures import load_int8_fixture

ROOT = Path(__file__).resolve().parent


class Trace:
    def __init__(self, path, limit=100000):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", newline="", encoding="utf-8")
        self.writer = None
        self.limit, self.rows, self.dropped = limit, 0, 0

    def __call__(self, event):
        if self.rows >= self.limit:
            self.dropped += 1
            return
        if self.writer is None:
            self.writer = csv.DictWriter(self.handle, fieldnames=list(event))
            self.writer.writeheader()
        self.writer.writerow(event)
        self.rows += 1

    def close(self):
        self.handle.close()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "sweep"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"resnet8/artifacts/axi_conv0")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--trace-limit", type=int, default=100000)
    parser.add_argument("--resident-parameters", action="store_true")
    args = parser.parse_args()
    if args.trace_limit < 0:
        parser.error("trace-limit must be nonnegative")
    fixture = load_int8_fixture()
    began = time.perf_counter()
    rows, sensitivity = [], []
    if args.command == "run":
        config = json.loads(args.config.read_text()) if args.config else {}
        if set(config)-{"description", "architecture", "axi", "mapping"}:
            raise ValueError("Unknown configuration section")
        arch = Architecture(**config.get("architecture", {"spatial_lanes": 32, "output_lanes": 104}))
        host = AxiConfig(**config.get("axi", {}))
        trace = Trace(args.trace, args.trace_limit) if args.trace else None
        try:
            result = run_axi_frame(arch, host, fixture, resident_parameters=args.resident_parameters, trace=trace)
        finally:
            if trace:
                trace.close()
        result["mapping"] = screen(arch, host, **config.get("mapping", {"dsp_fraction": 1, "dsp_per_mac": 1}))
        if trace:
            result["trace"] = {"path": str(args.trace), "rows": trace.rows, "dropped": trace.dropped}
        rows.append(result)
        print(f"{arch.macs} MACs / {arch.output_lanes} output lanes: {result['total_us']:.2f} us end-to-end, "
              f"{result['core_cycles']} raw-core clocks, {result['d2h_payload_bytes']} INT8 output bytes")
        print(f"All INT32 and INT8 outputs match. High-cost resource screen: {result['mapping']['passes_high_cost_scenario']}")
    else:
        if args.config or args.trace or args.resident_parameters:
            parser.error("Sweep uses its documented grid; use run for custom config/trace/resident mode")
        for bandwidth in (500, 1000, 2000):
            for p in (8, 16, 32, 64):
                for out in (8, 13, 26, 52, 104):
                    arch = Architecture(p, out)
                    for bus in (64, 128, 256):
                        for loader in (1, 4, 16):
                            host = AxiConfig(stream_bits=bus, loader_bytes_per_cycle=loader,
                                             h2d_mbps=bandwidth, d2h_mbps=bandwidth)
                            result = run_axi_frame(arch, host, fixture)
                            result["mappings"] = all_mappings(arch, host)
                            rows.append(result)
                            if len(rows) % 30 == 0:
                                print(f"Completed {len(rows)}/540 grid runs; all INT32/INT8 values match", flush=True)
        # Cold/warm and clock sensitivity for practical existing + expanded cores.
        for p, out in ((8, 8), (16, 26), (32, 26), (32, 52)):
            for clock in (75, 100, 150):
                for resident in (False, True):
                    arch = Architecture(p, out)
                    host = AxiConfig(clock_mhz=clock, stream_bits=256, loader_bytes_per_cycle=16,
                                     h2d_mbps=2000, d2h_mbps=2000)
                    sensitivity.append(run_axi_frame(arch, host, fixture, resident_parameters=resident))
        eligible = [r for r in rows if any(m["passes_high_cost_scenario"] for m in r["mappings"])]
        for bandwidth in (500, 1000, 2000):
            candidates = [r for r in eligible if r["axi"]["d2h_mbps"] == bandwidth]
            best = min(candidates, key=lambda r: (r["cycles"], r["macs"], r["architecture"]["output_lanes"]))
            print(f"Best high-cost-screen survivor at {bandwidth} MB/s: {best['macs']} MACs, "
                  f"{best['architecture']['output_lanes']} output lanes, "
                  f"{best['axi']['stream_bits']}-bit AXIS, loader {best['axi']['loader_bytes_per_cycle']} B/clock, "
                  f"{best['total_us']:.2f} us")
    payload = {"format": "conv0-int8-axis-v1", "scope": "Proposed AXIS data + AXI-Lite control Python model; not AXI4 DDR or implemented RTL",
               "fixture_identity": fixture["identity"], "elapsed_python_seconds": time.perf_counter()-began,
               "runs": rows, "sensitivity": sensitivity,
               "full_frame_runs": len(rows)+len(sensitivity),
               "int32_values_checked": sum(r["accumulator_values_checked"] for r in rows+sensitivity),
               "int8_values_checked": sum(r["int8_values_checked"] for r in rows+sensitivity),
               "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                                  ("simulate_axi_conv0.py", "cycle_sim/core.py", "cycle_sim/axi.py",
                                   "cycle_sim/axi_system.py", "cycle_sim/axi_resources.py")}}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/(args.command+".json"), payload)
    fields = ["macs", "output_lanes", "stream_bits", "loader_bytes_per_cycle", "bandwidth_mbps",
              "total_us", "cycles", "core_cycles", "ideal_core_cycles", "core_stall_cycles",
              "h2d_payload_bytes", "d2h_payload_bytes", "h2d_backpressure_cycles", "d2h_backpressure_cycles",
              "has_high_cost_mapping", "outputs_match"]
    with (args.output/(args.command+".csv")).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: v for k, v in {**r, **r["architecture"], **r["axi"],
                             "bandwidth_mbps": r["axi"]["d2h_mbps"],
                             "has_high_cost_mapping": any(m["passes_high_cost_scenario"] for m in
                                                          r.get("mappings", [r.get("mapping")]))}.items() if k in fields})
    print(f"Saved {len(rows)+len(sensitivity)} runs to {args.output}; clocks and resource costs are assumptions.")


if __name__ == "__main__":
    main()
