"""Run or sweep the clock-stepped Python Conv0 RTL model; NumPy only."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import csv

from cycle_sim import Architecture, Conv0Core, HostConfig, Inputs, run_frame
from cycle_sim.fixtures import DEFAULT_FIXTURE, load_fixture
from cycle_sim.resources import BOARD, estimate
from cycle_sim.trace import TraceCSV

ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_config(path):
    config = {} if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
    if set(config) - {"architecture", "host", "resources", "description"}:
        raise ValueError("Unknown configuration section")
    return config


def describe(report):
    resource = report["resources"]
    print(f"{report['mac_lanes']} MACs, {report['architecture']['output_lanes']} output lanes: "
          f"core {report['core_cycles']:,} cycles / {report['core_us']:.2f} us; "
          f"host total {report['total_cycles']:,} cycles / {report['total_us']:.2f} us")
    print(f"  checked {report['output_values_checked']:,} outputs; "
          f"stalls={report['core_stall_cycles']:,}; "
          f"DSP={resource['dsp']}/{resource['capacity']['dsp']}; "
          f"core BRAM36={resource['bram36_core']}/{resource['capacity']['bram36']}")
    if not resource["core_dsp_within_budget"]:
        print("  EXCEEDS DSP budget under the configured mapping assumption.")
    if not resource["standalone_iob_within_device_limit"]:
        print("  Standalone core pin count exceeds device I/O; an internal CPU/memory wrapper is required.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "sweep"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path)
        command.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
        command.add_argument("--output", type=Path)
    run = sub.choices["run"]
    run.add_argument("--frames", type=int, default=1, help="Reuse parameters after first frame")
    run.add_argument("--trace", type=Path)
    run.add_argument("--trace-limit", type=int, default=50000)
    sweep = sub.choices["sweep"]
    sweep.add_argument("--spatial-lanes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    sweep.add_argument("--output-lanes", nargs="+", type=int, default=[2, 4, 8])
    args = parser.parse_args()
    config = load_config(args.config)
    host = HostConfig(**config.get("host", {}))
    resources = config.get("resources", {})
    fixture = load_fixture(args.fixture)
    rtl = ROOT / "rtl/conv0_416_dsp.v"
    payload = {"format": "conv0-cycle-model-v1", "board": BOARD,
               "rtl_sha256": hashlib.sha256(rtl.read_bytes()).hexdigest(),
               "fixture_identity": fixture["identity"], "runs": [], "skipped": [],
               "validation_scope": "Python golden values and register/FSM schedule; no RTL co-simulation or timing closure"}
    if args.command == "run":
        if args.frames < 1:
            parser.error("--frames must be positive")
        arch = Architecture(**config.get("architecture", {}))
        resource = estimate(arch, host, **resources)
        core = Conv0Core(arch)
        core.step(Inputs(rst=True))
        trace = TraceCSV(args.trace, args.trace_limit) if args.trace else None
        try:
            for frame in range(args.frames):
                result = run_frame(core, fixture, host, reload_parameters=frame == 0, trace=trace)
                result.update(frame=frame, resources=resource)
                payload["runs"].append(result)
                describe(result)
        finally:
            if trace:
                trace.close()
                payload["trace"] = {"path": str(args.trace), "rows": trace.rows, "dropped": trace.dropped}
        output = args.output or ROOT / "resnet8/artifacts/cycle_sim/run.json"
    else:
        for spatial in args.spatial_lanes:
            for lanes in args.output_lanes:
                try:
                    arch = Architecture(spatial, lanes)
                except ValueError as error:
                    payload["skipped"].append({"spatial_lanes": spatial, "output_lanes": lanes, "reason": str(error)})
                    continue
                core = Conv0Core(arch)
                core.step(Inputs(rst=True))
                result = run_frame(core, fixture, host)
                result["resources"] = estimate(arch, host, **resources)
                payload["runs"].append(result)
                describe(result)
        if not payload["runs"]:
            parser.error("No valid architectures in sweep")
        output = args.output or ROOT / "resnet8/artifacts/cycle_sim/sweep.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        csv_path = output.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            fields = ["mac_lanes", "output_lanes", "core_cycles", "total_cycles", "core_us", "total_us",
                      "core_stall_cycles", "dsp", "bram36_core", "core_dsp_within_budget",
                      "standalone_iob_within_device_limit", "outputs_match"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for result in payload["runs"]:
                row = {**result, **result["architecture"], **result["resources"]}
                writer.writerow({name: row[name] for name in fields})
        print(f"Comparison CSV: {csv_path}")
    write_json(output, payload)
    print(f"Report: {output}")
    print("Times use an assumed clock. Resource estimates do not establish physical fit.")


if __name__ == "__main__":
    main()
