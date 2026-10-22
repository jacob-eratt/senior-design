"""Calibrated resource screening, explicitly separate from cycle simulation."""

import json
import math
from pathlib import Path

BOARD = json.loads(Path(__file__).with_name("kv260.json").read_text(encoding="utf-8"))


def estimate(arch, host, dsp_per_mac=2, reserve_dsp=0, reserve_bram36=0):
    if any(type(value) is not int for value in (dsp_per_mac, reserve_dsp, reserve_bram36)):
        raise ValueError("DSP mapping and resource reservations must be integers")
    if dsp_per_mac not in (1, 2):
        raise ValueError("Use measured 2 DSP/MAC or explicitly hypothetical 1 DSP/MAC")
    capacity = BOARD["capacity"]
    if not 0 <= reserve_dsp <= capacity["dsp"] or not 0 <= reserve_bram36 <= capacity["bram36"]:
        raise ValueError("Reservations must be within device capacities")
    baseline = arch.spatial_lanes == 16 and arch.output_lanes == 4 and dsp_per_mac == 2
    dsp = arch.macs*dsp_per_mac
    bram = arch.spatial_lanes  # Calibrated 3072x8 image copy maps to one RAMB36.
    fifo_payload = host.output_fifo_beats*arch.output_lanes*4
    fifo_lower_bound = math.ceil(fifo_payload/4608)
    return {"board": BOARD["board"], "capacity": capacity,
            "dsp": dsp, "bram36_core": bram, "image_replica_payload_bytes": 3072*arch.spatial_lanes,
            "weight_bytes": 702, "bias_bytes": 104,
            "accumulator_bits": arch.macs*32, "product_register_bits": arch.macs*16,
            "dsp_per_mac": dsp_per_mac,
            "dsp_basis": "observed baseline ratio" if dsp_per_mac == 2 else "hypothesis requiring RTL change and synthesis",
            "core_dsp_within_budget": dsp <= capacity["dsp"]-reserve_dsp,
            "core_bram_within_budget": bram <= capacity["bram36"]-reserve_bram36,
            "reserve_dsp": reserve_dsp, "reserve_bram36": reserve_bram36,
            "dsp_percent": 100*dsp/capacity["dsp"],
            "bram36_core_percent": 100*bram/capacity["bram36"],
            "core_lut_measured": BOARD["baseline"]["resources"]["lut"] if baseline else None,
            "core_ff_measured": BOARD["baseline"]["resources"]["ff"] if baseline else None,
            "fifo_payload_bytes": fifo_payload, "fifo_bram36_payload_lower_bound": fifo_lower_bound,
            "fifo_memory_note": "Additional adapter storage; BRAM lower bound ignores width/depth granularity. No adapter LUT/FF/BRAM implementation is claimed.",
            "standalone_iob_bits": 101+32*arch.output_lanes,
            "standalone_iob_within_device_limit": 101+32*arch.output_lanes <= capacity["iob"],
            "baseline_measured": baseline,
            "physical_fit_status": "not established; needs integrated interface, complete resource accounting, placement/routing and timing"}
