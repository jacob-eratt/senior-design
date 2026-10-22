"""Explicit, unmeasured DSP/LUT/banked-memory scenarios for INT8 AXIS designs."""

import math

from .resources import BOARD


def screen(arch, cfg, dsp_fraction=1.0, dsp_per_mac=2, requant_mapping="dsp"):
    if (dsp_fraction not in (0, 0.25, 0.5, 0.75, 1) or dsp_per_mac not in (1, 2)
            or requant_mapping not in ("dsp", "fabric")):
        raise ValueError("Unsupported mapping scenario")
    # Assign entire groups of 26 output channels to a mapping.
    dsp_mac = round(arch.spatial_lanes*dsp_fraction)*26
    fabric_mac = arch.macs-dsp_mac
    fabric_acc = dsp_mac if dsp_per_mac == 1 else 0
    core_dsp = dsp_mac*dsp_per_mac
    requant_dsp = 4*arch.output_lanes if requant_mapping == "dsp" else 0
    base = BOARD["baseline"]["resources"]
    core_lut_low = math.ceil(base["lut"]*arch.macs/416*0.75+64*fabric_mac+32*fabric_acc)
    core_lut_high = math.ceil(base["lut"]*arch.macs/416*1.5+160*fabric_mac+48*fabric_acc)
    quant_lut_range = ([96*arch.output_lanes]*2 if requant_mapping == "dsp"
                       else [64*arch.output_lanes, 192*arch.output_lanes])
    ff_high = math.ceil(base["ff"]*arch.macs/416*1.5+48*fabric_mac+32*fabric_acc+160*arch.output_lanes)
    # Proposal: each replica is interleaved across 32-bit write banks. One
    # BRAM36 per bank, one addressed bank/word/byte read per spatial lane.
    image_bram = arch.spatial_lanes*math.ceil(cfg.loader_bytes_per_cycle/4)
    # Input FIFO has the stream's fixed word width. The output ring must write
    # O quantized bytes and read stream-width bytes on one edge, even when O
    # exceeds the AXIS width. Propose power-of-two BYTE banks in dual-port
    # LUTRAM, avoiding a fictitious single narrow BRAM with many write ports.
    fifo_bram = math.ceil(cfg.stream_bits/64)*math.ceil(cfg.input_fifo_beats/512)
    output_banks = 1 << (max(arch.output_lanes, cfg.width)-1).bit_length()
    bank_depth = math.ceil(cfg.output_fifo_bytes/output_banks)
    output_fifo_lut = output_banks*(16*math.ceil(bank_depth/64)+8)
    reserve = {"dsp": 64, "lut": 8192, "ff": 8192, "bram36": 8}
    totals_high = {"dsp": core_dsp+requant_dsp+reserve["dsp"],
                   "lut": core_lut_high+quant_lut_range[1]+output_fifo_lut+reserve["lut"],
                   "ff": ff_high+reserve["ff"], "bram36": image_bram+fifo_bram+reserve["bram36"]}
    cap = BOARD["capacity"]
    high_failures = [k for k, v in totals_high.items() if v > cap[k]]
    totals_low = {**totals_high, "lut": core_lut_low+quant_lut_range[0]+output_fifo_lut+reserve["lut"]}
    return {"dsp_fraction": dsp_fraction, "dsp_per_mac": dsp_per_mac,
            "dsp_mac_lanes": dsp_mac, "fabric_mac_lanes": fabric_mac,
            "fabric_accumulator_lanes": fabric_acc, "core_dsp": core_dsp,
            "requant_mapping": requant_mapping, "requant_dsp_assumed": requant_dsp,
            "compute_and_requant_dsp": core_dsp+requant_dsp,
            "core_lut_range": [core_lut_low, core_lut_high], "requant_lut_range": quant_lut_range,
            "output_fifo_byte_banks": output_banks, "output_fifo_lut_assumed": output_fifo_lut,
            "image_bram36_proposed": image_bram, "fifo_bram36_proposed": fifo_bram,
            "system_reserve": reserve, "totals_with_reserve_high": totals_high,
            "passes_high_cost_scenario": not high_failures, "exceeds": high_failures,
            "passes_low_lut_cost_scenario": all(v <= cap[k] for k, v in totals_low.items()),
            "physical_fit_verified": False, "clock_verified": False,
            "note": "All new mappings, loader banking, requantizer and reserves are hypotheses requiring synthesis; LUT ranges are assumptions, not measured bounds."}


def all_mappings(arch, cfg):
    return [screen(arch, cfg, fraction, dsp, requant)
            for fraction in (0, 0.25, 0.5, 0.75, 1)
            for dsp in (1, 2) if fraction or dsp == 2
            for requant in ("dsp", "fabric")]
