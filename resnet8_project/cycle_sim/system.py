"""Cycle-stepped preload, finite output FIFO, and configurable host transport.

This is an explicit transaction-level adapter proposal, not an AXI/DMA model.
Host bandwidth uses decimal MB/s; integer credit accounting avoids roundoff.
"""

from collections import Counter, deque
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import math
import random

import numpy as np

from .core import Conv0Core, Inputs


@dataclass(frozen=True)
class HostConfig:
    clock_mhz: float = 100.0
    h2d_mbps: float | None = None
    d2h_mbps: float | None = None
    h2d_setup_us: float = 0.0
    launch_us: float = 0.0
    d2h_setup_us: float = 0.0
    completion_us: float = 0.0
    output_fifo_beats: int = 16
    output_stall_probability: float = 0.0
    seed: int = 2608
    final_stall_cycles: int = 0
    max_cycles: int = 2000000

    def __post_init__(self):
        for value in (self.output_fifo_beats, self.seed, self.final_stall_cycles, self.max_cycles):
            if type(value) is not int:
                raise ValueError("FIFO depth, seed, stall cycles, and timeout must be integers")
        if not math.isfinite(self.clock_mhz) or self.clock_mhz <= 0:
            raise ValueError("clock_mhz must be finite and positive")
        for value in (self.h2d_mbps, self.d2h_mbps):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("Bandwidth must be positive, or null for unlimited")
        for value in (self.h2d_setup_us, self.launch_us, self.d2h_setup_us, self.completion_us):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Delays must be finite and nonnegative")
        if self.output_fifo_beats < 1 or self.max_cycles < 1 or self.final_stall_cycles < 0:
            raise ValueError("Invalid FIFO size, timeout, or final stall count")
        if not 0 <= self.output_stall_probability < 1:
            raise ValueError("output_stall_probability must be in [0,1)")

    def delay_cycles(self, us):
        return math.ceil(Fraction(str(us)) * Fraction(str(self.clock_mhz)))

    def rate(self, mbps):
        return None if mbps is None else Fraction(str(mbps)) / Fraction(str(self.clock_mhz))


class Credits:
    def __init__(self, rate, max_burst_bytes):
        self.rate = rate
        self.units = 0
        self.denominator = 1 if rate is None else rate.denominator
        # The input adapter can assemble at least a full 32-bit bias word;
        # bounded credit accumulation never creates unbounded idle-time bursts.
        self.cap = max_burst_bytes * self.denominator

    def advance(self):
        if self.rate is not None:
            self.units = min(self.cap, self.units + self.rate.numerator)

    def take(self, count):
        if self.rate is None:
            return True
        if self.units < count * self.denominator:
            return False
        self.units -= count * self.denominator
        return True


def run_frame(core: Conv0Core, fixture, host=HostConfig(), reload_parameters=True, trace=None):
    """Run one batch-one frame, including host overhead, and check every result.

    Parameters may be retained across calls on the SAME core. Input is loaded
    for each frame. A host result appears only after its full beat crosses the
    modeled D2H link. Returning done is not enough to end the host transaction.
    """
    if core.busy:
        raise ValueError("run_frame requires an idle core")
    if not reload_parameters:
        if not (core.known_weights.all() and core.known_biases.all()):
            raise ValueError("Cannot reuse parameters before loading them")
        if not (np.array_equal(core.weights.reshape(-1), fixture["weights"]) and
                np.array_equal(core.biases, fixture["bias"])):
            raise ValueError("Resident parameters do not match the requested fixture")
    initial_cycle = core.cycle
    phases, states = Counter(), Counter()
    core_stalls = fifo_stalls = injected_stalls = 0
    fifo_peak = 0
    fifo = deque()
    fifo_bytes = 0
    beat_bytes = core.arch.output_lanes * 4
    capacity = host.output_fifo_beats * beat_bytes
    collected = np.empty(26624, dtype=np.int32)
    received = np.zeros(26624, dtype=bool)
    sent = np.zeros(26624, dtype=bool)
    rng = random.Random(host.seed)

    def tick(inputs, phase, **extra):
        if core.cycle - initial_cycle >= host.max_cycles:
            raise TimeoutError(f"Frame exceeded {host.max_cycles} cycles during {phase}")
        event = core.step(inputs)
        phases[phase] += 1
        if phase == "core":
            states[event["state"]] += 1
        if trace is not None:
            trace({**event, "phase": phase, "frame_cycle": core.cycle-initial_cycle,
                   "fifo_bytes_before_accept": fifo_bytes, **extra})
        return event

    for _ in range(host.delay_cycles(host.h2d_setup_us)):
        tick(Inputs(), "h2d_setup")
    pointers = {"image": 0, "weight": 0, "bias": 0}
    arrays = {"image": fixture["image"], "weight": fixture["weights"], "bias": fixture["bias"]}
    if not reload_parameters:
        pointers.update(weight=702, bias=26)
    credits = Credits(host.rate(host.h2d_mbps), 6)
    # A single shared H2D bandwidth budget feeds three independent write ports.
    # Fixed bias/weight/image arbitration; smaller writes may use credit when
    # a whole bias word does not yet fit. Each finite stream eventually finishes.
    # Each port accepts at most one word/clock.
    while any(pointers[k] < len(arrays[k]) for k in pointers):
        credits.advance()
        writes = {}
        for name in ("bias", "weight", "image"):
            p = pointers[name]
            if p < len(arrays[name]) and credits.take(4 if name == "bias" else 1):
                writes[name+"_write"] = (p, int(arrays[name][p]))
                pointers[name] += 1
        event = tick(Inputs(**writes), "preload")
        if sum(event["writes"].values()) != len(writes):
            raise AssertionError("A preload write was not accepted")
    for _ in range(host.delay_cycles(host.launch_us)):
        tick(Inputs(), "launch")

    d2h_rate = host.rate(host.d2h_mbps)
    d2h_units = 0
    d2h_denominator = 1 if d2h_rate is None else d2h_rate.denominator
    d2h_setup = host.delay_cycles(host.d2h_setup_us)
    elapsed = final_stalls = 0
    core_done_cycle = first_output_cycle = None
    d2h_bytes = 0
    while core_done_cycle is None or fifo:
        # Drain only previously enqueued data. New results from this clock
        # cannot reach the host until a later clock; one-cycle minimum latency.
        drained = 0
        if elapsed >= d2h_setup and fifo:
            if d2h_rate is None:
                budget = fifo_bytes
            else:
                d2h_units += d2h_rate.numerator
                budget = min(fifo_bytes, d2h_units // d2h_denominator)
            while fifo and budget:
                head = fifo[0]
                count = min(head[2], budget)
                head[2] -= count
                budget -= count
                fifo_bytes -= count
                drained += count
                if head[2] == 0:
                    address, data, _ = fifo.popleft()
                    collected[address:address+len(data)] = data
                    received[address:address+len(data)] = True
            if d2h_rate is not None:
                d2h_units -= drained * d2h_denominator
        if not fifo:
            d2h_units = 0  # Empty link cannot bank unused bandwidth for later.
        d2h_bytes += drained
        peek = core.peek()
        space = fifo_bytes + beat_bytes <= capacity
        injection = rng.random() < host.output_stall_probability
        if peek["out_valid"] and peek["out_addr"] == 26624-core.arch.output_lanes and final_stalls < host.final_stall_cycles:
            injection = True
            final_stalls += 1
        ready = space and not injection
        if peek["out_valid"] and not ready:
            core_stalls += 1
            fifo_stalls += int(not space)
            injected_stalls += int(space and injection)
        event = tick(Inputs(start=elapsed == 0, out_ready=ready),
                     "core" if core_done_cycle is None else "output_flush", d2h_bytes=drained)
        elapsed += 1
        if event["transfer"]:
            address, data = event["out_addr"], event["out_data"]
            region = slice(address, address+len(data))
            if sent[region].any():
                raise AssertionError(f"Duplicate output at {address}")
            if not np.array_equal(data, fixture["expected"][region]):
                raise AssertionError(f"Accumulator mismatch at output beat {address}")
            sent[region] = True
            fifo.append([address, data, beat_bytes])
            fifo_bytes += beat_bytes
            fifo_peak = max(fifo_peak, fifo_bytes)
            if first_output_cycle is None:
                first_output_cycle = elapsed
        if event["done"]:
            if core_done_cycle is not None or not sent.all():
                raise AssertionError("Invalid done pulse or incomplete output")
            core_done_cycle = elapsed
    if not received.all() or not np.array_equal(collected, fixture["expected"]):
        raise AssertionError("Host did not receive the exact complete tensor")
    for _ in range(host.delay_cycles(host.completion_us)):
        tick(Inputs(), "completion")
    total = core.cycle-initial_cycle
    if core_done_cycle != core.arch.ideal_core_cycles + core_stalls:
        raise AssertionError("Measured FSM cycles disagree with state accounting")
    return {"architecture": asdict(core.arch), "host": asdict(host),
            "mac_lanes": core.arch.macs, "mac_operations": 718848,
            "output_values_checked": int(received.sum()), "outputs_match": True,
            "output_int32_sha256": hashlib.sha256(collected.astype("<i4").tobytes()).hexdigest(),
            "core_cycles": core_done_cycle, "ideal_core_cycles": core.arch.ideal_core_cycles,
            "first_output_cycle_from_start": first_output_cycle,
            "total_cycles": total, "core_us": core_done_cycle/host.clock_mhz,
            "total_us": total/host.clock_mhz, "phase_cycles": dict(phases),
            "sequential_frames_per_second": host.clock_mhz*1000000/total,
            "average_macs_per_core_cycle": 718848/core_done_cycle,
            "core_state_cycles": dict(states), "core_stall_cycles": core_stalls,
            "fifo_stall_cycles": fifo_stalls, "injected_stall_cycles": injected_stalls,
            "fifo_peak_bytes": fifo_peak, "fifo_capacity_bytes": capacity,
            "h2d_bytes": 3072 + (806 if reload_parameters else 0), "d2h_bytes": d2h_bytes,
            "reload_parameters": reload_parameters,
            "mac_active_fraction": (core.arch.tiles*27)/core_done_cycle,
            "clock_status": "assumed, not timing-closed",
            "adapter_status": "proposed transaction-level model; no selected AXI/DMA interface"}
