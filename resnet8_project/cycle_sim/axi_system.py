"""Clock/data model: AXIS loader -> INT8 MAC/INT32 sum -> requant/ReLU -> AXIS.

Streams stand in for DMA-facing interfaces. DDR bursts, AXI4 memory-mapped
AR/R/AW/W/B channels, caches, clock crossings and software drivers are NOT
implemented. DMA setup, bandwidth, and service gaps are explicit assumptions.
"""

from collections import Counter, deque
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import math
import random

import numpy as np

from .axi import Beat, LiteRegisters, StreamMonitor, stream_fields
from .core import Architecture, Conv0Core, Inputs
from .system import Credits


@dataclass(frozen=True)
class AxiConfig:
    clock_mhz: float = 100
    stream_bits: int = 128
    loader_bytes_per_cycle: int = 4
    input_fifo_beats: int = 4
    output_fifo_bytes: int = 2048
    h2d_mbps: float = 1000
    d2h_mbps: float = 1000
    dma_setup_us: float = 2
    d2h_setup_us: float = 0
    completion_us: float = 1
    burst_beats: int = 16
    burst_gap_cycles: int = 2
    sink_stall_probability: float = 0
    axil_aw_delay: int = 0
    axil_w_delay: int = 2
    seed: int = 2608
    max_cycles: int = 2000000

    def __post_init__(self):
        for name in ("stream_bits", "loader_bytes_per_cycle", "input_fifo_beats", "output_fifo_bytes",
                     "burst_beats", "burst_gap_cycles", "axil_aw_delay", "axil_w_delay", "seed", "max_cycles"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.stream_bits not in (32, 64, 128, 256):
            raise ValueError("Supported stream widths: 32,64,128,256 bits")
        if self.loader_bytes_per_cycle not in (1, 4, 16, 32):
            raise ValueError("Loader widths: 1,4,16,32 bytes per clock")
        if min(self.input_fifo_beats, self.output_fifo_bytes, self.burst_beats, self.max_cycles) < 1:
            raise ValueError("Buffer sizes, burst length and timeout must be positive")
        if min(self.burst_gap_cycles, self.axil_aw_delay, self.axil_w_delay) < 0:
            raise ValueError("Cycle delays must be nonnegative")
        for name in ("clock_mhz", "h2d_mbps", "d2h_mbps"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("dma_setup_us", "d2h_setup_us", "completion_us"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.sink_stall_probability < 1:
            raise ValueError("Invalid stall probability")

    @property
    def width(self):
        return self.stream_bits//8

    def clocks(self, us):
        return math.ceil(Fraction(str(us))*Fraction(str(self.clock_mhz)))


class QuantPipeline:
    """Three registered stages, II=1 beat: multiply, round/shift, ReLU/clip.

    One globally stalled pipeline preserves all values when its output cannot
    advance. Hardware must provide one 32x32 multiplier per output lane; these
    are separate from the INT8 MACs and are charged in resource screening.
    """
    def __init__(self, multiplier, shift):
        self.multiplier = np.asarray(multiplier, dtype=np.int64)
        self.shift = np.asarray(shift, dtype=np.int64)
        if self.multiplier.shape != (26,) or self.shift.shape != (26,):
            raise ValueError("Expected 26 per-channel quantization parameters")
        if np.any(self.multiplier <= 0) or np.any(self.shift < 1) or np.any(self.shift > 62):
            raise ValueError("Invalid quantization parameters")
        self.stages = [None, None, None]

    def step(self, incoming, advance):
        if not advance:
            if incoming is not None:
                raise AssertionError("Input accepted into stalled requantizer")
            return None
        a, b, emitted = self.stages
        c_next = None if b is None else (b[0], np.clip(b[1], 0, 127).astype(np.int8).tobytes())
        b_next = None
        if a is not None:
            address, product, shifts = a
            magnitude = (np.abs(product)+(np.int64(1) << (shifts-1))) >> shifts
            b_next = (address, np.where(product < 0, -magnitude, magnitude))
        a_next = None
        if incoming is not None:
            address, values = incoming
            channels = (address+np.arange(len(values))) % 26
            a_next = (address, values.astype(np.int64)*self.multiplier[channels], self.shift[channels])
        self.stages = [a_next, b_next, c_next]
        return emitted


class Accelerator:
    def __init__(self, arch, config, fixture, resident_parameters=False):
        self.arch, self.config = arch, config
        if config.output_fifo_bytes < max(arch.output_lanes, config.width):
            raise ValueError("Output FIFO must fit a whole core beat and AXIS beat")
        self.core = Conv0Core(arch)
        self.lite = LiteRegisters()
        self.rx = deque()
        self.rx_bytes = 0
        self.loaded = {0: 0, 1: 0, 2: 0, 3: 0}
        self.sizes = {0: 3072, 1: 702, 2: 104, 3: 208}
        self.raw = {k: bytearray(n) for k, n in self.sizes.items()}
        self.packet_dest = None
        self.packet_bytes = 0
        self.pipeline = None
        self.tx = bytearray()
        self.sent = self.produced = 0
        self.raw_done = self.stream_done = False
        self.in_monitor, self.out_monitor = StreamMonitor(), StreamMonitor()
        self.rx_peak = self.tx_peak = self.q_stalls = 0
        if resident_parameters:
            for dest, data in parameter_packets(fixture):
                self._write(dest, data)

    @property
    def load_complete(self):
        return self.loaded == self.sizes and not self.rx

    def _write(self, dest, data):
        """One explicitly scheduled local loader write, broadcast to image replicas.

        Width>1 proposes banked/wide memories, not the existing byte-write RTL.
        Bias/quant words become visible only when all bytes have arrived.
        """
        if self.core.busy:
            raise AssertionError("Loader wrote active compute memories")
        offset = self.loaded[dest]
        end = offset+len(data)
        if end > self.sizes[dest]:
            raise ValueError("Packet exceeds destination memory")
        self.raw[dest][offset:end] = data
        self.loaded[dest] = end
        if dest in (0, 1):
            values = np.frombuffer(data, dtype=np.int8)
            target = self.core.image if dest == 0 else self.core.weights.reshape(-1)
            known = self.core.known_image if dest == 0 else self.core.known_weights.reshape(-1)
            target[offset:end], known[offset:end] = values, True
        elif dest == 2:
            for word in range(offset//4, end//4):
                self.core.biases[word] = int.from_bytes(self.raw[2][4*word:4*word+4], "little", signed=True)
                self.core.known_biases[word] = True
        elif end == self.sizes[3]:
            q = np.frombuffer(self.raw[3], dtype="<i4").reshape(26, 2).copy()
            self.pipeline = QuantPipeline(q[:, 0], q[:, 1])

    def step(self, source=None, sink_ready=True, master=None):
        cfg = self.config
        status = int(self.core.busy or bool(self.tx) or bool(self.pipeline and any(s is not None for s in self.pipeline.stages)))
        status |= 2*int(self.stream_done) | 4*int(self.raw_done)
        lite = self.lite.step(master, status=status, output_bytes=self.sent,
                              can_start=self.load_complete and not self.core.busy and not self.raw_done)
        # Drain previously accepted input; fresh AXIS data cannot bypass the FIFO.
        local_write_bytes = 0
        local_write_dest = local_write_address = local_write_tdata = None
        if self.rx and not self.core.busy:
            dest, data = self.rx[0]
            n = min(len(data), cfg.loader_bytes_per_cycle)
            local_write_dest, local_write_address = dest, self.loaded[dest]
            local_write_tdata = bytes(data[:n]).hex()
            self._write(dest, bytes(data[:n]))
            del data[:n]
            self.rx_bytes -= n
            local_write_bytes = n
            if not data:
                self.rx.popleft()
        source_ready = not self.core.busy and not self.raw_done and self.rx_bytes+cfg.width <= cfg.input_fifo_beats*cfg.width
        self.in_monitor.sample(source, source_ready)
        if source is not None and source_ready:
            if len(source.data) > cfg.width or source.dest not in self.sizes:
                raise ValueError("Invalid AXIS load beat")
            if self.packet_dest is None:
                if self.loaded[source.dest] or any(d == source.dest for d, _ in self.rx):
                    raise ValueError("Duplicate destination packet")
                self.packet_dest = source.dest
            if source.dest != self.packet_dest:
                raise ValueError("Packet interleaving unsupported")
            self.packet_bytes += len(source.data)
            if self.packet_bytes > self.sizes[source.dest]:
                raise ValueError("Oversized input packet")
            if source.last != (self.packet_bytes == self.sizes[source.dest]):
                raise ValueError("TLAST does not match destination length")
            self.rx.append((source.dest, bytearray(source.data)))
            self.rx_bytes += len(source.data)
            if source.last:
                self.packet_dest, self.packet_bytes = None, 0
        self.rx_peak = max(self.rx_peak, self.rx_bytes)
        count = min(cfg.width, 26624-self.sent)
        output = Beat(bytes(self.tx[:count]), self.sent+count == 26624) if count and len(self.tx) >= count else None
        self.out_monitor.sample(output, sink_ready)
        if output is not None and sink_ready:
            del self.tx[:count]
            self.sent += count
            self.stream_done = output.last
        last_stage = self.pipeline.stages[-1] if self.pipeline else None
        advance = last_stage is None or len(self.tx)+len(last_stage[1]) <= cfg.output_fifo_bytes
        self.q_stalls += int(not advance)
        core = self.core.step(Inputs(start=lite["start"], out_ready=advance and self.pipeline is not None))
        incoming = (core["out_addr"], core["out_data"]) if core["transfer"] else None
        if self.pipeline:
            emitted = self.pipeline.step(incoming, advance)
            if emitted is not None:
                address, data = emitted
                if address != self.produced:
                    raise AssertionError("Quantized output reordering")
                self.produced += len(data)
                self.tx.extend(data)
        self.tx_peak = max(self.tx_peak, len(self.tx))
        self.raw_done |= core["done"]
        return {"core": core, "lite": lite, "s_beat": source, "s_ready": source_ready,
                "m_beat": output, "m_ready": sink_ready, "local_write_bytes": local_write_bytes,
                "local_write_dest": local_write_dest, "local_write_address": local_write_address,
                "local_write_tdata": local_write_tdata or "",
                "rx_fifo_bytes": self.rx_bytes, "tx_fifo_bytes": len(self.tx), "requant_advance": advance}


def parameter_packets(fixture):
    return [(1, fixture["weights"].tobytes()), (2, fixture["bias"].astype("<i4").tobytes()),
            (3, fixture["quant"].astype("<i4").tobytes())]


def run_axi_frame(arch, config, fixture, *, resident_parameters=False, trace=None):
    """Execute one timed frame; resident mode explicitly starts with cached parameters."""
    device = Accelerator(arch, config, fixture, resident_parameters)
    packets = ([] if resident_parameters else parameter_packets(fixture))+[(0, fixture["image"].tobytes())]
    beats = deque(Beat(data[i:i+config.width], i+config.width >= len(data), dest)
                  for dest, data in packets for i in range(0, len(data), config.width))
    # Two-beat credit cap retains fractional-beat residue (e.g. 10 B/clock
    # on a 16-byte bus), without banking unbounded idle-time bandwidth.
    rx_credit = Credits(Fraction(str(config.h2d_mbps))/Fraction(str(config.clock_mhz)), 2*config.width)
    tx_credit = Credits(Fraction(str(config.d2h_mbps))/Fraction(str(config.clock_mhz)), 2*config.width)
    phases = Counter()
    source = None
    source_gap = sink_gap = rx_burst = tx_burst = 0
    rng = random.Random(config.seed)
    control_begin = start = raw_done = stream_done = first_output = None
    aw_sent = w_sent = b_received = ar_sent = r_received = False
    received = bytearray()
    accum_checked = core_stalls = 0
    dma_setup, output_setup = config.clocks(config.dma_setup_us), config.clocks(config.d2h_setup_us)
    completion = config.clocks(config.completion_us)
    for cycle in range(config.max_cycles):
        rx_credit.advance()
        tx_credit.advance()
        if cycle >= dma_setup and source_gap == 0 and source is None and beats and rx_credit.take(config.width):
            source = beats.popleft()
        if source_gap:
            source_gap -= 1
        master = {}
        if device.load_complete and control_begin is None:
            control_begin = cycle
        if control_begin is not None and not b_received:
            age = cycle-control_begin
            master = {"awvalid": not aw_sent and age >= config.axil_aw_delay, "awaddr": 0,
                      "wvalid": not w_sent and age >= config.axil_w_delay, "wdata": 1, "wstrb": 15,
                      "bready": True}
        if stream_done is not None and cycle > stream_done+completion:
            master.update(arvalid=not ar_sent, araddr=4, rready=True)
        sink_ready = (start is not None and cycle-start >= output_setup and sink_gap == 0
                      and tx_credit.units >= config.width*tx_credit.denominator
                      and rng.random() >= config.sink_stall_probability)
        if sink_gap:
            sink_gap -= 1
        event = device.step(source, sink_ready, master)
        core, lite = event["core"], event["lite"]
        if lite["awvalid"] and lite["awready"]:
            aw_sent = True
        if lite["wvalid"] and lite["wready"]:
            w_sent = True
        if lite["bvalid"] and lite["bready"]:
            if lite["bresp"]:
                raise AssertionError("AXI-Lite start rejected")
            b_received = True
        if lite["arvalid"] and lite["arready"]:
            ar_sent = True
        if lite["rvalid"] and lite["rready"]:
            if lite["rresp"] or not lite["rdata"] & 2:
                raise AssertionError("AXI-Lite status did not report completion")
            r_received = True
        if core["start_accepted"]:
            start = cycle
        if core["done"]:
            raw_done = cycle
        if core["out_valid"] and not core["out_ready"]:
            core_stalls += 1
        if core["transfer"]:
            address, data = core["out_addr"], core["out_data"]
            if address != accum_checked or not np.array_equal(data, fixture["expected"][address:address+len(data)]):
                raise AssertionError("Raw accumulator mismatch/order failure")
            accum_checked += len(data)
        if source is not None and event["s_ready"]:
            rx_burst += 1
            if source.last or rx_burst == config.burst_beats:
                source_gap, rx_burst = config.burst_gap_cycles, 0
            source = None
        output = event["m_beat"]
        if output is not None and sink_ready:
            assert tx_credit.take(config.width)
            offset = len(received)
            if output.data != fixture["expected_int8"][offset:offset+len(output.data)].tobytes():
                raise AssertionError("Requantized AXIS output mismatch")
            received.extend(output.data)
            if first_output is None:
                first_output = cycle
            tx_burst += 1
            if tx_burst == config.burst_beats:
                sink_gap, tx_burst = config.burst_gap_cycles, 0
            if output.last:
                if len(received) != 26624:
                    raise AssertionError("Premature TLAST")
                stream_done = cycle
        phase = ("dma_setup" if cycle < dma_setup else "load" if control_begin is None else
                 "control" if start is None else "compute" if raw_done is None else
                 "drain" if stream_done is None else "completion")
        phases[phase] += 1
        if trace:
            trace({"cycle": cycle, "phase": phase, "core_state": core["state"], "tile": core["tile"],
                   "tap": core["tap"], "read_addresses": str(core["read_addresses"]),
                   "product0": core["product0_before"], "acc0": core["acc0_after"],
                   "core_valid": core["out_valid"], "core_ready": core["out_ready"],
                   "core_output_address": core["out_addr"], "core_done": core["done"],
                   "local_write_bytes": event["local_write_bytes"], "rx_fifo_bytes": event["rx_fifo_bytes"],
                   "local_write_dest": event["local_write_dest"],
                   "local_write_address": event["local_write_address"],
                   "local_write_tdata": event["local_write_tdata"],
                   "tx_fifo_bytes": event["tx_fifo_bytes"], "requant_advance": event["requant_advance"],
                   **stream_fields("s_axis_", event["s_beat"], event["s_ready"], config.width),
                   **stream_fields("m_axis_", output, sink_ready, config.width),
                   **{"axil_"+k: v for k, v in lite.items()}})
        if r_received:
            break
    else:
        raise TimeoutError("AXI frame timed out")
    if accum_checked != 26624 or bytes(received) != fixture["expected_int8"].tobytes():
        raise AssertionError("Incomplete frame")
    if raw_done-start+1 != arch.ideal_core_cycles+core_stalls:
        raise AssertionError("Core cycle accounting mismatch")
    return {"architecture": asdict(arch), "axi": asdict(config), "macs": arch.macs,
            "resident_parameters": resident_parameters, "outputs_match": True,
            "accumulator_values_checked": accum_checked, "int8_values_checked": len(received),
            "output_sha256": hashlib.sha256(received).hexdigest(), "cycles": cycle+1,
            "total_us": (cycle+1)/config.clock_mhz, "core_cycles": raw_done-start+1,
            "ideal_core_cycles": arch.ideal_core_cycles, "core_stall_cycles": core_stalls,
            "core_us": (raw_done-start+1)/config.clock_mhz,
            "start_cycle": start, "core_done_cycle": raw_done, "first_output_cycle": first_output,
            "stream_done_cycle": stream_done, "output_tail_cycles": stream_done-raw_done,
            "phase_cycles": dict(phases), "h2d_payload_bytes": device.in_monitor.bytes,
            "d2h_payload_bytes": len(received), "h2d_beats": device.in_monitor.beats,
            "d2h_beats": device.out_monitor.beats, "input_packets": device.in_monitor.packets,
            "output_packets": device.out_monitor.packets, "h2d_backpressure_cycles": device.in_monitor.stalls,
            "d2h_backpressure_cycles": device.out_monitor.stalls,
            "rx_fifo_peak_bytes": device.rx_peak, "tx_fifo_peak_bytes": device.tx_peak,
            "requant_stall_cycles": device.q_stalls,
            "status": "Python cycle prediction; proposed loader/requant/AXI wrapper, no HDL or timing closure"}
