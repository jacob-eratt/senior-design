"""Register-transfer model of conv0_416_dsp.v, plus spatial-lane experiments.

step() samples inputs and old registers at one rising edge, then commits updates.
It does NOT call the software convolution to produce DUT results. Each active
MAC receives one product from the preceding pipeline register on that clock.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass(frozen=True)
class Architecture:
    spatial_lanes: int = 16
    output_lanes: int = 4

    def __post_init__(self):
        if type(self.spatial_lanes) is not int or type(self.output_lanes) is not int:
            raise ValueError("Lane counts must be integers")
        if self.spatial_lanes not in (1, 2, 4, 8, 16, 32, 64):
            raise ValueError("spatial_lanes must be 1,2,4,8,16,32,64 for this Conv0 family")
        if self.output_lanes < 1 or self.macs % self.output_lanes:
            raise ValueError("output_lanes must be a positive divisor of spatial_lanes*26")

    @property
    def macs(self):
        return self.spatial_lanes * 26

    @property
    def tiles(self):
        return 1024 // self.spatial_lanes

    @property
    def tile_beats(self):
        return self.macs // self.output_lanes

    @property
    def ideal_core_cycles(self):
        # Counts the accepted start edge and final accepted output edge.
        return 1 + self.tiles * (1 + 27 + 2 + self.tile_beats)


class State(str, Enum):
    IDLE = "IDLE"
    INIT = "INIT"
    ISSUE = "ISSUE"
    DRAIN = "DRAIN"
    EMIT = "EMIT"


@dataclass(frozen=True)
class Inputs:
    rst: bool = False
    start: bool = False
    out_ready: bool = True
    image_write: tuple[int, int] | None = None
    weight_write: tuple[int, int] | None = None
    bias_write: tuple[int, int] | None = None


def signed(value, bits):
    """Wire-width truncation then two's-complement interpretation."""
    value = int(value) & ((1 << bits) - 1)
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


class UninitializedMemory(RuntimeError):
    pass


class Conv0Core:
    """416-lane default has the existing RTL's state/pipeline/handshake schedule.

    Memories are represented by one logical image; broadcast preload guarantees
    all RTL replicas are identical. Resource accounting counts all replicas.
    Uninitialized data consumption raises an error rather than propagating Xs.
    Invalid-cycle out_data and internal uninitialized register bits are not
    modeled as four-state Verilog. This is a synchronous functional model.
    """

    def __init__(self, architecture=Architecture()):
        self.arch = architecture
        self.image = np.zeros(3072, dtype=np.int8)
        self.weights = np.zeros((26, 27), dtype=np.int8)
        self.biases = np.zeros(26, dtype=np.int32)
        self.known_image = np.zeros(3072, dtype=bool)
        self.known_weights = np.zeros((26, 27), dtype=bool)
        self.known_biases = np.zeros(26, dtype=bool)
        self.samples_q = np.zeros(self.arch.spatial_lanes, dtype=np.int8)
        self.weights_q = np.zeros(26, dtype=np.int8)
        self.products_q = np.zeros((self.arch.spatial_lanes, 26), dtype=np.int16)
        self.acc = np.zeros((self.arch.spatial_lanes, 26), dtype=np.int32)
        self.cycle = 0
        self._reset_control()

    def _reset_control(self):
        self.state = State.IDLE
        self.tile = self.tap = self.ky = self.kx = self.ic = self.emit_beat = 0
        self.read_valid = self.product_valid = self.read_last = self.product_last = False
        self.done = False

    @property
    def busy(self):
        return self.state != State.IDLE

    def peek(self, inputs=Inputs()):
        valid = self.state == State.EMIT
        offset = self.emit_beat * self.arch.output_lanes
        return {"busy": self.busy, "done": self.done,
                "load_ready": not self.busy and not inputs.rst and not inputs.start,
                "out_valid": valid, "out_addr": self.tile * self.arch.macs + offset,
                "out_data": self.acc.reshape(-1)[offset:offset+self.arch.output_lanes].copy()
                if valid else None}

    def step(self, inputs=Inputs()):
        before = self.peek(inputs)
        state, tile, tap, beat = self.state, self.tile, self.tap, self.emit_beat
        read_valid, product_valid = self.read_valid, self.product_valid
        acc0 = int(self.acc[0, 0])
        product0 = int(self.products_q[0, 0])
        issue = state == State.ISSUE
        transfer = bool(before["out_valid"] and inputs.out_ready and not inputs.rst)
        writes = {"image": False, "weight": False, "bias": False}
        read_addresses = None

        # Nonblocking semantics: create new stages from OLD stage registers.
        next_products = (self.samples_q[:, None].astype(np.int16)
                         * self.weights_q[None, :].astype(np.int16)) if read_valid else self.products_q
        if inputs.rst:
            next_acc = np.zeros_like(self.acc)
        elif state == State.INIT:
            if not self.known_biases.all():
                raise UninitializedMemory("Bias initialization before all 26 biases were loaded")
            next_acc = np.broadcast_to(self.biases, self.acc.shape).copy()
        elif product_valid:
            # NumPy int32 add has the same modulo-2^32 behavior as RTL.
            next_acc = self.acc + self.products_q.astype(np.int32)
        else:
            next_acc = self.acc

        if issue and not inputs.rst:
            position = tile * self.arch.spatial_lanes + np.arange(self.arch.spatial_lanes)
            y = position // 32 + self.ky - 1
            x = position % 32 + self.kx - 1
            inside = (y >= 0) & (y < 32) & (x >= 0) & (x < 32)
            addresses = np.where(inside, (y * 32 + x) * 3 + self.ic, 0)
            if not self.known_image[addresses[inside]].all() or not self.known_weights[:, tap].all():
                raise UninitializedMemory("Convolution read from an unloaded input or weight")
            self.samples_q = np.where(inside, self.image[addresses], 0).astype(np.int8)
            self.weights_q = self.weights[:, tap].copy()
            read_addresses = [int(a) if ok else None for a, ok in zip(addresses, inside)]

        if before["load_ready"]:
            if inputs.image_write is not None:
                a, v = inputs.image_write
                a = int(a) & 4095
                if a < 3072:
                    self.image[a], self.known_image[a] = signed(v, 8), True
                    writes["image"] = True
            if inputs.weight_write is not None:
                a, v = inputs.weight_write
                a = int(a) & 1023
                if a < 702:
                    self.weights.flat[a], self.known_weights.flat[a] = signed(v, 8), True
                    writes["weight"] = True
            if inputs.bias_write is not None:
                a, v = inputs.bias_write
                a = int(a) & 31
                if a < 26:
                    self.biases[a], self.known_biases[a] = signed(v, 32), True
                    writes["bias"] = True

        if inputs.rst:
            self._reset_control()
        else:
            self.done = False
            if state == State.IDLE and inputs.start:
                self.tile = self.emit_beat = 0
                self.state = State.INIT
            elif state == State.INIT:
                self.tap = self.ky = self.kx = self.ic = 0
                self.state = State.ISSUE
            elif state == State.ISSUE:
                if tap == 26:
                    self.state = State.DRAIN
                else:
                    self.tap += 1
                    if self.ic == 2:
                        self.ic = 0
                        if self.kx == 2:
                            self.kx = 0
                            self.ky += 1
                        else:
                            self.kx += 1
                    else:
                        self.ic += 1
            elif state == State.DRAIN and product_valid and self.product_last:
                self.emit_beat = 0
                self.state = State.EMIT
            elif state == State.EMIT and inputs.out_ready:
                if beat == self.arch.tile_beats - 1:
                    self.emit_beat = 0
                    if tile == self.arch.tiles - 1:
                        self.state, self.done = State.IDLE, True
                    else:
                        self.tile += 1
                        self.state = State.INIT
                else:
                    self.emit_beat += 1
            self.product_last, self.read_last = self.read_last, issue and tap == 26
            self.product_valid, self.read_valid = read_valid, issue
        self.products_q, self.acc = next_products, next_acc
        self.cycle += 1
        return {"cycle": self.cycle, "state": state.value, "next_state": self.state.value,
                "tile": tile, "tap": tap, "emit_beat": beat,
                "read_valid": read_valid, "product_valid": product_valid,
                "read_addresses": read_addresses, "product0_before": product0,
                "acc0_before": acc0, "acc0_after": int(self.acc[0, 0]),
                "out_valid": before["out_valid"], "out_ready": inputs.out_ready,
                "out_addr": before["out_addr"], "out_data": before["out_data"],
                "transfer": transfer, "start_accepted": state == State.IDLE and inputs.start and not inputs.rst,
                "writes": writes, "done": self.done, "busy_after": self.busy}
