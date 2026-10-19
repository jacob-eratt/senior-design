"""Synchronous SRAM macro model (1 read port + 1 write port, 1-cycle read latency).

Semantics match a simple dual-port BRAM in READ_FIRST mode:
  * `read(addr)` in cycle t returns the word as it was at the start of cycle t;
    the caller latches it into a pipeline register, which is the read latency.
  * `write(addr, data)` in cycle t becomes visible from cycle t+1 (`clock()`).
  * More accesses per cycle than there are ports raises HardwareError, so a
    schedule with a structural hazard cannot silently pass verification.
"""

import numpy as np


class HardwareError(RuntimeError):
    """Port conflict, out-of-range address or other illegal hardware behaviour."""


class SRAM:
    def __init__(self, name, depth, word_shape, dtype, read_ports=1, write_ports=1, trace=None):
        self.name = name
        self.mem = np.zeros((depth, *word_shape), dtype=dtype)
        self.read_ports = read_ports
        self.write_ports = write_ports
        self.reads = 0
        self.writes = 0
        self._cycle_reads = 0
        self._pending = []
        self._trace = trace        # optional list collecting (mem, addr, data) writes

    @property
    def depth(self):
        return len(self.mem)

    def _check(self, addr):
        if not 0 <= addr < self.depth:
            raise HardwareError(f"{self.name}: address {addr} out of range [0, {self.depth})")

    def read(self, addr):
        self._check(addr)
        self._cycle_reads += 1
        if self._cycle_reads > self.read_ports:
            raise HardwareError(f"{self.name}: {self._cycle_reads} reads in one cycle")
        self.reads += 1
        return self.mem[addr].copy()

    def write(self, addr, data):
        self._check(addr)
        if len(self._pending) >= self.write_ports:
            raise HardwareError(f"{self.name}: {len(self._pending) + 1} writes in one cycle")
        self._pending.append((addr, np.asarray(data, dtype=self.mem.dtype)))

    def clock(self, cycle):
        for addr, data in self._pending:
            self.mem[addr] = data
            self.writes += 1
            if self._trace is not None:
                self._trace.append((cycle, self.name, addr, data.copy()))
        self._pending.clear()
        self._cycle_reads = 0

    def load(self, words, base=0):
        """Backdoor initialisation (host/DMA preload), not counted as accesses."""
        words = np.asarray(words, dtype=self.mem.dtype)
        self._check(base + len(words) - 1)
        self.mem[base:base + len(words)] = words
