"""Single-clock AXI-style protocol primitives; no vendor IP dependency.

AXI4-Stream supports contiguous low TKEEP bytes and non-interleaved packets.
AXI4-Lite supports one outstanding write and one read, with independent AW/W.
This is an executable interface proposal, not a complete AXI verification IP.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Beat:
    data: bytes
    last: bool = False
    dest: int = 0

    @property
    def keep(self):
        return (1 << len(self.data)) - 1


class StreamMonitor:
    def __init__(self):
        self.held = None
        self.beats = self.bytes = self.packets = self.stalls = 0

    def sample(self, beat, ready):
        if self.held is not None and beat != self.held:
            raise AssertionError("AXIS VALID/payload/TKEEP/TLAST/TDEST changed while stalled")
        if beat is not None and not beat.data:
            raise AssertionError("Zero-length AXIS beat is outside this interface contract")
        self.held = beat if beat is not None and not ready else None
        if beat is not None:
            self.stalls += not ready
            if ready:
                self.beats += 1
                self.bytes += len(beat.data)
                self.packets += beat.last


class LiteRegisters:
    """0x00 write START bit0, 0x04 read STATUS, 0x08 read output-byte count.

    STATUS bit0 busy, bit1 stream_done, bit2 raw_core_done. Responses use OKAY
    (0) or SLVERR (2). AW and W can arrive in either order. B/R remain stable
    until accepted. Reset reconstructs this object and discards pending traffic.
    """
    def __init__(self):
        self.aw = self.w = None
        self.b = self.r = None

    def step(self, master=None, *, status=0, output_bytes=0, can_start=True):
        m = master or {}
        event = {"awvalid": bool(m.get("awvalid", False)), "awready": self.aw is None and self.b is None,
                 "awaddr": m.get("awaddr", 0), "wvalid": bool(m.get("wvalid", False)),
                 "wready": self.w is None and self.b is None, "wdata": m.get("wdata", 0),
                 "wstrb": m.get("wstrb", 15), "bvalid": self.b is not None,
                 "bready": bool(m.get("bready", False)), "bresp": self.b or 0,
                 "arvalid": bool(m.get("arvalid", False)), "arready": self.r is None,
                 "araddr": m.get("araddr", 0), "rvalid": self.r is not None,
                 "rready": bool(m.get("rready", False)), "rdata": self.r[0] if self.r else 0,
                 "rresp": self.r[1] if self.r else 0, "start": False}
        # Commit only address/data held BEFORE this edge. No same-edge response.
        if self.aw is not None and self.w is not None and self.b is None:
            value, strobe = self.w
            request = bool(value & 1 and strobe & 1)
            legal = self.aw == 0 and (not request or can_start)
            event["start"] = legal and request
            self.b = 0 if legal else 2
            self.aw = self.w = None
        if event["awvalid"] and event["awready"]:
            self.aw = int(event["awaddr"])
        if event["wvalid"] and event["wready"]:
            self.w = (int(event["wdata"]) & 0xffffffff, int(event["wstrb"]) & 15)
        if event["bvalid"] and event["bready"]:
            self.b = None
        if event["arvalid"] and event["arready"]:
            a = event["araddr"]
            self.r = (status, 0) if a == 4 else ((output_bytes, 0) if a == 8 else (0, 2))
        if event["rvalid"] and event["rready"]:
            self.r = None
        return event


def stream_fields(prefix, beat, ready, width_bytes):
    return {prefix+"tvalid": beat is not None, prefix+"tready": bool(ready),
            prefix+"tdata": (beat.data.ljust(width_bytes, b"\0").hex() if beat else ""),
            prefix+"tkeep": hex(beat.keep) if beat else "0x0",
            prefix+"tlast": bool(beat and beat.last), prefix+"tdest": beat.dest if beat else 0}
