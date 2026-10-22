"""Bounded cycle CSV, useful in a spreadsheet or for a later RTL trace diff."""

import csv
import json
from pathlib import Path


class TraceCSV:
    FIELDS = ["cycle", "frame_cycle", "phase", "state", "next_state", "tile", "tap",
              "emit_beat", "read_valid", "product_valid", "read_addresses",
              "product0_before", "acc0_before", "acc0_after", "out_valid", "out_ready",
              "out_addr", "out_data", "transfer", "start_accepted", "writes", "done",
              "busy_after", "fifo_bytes_before_accept", "d2h_bytes"]

    def __init__(self, path, limit=50000):
        if limit < 1:
            raise ValueError("Trace limit must be positive")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.limit, self.rows, self.dropped = limit, 0, 0

    def __call__(self, event):
        if self.rows == self.limit:
            self.dropped += 1
            return
        row = {name: event.get(name, "") for name in self.FIELDS}
        for name in ("read_addresses", "out_data", "writes"):
            value = row[name]
            row[name] = json.dumps(value.tolist() if hasattr(value, "tolist") else value)
        self.writer.writerow(row)
        self.rows += 1

    def close(self):
        self.file.close()
