"""Python arithmetic/protocol checks; never launches an HDL tool."""

from dataclasses import replace
import unittest

import numpy as np

from cycle_sim import Architecture
from cycle_sim.axi import Beat, LiteRegisters, StreamMonitor
from cycle_sim.axi_system import Accelerator, AxiConfig, QuantPipeline, run_axi_frame
from cycle_sim.fixtures import load_int8_fixture


class AxiProtocolTests(unittest.TestCase):
    def test_lite_independent_aw_w_and_stable_responses(self):
        for first in ("aw", "w"):
            slave = LiteRegisters()
            address = {"awvalid": True, "awaddr": 0}
            data = {"wvalid": True, "wdata": 1, "wstrb": 15}
            a, b = (address, data) if first == "aw" else (data, address)
            self.assertFalse(slave.step(a)["start"])
            self.assertFalse(slave.step()["start"])
            self.assertFalse(slave.step(b)["bvalid"])
            commit = slave.step()
            self.assertTrue(commit["start"])
            self.assertFalse(commit["bvalid"])
            for _ in range(3):
                response = slave.step()
                self.assertTrue(response["bvalid"])
                self.assertEqual(response["bresp"], 0)
                self.assertFalse(response["start"])
            self.assertTrue(slave.step({"bready": True})["bvalid"])
            self.assertFalse(slave.step()["bvalid"])
            slave.step({"arvalid": True, "araddr": 4}, status=7)
            for _ in range(3):
                response = slave.step(status=0)
                self.assertEqual((response["rvalid"], response["rdata"]), (True, 7))
            slave.step({"rready": True})
            self.assertFalse(slave.step()["rvalid"])

    def test_lite_strobe_errors_and_early_start(self):
        for addr, strobe, allowed, expected_error, expected_start in (
            (0, 0, True, False, False), (12, 15, True, True, False),
            (0, 15, False, True, False), (0, 1, True, False, True),
        ):
            slave = LiteRegisters()
            slave.step({"awvalid": True, "awaddr": addr, "wvalid": True, "wdata": 1, "wstrb": strobe})
            self.assertEqual(slave.step(can_start=allowed)["start"], expected_start)
            self.assertEqual(bool(slave.step()["bresp"]), expected_error)

    def test_stream_stability_and_keep(self):
        beat = Beat(b"\x80\x7f\x00", True, 3)
        self.assertEqual(beat.keep, 7)
        monitor = StreamMonitor()
        monitor.sample(beat, False)
        monitor.sample(beat, False)
        monitor.sample(beat, True)
        self.assertEqual((monitor.bytes, monitor.beats, monitor.packets, monitor.stalls), (3, 1, 1, 2))
        for changed in (None, Beat(b"\x01", True, 3), Beat(beat.data, False, 3), Beat(beat.data, True, 2)):
            monitor = StreamMonitor()
            monitor.sample(beat, False)
            with self.assertRaises(AssertionError):
                monitor.sample(changed, True)

    def test_quant_pipeline_rounding_saturation_and_freeze(self):
        # Independent Python-integer oracle, including negative ties before ReLU.
        values = np.array([-2147483648, -257, -3, -1, 0, 1, 3, 253, 255, 2147483647], np.int32)
        multiplier = np.full(26, 1 << 30, np.int32)
        shifts = np.full(26, 31, np.int32)
        pipe = QuantPipeline(multiplier, shifts)
        self.assertIsNone(pipe.step((23, values), True))
        self.assertIsNone(pipe.step(None, True))
        before = pipe.stages[1][1].copy()
        for _ in range(5):
            self.assertIsNone(pipe.step(None, False))
            np.testing.assert_array_equal(before, pipe.stages[1][1])
        pipe.step(None, True)
        address, result = pipe.step(None, True)
        expected = []
        for value in values:
            product = int(value)*(1 << 30)
            magnitude = (abs(product)+(1 << 30)) >> 31
            rounded = -magnitude if product < 0 else magnitude
            expected.append(max(0, min(127, rounded)))
        self.assertEqual(address, 23)
        self.assertEqual(result, bytes(expected))


class AxiFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = load_int8_fixture()

    def test_fixture_quantization_matches_original_export(self):
        expected = []
        for address, acc in enumerate(self.fixture["expected"]):
            channel = address % 26
            product = int(acc)*int(self.fixture["multiplier"][channel])
            shift = int(self.fixture["shift"][channel])
            rounded = (abs(product)+(1 << (shift-1))) >> shift
            expected.append(max(0, min(127, -rounded if product < 0 else rounded)))
        self.assertEqual(bytes(expected), self.fixture["expected_int8"].tobytes())

    def test_full_frame_stalls_partial_input_beats_and_trace(self):
        rows = []
        cfg = AxiConfig(stream_bits=128, loader_bytes_per_cycle=1, input_fifo_beats=1,
                        output_fifo_bytes=128, sink_stall_probability=0.35,
                        h2d_mbps=1600, d2h_mbps=500, axil_aw_delay=4, axil_w_delay=0)
        r = run_axi_frame(Architecture(32, 52), cfg, self.fixture, trace=rows.append)
        self.assertEqual((r["h2d_payload_bytes"], r["d2h_payload_bytes"]), (4086, 26624))
        self.assertEqual((r["input_packets"], r["output_packets"]), (4, 1))
        self.assertGreater(r["h2d_backpressure_cycles"], 0)
        self.assertGreater(r["core_stall_cycles"], 0)
        self.assertLessEqual(r["tx_fifo_peak_bytes"], cfg.output_fifo_bytes)
        self.assertTrue(any(row["s_axis_tvalid"] and row["s_axis_tkeep"] != "0xffff" for row in rows))
        self.assertTrue(any(row["m_axis_tvalid"] and not row["m_axis_tready"] for row in rows))
        self.assertEqual(sum(r["phase_cycles"].values()), r["cycles"])
        self.assertEqual(r["core_cycles"], r["ideal_core_cycles"]+r["core_stall_cycles"])
        self.assertEqual(r["core_done_cycle"]-r["start_cycle"]+1, r["core_cycles"])

    def test_wide_loader_and_residency_preserve_data(self):
        arch = Architecture(16, 26)
        slow = AxiConfig(loader_bytes_per_cycle=1)
        narrow = run_axi_frame(arch, slow, self.fixture)
        wide = run_axi_frame(arch, replace(slow, loader_bytes_per_cycle=16), self.fixture)
        warm = run_axi_frame(arch, replace(slow, loader_bytes_per_cycle=16), self.fixture, resident_parameters=True)
        self.assertEqual(narrow["output_sha256"], wide["output_sha256"])
        self.assertEqual(wide["output_sha256"], warm["output_sha256"])
        self.assertLess(wide["cycles"], narrow["cycles"])
        self.assertLess(warm["cycles"], wide["cycles"])
        self.assertEqual(warm["h2d_payload_bytes"], 3072)
        self.assertEqual(warm["input_packets"], 1)

    def test_invalid_packet_and_timeout(self):
        device = Accelerator(Architecture(), AxiConfig(), self.fixture)
        with self.assertRaises(ValueError):
            device.step(Beat(b"\0", True, 0))
        with self.assertRaises(TimeoutError):
            run_axi_frame(Architecture(), AxiConfig(max_cycles=10), self.fixture)
        with self.assertRaises(ValueError):
            AxiConfig(h2d_mbps=0)
        with self.assertRaises(ValueError):
            Accelerator(Architecture(32, 104), AxiConfig(output_fifo_bytes=16), self.fixture)


if __name__ == "__main__":
    unittest.main()
