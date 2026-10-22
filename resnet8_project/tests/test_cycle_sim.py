"""Python-only cycle/data/protocol tests. No HDL simulator is invoked."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cycle_sim import Architecture, Conv0Core, HostConfig, Inputs, run_frame
from cycle_sim.core import UninitializedMemory
from cycle_sim.fixtures import load_fixture
from cycle_sim.resources import estimate
from cycle_sim.trace import TraceCSV


def preload(core, fixture):
    for i in range(3072):
        core.step(Inputs(image_write=(i, int(fixture["image"][i])),
                         weight_write=(i, int(fixture["weights"][i])) if i < 702 else None,
                         bias_write=(i, int(fixture["bias"][i])) if i < 26 else None))


class CycleModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = load_fixture()

    def test_golden_and_exact_pipeline_schedule(self):
        events = []
        result = run_frame(Conv0Core(), self.fixture, trace=lambda e: events.append(e) if e["phase"] == "core" else None)
        self.assertEqual(result["core_cycles"], 8577)
        self.assertEqual(result["phase_cycles"]["preload"], 3072)
        self.assertEqual(result["total_cycles"], 11650)
        self.assertEqual(result["core_state_cycles"],
                         {"IDLE": 1, "INIT": 64, "ISSUE": 1728, "DRAIN": 128, "EMIT": 6656})
        self.assertEqual([e["state"] for e in events[:32]],
                         ["IDLE", "INIT"] + ["ISSUE"]*27 + ["DRAIN"]*2 + ["EMIT"])
        self.assertFalse(events[2]["read_valid"])
        self.assertTrue(events[3]["read_valid"])
        self.assertFalse(events[3]["product_valid"])
        self.assertTrue(events[4]["product_valid"])
        self.assertEqual(result["first_output_cycle_from_start"], 32)
        self.assertEqual(result["output_values_checked"], 26624)

    def test_six_final_stalls_reproduce_reported_8583_cycles(self):
        held = []
        host = HostConfig(final_stall_cycles=6)
        result = run_frame(Conv0Core(), self.fixture, host,
                           trace=lambda e: held.append(e) if e["out_valid"] and not e["out_ready"] else None)
        self.assertEqual(result["core_cycles"], 8583)
        self.assertEqual(len(held), 6)
        self.assertTrue(all(e["out_addr"] == 26620 for e in held))
        for event in held:
            np.testing.assert_array_equal(event["out_data"], self.fixture["expected"][-4:])

    def test_reset_retains_memory_and_start_blocks_writes(self):
        core = Conv0Core()
        preload(core, self.fixture)
        poison = dict(image_write=(0, 77), weight_write=(0, -37), bias_write=(0, 987654))
        core.step(Inputs(start=True, **poison))
        for _ in range(10):
            core.step(Inputs(start=True, **poison))
        core.step(Inputs(rst=True, start=True, **poison))
        self.assertFalse(core.busy)
        self.assertFalse(core.done)
        self.assertFalse(core.read_valid or core.product_valid)
        np.testing.assert_array_equal(core.image, self.fixture["image"])
        np.testing.assert_array_equal(core.weights.reshape(-1), self.fixture["weights"])
        np.testing.assert_array_equal(core.biases, self.fixture["bias"])
        core.step(Inputs(image_write=(4095, 1), weight_write=(1023, 1), bias_write=(31, 1)))
        result = run_frame(core, self.fixture, reload_parameters=False)
        self.assertTrue(result["outputs_match"])
        # Reset while an EMIT beat is held; no ghost output on later clocks.
        core.step(Inputs(start=True))
        while not core.peek()["out_valid"]:
            core.step(Inputs(out_ready=False))
        core.step(Inputs(rst=True, out_ready=False))
        for _ in range(4):
            event = core.step()
            self.assertFalse(event["transfer"] or event["done"] or event["busy_after"])

    def test_padding_and_int32_wrap_independent_closed_form(self):
        # All -128 * -128 = 16384, making signs and 4/6/9-tap padding explicit.
        image = np.full(3072, -128, dtype=np.int8)
        weights = np.full(702, -128, dtype=np.int8)
        bias = np.full(26, 2147483647, dtype=np.int32)
        counts = np.full((32, 32), 9, dtype=np.int64)
        counts[[0, -1], :] = 6
        counts[:, [0, -1]] = 6
        for y in (0, 31):
            for x in (0, 31):
                counts[y, x] = 4
        wide = counts*3*16384+2147483647
        wrapped = ((wide+(1 << 31)) % (1 << 32) - (1 << 31)).astype(np.int32)
        expected = np.repeat(wrapped.reshape(-1), 26)
        result = run_frame(Conv0Core(), {"image": image, "weights": weights, "bias": bias, "expected": expected})
        self.assertTrue(result["outputs_match"])

    def test_host_bandwidth_fifo_and_residency(self):
        core = Conv0Core()
        host = HostConfig(h2d_mbps=100, d2h_mbps=100, output_fifo_beats=1,
                          h2d_setup_us=2, launch_us=1, d2h_setup_us=3, completion_us=2)
        cold = run_frame(core, self.fixture, host)
        warm = run_frame(core, self.fixture, host, reload_parameters=False)
        self.assertEqual(cold["phase_cycles"]["preload"], 3878)
        self.assertEqual(warm["phase_cycles"]["preload"], 3072)
        self.assertEqual(cold["h2d_bytes"], 3878)
        self.assertEqual(warm["h2d_bytes"], 3072)
        self.assertEqual(cold["d2h_bytes"], 106496)
        self.assertEqual(cold["total_cycles"]-warm["total_cycles"], 806)
        self.assertGreater(cold["fifo_stall_cycles"], 0)
        self.assertLessEqual(cold["fifo_peak_bytes"], 16)
        self.assertEqual(sum(cold["phase_cycles"].values()), cold["total_cycles"])
        self.assertEqual(cold["core_cycles"], 8577+cold["core_stall_cycles"])
        self.assertGreater(cold["phase_cycles"]["output_flush"], 0)

    def test_random_stalls_are_deterministic_and_conserve_results(self):
        host = HostConfig(output_stall_probability=0.25, seed=42)
        a = run_frame(Conv0Core(), self.fixture, host)
        b = run_frame(Conv0Core(), self.fixture, host)
        self.assertEqual(a, b)
        self.assertGreater(a["injected_stall_cycles"], 0)

    def test_architecture_sweep_preserves_arithmetic(self):
        for spatial, output in ((1, 2), (4, 4), (8, 8), (32, 8)):
            with self.subTest(spatial=spatial, output=output):
                arch = Architecture(spatial, output)
                result = run_frame(Conv0Core(arch), self.fixture)
                self.assertEqual(result["core_cycles"], 1+arch.tiles*30+26624//output)
                self.assertTrue(result["outputs_match"])

    def test_resources_use_observed_two_dsp_mapping(self):
        actual = estimate(Architecture(), HostConfig())
        self.assertEqual((actual["dsp"], actual["bram36_core"], actual["core_lut_measured"]), (832, 16, 12121))
        self.assertFalse(actual["standalone_iob_within_device_limit"])
        bigger = estimate(Architecture(32, 4), HostConfig())
        self.assertFalse(bigger["core_dsp_within_budget"])
        self.assertIsNone(bigger["core_lut_measured"])
        self.assertFalse(estimate(Architecture(), HostConfig(), reserve_dsp=500)["core_dsp_within_budget"])
        hypothetical = estimate(Architecture(32, 4), HostConfig(), dsp_per_mac=1)
        self.assertTrue(hypothetical["core_dsp_within_budget"])
        self.assertFalse(hypothetical["baseline_measured"])

    def test_invalid_configuration_uninitialized_data_and_timeout(self):
        with self.assertRaises(ValueError):
            Architecture(16, 3)
        with self.assertRaises(ValueError):
            Architecture(16.0, 4)
        with self.assertRaises(ValueError):
            HostConfig(output_fifo_beats=1.5)
        with self.assertRaises(ValueError):
            HostConfig(d2h_mbps=0)
        core = Conv0Core()
        with self.assertRaises(ValueError):
            run_frame(core, self.fixture, reload_parameters=False)
        core.step(Inputs(start=True))
        with self.assertRaises(UninitializedMemory):
            core.step()
        with self.assertRaises(TimeoutError):
            run_frame(Conv0Core(), self.fixture, HostConfig(max_cycles=10))

    def test_trace_is_bounded_and_serializable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"trace.csv"
            trace = TraceCSV(path, limit=8)
            result = run_frame(Conv0Core(), self.fixture, trace=trace)
            trace.close()
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            self.assertEqual(trace.dropped, result["total_cycles"]-8)
            self.assertEqual(rows[0]["phase"], "preload")


if __name__ == "__main__":
    unittest.main()
