"""Resource decisions must preserve measured evidence and disclose uncertainty."""

import unittest

from cycle_sim import Architecture, HostConfig
from cycle_sim.resources import BOARD
from explore_conv0 import MAPPINGS, mapping_screen


class ResourceStudyTests(unittest.TestCase):
    def test_baseline_uses_report_and_accounts_for_external_fifo(self):
        row = mapping_screen(Architecture(), HostConfig(), "observed_two_dsp")
        baseline = BOARD["baseline"]["resources"]
        self.assertEqual(row["dsp"], baseline["dsp"])
        self.assertEqual(row["lut_scenario_low"], baseline["lut"])
        self.assertEqual(row["lut_scenario_high"], baseline["lut"])
        self.assertGreater(row["bram36_with_proposed_fifo"], baseline["bram36"])
        self.assertTrue(row["measured_compute_core"])
        self.assertFalse(row["physical_fit_verified"])

    def test_large_design_cannot_hide_dsp_overcommit(self):
        arch = Architecture(32, 8)
        observed = mapping_screen(arch, HostConfig(), "observed_two_dsp")
        hypothetical = mapping_screen(arch, HostConfig(), "dsp_multiply_fabric_accumulate")
        self.assertFalse(observed["dsp_bram_device_screen"])
        self.assertFalse(observed["conservative_scenario_screen"])
        self.assertTrue(hypothetical["conservative_scenario_screen"])
        self.assertFalse(hypothetical["measured_compute_core"])
        self.assertFalse(hypothetical["mapping_timing_verified"])

    def test_fabric_tradeoff_and_uncertainty_are_explicit(self):
        rows = [mapping_screen(Architecture(8, 8), HostConfig(), m) for m in MAPPINGS]
        self.assertEqual(rows[-1]["dsp"], 0)
        self.assertEqual(rows[-1]["fabric_mac_lanes"], 208)
        self.assertGreater(rows[-1]["lut_scenario_high"], rows[0]["lut_scenario_high"])
        for row in rows:
            self.assertLess(row["lut_scenario_low"], row["lut_scenario_high"])
            self.assertFalse(row["physical_fit_verified"])
            self.assertFalse(row["measured_compute_core"])


if __name__ == "__main__":
    unittest.main()
