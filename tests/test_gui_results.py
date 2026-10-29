"""Snapshot identity and archives are testable without Qt or OpenGL."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from crazyflie_sim.sim.archive import load_timeseries
from crazyflie_sim.sim.config import load_config, params_overridden
from crazyflie_sim.sim.metrics import compute_metrics
from crazyflie_sim.sim.simulation import Simulation
from crazyflie_sim.visualization.run_result import RunResult


REPO = Path(__file__).resolve().parents[1]


class TestGuiResults(unittest.TestCase):
    def make_run(self, event=0.0):
        cfg = load_config(REPO / "crazyflie_sim/configs/hover_wind_dryden.yaml")
        cfg.controller_mode = "baseline"
        cfg.scenario.duration_s = 0.06
        cfg.scenario.wind.start_time = event
        with params_overridden(cfg.params_overrides):
            sim = Simulation.from_scenario(cfg.scenario, cfg.controller_mode, cfg.gains())
            sim.run()
        return cfg, sim

    def test_shared_metrics_keep_comparison_import_compatible(self):
        from crazyflie_sim.experiments.compare import compute_metrics as legacy

        self.assertIs(legacy, compute_metrics)
        cfg, sim = self.make_run()
        result = RunResult.capture(cfg, sim, "Done")
        with params_overridden(cfg.params_overrides):
            expected = compute_metrics(cfg.scenario, sim.results(), cfg.metrics)
        self.assertEqual(result.metrics.keys(), expected.keys())
        for key in expected:
            np.testing.assert_equal(result.metrics[key], expected[key])

    def test_archive_round_trip_is_complete_and_does_not_overwrite(self):
        cfg, sim = self.make_run()
        result = RunResult.capture(cfg, sim, "Done")
        expected = sim.results()["pos"].copy()
        sim.results()["pos"][:] = 999
        cfg.scenario.duration_s = 99
        result.metrics["unavailable"] = float("nan")
        with tempfile.TemporaryDirectory() as tmp:
            first, second = result.export(tmp), result.export(tmp)
            self.assertNotEqual(first, second)
            self.assertEqual({p.name for p in first.iterdir()},
                             {"timeseries.npz", "config.yaml", "env.json", "metrics.json"})
            np.testing.assert_array_equal(load_timeseries(first / "timeseries.npz")["pos"], expected)
            self.assertEqual(load_config(first / "config.yaml").scenario.duration_s, 0.06)
            metrics = json.loads((first / "metrics.json").read_text())
            self.assertIsNone(metrics["metrics"]["unavailable"])
            self.assertTrue(metrics["interval"]["event_reached"])

    def test_event_not_reached_has_no_post_event_statistics(self):
        cfg, sim = self.make_run(event=10)
        result = RunResult.capture(cfg, sim, "Done")
        self.assertFalse(result.interval["event_reached"])
        self.assertIsNone(result.metrics["rms_err_post_wind_m"])
        self.assertIsNone(result.metrics["max_err_post_wind_m"])
        self.assertIn("Event not reached", result.summary())
        self.assertIsNotNone(result.metrics["final_err_m"])

    def test_trajectory_uses_full_run_and_invalid_samples_are_explicit(self):
        cfg = load_config(REPO / "crazyflie_sim/configs/benchmark/02_trajectory.yaml")
        cfg.scenario.duration_s = 0.06
        with params_overridden(cfg.params_overrides):
            sim = Simulation.from_scenario(cfg.scenario, cfg.controller_mode, cfg.gains())
            sim.run()
        result = RunResult.capture(cfg, sim, "Done")
        self.assertEqual(result.interval["start_s"], 0)
        self.assertAlmostEqual(result.metrics["rms_err_m"],
                               float(np.sqrt(np.mean(sim.results()["err"] ** 2))))
        sim.results()["pos"][-1, 0] = float("nan")
        result = RunResult.capture(cfg, sim, "Error", "invalid plant state")
        self.assertIn("Non-finite data", result.summary())
        self.assertIn("invalid plant state", result.summary())


if __name__ == "__main__":
    unittest.main()
