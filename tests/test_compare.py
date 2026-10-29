import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from crazyflie_sim.experiments import compare
from crazyflie_sim.sim import archive
from crazyflie_sim.sim.config import RunConfig
from crazyflie_sim.sim.scenario import RotorFaultSpec, Scenario


def _cfg(**scenario):
    return RunConfig(name="t", scenario=Scenario(**scenario))


class TestMetrics(unittest.TestCase):
    def test_metrics_follow_sim_mode(self):
        for sim_mode, key in (
            ("hover_wind", "rms_err_post_wind_m"),
            ("hover_rotor_fault", "max_err_post_fault_m"),
            ("trajectory", "rms_err_m"),
        ):
            with self.subTest(sim_mode=sim_mode):
                cfg = _cfg(sim_mode=sim_mode, duration_s=0.3, initial_pos=(0.0, 0.0, 1.0),
                           rotor_fault=RotorFaultSpec(start_time_s=0.1))
                with contextlib.redirect_stdout(io.StringIO()):
                    runs, results = compare.compare(cfg, ["baseline"])
                metrics = results["baseline"]
                self.assertIn(key, metrics)
                for common in ("iae", "control_energy", "pwm_saturation_ratio", "all_finite", "wall_time_s"):
                    self.assertIn(common, metrics)
                self.assertTrue(metrics["all_finite"])

    def test_every_controller_flies_the_same_scenario(self):
        cfg = _cfg(sim_mode="hover_wind", duration_s=0.2)
        with contextlib.redirect_stdout(io.StringIO()):
            runs, _ = compare.compare(cfg, ["baseline", "mellinger"])
        np.testing.assert_array_equal(runs["baseline"]["wind"], runs["mellinger"]["wind"])
        np.testing.assert_array_equal(runs["baseline"]["t"], runs["mellinger"]["t"])


class TestDefaults(unittest.TestCase):
    def test_default_controllers_come_from_the_registry(self):
        from crazyflie_sim.controllers import registry

        with mock.patch.object(compare, "compare", return_value=({}, {})) as fake, \
                mock.patch.object(compare, "write_case"), \
                tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(archive, "RUNS_ROOT", Path(tmp)), \
                contextlib.redirect_stdout(io.StringIO()):
            compare.main(["hover_wind_default.yaml"])
        self.assertEqual(fake.call_args.args[1], list(registry.available()))


class TestOutputs(unittest.TestCase):
    def test_main_writes_an_archive(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(archive, "RUNS_ROOT", Path(tmp)):
            with contextlib.redirect_stdout(io.StringIO()):
                compare.main([
                    "hover_rotor_fault_default",
                    "--controllers", "baseline", "mellinger",
                    "--fault-cases", "failure", "power_loss_50pct",
                    "--duration", "0.4",
                ])
            (run_dir,) = Path(tmp).iterdir()
            for name in ("config.yaml", "env.json"):
                self.assertTrue((run_dir / name).exists(), name)
            for case in ("failure", "power_loss_50pct"):
                case_dir = run_dir / case
                for name in ("baseline.npz", "mellinger.npz", "metrics.json",
                             "summary.md", "overview.png"):
                    self.assertTrue((case_dir / name).exists(), (case, name))
                metrics = json.loads((case_dir / "metrics.json").read_text())
                self.assertEqual(set(metrics), {"baseline", "mellinger"})
                with np.load(case_dir / "baseline.npz") as data:
                    self.assertEqual(data["pos"].shape[1], 3)


if __name__ == "__main__":
    unittest.main()
