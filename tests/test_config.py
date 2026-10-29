import tempfile
import unittest
from pathlib import Path

import numpy as np

from crazyflie_sim.sim.config import from_dict, load_config, params_overridden
from crazyflie_sim.sim.simulation import Simulation

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "crazyflie_sim" / "configs").glob("*.yaml"))


class TestEveryConfig(unittest.TestCase):
    def test_default_is_a_full_run_without_disturbances(self):
        from crazyflie_sim.runsim import load_run_config, parse_args

        cfg = load_run_config(parse_args([]))
        self.assertEqual(cfg.scenario.sim_mode, "hovering")
        self.assertEqual(cfg.scenario.event(), (None, ""))
        np.testing.assert_array_equal(cfg.scenario.initial_pos, [0., 0., 1.])
        np.testing.assert_array_equal(cfg.scenario.hover_setpoint, [0., 0., 1.])
        with params_overridden(cfg.params_overrides):
            sim = Simulation.from_scenario(cfg.scenario, cfg.controller_mode, cfg.gains())
            run = sim.run()
        self.assertAlmostEqual(run["t"][-1], 20.)
        np.testing.assert_array_equal(run["wind"], np.zeros_like(run["wind"]))
        np.testing.assert_array_equal(run["desired_pos"], np.tile([0., 0., 1.], (len(run["t"]), 1)))
        self.assertFalse(sim.quad.rotor_fault.enabled)
        self.assertTrue(np.all(np.isfinite(run["pos"])))

    def test_every_config_loads_and_flies(self):
        self.assertGreater(len(CONFIGS), 5)
        for path in CONFIGS:
            with self.subTest(config=path.name):
                cfg = load_config(path)
                with params_overridden(cfg.params_overrides):
                    sim = Simulation.from_scenario(
                        cfg.scenario, cfg.controller_mode, cfg.gains(), duration_s=0.1
                    )
                    run = sim.run()
                self.assertTrue(np.all(np.isfinite(run["pos"])))


class TestLoader(unittest.TestCase):
    def _write(self, directory, name, text):
        path = Path(directory) / name
        path.write_text(text)
        return path

    def test_extends_merges_mappings(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, "base.yaml", "sim_mode: hover_wind\nduration_s: 5.0\n"
                        "wind: {start_time: 1.0, x: {mode: constant, constant: 0.2}}\n")
            child = self._write(tmp, "child.yaml", "extends: base.yaml\nname: child\n"
                                "wind: {x: {constant: 0.3}}\n")
            cfg = load_config(child)
        self.assertEqual(cfg.name, "child")
        self.assertEqual(cfg.scenario.duration_s, 5.0)
        self.assertEqual(cfg.scenario.wind.start_time, 1.0)
        self.assertEqual((cfg.scenario.wind.x.mode, cfg.scenario.wind.x.constant), ("constant", 0.3))

    def test_errors_name_the_key(self):
        cases = (
            ({"sim_mode": "hover"}, "sim_mode"),
            ({"controller_mode": "nope"}, "controller_mode"),
            ({"controller_gains": {"nope": {}}}, "controller_gains"),
            ({"bogus": 1}, "bogus"),
            ({"wind": {"FX_MODE": "sin"}}, "UPPERCASE"),
            ({"params_overrides": {"g": 9.0}}, "params_overrides"),
            ({"initial_pos": [0, 0]}, "initial_pos"),
        )
        for raw, needle in cases:
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, needle):
                    from_dict(raw)

    def test_loading_touches_no_global_state(self):
        import crazyflie_sim.model.params as params

        tau = params.battery_sag_tau
        cfg = from_dict({"params_overrides": {"battery_sag_tau": 0.9}})
        self.assertEqual(params.battery_sag_tau, tau)
        with params_overridden(cfg.params_overrides):
            self.assertEqual(params.battery_sag_tau, 0.9)
        self.assertEqual(params.battery_sag_tau, tau)

    def test_vehicle_keys_point_to_the_profile(self):
        for key, where in (("mass", "plant_overrides.mass"), ("I", "plant_overrides.inertia"),
                           ("motor_tau_up", "plant_overrides.motor_tau_up")):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, where):
                    from_dict({"params_overrides": {key: 1.0}})
        cfg = from_dict({"plant_overrides": {"mass": 0.05}})
        sim = Simulation.from_scenario(cfg.scenario, "baseline", duration_s=0.01)
        self.assertEqual(sim.quad.mass, 0.05)                    # the plant is perturbed
        self.assertEqual(sim.controller.mass, 0.040)             # the controller is not

    def test_archived_snapshot_loads_back(self):
        import yaml

        for path in CONFIGS:
            with self.subTest(config=path.name):
                cfg = load_config(path)
                again = from_dict(yaml.safe_load(yaml.safe_dump(cfg.to_dict())))
                self.assertEqual(again.to_dict(), cfg.to_dict())


if __name__ == "__main__":
    unittest.main()
