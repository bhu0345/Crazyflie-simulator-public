"""View-config and payload plumbing between runsim.py and the two UI backends.

The PyQtGraph backend is stubbed with a fake ``visualization.quad_plot_pg``
module so these tests run
without PyQt6; each probe runs in a subprocess for module-state isolation.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


_REPO = Path(__file__).resolve().parents[1]

_PROBE_PREAMBLE = r"""
import json
import sys
import types
from pathlib import Path

import numpy as np

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo))

fake_pg = types.ModuleType("crazyflie_sim.visualization.quad_plot_pg")


def _clean(value):
    if isinstance(value, np.ndarray):
        return [float(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def plot_quad_3d(waypoints, get_world_frame, frames=400, history_window_s=None,
                 fps=50, view_config=None, **kwargs):
    payload = get_world_frame(0)
    get_world_frame(1)
    print("VIEW_CONFIG", json.dumps({k: _clean(v) for k, v in (view_config or {}).items()}))
    print("PAYLOAD_KEYS", json.dumps(sorted(payload.keys())))
    for key in ("motor_thrusts", "effectiveness_truth", "effectiveness_estimate"):
        print("FIELD", key, json.dumps(_clean(payload.get(key))))


fake_pg.plot_quad_3d = plot_quad_3d
sys.modules["crazyflie_sim.visualization.quad_plot_pg"] = fake_pg
"""

_FAULT_PROBE = _PROBE_PREAMBLE + r"""
import runpy

sys.argv = [
    str(repo / "runsim.py"),
    "--config",
    str(repo / "crazyflie_sim" / "configs" / "hover_rotor_fault_default.yaml"),
]
runpy.run_path(str(repo / "runsim.py"), run_name="__main__")
"""

_WIND_PROBE = _PROBE_PREAMBLE + r"""
from crazyflie_sim import runsim

runsim.main([str(repo / "crazyflie_sim" / "configs" / "hover_wind_default.yaml"), "--controller", "baseline"])
"""


def _run_probe(script):
    result = subprocess.run(
        [sys.executable, "-c", script, str(_REPO)],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr + result.stdout)
    view_config = None
    payload_keys = None
    fields = {}
    for line in result.stdout.splitlines():
        if line.startswith("VIEW_CONFIG "):
            view_config = json.loads(line[len("VIEW_CONFIG "):])
        elif line.startswith("PAYLOAD_KEYS "):
            payload_keys = json.loads(line[len("PAYLOAD_KEYS "):])
        elif line.startswith("FIELD "):
            _prefix, key, value = line.split(" ", 2)
            fields[key] = json.loads(value)
    return view_config, payload_keys, fields


class TestViewConfigFaultMode(unittest.TestCase):
    def test_fault_mode_view_config_and_payload(self):
        view_config, payload_keys, fields = _run_probe(_FAULT_PROBE)

        self.assertEqual(view_config["sim_mode"], "hover_rotor_fault")
        # hover_rotor_fault_default.yaml injects the fault at 3 s.
        self.assertAlmostEqual(float(view_config["event_time_s"]), 3.0)
        self.assertEqual(view_config["event_label"], "rotor fault")
        # Baseline PID exposes no effectiveness estimate: thrusts are plotted.
        self.assertEqual(view_config["rotor_plot"], "thrust")
        self.assertFalse(bool(view_config["wind_as_velocity"]))

        for key in (
            "wind_force_world",
            "controller_status_label",
            "controller_status_value",
            "motor_thrusts",
            "effectiveness_truth",
            "effectiveness_estimate",
        ):
            self.assertIn(key, payload_keys)

        motor_thrusts = fields["motor_thrusts"]
        self.assertEqual(len(motor_thrusts), 4)
        self.assertTrue(all(np.isfinite(motor_thrusts)))
        self.assertGreater(float(np.sum(motor_thrusts)), 0.15)

        # Pre-fault: truth all healthy; the PID has no estimate.
        self.assertEqual(fields["effectiveness_truth"], [1.0, 1.0, 1.0, 1.0])
        self.assertIsNone(fields["effectiveness_estimate"])


class TestViewConfigWindMode(unittest.TestCase):
    def test_wind_mode_view_config_and_payload(self):
        view_config, payload_keys, fields = _run_probe(_WIND_PROBE)

        self.assertEqual(view_config["sim_mode"], "hover_wind")
        self.assertAlmostEqual(
            float(view_config["event_time_s"]), 2.0
        )
        self.assertEqual(view_config["event_label"], "wind on")
        self.assertTrue(str(view_config["wind_formula"]).strip())
        # Baseline PID exposes no effectiveness estimate.
        self.assertEqual(view_config["rotor_plot"], "thrust")

        self.assertIn("motor_thrusts", payload_keys)
        self.assertEqual(len(fields["motor_thrusts"]), 4)
        # No fault configured in hover_wind: no truth vector.
        self.assertIsNone(fields["effectiveness_truth"])
        self.assertIsNone(fields["effectiveness_estimate"])


class TestMatplotlibPayloadParsing(unittest.TestCase):
    """crazyflie_sim.visualization.quad_plot._parse_payload without creating any figure."""

    @classmethod
    def setUpClass(cls):
        import matplotlib

        matplotlib.use("Agg")
        from crazyflie_sim.visualization import quad_plot

        cls.quad_plot = quad_plot

    def test_dict_payload_parses_new_fields(self):
        payload = {
            "frame": np.zeros((3, 6)),
            "time": 1.5,
            "controller_status_label": "OK_INTERIOR",
            "controller_status_value": 0.0,
            "motor_thrusts": [0.05, 0.0, 0.06, 0.05],
            "effectiveness_truth": (1.0, 0.0, 1.0, 1.0),
            "effectiveness_estimate": [0.9, 0.1, 0.95, 1.0],
        }
        parsed = self.quad_plot._parse_payload(payload)
        (
            _frame,
            sim_time,
            _wind,
            _yaw_rate,
            _err,
            _err_max_src,
            _err_max,
            _waiting,
            _wait_label,
            _mode,
            _fault_case,
            _fault_rotor,
            _fault_start,
            _fault_state,
            status_label,
            status_value,
            motor_thrusts,
            effectiveness_truth,
            effectiveness_estimate,
        ) = parsed
        self.assertAlmostEqual(sim_time, 1.5)
        self.assertEqual(status_label, "OK_INTERIOR")
        self.assertAlmostEqual(status_value, 0.0)
        np.testing.assert_allclose(motor_thrusts, [0.05, 0.0, 0.06, 0.05])
        np.testing.assert_allclose(effectiveness_truth, [1.0, 0.0, 1.0, 1.0])
        np.testing.assert_allclose(effectiveness_estimate, [0.9, 0.1, 0.95, 1.0])

    def test_raw_frame_payload_defaults(self):
        parsed = self.quad_plot._parse_payload(np.zeros((3, 6)))
        (
            _frame,
            sim_time,
            _wind,
            _yaw_rate,
            _err,
            _err_max_src,
            _err_max,
            _waiting,
            _wait_label,
            _mode,
            _fault_case,
            _fault_rotor,
            _fault_start,
            _fault_state,
            status_label,
            status_value,
            motor_thrusts,
            effectiveness_truth,
            effectiveness_estimate,
        ) = parsed
        self.assertFalse(np.isfinite(sim_time))
        self.assertEqual(status_label, "")
        self.assertFalse(np.isfinite(status_value))
        self.assertIsNone(motor_thrusts)
        self.assertIsNone(effectiveness_truth)
        self.assertIsNone(effectiveness_estimate)


if __name__ == "__main__":
    unittest.main()
