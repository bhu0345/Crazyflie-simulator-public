import unittest

import numpy as np

from crazyflie_sim.model.rotor_fault import RotorFaultConfig
from crazyflie_sim.sim.scenario import RotorFaultSpec, Scenario, rotor_fault_config


class TestScenario(unittest.TestCase):
    def test_hover_reference_and_no_wind_outside_hover_wind(self):
        scenario = Scenario(sim_mode="hover_rotor_fault", hover_setpoint=(0.1, 0.2, 1.5), hover_yaw=0.3)
        desired, plot_waypoints, duration = scenario.reference()
        state = desired(0.0)
        np.testing.assert_allclose(state.pos, [0.1, 0.2, 1.5])
        np.testing.assert_allclose(state.vel, np.zeros(3))
        np.testing.assert_allclose(state.acc, np.zeros(3))
        self.assertEqual(state.yaw, 0.3)
        self.assertEqual(duration, 50.0)
        self.assertIsNone(scenario.active_wind())
        self.assertIs(Scenario(sim_mode="hover_wind").active_wind().__class__.__name__, "WindModel")

    def test_fault_only_in_fault_mode(self):
        spec = RotorFaultSpec(case="power_loss_25pct", rotor=2, start_time_s=4.0)
        fault = Scenario(sim_mode="hover_rotor_fault", rotor_fault=spec).active_fault()
        self.assertIsInstance(fault, RotorFaultConfig)
        self.assertTrue(fault.enabled)
        self.assertEqual(fault.rotor_index, 2)
        self.assertAlmostEqual(fault.start_time_s, 4.0)
        self.assertAlmostEqual(fault.effectiveness, 0.75)
        self.assertFalse(Scenario(sim_mode="hover_wind", rotor_fault=spec).active_fault().enabled)
        disabled = RotorFaultSpec(enabled=False)
        self.assertFalse(Scenario(rotor_fault=disabled).active_fault().enabled)

    def test_hovering_has_no_disturbances_even_if_specs_are_configured(self):
        scenario = Scenario(sim_mode="hovering")
        self.assertIsNone(scenario.active_wind())
        self.assertFalse(scenario.active_fault().enabled)
        self.assertEqual(scenario.event(), (None, ""))
        desired, _, _ = scenario.reference()
        for t in (0., 3., 20.):
            np.testing.assert_array_equal(desired(t).pos, scenario.hover_setpoint)
            np.testing.assert_array_equal(desired(t).vel, np.zeros(3))

    def test_named_cases(self):
        for case, expected in {
            "power_loss_25pct": 0.75,
            "power_loss_50pct": 0.50,
            "power_loss_75pct": 0.25,
            "failure": 0.0,
        }.items():
            with self.subTest(case=case):
                self.assertAlmostEqual(rotor_fault_config(case, 1, 2.0).effectiveness, expected)
        with self.assertRaises(ValueError):
            rotor_fault_config("half_gone")

    def test_multiple_rotors_and_onsets(self):
        fault = rotor_fault_config("power_loss_25pct", [1, 2], 6.0)
        self.assertEqual(fault.rotor_index, [1, 2])
        self.assertEqual(fault.rotor_slots, (0, 1))
        fault = rotor_fault_config("failure", [1, 3], [2.0, 2.5])
        self.assertEqual(fault.start_times_s, (2.0, 2.5))
        self.assertEqual(fault.active_rotor_slots(2.25), (0,))
        self.assertEqual(fault.active_rotor_slots(2.5), (0, 2))

    def test_trajectory_prepends_the_start_point(self):
        scenario = Scenario(sim_mode="trajectory", initial_pos=(0.0, 0.0, 0.0),
                            user_waypoints=((1.0, 0.0, 1.0), (1.0, 1.0, 1.0)))
        waypoints = scenario.waypoints()
        np.testing.assert_allclose(waypoints[0], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(waypoints.shape, (3, 4))
        _, _, duration = scenario.reference()
        self.assertAlmostEqual(duration, (np.sqrt(2.0) + 1.0) / 0.3 + 2.0)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            Scenario(sim_mode="hover")


if __name__ == "__main__":
    unittest.main()
