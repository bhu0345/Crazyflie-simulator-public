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

    def test_takeoff_climbs_from_the_start_and_is_off_by_default(self):
        scenario = Scenario(sim_mode="hovering", initial_pos=(0.0, 0.0, 0.0),
                            hover_setpoint=(0.2, 0.0, 1.0), hover_yaw=0.4, takeoff_duration_s=2.0)
        desired, plot_waypoints, duration = scenario.reference()
        np.testing.assert_allclose(plot_waypoints, [[0.0, 0.0, 0.0], [0.2, 0.0, 1.0]])
        self.assertEqual(duration, 50.0)
        np.testing.assert_allclose(desired(0.0).pos, np.zeros(3))
        np.testing.assert_allclose(desired(1.0).pos, [0.1, 0.0, 0.5])
        np.testing.assert_allclose(desired(1.0).vel, [0.1875, 0.0, 0.9375])  # peak 1.875 * d / T
        for t in (0.0, 2.0, 5.0):
            np.testing.assert_allclose(desired(t).acc, np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(desired(5.0).pos, [0.2, 0.0, 1.0])
        np.testing.assert_allclose(desired(5.0).vel, np.zeros(3))
        self.assertEqual(desired(1.0).yaw, 0.4)
        stepped = Scenario(sim_mode="hovering", initial_pos=(0.0, 0.0, 0.0)).reference()[0]
        np.testing.assert_allclose(stepped(0.0).pos, [0.0, 0.0, 1.0])

    def test_timed_waypoints_are_reached_on_time_without_overshoot(self):
        rows = ((2.0, 0.0, 0.0, 1.0, 0.0), (4.0, 0.6, 0.0, 1.0, 0.0), (6.0, 0.6, 0.6, 1.2, 3.0),
                (7.0, 0.6, 0.6, 1.2, -3.0), (9.0, 0.0, 0.6, 0.5, -3.0))
        scenario = Scenario(sim_mode="trajectory", waypoint_source="timed", timed_waypoints=rows)
        desired, plot_waypoints, duration = scenario.reference()
        self.assertEqual(duration, 11.0)
        self.assertEqual(len(plot_waypoints), 6)  # the start at t = 0 is prepended
        for t, x, y, z, yaw in ((0.0, 0.0, 0.0, 0.0, 0.0),) + rows:
            state = desired(t)
            np.testing.assert_allclose(state.pos, [x, y, z], atol=1e-12)
            self.assertAlmostEqual(np.cos(state.yaw - yaw), 1.0)
        times = np.linspace(0.0, duration, 2201)
        states = [desired(t) for t in times]
        pos = np.array([s.pos for s in states])
        vel = np.array([s.vel for s in states])
        self.assertTrue(np.all(pos.min(axis=0) >= [0.0, 0.0, 0.0]))
        self.assertTrue(np.all(pos.max(axis=0) <= [0.6, 0.6, 1.2] + np.array(1e-12)))
        np.testing.assert_allclose(np.gradient(pos, times, axis=0)[2:-2], vel[2:-2], atol=2e-3)
        # 3.0 -> -3.0 rad is a short turn through +/-pi, not a 6 rad unwind.
        yaw = np.unwrap([s.yaw for s in states])
        self.assertLess(np.ptp(yaw[(times >= 6.0) & (times <= 7.0)]), 0.3)
        np.testing.assert_allclose(desired(10.0).vel, np.zeros(3))

    def test_timed_waypoints_validation(self):
        cases = (
            ((), "list of"),
            (((1.0, 0.0, 0.0, 1.0),), "list of"),
            (((-1.0, 0.0, 0.0, 1.0, 0.0),), "negative"),
            (((0.0, 1.0, 0.0, 1.0, 0.0), (2.0, 0.0, 0.0, 1.0, 0.0)), "initial_pos"),
            (((0.0, 0.0, 0.0, 0.0, 0.0),), "after t = 0"),
            (((2.0, 0.0, 0.0, 1.0, 0.0), (2.0, 1.0, 0.0, 1.0, 0.0)), "increase"),
        )
        for rows, needle in cases:
            with self.subTest(rows=rows):
                scenario = Scenario(sim_mode="trajectory", waypoint_source="timed", timed_waypoints=rows)
                with self.assertRaisesRegex(ValueError, needle):
                    scenario.reference()


if __name__ == "__main__":
    unittest.main()
