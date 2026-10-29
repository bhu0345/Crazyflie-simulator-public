import unittest

import numpy as np

from crazyflie_sim.model.rotor_fault import RotorFaultConfig
from crazyflie_sim.model.wind import AxisWind, WindModel
from crazyflie_sim.sim.scenario import RotorFaultSpec, Scenario
from crazyflie_sim.sim.simulation import FunctionWind, Simulation, run


class TestSimulation(unittest.TestCase):
    def test_logged_reference_matches_the_recorded_state_time(self):
        from crazyflie_sim.controllers.base import DesiredState
        from crazyflie_sim.runsim import make_frame_source

        def reference(t):
            return DesiredState(np.array([t, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]),
                                np.zeros(3), 0.0, 0.0)

        sim = Simulation("mellinger", duration_s=0.04, reference=reference, initial_pos=(0, 0, 1))
        frames = make_frame_source(sim, Scenario(sim_mode="trajectory"), "mellinger")
        payload = frames(0)
        run = sim.results()
        expected = np.array([reference(t).pos for t in run["t"]])
        np.testing.assert_allclose(run["desired_pos"], expected, atol=1e-15)
        np.testing.assert_allclose(run["err"], np.linalg.norm(run["pos"] - expected, axis=1))
        self.assertAlmostEqual(run["err"][-1], payload["position_error_norm"])

    def test_stepping_equals_run(self):
        scenario = Scenario(sim_mode="hover_rotor_fault", duration_s=0.4, initial_pos=(0, 0, 1),
                            rotor_fault=RotorFaultSpec(start_time_s=0.2))
        batch = run(scenario, "mellinger")
        sim = Simulation.from_scenario(scenario, "mellinger")
        while not sim.done:
            sim.step()
        stepped = sim.results()
        for key in ("t", "pos", "att", "motor_pwm"):
            np.testing.assert_array_equal(batch[key], stepped[key])
        self.assertEqual(stepped["t"][-1], 0.4)

    def test_time_is_k_dt(self):
        sim = Simulation("baseline", duration_s=0.1)
        for _ in range(37):
            sim.step()
        self.assertEqual(sim.t, 37 * sim.dt)

    def test_off_grid_fault_splits_the_plant_step(self):
        fault = RotorFaultConfig(enabled=True, rotor_index=1, start_time_s=0.1011, effectiveness=0.0)
        kwargs = dict(duration_s=0.2, initial_pos=(0, 0, 1), rotor_fault=fault)
        split = Simulation("baseline", plant_event_times=[0.1011], **kwargs).run()
        whole = Simulation("baseline", **kwargs).run()
        self.assertFalse(np.array_equal(split["pos"], whole["pos"]))
        np.testing.assert_array_equal(split["pos"][:50], whole["pos"][:50])

    def test_velocity_wind_is_not_a_force(self):
        force = WindModel(x=AxisWind(mode="constant", constant=0.3), y=AxisWind(), z=AxisWind(), start_time=0.0)
        velocity = WindModel(x=AxisWind(mode="constant", constant=0.3), y=AxisWind(), z=AxisWind(),
                             start_time=0.0, as_velocity=True)
        a = Simulation("baseline", duration_s=0.2, initial_pos=(0, 0, 1), wind=force).run()
        b = Simulation("baseline", duration_s=0.2, initial_pos=(0, 0, 1), wind=velocity).run()
        np.testing.assert_array_equal(a["wind"], b["wind"])
        self.assertFalse(np.array_equal(a["pos"], b["pos"]))

    def test_function_wind(self):
        out = Simulation("baseline", duration_s=0.02, wind=FunctionWind(lambda t: [t, 0.0, 0.0])).run()
        np.testing.assert_allclose(out["wind"][:, 0], out["t"] - 0.002)

    def test_rotor_speed_params_override_selects_the_plant(self):
        from crazyflie_sim.sim.config import params_overridden

        self.assertEqual(Simulation("baseline", duration_s=0.01).quad.actuator_plant_backend,
                         "legacy_thrust_euler")
        with params_overridden({"rotor_speed_model_enabled": True}):
            sim = Simulation("baseline", duration_s=0.01)
        self.assertEqual(sim.quad.actuator_plant_backend, "rotor_speed_exact_zoh")
        sim = Simulation("baseline", duration_s=0.01, actuator_plant_backend="thrust_exact_zoh")
        self.assertEqual(sim.quad.actuator_plant_backend, "thrust_exact_zoh")

    def test_step_past_the_end_raises(self):
        sim = Simulation("baseline", duration_s=0.004)
        sim.run()
        with self.assertRaises(RuntimeError):
            sim.step()


if __name__ == "__main__":
    unittest.main()
