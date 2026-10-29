import unittest

import numpy as np

from crazyflie_sim.controllers import registry
from crazyflie_sim.controllers.base import ControllerBase
from crazyflie_sim.model.actuator_profile import get_actuator_profile
from crazyflie_sim.sim.scenario import DT, Scenario
from crazyflie_sim.sim.simulation import Simulation, run


class HoverThrust(ControllerBase):
    """Open-loop equal PWM: the smallest possible new controller."""

    def __init__(self, dt, pwm=30000.0):
        self.dt, self.pwm = dt, float(pwm)

    def set_dt(self, dt):
        self.dt = dt

    def reset(self):
        pass

    def run(self, quad, des_state, sim_time=0.0):
        return np.full(4, self.pwm)

    def diagnostics(self):
        return {}


class TestRegistry(unittest.TestCase):
    def test_every_controller_is_registered_once(self):
        names = registry.available()
        for name in ("baseline", "mellinger", "full_state_feedback"):
            self.assertIn(name, names)
        colors = [registry.get(n).color for n in names]
        self.assertEqual(len(colors), len(set(colors)))

    def test_fixed_kwargs_are_constructor_arguments(self):
        registry.register("test_fixed_pwm", label="Test", color="#000001", pwm=12345.0)(HoverThrust)
        try:
            self.assertEqual(registry.build("test_fixed_pwm", DT).pwm, 12345.0)
            self.assertEqual(registry.build("test_fixed_pwm", DT, {"pwm": 1.0}).pwm, 1.0)
        finally:
            registry._REGISTRY.pop("test_fixed_pwm")

    def test_unknown_names_and_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            registry.get("nope")
        with self.assertRaisesRegex(ValueError, "unknown parameter"):
            registry.build("mellinger", DT, {"kp_typo": 1.0})
        with self.assertRaisesRegex(ValueError, "dt"):
            registry.build("mellinger", DT, {"dt": 0.01})

    def test_every_controller_gets_the_run_vehicle(self):
        standard = get_actuator_profile("crazyflie21_standard")
        for mode in registry.available():
            with self.subTest(mode=mode):
                stock = registry.build(mode, DT)
                built = registry.build(mode, DT, actuator_profile="crazyflie21_standard")
                self.assertIs(built.actuator_profile, standard)
                self.assertIsNot(stock.actuator_profile, standard)

    def test_a_controller_without_a_profile_argument_still_builds(self):
        registry.register("test_hover_thrust", label="Test", color="#000000")(HoverThrust)
        try:
            registry.build("test_hover_thrust", DT, {"pwm": 1.0}, actuator_profile="crazyflie21_standard")
        finally:
            registry._REGISTRY.pop("test_hover_thrust")

    def test_a_new_controller_is_one_decorator(self):
        registry.register("test_hover_thrust", label="Test", color="#000000")(HoverThrust)
        try:
            out = run(Scenario(sim_mode="hover_wind", duration_s=0.05), "test_hover_thrust",
                      {"pwm": 25000.0})
            np.testing.assert_array_equal(out["motor_pwm"], np.full((25, 4), 25000.0))
        finally:
            registry._REGISTRY.pop("test_hover_thrust")

    def test_unregistered_instance_also_flies(self):
        out = Simulation(HoverThrust(DT), duration_s=0.02).run()
        self.assertEqual(out["mode"], "HoverThrust")


if __name__ == "__main__":
    unittest.main()
