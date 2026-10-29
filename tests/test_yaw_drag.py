import unittest

import numpy as np

import crazyflie_sim.model.params as params
import crazyflie_sim.model.actuators as actuators
from crazyflie_sim.model.actuator_profile import CRAZYFLIE21_PHYSICAL as CF
from crazyflie_sim.model.quadcopter import Quadcopter


class TestQuadraticYawDrag(unittest.TestCase):
    def _yaw_acceleration(self, yaw_rate, **plant):
        quad = Quadcopter((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), **plant)
        state = quad.state.copy()
        state[11] = yaw_rate
        return quad.state_dot(
            state,
            0.0,
            CF.mass * params.g,
            np.zeros((3, 1)),
            np.zeros(3),
        )[11]

    def test_analytic_geometry_coefficient(self):
        # 0.5 rho Cd A d^3 for the 30 mm x 100 mm envelope, times the scale 3
        geometric = 0.5 * 1.225 * 1.1 * 0.003 * 0.05 ** 3
        self.assertAlmostEqual(geometric, 2.5265625e-7, places=16)
        self.assertAlmostEqual(CF.yaw_drag_coefficient, 3.0 * geometric, places=16)

    def test_drag_is_zero_odd_and_opposes_rotation(self):
        linear_only = CF.angular_damping[2]
        self.assertEqual(self._yaw_acceleration(0.0), 0.0)
        positive = self._yaw_acceleration(100.0)
        negative = self._yaw_acceleration(-100.0)
        self.assertLess(positive, 0.0)
        self.assertGreater(negative, 0.0)
        self.assertAlmostEqual(positive, -negative, places=10)
        expected = -(
            linear_only * 100.0
            + CF.yaw_drag_coefficient * 100.0 ** 2
        ) / CF.inertia[2, 2]
        self.assertAlmostEqual(positive, expected, places=10)

    def test_plant_scale_override(self):
        linear = -CF.angular_damping[2] * 100.0 / CF.inertia[2, 2]
        nominal = self._yaw_acceleration(100.0) - linear
        for scale in (0.0, 0.5, 2.0):
            scaled = self._yaw_acceleration(100.0, yaw_drag_scale=scale) - linear
            self.assertAlmostEqual(scaled, scale * nominal, places=10)

    def test_nominal_zero_yaw_command_damps_yaw_rate(self):
        quad = Quadcopter((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        quad.state[11] = 5.0
        hover_pwm = actuators.wrench_to_pwm(
            CF.mass * params.g,
            np.zeros((3, 1)),
            quad.battery_voltage,
        )
        initial_rate = abs(quad.omega()[2])
        for _ in range(100):
            quad.update(0.002, hover_pwm)
        self.assertLess(abs(quad.omega()[2]), initial_rate)
        self.assertTrue(np.all(np.isfinite(quad.state)))


if __name__ == "__main__":
    unittest.main()
