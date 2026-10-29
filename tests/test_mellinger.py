import unittest

import numpy as np

import crazyflie_sim.model.actuators as actuators
import crazyflie_sim.model.params as params
from crazyflie_sim.controllers import mellinger as mellinger_controller
from crazyflie_sim.controllers.base import DesiredState
from crazyflie_sim.model.quadcopter import Quadcopter
from crazyflie_sim.sim.scenario import Scenario
from crazyflie_sim.sim.simulation import run as sim_run
from crazyflie_sim.model.actuator_profile import CRAZYFLIE21_PHYSICAL as CF


class TestMellingerController(unittest.TestCase):
    def setUp(self):
        self.dt = 1.0 / 500.0
        self.backend = mellinger_controller.MellingerController(dt=self.dt)
        self.backend.set_dt(self.dt)
        self.backend.reset()

    def test_hover_at_setpoint_has_hover_thrust_and_small_moment(self):
        quad = Quadcopter((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        des_state = DesiredState(
            pos=np.array([0.0, 0.0, 1.0]),
            vel=np.zeros(3),
            acc=np.zeros(3),
            yaw=0.0,
            yawdot=0.0,
        )

        F, M = self.backend.run_wrench(quad, des_state)

        self.assertAlmostEqual(F, CF.mass * params.g, places=4)
        self.assertTrue(np.linalg.norm(M) < 1e-8)

    def test_positive_x_error_generates_positive_pitch_moment(self):
        quad = Quadcopter((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        des_state = DesiredState(
            pos=np.array([0.5, 0.0, 1.0]),
            vel=np.zeros(3),
            acc=np.zeros(3),
            yaw=0.0,
            yawdot=0.0,
        )

        _, M = self.backend.run_wrench(quad, des_state)

        self.assertGreater(float(M[1, 0]), 0.0)
        self.assertAlmostEqual(float(M[0, 0]), 0.0, places=8)

    def test_yaw_error_generates_nonzero_yaw_moment(self):
        quad = Quadcopter((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        des_state = DesiredState(
            pos=np.array([0.0, 0.0, 1.0]),
            vel=np.zeros(3),
            acc=np.zeros(3),
            yaw=np.pi / 6.0,
            yawdot=0.0,
        )

        _, M = self.backend.run_wrench(quad, des_state)

        self.assertGreater(float(M[2, 0]), 0.0)

    def test_reset_clears_integrators(self):
        quad = Quadcopter((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        des_state = DesiredState(
            pos=np.array([0.3, -0.2, 1.2]),
            vel=np.zeros(3),
            acc=np.zeros(3),
            yaw=0.0,
            yawdot=0.0,
        )

        self.backend.run_wrench(quad, des_state)
        self.assertGreater(np.linalg.norm(self.backend.diagnostics()["i_error_pos"]), 0.0)

        self.backend.reset()

        self.assertTrue(
            np.allclose(self.backend.diagnostics()["i_error_pos"], np.zeros(3))
        )
        self.assertTrue(
            np.allclose(self.backend.diagnostics()["i_error_att"], np.zeros(3))
        )

    def test_public_run_returns_four_pwm_commands(self):
        quad = Quadcopter((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        des_state = DesiredState(
            pos=np.array([0.0, 0.0, 1.0]),
            vel=np.zeros(3),
            acc=np.zeros(3),
            yaw=0.0,
            yawdot=0.0,
        )

        motor_pwm = self.backend.run(quad, des_state)
        motor_forces = actuators.pwm_to_motor_forces(motor_pwm, quad.battery_voltage)
        F, _ = actuators.motor_forces_to_wrench(motor_forces)

        self.assertEqual(motor_pwm.shape, (4,))
        self.assertTrue(np.all(motor_pwm >= 0.0))
        self.assertTrue(np.all(motor_pwm <= 65535.0))
        self.assertAlmostEqual(F, CF.mass * params.g, places=4)

    def test_hover_wind_short_run_stays_bounded(self):
        scenario = Scenario(sim_mode="hover_wind", duration_s=6.0)
        run = sim_run(scenario, "mellinger")
        errs, t = run["err"], run["t"]
        errs_post = errs[t >= scenario.wind.start_time]

        self.assertLess(float(np.max(errs_post)), 0.25)
        self.assertLess(float(np.sqrt(np.mean(errs_post ** 2))), 0.20)


if __name__ == "__main__":
    unittest.main()
