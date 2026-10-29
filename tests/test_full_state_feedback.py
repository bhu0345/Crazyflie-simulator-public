import unittest

import numpy as np

import crazyflie_sim.model.actuators as actuators
import crazyflie_sim.model.params as params
from crazyflie_sim.controllers.base import DesiredState
from crazyflie_sim.controllers import full_state_feedback as full_state_feedback_controller
from crazyflie_sim.model.quadcopter import Quadcopter
from crazyflie_sim.sim.scenario import Scenario
from crazyflie_sim.sim.simulation import run as sim_run
from crazyflie_sim.model.actuator_profile import CRAZYFLIE21_PHYSICAL as CF


class TestFullStateFeedbackController(unittest.TestCase):
    def setUp(self):
        self.dt = 1.0 / 500.0
        self.backend = full_state_feedback_controller.FullStateFeedbackController(
            dt=self.dt
        )
        self.backend.set_dt(self.dt)
        self.backend.reset()

    def _hover_state(self, pos=None, yaw=0.0):
        return DesiredState(
            pos=np.array([0.0, 0.0, 1.0] if pos is None else pos, dtype=float),
            vel=np.zeros(3),
            acc=np.zeros(3),
            yaw=float(yaw),
            yawdot=0.0,
        )

    def test_hover_at_setpoint_has_hover_thrust_and_small_moment(self):
        quad = Quadcopter((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))

        F, M = self.backend.run_wrench(quad, self._hover_state())

        self.assertAlmostEqual(F, CF.mass * params.g, places=4)
        self.assertTrue(np.linalg.norm(M) < 1e-8)

    def test_positive_x_error_generates_positive_pitch_moment(self):
        quad = Quadcopter((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))

        _, M = self.backend.run_wrench(quad, self._hover_state(pos=[0.5, 0.0, 1.0]))

        self.assertGreater(float(M[1, 0]), 0.0)
        self.assertAlmostEqual(float(M[0, 0]), 0.0, places=8)

    def test_yaw_error_generates_nonzero_yaw_moment(self):
        quad = Quadcopter((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))

        _, M = self.backend.run_wrench(quad, self._hover_state(yaw=np.pi / 6.0))

        self.assertGreater(float(M[2, 0]), 0.0)

    def test_reset_clears_integrator(self):
        quad = Quadcopter((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))

        self.backend.run_wrench(quad, self._hover_state(pos=[0.05, -0.02, 1.01]))
        self.assertGreater(np.linalg.norm(self.backend.diagnostics()["e_int"]), 0.0)

        self.backend.reset()

        self.assertTrue(np.allclose(self.backend.diagnostics()["e_int"], np.zeros(3)))

    def test_public_run_returns_four_pwm_commands(self):
        quad = Quadcopter((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))

        motor_pwm = self.backend.run(quad, self._hover_state())
        motor_forces = actuators.pwm_to_motor_forces(motor_pwm, quad.battery_voltage)
        F, _ = actuators.motor_forces_to_wrench(motor_forces)

        self.assertEqual(motor_pwm.shape, (4,))
        self.assertTrue(np.all(motor_pwm >= 0.0))
        self.assertTrue(np.all(motor_pwm <= 65535.0))
        self.assertAlmostEqual(F, CF.mass * params.g, places=4)

    def test_lqr_closed_loop_poles_are_stable(self):
        diagnostics = self.backend.diagnostics()
        A_cl = diagnostics["A_aug"] - diagnostics["B_aug"].dot(diagnostics["K"])
        eigvals = np.linalg.eigvals(A_cl)

        self.assertLess(float(np.max(eigvals.real)), 0.0)

    def test_hover_wind_short_run_stays_bounded(self):
        scenario = Scenario(sim_mode="hover_wind", duration_s=6.0)
        run = sim_run(scenario, "full_state_feedback")
        errs, t = run["err"], run["t"]
        errs_post = errs[t >= scenario.wind.start_time]

        self.assertLess(float(np.max(errs_post)), 0.25)
        self.assertLess(float(np.sqrt(np.mean(errs_post ** 2))), 0.20)


if __name__ == "__main__":
    unittest.main()
