import unittest

import numpy as np

from crazyflie_sim.controllers.base import DesiredState
from crazyflie_sim.controllers.pid import CascadedPIDController
from crazyflie_sim.model.quadcopter import Quadcopter


class TestCascadedPID(unittest.TestCase):
    def test_run(self):
        quad = Quadcopter((0, 0, 0), [0, 0, np.pi / 6])
        des_state = DesiredState(
            pos=np.array([0.0, 0.0, 1.0]),
            vel=np.zeros(3),
            acc=np.zeros(3),
            yaw=0.0,
            yawdot=0.0,
        )
        controller = CascadedPIDController(dt=0.002)
        motor_pwm = controller.run(quad, des_state)
        diagnostics = controller.diagnostics()
        self.assertEqual(motor_pwm.shape, (4,))
        self.assertGreater(diagnostics["force_cmd"], 0.0)
        self.assertEqual(diagnostics["moment_cmd"].shape, (3,))

    def test_gain_override_is_per_instance(self):
        tuned = CascadedPIDController(dt=0.002, POS_XY_KP=3.0)
        stock = CascadedPIDController(dt=0.002)
        self.assertEqual(tuned.gains.POS_XY_KP, 3.0)
        self.assertNotEqual(stock.gains.POS_XY_KP, 3.0)
        with self.assertRaisesRegex(ValueError, "unknown PID gain"):
            CascadedPIDController(dt=0.002, NOT_A_GAIN=1.0)


if __name__ == "__main__":
    unittest.main()
