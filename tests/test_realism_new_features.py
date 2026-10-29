import unittest

import numpy as np

import crazyflie_sim.model.params as params
from crazyflie_sim.model.control_link import (
    ControlChannel,
    build_control_link,
    validate_control_link_spec,
)
from crazyflie_sim.model.quadcopter import Quadcopter
from crazyflie_sim.utils.utils import quat_to_rpy, rpy_to_quat, quat_to_rot_body_to_world, RPYToRot
from crazyflie_sim.model.actuator_profile import CRAZYFLIE21_PHYSICAL as CF


class TestControlLink(unittest.TestCase):
    def test_disabled_returns_none(self):
        self.assertIsNone(build_control_link({"enabled": False}))

    def test_delay_holds_then_releases(self):
        link = ControlChannel(delay_steps=2, seed=0)
        c0 = np.array([1.0, 2.0, 3.0, 4.0])
        c1 = np.array([5.0, 6.0, 7.0, 8.0])
        c2 = np.array([9.0, 10.0, 11.0, 12.0])
        c3 = np.array([13.0, 14.0, 15.0, 16.0])
        # Pipeline filling: hold first command until delay elapses.
        self.assertTrue(np.allclose(link.apply(c0), c0))
        self.assertTrue(np.allclose(link.apply(c1), c0))
        self.assertTrue(np.allclose(link.apply(c2), c0))
        self.assertTrue(np.allclose(link.apply(c3), c1))

    def test_dropout_reuses_last_applied(self):
        link = ControlChannel(dropout_probability=1.0, seed=0)
        first = np.array([1.0, 1.0, 1.0, 1.0])
        second = np.array([9.0, 9.0, 9.0, 9.0])
        self.assertTrue(np.allclose(link.apply(first), first))
        self.assertTrue(np.allclose(link.apply(second), first))
        self.assertTrue(link.last_dropped)

    def test_validate_rejects_bad_probability(self):
        with self.assertRaises(ValueError):
            validate_control_link_spec({"enabled": True, "dropout_probability": 2.0})


class TestQuaternionAttitude(unittest.TestCase):
    def test_rpy_quat_roundtrip_near_hover(self):
        for rpy in ((0.0, 0.0, 0.0), (0.2, -0.1, 0.3), (-0.4, 0.2, -1.0)):
            q = rpy_to_quat(*rpy)
            recovered = quat_to_rpy(q)
            self.assertTrue(np.allclose(recovered, rpy, atol=1.0e-9))
            R_q = quat_to_rot_body_to_world(q)
            R_e = RPYToRot(*rpy).T
            self.assertTrue(np.allclose(R_q, R_e, atol=1.0e-9))

    def test_quaternion_mode_survives_large_roll(self):
        quad = Quadcopter(
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
            use_quaternion=True,
        )
        # Command a strong roll moment; Euler path would approach singularity.
        hover = CF.mass * params.g / 4.0
        pwm = np.full(4, 30000.0)
        # Bias rotors to produce roll: reduce M1/M2, increase M3/M4 via PWM map
        # is awkward; instead poke omega and integrate free dynamics briefly.
        quad.state[9] = 8.0  # large roll rate
        for _ in range(200):
            quad.update(0.002, pwm)
            self.assertTrue(np.all(np.isfinite(quad.state)))
            self.assertTrue(np.all(np.isfinite(quad.quat)))
            self.assertAlmostEqual(float(np.linalg.norm(quad.quat)), 1.0, places=5)

    def test_default_path_still_euler(self):
        quad = Quadcopter((0.0, 0.0, 1.0), (0.1, -0.05, 0.2))
        self.assertFalse(quad._use_quaternion)


if __name__ == "__main__":
    unittest.main()
