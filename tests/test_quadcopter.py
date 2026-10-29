import unittest

import numpy as np

import crazyflie_sim.model.actuators as actuators
import crazyflie_sim.model.params as params
from crazyflie_sim.model.quadcopter import Quadcopter
from crazyflie_sim.model.rotor_fault import RotorFaultConfig
from crazyflie_sim.model.actuator_profile import CRAZYFLIE21_PHYSICAL as CF


class TestQuadcopterMethods(unittest.TestCase):
    def test_world_frame_has_expected_shape(self):
        quadcopter = Quadcopter((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        frame = quadcopter.world_frame()
        self.assertEqual((3, 6), frame.shape)

    def test_state_dot_returns_finite_vector(self):
        quadcopter = Quadcopter((0.0, 0.0, 0.0), (0.0, 0.0, np.pi / 6.0))
        force = CF.mass * params.g
        moment = np.array([[1.0e-5, -2.0e-5, 3.0e-5]]).T
        state_dot = quadcopter.state_dot(
            quadcopter.state,
            0.0,
            force,
            moment,
            np.zeros(3),
        )
        self.assertEqual(state_dot.shape, (12,))
        self.assertTrue(np.all(np.isfinite(state_dot)))

    def test_update_accepts_pwm_command(self):
        quadcopter = Quadcopter((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        hover_pwm = actuators.wrench_to_pwm(
            CF.mass * params.g,
            np.zeros((3, 1), dtype=float),
            quadcopter.battery_voltage,
        )
        quadcopter.update(0.01, hover_pwm, external_force_world=np.zeros(3))
        self.assertEqual(quadcopter.motor_pwm.shape, (4,))
        self.assertTrue(np.all(quadcopter.motor_thrusts >= 0.0))

    def test_attitude_matches_initial_yaw(self):
        quadcopter = Quadcopter((0.0, 0.0, 0.0), (0.0, 0.0, np.pi / 2.0))
        _, _, yaw = quadcopter.attitude()
        self.assertAlmostEqual(yaw, np.pi / 2.0, places=6)

    def test_rotor_fault_changes_realized_wrench_after_fault_onset(self):
        hover_pwm = actuators.wrench_to_pwm(
            CF.mass * params.g,
            np.zeros((3, 1), dtype=float),
            params.battery_voltage_nominal,
        )
        nominal_quad = Quadcopter((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        faulted_quad = Quadcopter(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            rotor_fault=RotorFaultConfig(
                enabled=True,
                rotor_index=1,
                start_time_s=0.5,
                effectiveness=0.0,
                label="failure",
            ),
        )

        nominal_quad.update(0.01, hover_pwm, external_force_world=np.zeros(3), sim_time=0.6)
        faulted_quad.update(0.01, hover_pwm, external_force_world=np.zeros(3), sim_time=0.6)

        F_nominal, M_nominal = actuators.motor_forces_to_wrench(nominal_quad.motor_thrusts)
        F_faulted, M_faulted = actuators.motor_forces_to_wrench(faulted_quad.motor_thrusts)

        self.assertLess(F_faulted, F_nominal)
        self.assertGreater(np.linalg.norm(M_faulted), np.linalg.norm(M_nominal) + 1.0e-9)
        self.assertTrue(faulted_quad.rotor_fault_active)


if __name__ == "__main__":
    unittest.main()
