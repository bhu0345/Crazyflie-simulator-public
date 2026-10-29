import unittest

import numpy as np

import crazyflie_sim.model.actuators as actuators
import crazyflie_sim.model.params as params
from crazyflie_sim.model.rotor_fault import RotorFaultConfig
from crazyflie_sim.model.actuator_profile import CRAZYFLIE21_PHYSICAL as CF


class TestActuatorModel(unittest.TestCase):
    def test_hover_wrench_round_trips_through_pwm(self):
        motor_pwm = actuators.wrench_to_pwm(
            total_thrust=CF.mass * params.g,
            moment=np.zeros((3, 1), dtype=float),
            battery_voltage=params.battery_voltage_nominal,
        )
        motor_forces = actuators.pwm_to_motor_forces(
            motor_pwm,
            params.battery_voltage_nominal,
        )
        total_thrust, moment = actuators.motor_forces_to_wrench(motor_forces)

        self.assertEqual(motor_pwm.shape, (4,))
        self.assertTrue(np.all(motor_pwm >= 0.0))
        self.assertTrue(np.all(motor_pwm <= params.pwm_max))
        self.assertAlmostEqual(total_thrust, CF.mass * params.g, places=4)
        self.assertTrue(np.allclose(moment, np.zeros((3, 1)), atol=1e-8))

    def test_high_thrust_command_is_pwm_capped(self):
        motor_pwm = actuators.wrench_to_pwm(
            total_thrust=2.0 * CF.max_total_thrust,
            moment=np.zeros((3, 1), dtype=float),
            battery_voltage=params.battery_voltage_nominal,
        )

        self.assertTrue(np.all(motor_pwm <= params.pwm_max))
        self.assertGreater(float(np.max(motor_pwm)), 0.99 * params.pwm_max)

    def test_pwm_to_motor_forces_returns_zero_when_supply_is_too_low(self):
        motor_pwm = np.full(4, params.pwm_max, dtype=float)

        motor_forces = actuators.pwm_to_motor_forces(
            motor_pwm,
            battery_voltage=0.0,
        )

        self.assertTrue(np.allclose(motor_forces, np.zeros(4)))

    def test_rotor_fault_is_inactive_before_start_time(self):
        motor_forces = np.array([0.08, 0.08, 0.08, 0.08], dtype=float)
        fault = RotorFaultConfig(
            enabled=True,
            rotor_index=2,
            start_time_s=5.0,
            effectiveness=0.75,
            label="power_loss_25pct",
        )

        faulted = actuators.apply_rotor_fault(motor_forces, fault, sim_time=4.99)

        self.assertTrue(np.allclose(faulted, motor_forces))

    def test_rotor_fault_only_scales_selected_rotor(self):
        motor_forces = np.array([0.05, 0.06, 0.07, 0.08], dtype=float)
        fault = RotorFaultConfig(
            enabled=True,
            rotor_index=3,
            start_time_s=0.0,
            effectiveness=0.75,
            label="power_loss_25pct",
        )

        faulted = actuators.apply_rotor_fault(motor_forces, fault, sim_time=0.0)

        self.assertTrue(np.allclose(faulted[[0, 1, 3]], motor_forces[[0, 1, 3]]))
        self.assertAlmostEqual(float(faulted[2]), 0.75 * float(motor_forces[2]))

    def test_rotor_fault_scales_multiple_selected_rotors(self):
        motor_forces = np.array([0.05, 0.06, 0.07, 0.08], dtype=float)
        fault = RotorFaultConfig(
            enabled=True,
            rotor_index=[1, 2],
            start_time_s=0.0,
            effectiveness=0.75,
            label="power_loss_25pct",
        )

        faulted = actuators.apply_rotor_fault(motor_forces, fault, sim_time=0.0)

        self.assertTrue(np.allclose(faulted[:2], 0.75 * motor_forces[:2]))
        self.assertTrue(np.allclose(faulted[2:], motor_forces[2:]))

    def test_multiple_rotors_can_fail_at_distinct_times(self):
        motor_forces = np.array([0.05, 0.06, 0.07, 0.08], dtype=float)
        fault = RotorFaultConfig(
            enabled=True,
            rotor_index=[1, 2],
            start_time_s=[0.5, 1.0],
            effectiveness=0.0,
            label="sequential_failure",
        )

        first = actuators.apply_rotor_fault(motor_forces, fault, sim_time=0.75)
        both = actuators.apply_rotor_fault(motor_forces, fault, sim_time=1.0)

        self.assertEqual(fault.start_times_s, (0.5, 1.0))
        self.assertAlmostEqual(fault.start_time_s, 0.5)
        self.assertTrue(np.allclose(first, [0.0, 0.06, 0.07, 0.08]))
        self.assertTrue(np.allclose(both, [0.0, 0.0, 0.07, 0.08]))

    def test_rotor_fault_75_percent_effectiveness_is_supported(self):
        motor_forces = np.array([0.04, 0.04, 0.04, 0.04], dtype=float)
        fault = RotorFaultConfig(
            enabled=True,
            rotor_index=1,
            start_time_s=0.0,
            effectiveness=0.75,
            label="power_loss_25pct",
        )

        faulted = actuators.apply_rotor_fault(motor_forces, fault, sim_time=1.0)

        self.assertAlmostEqual(float(faulted[0]), 0.03, places=8)

    def test_rotor_fault_zero_effectiveness_models_full_failure(self):
        motor_forces = np.array([0.04, 0.04, 0.04, 0.04], dtype=float)
        fault = RotorFaultConfig(
            enabled=True,
            rotor_index=4,
            start_time_s=0.0,
            effectiveness=0.0,
            label="failure",
        )

        faulted = actuators.apply_rotor_fault(motor_forces, fault, sim_time=1.0)

        self.assertAlmostEqual(float(faulted[3]), 0.0, places=8)
        self.assertTrue(np.allclose(faulted[:3], motor_forces[:3]))


if __name__ == "__main__":
    unittest.main()
