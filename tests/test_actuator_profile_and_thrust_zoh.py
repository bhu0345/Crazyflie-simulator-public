import dataclasses
import unittest

import numpy as np

import crazyflie_sim.model.actuators as actuators
import crazyflie_sim.model.params as params
from crazyflie_sim.model.actuator_profile import (
    ACTUATOR_PROFILE_REGISTRY,
    CRAZYFLIE21_PHYSICAL,
    CRAZYFLIE21_GRAEFE,
    CRAZYFLIE21_THRUST_RICH,
    ActuatorProfile,
    _x_wrench_matrix,
    available_actuator_profiles,
    get_actuator_profile,
)
from crazyflie_sim.model.quadcopter import (
    LEGACY_THRUST_EULER,
    ROTOR_SPEED_EXACT_ZOH,
    THRUST_EXACT_ZOH,
    Quadcopter,
)
from crazyflie_sim.model.rotor_fault import RotorFaultConfig
from crazyflie_sim.model.actuator_profile import CRAZYFLIE21_PHYSICAL as CF

# A medium-size, non-Crazyflie airframe: checks that a profile's mass,
# inertia, lag and PWM scaling reach the plant instead of model.params.
AUTHORITY_RICH_QUAD = ActuatorProfile(
    name="authority_rich_quad",
    wrench_matrix=_x_wrench_matrix(0.15, 0.008),
    command_lower=np.zeros(4),
    command_upper=np.full(4, 0.50),
    tau_up=0.018,
    tau_down=0.030,
    tau_uncertainty=0.006,
    mass=0.100,
    inertia=np.diag([3.240e-3, 3.240e-3, 5.832e-3]),
    angular_damping=np.array([5.0e-4, 5.0e-4, 2.0e-4]),
    yaw_drag_coefficient=3.0e-5,
    pwm_map=2.50 * np.asarray(CF.pwm_map, dtype=float),
    rotor_speed_limit=3000.0,
)


def _pwm_for_forces(forces, profile=CRAZYFLIE21_PHYSICAL):
    thrust_command = actuators.motor_forces_to_thrust_command(
        np.asarray(forces, dtype=float), actuator_profile=profile
    )
    return actuators.cap_motor_pwm(
        actuators.thrust_command_to_pwm(
            thrust_command,
            params.battery_voltage_nominal,
            actuator_profile=profile,
        )
    )


class TestActuatorProfile(unittest.TestCase):
    def test_registry_is_complete_and_default_is_crazyflie(self):
        self.assertEqual(
            set(available_actuator_profiles()),
            {
                "crazyflie21_physical",
                "crazyflie21_graefe",
                "crazyflie21_standard",
                "crazyflie21_thrust_rich",
            },
        )
        self.assertIs(get_actuator_profile(), CRAZYFLIE21_PHYSICAL)
        self.assertIs(get_actuator_profile(AUTHORITY_RICH_QUAD), AUTHORITY_RICH_QUAD)
        self.assertEqual(len(ACTUATOR_PROFILE_REGISTRY), 4)

    def test_profile_and_array_payloads_are_immutable(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            CRAZYFLIE21_PHYSICAL.mass = 1.0
        with self.assertRaises(ValueError):
            CRAZYFLIE21_PHYSICAL.command_upper[0] = 1.0
        with self.assertRaises(ValueError):
            CRAZYFLIE21_PHYSICAL.command_upper.flags.writeable = True

    def test_validation_rejects_negative_commands_and_bad_inertia(self):
        config = CRAZYFLIE21_PHYSICAL.to_config()
        config["name"] = "invalid_negative"
        config["command_lower"][0] = -1.0e-3
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            ActuatorProfile(**config)

        config = CRAZYFLIE21_PHYSICAL.to_config()
        config["name"] = "invalid_inertia"
        config["inertia"][0][0] = -1.0
        with self.assertRaisesRegex(ValueError, "positive definite"):
            ActuatorProfile(**config)

    def test_crazyflie_thrust_rich_keeps_geometry_and_raises_force_cap(self):
        self.assertEqual(CRAZYFLIE21_THRUST_RICH.mass, CRAZYFLIE21_PHYSICAL.mass)
        np.testing.assert_allclose(
            CRAZYFLIE21_THRUST_RICH.inertia,
            CRAZYFLIE21_PHYSICAL.inertia,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            CRAZYFLIE21_THRUST_RICH.wrench_matrix,
            CRAZYFLIE21_PHYSICAL.wrench_matrix,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            CRAZYFLIE21_THRUST_RICH.command_upper,
            np.full(4, 0.50),
            rtol=0.0,
            atol=0.0,
        )
        self.assertGreater(
            float(np.max(CRAZYFLIE21_THRUST_RICH.command_upper)),
            float(np.max(CRAZYFLIE21_PHYSICAL.command_upper)),
        )
        self.assertGreater(
            CRAZYFLIE21_THRUST_RICH.rotor_speed_limit,
            CRAZYFLIE21_PHYSICAL.rotor_speed_limit,
        )
        pwm_force = actuators.pwm_to_motor_forces(
            np.full(4, params.pwm_max),
            params.battery_voltage_nominal,
            actuator_profile=CRAZYFLIE21_THRUST_RICH,
        )
        self.assertTrue(
            np.all(pwm_force + 1.0e-9 >= CRAZYFLIE21_THRUST_RICH.command_upper)
        )

    def test_profile_pwm_map_round_trip(self):
        # Stay above this profile's PWM dead-zone-derived armed floor.
        desired = np.array([0.08, 0.14, 0.22, 0.36], dtype=float)
        pwm = _pwm_for_forces(desired, AUTHORITY_RICH_QUAD)
        realized = actuators.pwm_to_motor_forces(
            pwm,
            params.battery_voltage_nominal,
            actuator_profile=AUTHORITY_RICH_QUAD,
        )
        np.testing.assert_allclose(realized, desired, rtol=1.0e-12, atol=1.0e-12)

class TestThrustExactZohPlant(unittest.TestCase):
    def _quad(self, **kwargs):
        return Quadcopter(
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
            actuator_profile=CRAZYFLIE21_PHYSICAL,
            actuator_plant_backend=THRUST_EXACT_ZOH,
            **kwargs,
        )

    def test_legacy_boolean_mapping_is_preserved(self):
        legacy = Quadcopter((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        rpm = Quadcopter(
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
            use_rotor_speed_model=True,
        )
        explicit = Quadcopter(
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
            use_rotor_speed_model=True,
            actuator_plant_backend=THRUST_EXACT_ZOH,
        )

        self.assertEqual(legacy.actuator_plant_backend, LEGACY_THRUST_EULER)
        self.assertEqual(rpm.actuator_plant_backend, ROTOR_SPEED_EXACT_ZOH)
        self.assertEqual(explicit.actuator_plant_backend, THRUST_EXACT_ZOH)
        self.assertFalse(explicit._use_rotor_speed_model)

    def test_invalid_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "actuator_plant_backend"):
            Quadcopter(
                (0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0),
                actuator_plant_backend="not_a_backend",
            )

    def test_asymmetric_thrust_update_matches_exact_zoh(self):
        quad = self._quad()
        before = quad.motor_thrusts.copy()
        desired = np.array([0.13, 0.04, 0.12, 0.05], dtype=float)
        pwm = _pwm_for_forces(desired)
        mapped = np.clip(
            actuators.pwm_to_motor_forces(
                pwm,
                quad.battery_voltage,
                actuator_profile=CRAZYFLIE21_PHYSICAL,
            ),
            CRAZYFLIE21_PHYSICAL.command_lower,
            CRAZYFLIE21_PHYSICAL.command_upper,
        )
        dt = 0.007
        tau = np.where(
            mapped >= before,
            CRAZYFLIE21_PHYSICAL.tau_up,
            CRAZYFLIE21_PHYSICAL.tau_down,
        )
        expected = np.exp(-dt / tau) * before + (
            1.0 - np.exp(-dt / tau)
        ) * mapped

        quad.update(dt, pwm, sim_time=0.0)

        np.testing.assert_allclose(
            quad.motor_thrusts, expected, rtol=1.0e-13, atol=1.0e-13
        )

    def test_fault_effectiveness_is_inside_exact_zoh_target(self):
        fault = RotorFaultConfig(
            enabled=True,
            rotor_index=1,
            start_time_s=0.0,
            effectiveness=0.25,
            mechanism="motor_loe",
        )
        quad = self._quad(rotor_fault=fault)
        before = quad.motor_thrusts.copy()
        pwm = _pwm_for_forces(np.full(4, 0.12, dtype=float))
        healthy_target = actuators.pwm_to_motor_forces(
            pwm,
            quad.battery_voltage,
            actuator_profile=CRAZYFLIE21_PHYSICAL,
        )
        target = healthy_target.copy()
        target[0] *= 0.25
        dt = 0.01
        tau = np.where(
            target >= before,
            CRAZYFLIE21_PHYSICAL.tau_up,
            CRAZYFLIE21_PHYSICAL.tau_down,
        )
        decay = np.exp(-dt / tau)

        quad.update(dt, pwm, sim_time=0.0)

        np.testing.assert_allclose(
            quad.motor_thrusts,
            decay * before + (1.0 - decay) * target,
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(quad.commanded_motor_thrusts, target)

    def test_prop_fault_keeps_separate_yaw_reaction_semantics(self):
        effectiveness = 0.25
        fault = RotorFaultConfig(
            enabled=True,
            rotor_index=2,
            start_time_s=0.0,
            effectiveness=effectiveness,
            mechanism="prop_loe",
        )
        quad = self._quad(rotor_fault=fault)
        force_before = quad.motor_thrusts.copy()
        yaw_before = quad.rotor_yaw_equivalent_forces.copy()
        pwm = _pwm_for_forces(np.full(4, 0.12, dtype=float))
        healthy_target = actuators.pwm_to_motor_forces(
            pwm,
            quad.battery_voltage,
            actuator_profile=CRAZYFLIE21_PHYSICAL,
        )
        force_target = healthy_target.copy()
        force_target[1] *= effectiveness
        yaw_target = healthy_target.copy()
        yaw_target[1] *= effectiveness ** params.prop_yaw_torque_loss_exponent
        dt = 0.01
        force_tau = np.where(
            force_target >= force_before,
            CRAZYFLIE21_PHYSICAL.tau_up,
            CRAZYFLIE21_PHYSICAL.tau_down,
        )
        yaw_tau = np.where(
            yaw_target >= yaw_before,
            CRAZYFLIE21_PHYSICAL.tau_up,
            CRAZYFLIE21_PHYSICAL.tau_down,
        )

        quad.update(dt, pwm, sim_time=0.0)

        force_decay = np.exp(-dt / force_tau)
        yaw_decay = np.exp(-dt / yaw_tau)
        np.testing.assert_allclose(
            quad.motor_thrusts,
            force_decay * force_before + (1.0 - force_decay) * force_target,
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            quad.rotor_yaw_equivalent_forces,
            yaw_decay * yaw_before + (1.0 - yaw_decay) * yaw_target,
            rtol=1.0e-13,
            atol=1.0e-13,
        )

    def test_thrust_and_rpm_exact_zoh_are_distinct_transient_models(self):
        thrust_quad = self._quad()
        rpm_quad = Quadcopter(
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
            actuator_profile=CRAZYFLIE21_PHYSICAL,
            actuator_plant_backend=ROTOR_SPEED_EXACT_ZOH,
        )
        pwm = _pwm_for_forces(np.full(4, 0.15, dtype=float))

        thrust_quad.update(0.01, pwm, sim_time=0.0)
        rpm_quad.update(0.01, pwm, sim_time=0.0)

        self.assertGreater(
            float(np.linalg.norm(thrust_quad.motor_thrusts - rpm_quad.motor_thrusts)),
            1.0e-5,
        )

    def test_profile_defaults_are_injected_into_plant(self):
        quad = Quadcopter(
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
            actuator_profile=AUTHORITY_RICH_QUAD,
            actuator_plant_backend=THRUST_EXACT_ZOH,
        )
        self.assertIs(quad.actuator_profile, AUTHORITY_RICH_QUAD)
        self.assertEqual(quad.mass, AUTHORITY_RICH_QUAD.mass)
        self.assertEqual(quad.motor_tau_up, AUTHORITY_RICH_QUAD.tau_up)
        np.testing.assert_allclose(quad.inertia, AUTHORITY_RICH_QUAD.inertia)


if __name__ == "__main__":
    unittest.main()
