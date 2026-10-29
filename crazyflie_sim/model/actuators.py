import numpy as np

import crazyflie_sim.model.params as params
from crazyflie_sim.model.actuator_profile import get_actuator_profile
from crazyflie_sim.model.rotor_fault import RotorFaultConfig


def _as_motor_vector(values):
    return np.asarray(values, dtype=float).reshape(4,)


def _as_moment_vector(moment):
    return np.asarray(moment, dtype=float).reshape(3,)


def _profile(profile):
    """A profile instance, registry name, or None (the default vehicle)."""
    return get_actuator_profile(profile)


def rotor_speed_to_thrust(rotor_speed_rad_s, thrust_coefficient=None):
    """Evaluate the calibrated monotone map ``f = c_T * Omega**2``.

    ``thrust_coefficient`` None is the default vehicle's ``c_T``.
    """
    omega = np.maximum(np.asarray(rotor_speed_rad_s, dtype=float), 0.0)
    coefficient = (
        get_actuator_profile().nominal_thrust_coefficient
        if thrust_coefficient is None
        else np.asarray(thrust_coefficient, dtype=float)
    )
    return np.maximum(coefficient, 0.0) * omega * omega


def thrust_to_rotor_speed(thrust, thrust_coefficient=None):
    """Invert the nominal quadratic thrust map with safe zero handling."""
    force = np.maximum(np.asarray(thrust, dtype=float), 0.0)
    coefficient = (
        get_actuator_profile().nominal_thrust_coefficient
        if thrust_coefficient is None
        else np.asarray(thrust_coefficient, dtype=float)
    )
    return np.sqrt(force / np.maximum(coefficient, 1.0e-16))


def max_body_moments(actuator_profile=None):
    """Max realizable body moments [Mx, My, Mz] for the quadrotor mixer."""
    profile = _profile(actuator_profile)
    f_max = profile.max_rotor_thrust
    return np.array(
        [
            2.0 * profile.mixer_arm * f_max,
            2.0 * profile.mixer_arm * f_max,
            2.0 * profile.thrust_to_torque * f_max,
        ],
        dtype=float,
    )


def mix_wrench_to_motor_forces(total_thrust, moment, actuator_profile=None):
    """Map a body wrench to uncapped per-rotor forces."""
    mx, my, mz = _as_moment_vector(moment)
    profile = _profile(actuator_profile)
    target = np.array([float(total_thrust), mx, my, mz], dtype=float)
    try:
        return np.linalg.solve(profile.wrench_matrix, target)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(profile.wrench_matrix, target, rcond=None)[0]


def motor_forces_to_wrench(
    motor_forces, yaw_equivalent_forces=None, actuator_profile=None
):
    """Map realized rotor forces to body wrench.

    ``yaw_equivalent_forces`` is an optional plant-only reaction-torque
    channel.  It permits propeller damage to change the yaw torque/thrust
    ratio without corrupting the reduced force/roll/pitch wrench.  Omitting it
    gives the fixed-ratio mixer.
    """
    motor_forces = _as_motor_vector(motor_forces)
    yaw_forces = (
        motor_forces
        if yaw_equivalent_forces is None
        else _as_motor_vector(yaw_equivalent_forces)
    )
    profile = _profile(actuator_profile)
    wrench = np.empty(4, dtype=float)
    wrench[:3] = profile.wrench_matrix[:3].dot(motor_forces)
    wrench[3] = profile.wrench_matrix[3].dot(yaw_forces)
    return float(wrench[0]), wrench[1:].reshape(3, 1)


def apply_rotor_fault(motor_forces, rotor_fault=None, sim_time=None):
    """Scale one realized rotor thrust command when a fault is active."""
    motor_forces = _as_motor_vector(motor_forces)
    if rotor_fault is None:
        return motor_forces.copy()
    if not isinstance(rotor_fault, RotorFaultConfig):
        raise TypeError("rotor_fault must be a RotorFaultConfig or None.")
    return motor_forces * np.asarray(
        rotor_fault.effectiveness_vector(sim_time), dtype=float
    )


def motor_forces_to_thrust_command(motor_forces, actuator_profile=None):
    """Convert per-rotor force setpoints to firmware-style thrust commands."""
    motor_forces = _as_motor_vector(motor_forces)
    upper = _profile(actuator_profile).command_upper
    return np.maximum(motor_forces, 0.0) * params.pwm_max / upper


def thrust_command_to_pwm(
    thrust_command, battery_voltage, actuator_profile=None
):
    """Apply the Bitcraze battery-compensation inverse thrust curve."""
    thrust_command = _as_motor_vector(thrust_command)
    profile = _profile(actuator_profile)
    supply_voltage = float(np.clip(battery_voltage, 0.0, params.battery_voltage_max))
    if supply_voltage < 2.0:
        return np.zeros(4, dtype=float)

    upper = profile.command_upper
    thrust = (thrust_command / params.pwm_max) * upper
    pwm = np.zeros(4, dtype=float)
    minimum_force = params.thrust_cmd_min_per_motor * (
        upper / params.thrust_cmd_max_per_motor
    )
    valid = thrust >= minimum_force
    if not np.any(valid):
        return pwm

    c0, c1, c2, c3 = profile.pwm_map
    p = -c2 / (3.0 * c3)
    r = c1 / (3.0 * c3)

    thrust_valid = thrust[valid]
    q = p ** 3 + (c2 * c1 - 3.0 * c3 * (c0 - thrust_valid)) / (6.0 * c3 ** 2)
    qrp = np.sqrt(np.maximum(q ** 2 + (r - p ** 2) ** 3, 0.0))
    motor_voltage = np.cbrt(q + qrp) + np.cbrt(q - qrp) + p
    pwm[valid] = params.pwm_max * (motor_voltage / supply_voltage)
    return pwm


def cap_motor_pwm(pwm_command, min_pwm=0.0):
    """Match Crazyflie capping: reduce all motors equally if one saturates."""
    pwm_command = _as_motor_vector(pwm_command)
    reduction = max(0.0, float(np.max(pwm_command)) - params.pwm_max)
    capped = pwm_command - reduction
    capped = np.maximum(capped, float(min_pwm))
    return np.clip(capped, float(min_pwm), params.pwm_max)


def wrench_to_pwm(
    total_thrust,
    moment,
    battery_voltage,
    min_pwm=0.0,
    actuator_profile=None,
):
    motor_forces = mix_wrench_to_motor_forces(
        total_thrust, moment, actuator_profile=actuator_profile
    )
    thrust_command = motor_forces_to_thrust_command(
        motor_forces, actuator_profile=actuator_profile
    )
    pwm_uncapped = thrust_command_to_pwm(
        thrust_command, battery_voltage, actuator_profile=actuator_profile
    )
    return cap_motor_pwm(pwm_uncapped, min_pwm=min_pwm)


def pwm_to_motor_forces(motor_pwm, battery_voltage, actuator_profile=None):
    """Static per-rotor thrust generated by the commanded PWM."""
    profile = _profile(actuator_profile)
    motor_pwm = np.clip(_as_motor_vector(motor_pwm), params.pwm_min, params.pwm_max)
    supply_voltage = float(
        np.clip(battery_voltage, 0.0, params.battery_voltage_max)
    )
    if supply_voltage < 2.0:
        return np.zeros(4, dtype=float)

    motor_voltage = supply_voltage * (motor_pwm / params.pwm_max)
    c0, c1, c2, c3 = profile.pwm_map
    motor_forces = c0 + c1 * motor_voltage + c2 * motor_voltage ** 2 + c3 * motor_voltage ** 3
    return np.maximum(motor_forces, 0.0)
