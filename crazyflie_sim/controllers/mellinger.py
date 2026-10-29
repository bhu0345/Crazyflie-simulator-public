import numpy as np

import crazyflie_sim.model.actuators as actuators
import crazyflie_sim.model.params as params
from crazyflie_sim.model.actuator_profile import get_actuator_profile
from .base import ControllerBase
from .registry import register
from .common import clip_sym, translational_pid_target_thrust

# Bitcraze Crazyflie firmware defaults from controller_mellinger.c.
#
# The translational XY gains below are intentionally retuned for this simulator's
# brushless actuator path, drag model, and disturbance levels. The stock firmware
# values were too soft for the current plant and produced large sustained drift
# under the configured hover-wind benchmark.
CONTROL_DT = 1.0 / 500.0
MIX_CMD_MAX = 32000.0

KP_XY = 3.0
KD_XY = 2.2
KI_XY = 0.3
I_RANGE_XY = 2.0

KP_Z = 1.25
KD_Z = 0.4
KI_Z = 0.05
I_RANGE_Z = 0.4

KR_XY = 70000.0
KW_XY = 20000.0
KI_M_XY = 0.0
I_RANGE_M_XY = 1.0

KR_Z = 60000.0
KW_Z = 12000.0
KI_M_Z = 500.0
I_RANGE_M_Z = 1500.0

KD_OMEGA_RP = 200.0


def _normalize(vec, fallback):
    vec = np.asarray(vec, dtype=float).reshape(3,)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.asarray(fallback, dtype=float).reshape(3,)
    return vec / norm


@register("mellinger", label="Mellinger", color="#9467bd")
class MellingerController(ControllerBase):
    """Simulator-side adaptation of Bitcraze's Mellinger controller.

    The Bitcraze firmware computes thrust in physical force units and moment-like
    mixer commands in the range +/-32000. This backend keeps the same control
    law and gain structure, then maps those mixer commands to physical body
    moments expected by the simulator plant.
    """

    def __init__(
        self,
        dt=CONTROL_DT,
        actuator_profile=None,
        mass=None,
        kp_xy=KP_XY,
        kd_xy=KD_XY,
        ki_xy=KI_XY,
        i_range_xy=I_RANGE_XY,
        kp_z=KP_Z,
        kd_z=KD_Z,
        ki_z=KI_Z,
        i_range_z=I_RANGE_Z,
        kR_xy=KR_XY,
        kw_xy=KW_XY,
        ki_m_xy=KI_M_XY,
        i_range_m_xy=I_RANGE_M_XY,
        kR_z=KR_Z,
        kw_z=KW_Z,
        ki_m_z=KI_M_Z,
        i_range_m_z=I_RANGE_M_Z,
        kd_omega_rp=KD_OMEGA_RP,
    ):
        self.dt = float(dt)
        self.actuator_profile = get_actuator_profile(actuator_profile)
        self.mass = float(self.actuator_profile.mass if mass is None else mass)
        self.max_thrust = self.actuator_profile.max_total_thrust

        self.kp_xy = float(kp_xy)
        self.kd_xy = float(kd_xy)
        self.ki_xy = float(ki_xy)
        self.i_range_xy = float(i_range_xy)

        self.kp_z = float(kp_z)
        self.kd_z = float(kd_z)
        self.ki_z = float(ki_z)
        self.i_range_z = float(i_range_z)

        self.kR_xy = float(kR_xy)
        self.kw_xy = float(kw_xy)
        self.ki_m_xy = float(ki_m_xy)
        self.i_range_m_xy = float(i_range_m_xy)

        self.kR_z = float(kR_z)
        self.kw_z = float(kw_z)
        self.ki_m_z = float(ki_m_z)
        self.i_range_m_z = float(i_range_m_z)
        self.kd_omega_rp = float(kd_omega_rp)

        self._max_m = actuators.max_body_moments(self.actuator_profile)
        self.reset()

    def set_dt(self, dt):
        self.dt = float(dt)

    def _reset_integrators(self):
        self.i_error_pos = np.zeros(3, dtype=float)
        self.i_error_att = np.zeros(3, dtype=float)
        self.prev_omega = np.full(2, np.nan, dtype=float)
        self.prev_omega_sp = np.zeros(2, dtype=float)

    def reset(self):
        self._reset_integrators()
        self._last_target_thrust = np.zeros(3, dtype=float)
        self._last_z_axis_des = np.array([0.0, 0.0, 1.0], dtype=float)
        self._last_eR = np.zeros(3, dtype=float)
        self._last_ew = np.zeros(3, dtype=float)
        self._last_m_cmd = np.zeros(3, dtype=float)
        self._last_force_cmd = 0.0
        self._last_moment = np.zeros(3, dtype=float)
        self._last_motor_pwm = np.zeros(4, dtype=float)

    def _rotation_body_to_world(self, quad):
        return quad.rotation_body_to_world()

    def run_wrench(self, quad, des_state, sim_time=0.0):
        del sim_time

        dt = max(float(self.dt), 1e-6)
        omega = np.asarray(quad.omega(), dtype=float).reshape(3,)
        R = self._rotation_body_to_world(quad)

        target_thrust = translational_pid_target_thrust(
            quad,
            des_state,
            self.i_error_pos,
            self.dt,
            self.mass,
            self.kp_xy,
            self.kd_xy,
            self.ki_xy,
            self.i_range_xy,
            self.kp_z,
            self.kd_z,
            self.ki_z,
            self.i_range_z,
        )

        z_axis = R[:, 2]
        current_thrust = float(np.dot(target_thrust, z_axis))

        z_axis_des = _normalize(target_thrust, z_axis)
        x_c_des = np.array(
            [np.cos(float(des_state.yaw)), np.sin(float(des_state.yaw)), 0.0],
            dtype=float,
        )
        y_axis_des = _normalize(np.cross(z_axis_des, x_c_des), R[:, 1])
        x_axis_des = _normalize(np.cross(y_axis_des, z_axis_des), R[:, 0])
        R_des = np.column_stack((x_axis_des, y_axis_des, z_axis_des))

        eR_mat = R_des.T.dot(R) - R.T.dot(R_des)
        eR = np.array([eR_mat[2, 1], eR_mat[0, 2], eR_mat[1, 0]], dtype=float)

        omega_sp = np.array([0.0, 0.0, float(des_state.yawdot)], dtype=float)
        ew = omega_sp - omega

        err_d_roll = 0.0
        err_d_pitch = 0.0
        if np.isfinite(self.prev_omega[0]):
            err_d_roll = (
                (omega_sp[0] - self.prev_omega_sp[0]) - (omega[0] - self.prev_omega[0])
            ) / dt
            err_d_pitch = (
                (omega_sp[1] - self.prev_omega_sp[1]) - (omega[1] - self.prev_omega[1])
            ) / dt
        self.prev_omega[:] = omega[:2]
        self.prev_omega_sp[:] = omega_sp[:2]

        self.i_error_att[0] = clip_sym(
            self.i_error_att[0] + (-eR[0]) * dt, self.i_range_m_xy
        )
        self.i_error_att[1] = clip_sym(
            self.i_error_att[1] + (-eR[1]) * dt, self.i_range_m_xy
        )
        self.i_error_att[2] = clip_sym(
            self.i_error_att[2] + (-eR[2]) * dt, self.i_range_m_z
        )

        moment_cmd = np.array(
            [
                -self.kR_xy * eR[0]
                + self.kw_xy * ew[0]
                + self.ki_m_xy * self.i_error_att[0]
                + self.kd_omega_rp * err_d_roll,
                -self.kR_xy * eR[1]
                + self.kw_xy * ew[1]
                + self.ki_m_xy * self.i_error_att[1]
                + self.kd_omega_rp * err_d_pitch,
                -self.kR_z * eR[2]
                + self.kw_z * ew[2]
                + self.ki_m_z * self.i_error_att[2],
            ],
            dtype=float,
        )

        if current_thrust <= 0.0:
            self._last_target_thrust = target_thrust
            self._last_z_axis_des = z_axis_des
            self._last_eR = eR
            self._last_ew = ew
            self._last_m_cmd.fill(0.0)
            self._last_force_cmd = 0.0
            self._last_moment.fill(0.0)
            self._last_motor_pwm.fill(0.0)
            self._reset_integrators()
            return 0.0, np.zeros((3, 1), dtype=float)

        moment_cmd = np.clip(moment_cmd, -MIX_CMD_MAX, MIX_CMD_MAX)
        force_cmd = float(np.clip(current_thrust, 0.0, self.max_thrust))
        moment = self._max_m * (moment_cmd / MIX_CMD_MAX)

        self._last_target_thrust = target_thrust
        self._last_z_axis_des = z_axis_des
        self._last_eR = eR
        self._last_ew = ew
        self._last_m_cmd = moment_cmd
        self._last_force_cmd = force_cmd
        self._last_moment = moment.copy()

        return force_cmd, moment.reshape(3, 1)

    def run(self, quad, des_state, sim_time=0.0):
        force_cmd, moment = self.run_wrench(quad, des_state, sim_time=sim_time)
        motor_pwm = actuators.wrench_to_pwm(
            total_thrust=force_cmd,
            moment=moment,
            battery_voltage=quad.battery_voltage,
            actuator_profile=self.actuator_profile,
        )
        self._last_motor_pwm = motor_pwm.copy()
        return motor_pwm

    def diagnostics(self):
        return {
            "target_thrust": self._last_target_thrust.copy(),
            "z_axis_desired": self._last_z_axis_des.copy(),
            "eR": self._last_eR.copy(),
            "ew": self._last_ew.copy(),
            "moment_cmd": self._last_m_cmd.copy(),
            "force_cmd": float(self._last_force_cmd),
            "moment_si": self._last_moment.copy(),
            "motor_pwm": self._last_motor_pwm.copy(),
            "i_error_pos": self.i_error_pos.copy(),
            "i_error_att": self.i_error_att.copy(),
            "status_label": "|eR|",
            "status_value": float(np.linalg.norm(self._last_eR)),
        }
