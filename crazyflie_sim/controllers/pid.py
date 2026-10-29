import numpy as np
import crazyflie_sim.model.actuators as actuators
import crazyflie_sim.model.params as params
from crazyflie_sim.model.actuator_profile import get_actuator_profile
from math import sin, cos
from types import SimpleNamespace

from .base import ControllerBase
from .registry import register
from .common import clip_sym, wrap_pi, wrap_deg

# Crazyflie-like cascaded PID controller.

CONTROL_DT = 1.0 / 500.0
OUTER_LOOP_DIV = 5  # 500 Hz inner loops, 100 Hz outer loops

# Yaw reference mode
USE_TRAJECTORY_YAW = True
HOLD_YAW = 0.0

# Outer loops
POS_XY_KP, POS_XY_KI, POS_XY_KD = 1.6, 8.0, 0.0
VEL_XY_KP, VEL_XY_KI, VEL_XY_KD = 22.0, 0.5, 0.0
POS_Z_KP, POS_Z_KI, POS_Z_KD = 2.0, 0.5, 0.0
VEL_Z_KP, VEL_Z_KI, VEL_Z_KD = 25.0, 15.0, 0.0

# Inner loops
ATT_RP_KP, ATT_RP_KI, ATT_RP_KD = 6.0, 3.0, 0.0
ATT_YAW_KP, ATT_YAW_KI, ATT_YAW_KD = 6.0, 1.0, 0.35
RATE_RP_KP, RATE_RP_KI, RATE_RP_KD = 200.0, 400.0, 2.5
RATE_YAW_KP, RATE_YAW_KI, RATE_YAW_KD = 120.0, 16.7, 0.0

# Integrator limits from firmware (deg or deg/s channels)
ATT_RP_INT_LIM = 20.0
ATT_YAW_INT_LIM = 360.0
RATE_RP_INT_LIM = 33.3
RATE_YAW_INT_LIM = 166.7

# Simulator-side practical limits (not from firmware)
MAX_TILT = np.deg2rad(50.0)
RATE_RP_SP_MAX = 360.0
RATE_YAW_SP_MAX = 200.0
POS_Z_INT_LIM = 1.5
POS_XY_INT_LIM = 2.0
VEL_XY_INT_LIM = 2.0
VEL_Z_INT_LIM = 2.0
MOTOR_MIX_CMD_MAX = 32767.0

# Thin aliases to the shared helpers (kept so the run_wrench body below stays
# verbatim relative to the pre-refactor module-level implementation).
_clip_sym = clip_sym
_wrap_pi = wrap_pi
_wrap_deg = wrap_deg


GAIN_NAMES = (
    "USE_TRAJECTORY_YAW",
    "HOLD_YAW",
    "OUTER_LOOP_DIV",
    "POS_XY_KP",
    "POS_XY_KI",
    "POS_XY_KD",
    "VEL_XY_KP",
    "VEL_XY_KI",
    "VEL_XY_KD",
    "POS_Z_KP",
    "POS_Z_KI",
    "POS_Z_KD",
    "VEL_Z_KP",
    "VEL_Z_KI",
    "VEL_Z_KD",
    "ATT_RP_KP",
    "ATT_RP_KI",
    "ATT_RP_KD",
    "ATT_YAW_KP",
    "ATT_YAW_KI",
    "ATT_YAW_KD",
    "RATE_RP_KP",
    "RATE_RP_KI",
    "RATE_RP_KD",
    "RATE_YAW_KP",
    "RATE_YAW_KI",
    "RATE_YAW_KD",
    "ATT_RP_INT_LIM",
    "ATT_YAW_INT_LIM",
    "RATE_RP_INT_LIM",
    "RATE_YAW_INT_LIM",
    "MAX_TILT",
    "RATE_RP_SP_MAX",
    "RATE_YAW_SP_MAX",
    "POS_Z_INT_LIM",
    "POS_XY_INT_LIM",
    "VEL_XY_INT_LIM",
    "VEL_Z_INT_LIM",
    "MOTOR_MIX_CMD_MAX"
)


@register("baseline", label="Cascaded PID", color="#1f77b4")
class CascadedPIDController(ControllerBase):
    """Crazyflie-like cascaded PID controller.

    The module constants above are the default gains.  Keyword arguments with
    the same UPPERCASE names override them for this instance only, e.g.
    ``CascadedPIDController(dt, POS_XY_KP=2.0)``.  ``actuator_profile`` is the
    vehicle (mass, thrust and moment limits, mixer, PWM map); None is the
    default crazyflie21_physical.
    """

    def __init__(self, dt=CONTROL_DT, actuator_profile=None, **gains):
        unknown = sorted(set(gains) - set(GAIN_NAMES))
        if unknown:
            raise ValueError(
                "unknown PID gain(s) {}; valid: {}".format(unknown, list(GAIN_NAMES))
            )
        values = {name: globals()[name] for name in GAIN_NAMES}
        values.update(gains)
        self.gains = SimpleNamespace(**values)
        self.actuator_profile = get_actuator_profile(actuator_profile)
        self.mass = self.actuator_profile.mass
        self.max_thrust = self.actuator_profile.max_total_thrust
        self.dt = float(dt)
        self._state = {}
        self.reset()

    def set_dt(self, dt):
        self.dt = float(dt)

    def reset(self):
        self._state["tick"] = 0

        # outer-loop integrators
        self._state["i_pos_x"] = 0.0
        self._state["i_pos_y"] = 0.0
        self._state["i_pos_z"] = 0.0
        self._state["i_vel_x"] = 0.0
        self._state["i_vel_y"] = 0.0
        self._state["i_vel_z"] = 0.0

        # inner-loop integrators
        self._state["i_att_roll"] = 0.0
        self._state["i_att_pitch"] = 0.0
        self._state["i_att_yaw"] = 0.0
        self._state["i_rate_roll"] = 0.0
        self._state["i_rate_pitch"] = 0.0
        self._state["i_rate_yaw"] = 0.0

        # derivatives / memory
        self._state["prev_vel_x"] = 0.0
        self._state["prev_vel_y"] = 0.0
        self._state["prev_vel_z"] = 0.0
        self._state["prev_p_dps"] = 0.0
        self._state["prev_q_dps"] = 0.0
        self._state["prev_r_dps"] = 0.0

        # held setpoints from outer loop
        self._state["des_phi"] = 0.0
        self._state["des_theta"] = 0.0
        self._state["des_psi"] = 0.0
        self._state["des_psi_dot"] = 0.0
        self._state["thrust_cmd"] = self.mass * params.g
        self._state["p_sp_dps"] = 0.0
        self._state["q_sp_dps"] = 0.0
        self._state["r_sp_dps"] = 0.0
        self._state["force_cmd"] = self.mass * params.g
        self._state["moment_cmd"] = np.zeros(3, dtype=float)
        self._state["motor_pwm"] = np.zeros(4, dtype=float)

    def run_wrench(self, quad, des_state):
        g = self.gains
        x, y, z = quad.position()
        x_dot, y_dot, z_dot = quad.velocity()
        phi, theta, psi = quad.attitude()
        p, q, r = quad.omega()

        des_x, des_y, des_z = des_state.pos
        des_x_dot, des_y_dot, des_z_dot = des_state.vel
        des_x_ddot, des_y_ddot, des_z_ddot = des_state.acc

        if g.USE_TRAJECTORY_YAW:
            des_psi = des_state.yaw
            des_psi_dot = des_state.yawdot
        else:
            des_psi = g.HOLD_YAW
            des_psi_dot = 0.0
        self._state["des_psi"] = float(des_psi)
        self._state["des_psi_dot"] = float(des_psi_dot)

        dt = self.dt
        self._state["tick"] += 1
        update_outer = (self._state["tick"] == 1) or (self._state["tick"] % g.OUTER_LOOP_DIV == 0)

        # ----------------------
        # Outer loops (100 Hz)
        # ----------------------
        if update_outer:
            # position -> velocity setpoint
            ex, ey, ez = des_x - x, des_y - y, des_z - z
            self._state["i_pos_x"] = _clip_sym(self._state["i_pos_x"] + ex * dt, g.POS_XY_INT_LIM)
            self._state["i_pos_y"] = _clip_sym(self._state["i_pos_y"] + ey * dt, g.POS_XY_INT_LIM)
            self._state["i_pos_z"] = _clip_sym(self._state["i_pos_z"] + ez * dt, g.POS_Z_INT_LIM)
            vx_sp = des_x_dot + g.POS_XY_KP * ex + g.POS_XY_KI * self._state["i_pos_x"] + g.POS_XY_KD * 0.0
            vy_sp = des_y_dot + g.POS_XY_KP * ey + g.POS_XY_KI * self._state["i_pos_y"] + g.POS_XY_KD * 0.0
            vz_sp = des_z_dot + g.POS_Z_KP * ez + g.POS_Z_KI * self._state["i_pos_z"] + g.POS_Z_KD * 0.0

            # velocity -> acceleration setpoint
            evx, evy, evz = vx_sp - x_dot, vy_sp - y_dot, vz_sp - z_dot
            self._state["i_vel_x"] = _clip_sym(self._state["i_vel_x"] + evx * dt, g.VEL_XY_INT_LIM)
            self._state["i_vel_y"] = _clip_sym(self._state["i_vel_y"] + evy * dt, g.VEL_XY_INT_LIM)
            self._state["i_vel_z"] = _clip_sym(self._state["i_vel_z"] + evz * dt, g.VEL_Z_INT_LIM)

            ax_cmd = des_x_ddot + g.VEL_XY_KP * evx + g.VEL_XY_KI * self._state["i_vel_x"] + g.VEL_XY_KD * 0.0
            ay_cmd = des_y_ddot + g.VEL_XY_KP * evy + g.VEL_XY_KI * self._state["i_vel_y"] + g.VEL_XY_KD * 0.0
            az_cmd = des_z_ddot + g.VEL_Z_KP * evz + g.VEL_Z_KI * self._state["i_vel_z"] + g.VEL_Z_KD * 0.0

            # acceleration -> attitude + thrust
            des_phi = (1.0 / params.g) * (ax_cmd * sin(des_psi) - ay_cmd * cos(des_psi))
            des_theta = (1.0 / params.g) * (ax_cmd * cos(des_psi) + ay_cmd * sin(des_psi))
            self._state["des_phi"] = float(np.clip(des_phi, -g.MAX_TILT, g.MAX_TILT))
            self._state["des_theta"] = float(np.clip(des_theta, -g.MAX_TILT, g.MAX_TILT))

            max_upward_acc = self.max_thrust / self.mass - params.g
            az_cmd = float(np.clip(az_cmd, -0.95 * params.g, max_upward_acc))
            self._state["thrust_cmd"] = float(np.clip(self.mass * (params.g + az_cmd), 0.0, self.max_thrust))

        F = self._state["thrust_cmd"]
        des_phi = self._state["des_phi"]
        des_theta = self._state["des_theta"]

        # ----------------------
        # Attitude loop (500 Hz)
        # ----------------------
        phi_d = np.degrees(phi)
        theta_d = np.degrees(theta)
        psi_d = np.degrees(psi)
        des_phi_d = np.degrees(des_phi)
        des_theta_d = np.degrees(des_theta)
        des_psi_d = np.degrees(des_psi)
        des_psi_dot_dps = np.degrees(des_psi_dot)

        e_roll = _wrap_deg(des_phi_d - phi_d)
        e_pitch = _wrap_deg(des_theta_d - theta_d)
        e_yaw = _wrap_deg(des_psi_d - psi_d)

        self._state["i_att_roll"] = _clip_sym(self._state["i_att_roll"] + e_roll * dt, g.ATT_RP_INT_LIM)
        self._state["i_att_pitch"] = _clip_sym(self._state["i_att_pitch"] + e_pitch * dt, g.ATT_RP_INT_LIM)
        self._state["i_att_yaw"] = _clip_sym(self._state["i_att_yaw"] + e_yaw * dt, g.ATT_YAW_INT_LIM)

        # derivative on measurement (yaw D only in table)
        phi_dot_dps = np.degrees(p)
        theta_dot_dps = np.degrees(q)
        psi_dot_dps = np.degrees(r)

        p_sp = g.ATT_RP_KP * e_roll + g.ATT_RP_KI * self._state["i_att_roll"] - g.ATT_RP_KD * phi_dot_dps
        q_sp = g.ATT_RP_KP * e_pitch + g.ATT_RP_KI * self._state["i_att_pitch"] - g.ATT_RP_KD * theta_dot_dps
        r_sp = des_psi_dot_dps + g.ATT_YAW_KP * e_yaw + g.ATT_YAW_KI * self._state["i_att_yaw"] - g.ATT_YAW_KD * psi_dot_dps
        p_sp = _clip_sym(p_sp, g.RATE_RP_SP_MAX)
        q_sp = _clip_sym(q_sp, g.RATE_RP_SP_MAX)
        r_sp = _clip_sym(r_sp, g.RATE_YAW_SP_MAX)
        self._state["p_sp_dps"] = p_sp
        self._state["q_sp_dps"] = q_sp
        self._state["r_sp_dps"] = r_sp

        # ----------------------
        # Rate loop (500 Hz)
        # ----------------------
        p_dps, q_dps, r_dps = np.degrees([p, q, r])
        ep, eq, er = p_sp - p_dps, q_sp - q_dps, r_sp - r_dps

        self._state["i_rate_roll"] = _clip_sym(self._state["i_rate_roll"] + ep * dt, g.RATE_RP_INT_LIM)
        self._state["i_rate_pitch"] = _clip_sym(self._state["i_rate_pitch"] + eq * dt, g.RATE_RP_INT_LIM)
        self._state["i_rate_yaw"] = _clip_sym(self._state["i_rate_yaw"] + er * dt, g.RATE_YAW_INT_LIM)

        p_dot_dps2 = (p_dps - self._state["prev_p_dps"]) / dt
        q_dot_dps2 = (q_dps - self._state["prev_q_dps"]) / dt
        r_dot_dps2 = (r_dps - self._state["prev_r_dps"]) / dt
        self._state["prev_p_dps"] = p_dps
        self._state["prev_q_dps"] = q_dps
        self._state["prev_r_dps"] = r_dps

        u_roll = g.RATE_RP_KP * ep + g.RATE_RP_KI * self._state["i_rate_roll"] - g.RATE_RP_KD * p_dot_dps2
        u_pitch = g.RATE_RP_KP * eq + g.RATE_RP_KI * self._state["i_rate_pitch"] - g.RATE_RP_KD * q_dot_dps2
        u_yaw = g.RATE_YAW_KP * er + g.RATE_YAW_KI * self._state["i_rate_yaw"] - g.RATE_YAW_KD * r_dot_dps2

        # Map firmware-like rate outputs to realizable body moments.
        u_roll = _clip_sym(u_roll, g.MOTOR_MIX_CMD_MAX)
        u_pitch = _clip_sym(u_pitch, g.MOTOR_MIX_CMD_MAX)
        u_yaw = _clip_sym(u_yaw, g.MOTOR_MIX_CMD_MAX)

        max_mx, max_my, max_mz = actuators.max_body_moments(self.actuator_profile)
        M = np.array([
            max_mx * (u_roll / g.MOTOR_MIX_CMD_MAX),
            max_my * (u_pitch / g.MOTOR_MIX_CMD_MAX),
            max_mz * (u_yaw / g.MOTOR_MIX_CMD_MAX)
        ]).reshape(3, 1)

        self._state["force_cmd"] = float(F)
        self._state["moment_cmd"] = M.reshape(3,).copy()

        return F, M

    def run(self, quad, des_state, sim_time=0.0):
        del sim_time  # the PID has no time gating
        F, M = self.run_wrench(quad, des_state)
        motor_pwm = actuators.wrench_to_pwm(
            total_thrust=F,
            moment=M,
            battery_voltage=quad.battery_voltage,
            actuator_profile=self.actuator_profile,
        )
        self._state["motor_pwm"] = motor_pwm.copy()
        return motor_pwm

    def diagnostics(self):
        return {
            "des_phi": float(self._state["des_phi"]),
            "des_theta": float(self._state["des_theta"]),
            "des_psi": float(self._state["des_psi"]),
            "des_psi_dot": float(self._state["des_psi_dot"]),
            "thrust_cmd": float(self._state["thrust_cmd"]),
            "force_cmd": float(self._state["force_cmd"]),
            "moment_cmd": self._state["moment_cmd"].copy(),
            "motor_pwm": self._state["motor_pwm"].copy(),
            "p_sp_dps": float(self._state["p_sp_dps"]),
            "q_sp_dps": float(self._state["q_sp_dps"]),
            "r_sp_dps": float(self._state["r_sp_dps"]),
        }
