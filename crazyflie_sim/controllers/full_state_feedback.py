import numpy as np
from scipy.linalg import solve_continuous_are

import crazyflie_sim.model.actuators as actuators
import crazyflie_sim.model.params as params
from crazyflie_sim.model.actuator_profile import get_actuator_profile
from crazyflie_sim.model import linearization
from .base import ControllerBase
from .registry import register
from .common import clip_sym, wrap_pi


CONTROL_DT = 1.0 / 500.0

# Bryson-style default tuning. The values are set a little tighter than the
# nominal "1 unit" rule so the controller remains useful with motor lag and the
# default wind benchmark.
DEFAULT_STATE_LIMITS = np.array(
    [
        0.50, 0.50, 0.50,  # position [m]
        0.80, 0.80, 0.80,  # velocity [m/s]
        0.35, 0.35, 0.50,  # attitude [rad]
        4.00, 4.00, 5.00,  # body rates [rad/s]
    ],
    dtype=float,
)
DEFAULT_INT_LIMITS = np.array([4.00, 4.00, 4.00], dtype=float)
DEFAULT_INPUT_LIMITS = np.array(
    [
        0.40,   # total thrust delta [N]
        0.014,  # roll moment [N*m]
        0.014,  # pitch moment [N*m]
        0.0023, # yaw moment [N*m]
    ],
    dtype=float,
)

DEFAULT_Q_SCALE = 1.0
DEFAULT_R_SCALE = 1.0
DEFAULT_INT_Q = np.array([512.0, 512.0, 512.0], dtype=float)
DEFAULT_MAX_TILT = np.deg2rad(50.0)


def _as_vector3(value, name):
    arr = np.asarray(value, dtype=float).reshape(3,)
    if arr.shape != (3,):
        raise ValueError("{} must be shape (3,).".format(name))
    return arr


def _sanitize_hover_A(A):
    A_clean = np.asarray(A, dtype=float).copy()

    # The quadratic drag derivative at exactly zero velocity is analytically
    # zero; finite differences leave tiny self-damping artifacts in these slots.
    for idx in range(3, 6):
        if abs(A_clean[idx, idx]) < 1.0e-5:
            A_clean[idx, idx] = 0.0
    return A_clean


def _default_Q(use_integral):
    state_limits = DEFAULT_STATE_LIMITS.copy()
    q_diag = 1.0 / np.square(state_limits)
    if not use_integral:
        return np.diag(DEFAULT_Q_SCALE * q_diag)

    int_diag = DEFAULT_INT_Q.copy()
    return np.diag(DEFAULT_Q_SCALE * np.concatenate((q_diag, int_diag)))


def _default_R():
    return np.diag(DEFAULT_R_SCALE / np.square(DEFAULT_INPUT_LIMITS))


@register("full_state_feedback", label="Full-state LQR", color="#bcbd22")
class FullStateFeedbackController(ControllerBase):
    """Continuous-time LQR full-state feedback controller.

    The controller is designed around the hover linearization in wrench
    coordinates u = [F, Mx, My, Mz]. It tracks position, velocity, attitude,
    yaw, and body-rate references with optional position-integral augmentation.
    """

    def __init__(
        self,
        dt=CONTROL_DT,
        actuator_profile=None,
        mass=None,
        Q=None,
        R=None,
        use_integral=True,
        int_limit=None,
        max_tilt=None,
    ):
        self.dt = float(dt)
        self.actuator_profile = get_actuator_profile(actuator_profile)
        self.mass = float(self.actuator_profile.mass if mass is None else mass)
        self.max_thrust = self.actuator_profile.max_total_thrust
        self.use_integral = bool(use_integral)
        self.int_limit = np.asarray(
            DEFAULT_INT_LIMITS if int_limit is None else int_limit,
            dtype=float,
        ).reshape(3,)
        self.max_tilt = float(DEFAULT_MAX_TILT if max_tilt is None else max_tilt)

        A, B, x_eq, u_eq = linearization.linearize_hover(actuator_profile=self.actuator_profile)
        self.A = _sanitize_hover_A(A)
        self.B = np.asarray(B, dtype=float).copy()
        self.x_eq = np.asarray(x_eq, dtype=float).copy()
        self.u_eq = np.asarray(u_eq, dtype=float).copy()
        self.u_eq[0] = self.mass * params.g

        self.A_aug, self.B_aug = self._build_design_model()
        self.Q = np.asarray(_default_Q(self.use_integral) if Q is None else Q, dtype=float)
        self.R = np.asarray(_default_R() if R is None else R, dtype=float)
        self.K = self._solve_gain(self.Q, self.R)

        self._max_m = actuators.max_body_moments(self.actuator_profile)
        self.reset()

    def _build_design_model(self):
        if not self.use_integral:
            return self.A.copy(), self.B.copy()

        A_aug = np.zeros((15, 15), dtype=float)
        A_aug[:12, :12] = self.A
        A_aug[12:15, 0:3] = np.eye(3)

        B_aug = np.zeros((15, 4), dtype=float)
        B_aug[:12, :] = self.B
        return A_aug, B_aug

    def _solve_gain(self, Q, R):
        expected_n = 15 if self.use_integral else 12
        if Q.shape != (expected_n, expected_n):
            raise ValueError("Q must have shape ({0}, {0}).".format(expected_n))
        if R.shape != (4, 4):
            raise ValueError("R must have shape (4, 4).")

        P = solve_continuous_are(self.A_aug, self.B_aug, Q, R)
        return np.linalg.solve(R, self.B_aug.T.dot(P))

    def set_dt(self, dt):
        self.dt = float(dt)

    def reset(self):
        self.e_int = np.zeros(3, dtype=float)
        self._last_error = np.zeros(12, dtype=float)
        self._last_aug_error = np.zeros(self.K.shape[1], dtype=float)
        self._last_x_ref = np.zeros(12, dtype=float)
        self._last_u_ref = self.u_eq.copy()
        self._last_u_unsat = self.u_eq.copy()
        self._last_force_cmd = float(self.u_eq[0])
        self._last_moment = np.zeros(3, dtype=float)
        self._last_motor_pwm = np.zeros(4, dtype=float)
        self._last_saturated = False

    def _current_state(self, quad):
        return np.concatenate(
            (
                np.asarray(quad.position(), dtype=float).reshape(3,),
                np.asarray(quad.velocity(), dtype=float).reshape(3,),
                np.asarray(quad.attitude(), dtype=float).reshape(3,),
                np.asarray(quad.omega(), dtype=float).reshape(3,),
            )
        )

    def _reference(self, des_state):
        pos_d = _as_vector3(des_state.pos, "des_state.pos")
        vel_d = _as_vector3(des_state.vel, "des_state.vel")
        acc_d = _as_vector3(des_state.acc, "des_state.acc")
        yaw_d = float(des_state.yaw)
        yawdot_d = float(des_state.yawdot)

        des_phi = (acc_d[0] * np.sin(yaw_d) - acc_d[1] * np.cos(yaw_d)) / params.g
        des_theta = (acc_d[0] * np.cos(yaw_d) + acc_d[1] * np.sin(yaw_d)) / params.g
        des_phi = clip_sym(des_phi, self.max_tilt)
        des_theta = clip_sym(des_theta, self.max_tilt)

        x_ref = np.zeros(12, dtype=float)
        x_ref[0:3] = pos_d
        x_ref[3:6] = vel_d
        x_ref[6:9] = np.array([des_phi, des_theta, yaw_d], dtype=float)
        x_ref[9:12] = np.array([0.0, 0.0, yawdot_d], dtype=float)

        u_ref = np.array(
            [self.mass * (params.g + acc_d[2]), 0.0, 0.0, 0.0],
            dtype=float,
        )
        u_ref[0] = float(np.clip(u_ref[0], 0.0, self.max_thrust))
        return x_ref, u_ref

    def _state_error(self, x, x_ref):
        e = np.asarray(x, dtype=float).reshape(12,) - np.asarray(x_ref, dtype=float).reshape(12,)
        e[8] = wrap_pi(e[8])
        return e

    def _control_from_error(self, e, e_int):
        if self.use_integral:
            aug_error = np.concatenate((e, e_int))
        else:
            aug_error = e
        u = self._last_u_ref - self.K.dot(aug_error)
        return u, aug_error

    def run_wrench(self, quad, des_state, sim_time=0.0):
        del sim_time

        dt = max(float(self.dt), 1.0e-6)
        x = self._current_state(quad)
        x_ref, u_ref = self._reference(des_state)
        self._last_u_ref = u_ref
        e = self._state_error(x, x_ref)

        candidate_int = self.e_int.copy()
        if self.use_integral:
            candidate_int = np.clip(
                self.e_int + e[0:3] * dt,
                -self.int_limit,
                self.int_limit,
            )

        u_unsat, aug_error = self._control_from_error(e, candidate_int)
        force_cmd = float(np.clip(u_unsat[0], 0.0, self.max_thrust))
        moment = np.clip(u_unsat[1:4], -self._max_m, self._max_m)
        saturated = (
            abs(force_cmd - float(u_unsat[0])) > 1.0e-10
            or np.any(np.abs(moment - u_unsat[1:4]) > 1.0e-10)
        )

        if self.use_integral and not saturated:
            self.e_int = candidate_int
        elif self.use_integral:
            _, aug_error = self._control_from_error(e, self.e_int)

        self._last_error = e.copy()
        self._last_aug_error = aug_error.copy()
        self._last_x_ref = x_ref.copy()
        self._last_u_unsat = u_unsat.copy()
        self._last_force_cmd = force_cmd
        self._last_moment = moment.copy()
        self._last_saturated = bool(saturated)

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
            "force_cmd": float(self._last_force_cmd),
            "moment_cmd": self._last_moment.copy(),
            "moment_si": self._last_moment.copy(),
            "motor_pwm": self._last_motor_pwm.copy(),
            "e": self._last_error.copy(),
            "e_int": self.e_int.copy(),
            "aug_error": self._last_aug_error.copy(),
            "x_ref": self._last_x_ref.copy(),
            "u_ref": self._last_u_ref.copy(),
            "u_unsat": self._last_u_unsat.copy(),
            "K": self.K.copy(),
            "A_aug": self.A_aug.copy(),
            "B_aug": self.B_aug.copy(),
            "saturated": bool(self._last_saturated),
            "status_label": "||e||",
            "status_value": float(np.linalg.norm(self._last_error)),
        }
