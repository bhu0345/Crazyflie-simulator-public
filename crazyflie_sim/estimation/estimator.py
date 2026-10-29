"""State-estimation layer: navigation filter behind an estimated view.

Two backends share :class:`EstimatedQuadView`:

- ``kalman`` (default): firmware-style error-state quaternion EKF.  IMU
  prediction, lighthouse *scalar sweep-angle* updates, and an accelerometer
  specific-force / gravity (thrust-axis) update.  This is the *navigation*
  EKF; it is not ``RotorThrustEKF``.
- ``complementary``: IMU prediction + delayed mocap Euler, kept as a
  regression ablation.

The view wraps the real Quadcopter so a controller can close the loop on the
ESTIMATE through the same interface; everything that is not a state accessor
still resolves to the truth.

Determinism: the estimator itself is deterministic; all randomness lives in
the sensors, which share one np.random.Generator (see build_estimator).
"""

import numpy as np

import crazyflie_sim.model.params as params
from crazyflie_sim.model.quadcopter import Quadcopter
from crazyflie_sim.model.sensors import IMU, Mocap
from crazyflie_sim.utils.utils import RPYToRot

_ESTIMATOR_KEYS = (
    "enabled",
    "backend",
    "imu",
    "mocap",
    "process_noise",
    "gains",
    "gyro_range_dps",
    "lighthouse_bias_shared",
    "lighthouse_yaw_update",
)
_IMU_KEYS = (
    "gyro_noise_std",
    "gyro_bias_std",
    "accel_noise_std",
    "accel_bias_std",
)
_MOCAP_KEYS = ("rate_hz", "pos_noise_std", "att_noise_std", "delay_steps")
_GAIN_KEYS = ("k_pos", "k_vel", "k_att", "k_bias")
_PROCESS_NOISE_KEYS = (
    "pos",
    "vel",
    "att",
    "gyro_bias",
    "accel",
    "accel_meas_std",
    "angle_meas_std",
)
_BACKENDS = ("kalman", "complementary")

# Defaults chosen in the Crazyflie 2.1 ballpark (BMI088-class IMU, mm-accurate
# 100 Hz mocap with ~20 ms pipeline latency).
_IMU_DEFAULTS = {
    "gyro_noise_std": 0.005,  # rad/s white noise per axis
    "gyro_bias_std": 0.02,  # rad/s constant per-run turn-on bias
    "accel_noise_std": 0.1,  # m/s^2 white noise per axis
    "accel_bias_std": 0.05,  # m/s^2 constant per-run bias
}
_MOCAP_DEFAULTS = {
    "rate_hz": 100.0,
    "pos_noise_std": 0.002,  # m per axis
    "att_noise_std": 0.01,  # rad per axis
    "delay_steps": 2,  # samples; latency ~= delay_steps / rate_hz (~20 ms)
}
_GAIN_DEFAULTS = {
    "k_pos": 0.4,
    "k_vel": 0.1,
    "k_att": 0.3,
    "k_bias": 0.01,  # Mahony-lite constant-bias learning (small on purpose)
}
_PROCESS_NOISE_DEFAULTS = {
    "pos": 0.02,
    "vel": 0.25,
    "att": 0.05,
    "gyro_bias": 1.0e-5,
    "accel": 0.50,
    "accel_meas_std": 0.35,
    # Larger than LH2_ANGLE_NOISE_STD white jitter.  Does not absorb the
    # unmodeled per-run LH2_ANGLE_BIAS_STD (Taffanel 2021 centimetre-class
    # Lighthouse accuracy versus plant truth).
    "angle_meas_std": 0.006,
}
_SPIN_YAW_RATE_MIN = 20.0  # rad/s; past the fault yaw impulse
_SPIN_YAW_RATE_CLEAR = 8.0
_SPIN_TILT_MAX = float(np.radians(15.0))


def _wrap_pi(angle):
    """Wrap angle(s) to [-pi, pi]."""
    return (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _tilt_rad(att):
    att = np.asarray(att, dtype=float).reshape(3,)
    return float(
        np.arccos(
            np.clip(np.cos(att[0]) * np.cos(att[1]), -1.0, 1.0)
        )
    )


class EstimatedQuadView:
    """Controller-facing view of a Quadcopter presenting the ESTIMATED state.

    The state accessors controllers actually read -- position(), velocity(),
    attitude(), omega(), rotation_world_to_body(), rotation_body_to_world() --
    are explicitly overridden to return the estimator's snapshot taken at view
    creation.  Everything else delegates to the wrapped Quadcopter via
    __getattr__: battery_voltage, rotor_fault, motor_thrusts, state,
    world_frame(), rotor_fault_state(), ... all resolve to the TRUTH.  In
    particular world_frame() and other plot/diagnostic helpers intentionally
    show the truth -- they are outputs, not controller inputs.
    """

    def __init__(self, quad, pos, vel, att, omega):
        self._quad = quad
        # Snapshots: later estimator updates must not leak into a live view.
        self._pos = np.asarray(pos, dtype=float).reshape(3).copy()
        self._vel = np.asarray(vel, dtype=float).reshape(3).copy()
        self._att = np.asarray(att, dtype=float).reshape(3).copy()
        self._omega = np.asarray(omega, dtype=float).reshape(3).copy()

    def __getattr__(self, name):
        # Only called for attributes not defined above: delegate to the truth.
        return getattr(self._quad, name)

    # --- estimated state accessors (same signatures as Quadcopter) ---------
    def position(self):
        return self._pos

    def velocity(self):
        return self._vel

    def attitude(self):
        return self._att

    def omega(self):
        return self._omega

    def rotation_world_to_body(self):
        phi, theta, psi = self._att
        return RPYToRot(float(phi), float(theta), float(psi))

    def rotation_body_to_world(self):
        return self.rotation_world_to_body().T


class ComplementaryEstimator:
    """Complementary filter: IMU prediction + mocap correction.

    States: est pos (3), vel (3), att RPY (3), gyro_bias (3).  The estimated
    body rates exposed through the view are the bias-corrected gyro
    measurements (gyro_bias is the only attitude-rate state, so omega is not
    filtered further -- this is the usual complementary-filter choice).
    """

    backend = "complementary"

    def __init__(
        self,
        quad,
        imu,
        mocap,
        dt,
        k_pos=0.4,
        k_vel=0.1,
        k_att=0.3,
        k_bias=0.01,
    ):
        self.imu = imu
        self.mocap = mocap
        self.dt = float(dt)
        self.k_pos = float(k_pos)
        self.k_vel = float(k_vel)
        self.k_att = float(k_att)
        self.k_bias = float(k_bias)
        self.prime(quad)

    def prime(self, quad):
        """(Re)initialize the estimate on a quad at its initial state.

        Position/attitude are primed from the first available mocap sample
        (the mocap emits at t = 0); if no sample is available yet we fall
        back to the truth -- justified because the estimator is primed on the
        known parked-on-the-pad initial state.  Mocap measures no velocity, so
        velocity is primed from the truth as well (the vehicle starts at
        rest).  The gyro-bias estimate starts at zero.
        """
        self.mocap.reset()
        sample = self.mocap.update(quad, 0.0)
        if sample is not None:
            self.pos_est, self.att_est = sample
        else:
            self.pos_est = np.asarray(quad.position(), dtype=float).reshape(3).copy()
            self.att_est = np.asarray(quad.attitude(), dtype=float).reshape(3).copy()
        self.vel_est = np.asarray(quad.velocity(), dtype=float).reshape(3).copy()
        self.gyro_bias_est = np.zeros(3, dtype=float)
        # Latest bias-corrected gyro measurement (bias estimate is zero here).
        self._omega_est = self.imu.gyro(quad.omega()) - self.gyro_bias_est
        self._spin_yaw_bias_frozen = False

    def update(self, quad, dt, t, external_force_world, wind_vel_world=None):
        """Advance the estimate to time ``t`` (post-plant-step)."""
        dt = float(dt)
        # Per-call sensor draw order is fixed: gyro, accel, then mocap.
        gyro_meas = self.imu.gyro(quad.omega())
        accel_meas = self.imu.accel(quad, external_force_world, wind_vel_world)
        self._omega_est = gyro_meas - self.gyro_bias_est

        # ---- predict --------------------------------------------------------
        # Attitude integrates bias-corrected gyro rates through the plant's
        # own RPY-rate convention (Quadcopter.euler_rates is a staticmethod).
        eulerdot = Quadcopter.euler_rates(
            float(self.att_est[0]),
            float(self.att_est[1]),
            float(self._omega_est[0]),
            float(self._omega_est[1]),
            float(self._omega_est[2]),
        )
        self.att_est = self.att_est + dt * eulerdot
        # pos/vel integrate the specific force rotated to the world frame by
        # the ESTIMATED attitude, plus gravity.
        bRw_est = RPYToRot(
            float(self.att_est[0]), float(self.att_est[1]), float(self.att_est[2])
        )
        a_world = bRw_est.T.dot(accel_meas) - np.array([0.0, 0.0, params.g])
        self.vel_est = self.vel_est + dt * a_world
        self.pos_est = self.pos_est + dt * self.vel_est

        # ---- correct (only when the mocap emits) ----------------------------
        sample = self.mocap.update(quad, t)
        if sample is not None:
            mocap_pos, mocap_att = sample
            pos_err = mocap_pos - self.pos_est
            self.pos_est = self.pos_est + self.k_pos * pos_err
            self.vel_est = self.vel_est + self.k_vel * pos_err
            att_err = _wrap_pi(mocap_att - self.att_est)
            omega_z = abs(float(self._omega_est[2]))
            rate_norm = float(np.linalg.norm(self._omega_est))
            att_gain = self.k_att / (1.0 + rate_norm / 5.0)
            tilt_est = _tilt_rad(self.att_est)
            if omega_z > _SPIN_YAW_RATE_MIN and tilt_est < _SPIN_TILT_MAX:
                self._spin_yaw_bias_frozen = True
            elif omega_z < _SPIN_YAW_RATE_CLEAR:
                self._spin_yaw_bias_frozen = False
            if self._spin_yaw_bias_frozen or (
                omega_z > _SPIN_YAW_RATE_CLEAR and tilt_est < _SPIN_TILT_MAX
            ):
                # Raw delayed Euler yaw is a DC heading error of about Ω τ.
                # After the spin is established, never integrate that into
                # gyro_bias[2] (that dump is what used to appear near 30 s).
                att_err = np.asarray(att_err, dtype=float).copy()
                att_err[2] = 0.0
            self.att_est = self.att_est + att_gain * att_err
            if self._spin_yaw_bias_frozen:
                delay_s = 0.0
                if self.mocap.rate_hz > 1.0e-9:
                    delay_s = float(self.mocap.delay_steps) / float(
                        self.mocap.rate_hz
                    )
                yaw_meas = _wrap_pi(
                    float(mocap_att[2]) + float(self._omega_est[2]) * delay_s
                )
                # Delay-compensated heading keeps roll/pitch and yaw in the
                # same Euler chart so accelerometer integration and the
                # world-frame COM pull stay consistent.
                self.att_est[2] = self.att_est[2] + self.k_att * _wrap_pi(
                    yaw_meas - float(self.att_est[2])
                )
                self.gyro_bias_est[:2] = (
                    self.gyro_bias_est[:2] - self.k_bias * att_err[:2]
                )
            else:
                self.gyro_bias_est = (
                    self.gyro_bias_est - self.k_bias * att_err
                )

    def view(self, quad):
        """Return a fresh EstimatedQuadView snapshot for one control tick."""
        return EstimatedQuadView(
            quad, self.pos_est, self.vel_est, self.att_est, self._omega_est
        )


# ---------------------------------------------------------------------------
# Config-driven construction
# ---------------------------------------------------------------------------


def _err(key, message):
    raise ValueError("estimator config key {!r}: {}".format(key, message))


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_nonneg_number(key, value):
    if not _is_number(value):
        _err(key, "expected a number, got {!r}".format(value))
    if value < 0:
        _err(key, "must be non-negative, got {!r}".format(value))
    return float(value)


def _require_positive_number(key, value):
    if not _is_number(value) or value <= 0:
        _err(key, "expected a positive number, got {!r}".format(value))
    return float(value)


def _require_nonneg_int(key, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _err(key, "expected a non-negative integer, got {!r}".format(value))
    return int(value)


def _sub_cfg(est_cfg, name, valid_keys, defaults, validators):
    value = est_cfg.get(name, {})
    if value is None:
        value = {}
    if not isinstance(value, dict):
        _err("estimator." + name, "expected a mapping, got {!r}".format(value))
    unknown = sorted(set(value) - set(valid_keys))
    if unknown:
        _err(
            "estimator." + name,
            "unknown sub-key(s) {}; valid sub-keys: {}".format(unknown, list(valid_keys)),
        )
    out = dict(defaults)
    for key, item in value.items():
        out[key] = validators[key]("estimator.{}.{}".format(name, key), item)
    return out


def build_estimator(est_cfg, quad, dt, seed=None):
    """Build a navigation estimator (+ IMU + mocap or lighthouse) from config.

    Schema (all sub-dicts optional except ``enabled``)::

        estimator:
          enabled: bool
          backend: kalman | complementary   # default kalman
          imu:   {gyro_noise_std, gyro_bias_std, accel_noise_std,
                  accel_bias_std}
          mocap: {rate_hz, pos_noise_std, att_noise_std, delay_steps}
          process_noise: {pos, vel, att, gyro_bias, accel, ...}
          gains: {k_pos, k_vel, k_att, k_bias}   # complementary only
          gyro_range_dps: float                  # optional IMU full-scale
                                                 # override (deg/s)
          lighthouse_bias_shared: bool           # kalman only, default false
          lighthouse_yaw_update: bool            # kalman only, default false

    Lighthouse V2 geometry/rate/delay are hardware constants
    (``model.lighthouse``), not estimator-config keys.  The IMU gyro range
    defaults to ``params.sim_gyro_range_dps`` for kalman and
    ``params.gyro_range_dps`` for complementary; ``gyro_range_dps`` overrides
    either, for studies of gyro saturation.

    Raises ValueError naming the offending key on schema violations.  Sensors
    share ONE np.random.default_rng(seed); the draw order is fixed (imu
    biases at construction, then per update: gyro noise, accel noise,
    lighthouse or mocap noise on emission).
    """
    if not isinstance(est_cfg, dict):
        _err("estimator", "expected a mapping, got {!r}".format(est_cfg))
    unknown = sorted(set(est_cfg) - set(_ESTIMATOR_KEYS))
    if unknown:
        _err(
            "estimator",
            "unknown sub-key(s) {}; valid sub-keys: {}".format(
                unknown, list(_ESTIMATOR_KEYS)
            ),
        )
    if "enabled" not in est_cfg:
        _err("estimator.enabled", "missing required key")
    if not isinstance(est_cfg["enabled"], bool):
        _err("estimator.enabled", "expected a boolean, got {!r}".format(est_cfg["enabled"]))
    for key in ("lighthouse_bias_shared", "lighthouse_yaw_update"):
        if not isinstance(est_cfg.get(key, False), bool):
            _err("estimator." + key, "expected a boolean, got {!r}".format(est_cfg[key]))

    backend = str(est_cfg.get("backend", "kalman")).strip().lower()
    if backend not in _BACKENDS:
        _err(
            "estimator.backend",
            "expected one of {}, got {!r}".format(list(_BACKENDS), est_cfg.get("backend")),
        )

    imu_cfg = _sub_cfg(
        est_cfg,
        "imu",
        _IMU_KEYS,
        _IMU_DEFAULTS,
        {key: _require_nonneg_number for key in _IMU_KEYS},
    )
    if est_cfg.get("gyro_range_dps") is not None:
        gyro_range_dps = _require_positive_number(
            "estimator.gyro_range_dps", est_cfg["gyro_range_dps"]
        )
    elif backend == "kalman":
        gyro_range_dps = float(getattr(params, "sim_gyro_range_dps", 4000.0))
    else:
        gyro_range_dps = float(params.gyro_range_dps)

    rng = np.random.default_rng(seed)
    imu = IMU(
        rng=rng,
        gyro_range_rad_s=float(np.radians(gyro_range_dps)),
        **imu_cfg,
    )
    if "mocap" in est_cfg:
        _sub_cfg(
            est_cfg,
            "mocap",
            _MOCAP_KEYS,
            _MOCAP_DEFAULTS,
            {
                "rate_hz": _require_positive_number,
                "pos_noise_std": _require_nonneg_number,
                "att_noise_std": _require_nonneg_number,
                "delay_steps": _require_nonneg_int,
            },
        )
    if "gains" in est_cfg:
        _sub_cfg(
            est_cfg,
            "gains",
            _GAIN_KEYS,
            _GAIN_DEFAULTS,
            {key: _require_nonneg_number for key in _GAIN_KEYS},
        )
    if backend == "complementary":
        mocap_cfg = _sub_cfg(
            est_cfg,
            "mocap",
            _MOCAP_KEYS,
            _MOCAP_DEFAULTS,
            {
                "rate_hz": _require_positive_number,
                "pos_noise_std": _require_nonneg_number,
                "att_noise_std": _require_nonneg_number,
                "delay_steps": _require_nonneg_int,
            },
        )
        gains = _sub_cfg(
            est_cfg,
            "gains",
            _GAIN_KEYS,
            _GAIN_DEFAULTS,
            {key: _require_nonneg_number for key in _GAIN_KEYS},
        )
        mocap = Mocap(rng=rng, **mocap_cfg)
        return ComplementaryEstimator(quad, imu, mocap, dt, **gains)

    from crazyflie_sim.model.lighthouse import Lighthouse

    process_noise = _sub_cfg(
        est_cfg,
        "process_noise",
        _PROCESS_NOISE_KEYS,
        _PROCESS_NOISE_DEFAULTS,
        {key: _require_nonneg_number for key in _PROCESS_NOISE_KEYS},
    )
    lighthouse = Lighthouse(
        rng, bias_shared_across_sensors=est_cfg.get("lighthouse_bias_shared", False)
    )
    from crazyflie_sim.estimation.nav_ekf import ErrorStateNavEKF

    return ErrorStateNavEKF(
        quad, imu, lighthouse, dt, process_noise=process_noise,
        lighthouse_yaw_update=est_cfg.get("lighthouse_yaw_update", False),
    )
