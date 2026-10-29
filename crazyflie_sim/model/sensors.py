"""Sensor models: MEMS IMU (gyro + accelerometer) and motion capture.

Lighthouse sweep angles live in ``model.lighthouse``.  They are not Euler
RPY and are not produced here.

Every sensor draws its randomness from a single np.random.Generator supplied
at construction, so a whole simulation run is deterministic given its seed.
Draw order is fixed: IMU biases at construction, then per call gyro noise,
accel noise, and (on emission) lighthouse or mocap noise -- see
estimation.estimator for how the sensors are driven.

The IMU clips gyro measurements at ``gyro_range_rad_s``.  Firmware BMI088
hardware is 2000 deg/s (``params.gyro_range_dps``) and saturates a
40 rad/s yaw-released spin.  The research-plant navigation EKF uses
``params.sim_gyro_range_dps`` (4000 deg/s) so that study is not first
limited by saturation; the filter still consumes whatever clipped
measurement the IMU returns.
"""

from collections import deque

import numpy as np

import crazyflie_sim.model.aero as aero
import crazyflie_sim.model.params as params


class IMU:
    """Gyroscope + accelerometer with constant per-run bias and white noise.

    Biases are drawn once at construction from N(0, *_bias_std) -- they model
    the turn-on bias of a real MEMS IMU, constant for the whole run.  Every
    measurement additionally carries zero-mean white noise N(0, *_noise_std).
    Noise draws are unconditional (even when a std is 0.0) so the RNG stream
    does not depend on the configured noise levels.
    """

    def __init__(
        self,
        gyro_noise_std,
        gyro_bias_std,
        accel_noise_std,
        accel_bias_std,
        rng,
        gyro_range_rad_s=None,
    ):
        self.gyro_noise_std = float(gyro_noise_std)
        self.gyro_bias_std = float(gyro_bias_std)
        self.accel_noise_std = float(accel_noise_std)
        self.accel_bias_std = float(accel_bias_std)
        self.rng = rng
        self.gyro_range_rad_s = float(
            np.radians(params.gyro_range_dps)
            if gyro_range_rad_s is None
            else gyro_range_rad_s
        )
        if self.gyro_range_rad_s <= 0.0:
            raise ValueError("gyro_range_rad_s must be positive")
        self.last_gyro_saturated = np.zeros(3, dtype=bool)
        # Constant per-run biases (gyro first: fixed draw order).
        self.gyro_bias = rng.normal(0.0, self.gyro_bias_std, size=3)
        self.accel_bias = rng.normal(0.0, self.accel_bias_std, size=3)

    def gyro(self, omega_true):
        """Measured body rates (rad/s): truth + constant bias + white noise."""
        raw = (
            np.asarray(omega_true, dtype=float).reshape(3)
            + self.gyro_bias
            + self.rng.normal(0.0, self.gyro_noise_std, size=3)
        )
        self.last_gyro_saturated = np.abs(raw) >= self.gyro_range_rad_s
        return np.clip(raw, -self.gyro_range_rad_s, self.gyro_range_rad_s)

    def accel(self, quad, external_force_world, wind_vel_world=None):
        """Measured specific force in the body frame (m/s^2).

        Specific force = every non-gravity force per unit mass, which is what
        a real accelerometer reports (gravity is excluded).  This mirrors the
        plant's non-gravity force terms in Quadcopter.state_dot exactly:

            f_body = (thrust_body + R_world_to_body @ (F_ext + F_aero)) / mass

        with thrust_body = [0, 0, sum(motor_thrusts) * ground_effect_factor]
        and the same aerodynamic force (parasitic + rotor drag on the relative
        airflow v - v_wind) the plant applies.  Plus constant bias and white
        noise.
        """
        thrust_body = np.array(
            [
                0.0,
                0.0,
                float(np.sum(quad.motor_thrusts)) * quad.ground_effect_factor,
            ]
        )
        aero_world = aero.aero_force_world(
            quad.velocity(),
            wind_vel_world,
            quad.rotation_world_to_body(),
        )
        ext = np.asarray(external_force_world, dtype=float).reshape(3)
        specific_force_body = (
            thrust_body + quad.rotation_world_to_body().dot(ext + aero_world)
        ) / quad.mass
        return (
            specific_force_body
            + self.accel_bias
            + self.rng.normal(0.0, self.accel_noise_std, size=3)
        )


class Mocap:
    """Motion-capture system: noisy position + RPY attitude at a fixed rate.

    update(quad, t) returns a (pos_meas, att_meas) tuple when ``t`` crosses
    the next multiple of 1/rate_hz (the first sample is emitted at t = 0),
    and None otherwise.  Float timestamps are compared with a small absolute
    tolerance so exact-multiple crossings are not missed to last-ulp error.

    delay_steps models the pipeline latency in units of emitted samples:
    the served measurement is the sample from ``delay_steps`` emissions ago,
    kept in a ring buffer of capacity delay_steps + 1.  Before the buffer
    has filled, the oldest available sample is served (so the effective
    latency ramps up over the first delay_steps emissions).
    """

    # Absolute timestamp tolerance (s): far above float ulp at sim timescales,
    # far below any realistic sample period.
    TOL_S = 1e-9

    def __init__(self, rate_hz, pos_noise_std, att_noise_std, delay_steps, rng):
        self.rate_hz = float(rate_hz)
        self.pos_noise_std = float(pos_noise_std)
        self.att_noise_std = float(att_noise_std)
        self.delay_steps = int(delay_steps)
        self.rng = rng
        self._period = 1.0 / self.rate_hz
        self._buffer = deque(maxlen=self.delay_steps + 1)
        self._next_sample_time = 0.0  # first sample at t = 0

    def reset(self):
        """Clear the delay buffer and re-arm the t = 0 sample."""
        self._buffer.clear()
        self._next_sample_time = 0.0

    def update(self, quad, t):
        """Possibly sample the quad; returns (pos_meas, att_meas) or None."""
        t = float(t)
        if t + self.TOL_S < self._next_sample_time:
            return None
        # Advance to the period multiple strictly after t; the tolerance keeps
        # last-ulp timestamps from double-emitting or skipping a multiple.
        idx = int(np.floor((t + self.TOL_S) / self._period))
        self._next_sample_time = (idx + 1) * self._period

        sample = (
            np.asarray(quad.position(), dtype=float).reshape(3)
            + self.rng.normal(0.0, self.pos_noise_std, size=3),
            np.asarray(quad.attitude(), dtype=float).reshape(3)
            + self.rng.normal(0.0, self.att_noise_std, size=3),
        )
        self._buffer.append(sample)
        oldest = self._buffer[0]
        return oldest[0].copy(), oldest[1].copy()
