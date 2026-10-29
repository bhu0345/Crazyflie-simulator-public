"""Error-state quaternion navigation EKF (firmware kalman_core structure).

This is the *navigation* filter.  It is not ``RotorThrustEKF`` (the paddle
thrust observer).  The error state is world position, world velocity, body
attitude error ``d0:2``, and a slow gyro-bias random walk.  The unit
quaternion lives outside the covariance and is injected after every update.

Prediction uses the existing IMU gyro / specific-force pair.  Corrections are
lighthouse *scalar sweep angles* plus an accelerometer specific-force /
gravity update that observes the thrust axis (body-z vs world vertical)
through quaternion kinematics — not ``euler_rates``.
"""

import numpy as np

import crazyflie_sim.model.params as params
from crazyflie_sim.model.lighthouse import _lh2_direction_bs, sweep_angle_and_grad
from crazyflie_sim.utils.utils import (
    normalize_quat,
    quat_mul,
    quat_to_rot_body_to_world,
    quat_to_rpy,
    rpy_to_quat,
)

_STATE_DIM = 12
_POS = slice(0, 3)
_VEL = slice(3, 6)
_ATT = slice(6, 9)
_BIAS = slice(9, 12)
_INNOV_GATE = 5.0
# Standard deviation of the Lighthouse heading measurement.  The firmware uses
# 0.01 rad; with the simulated sweep-angle noise (bias shared across sensors)
# the measured heading error has a standard deviation of 4.45 deg (0.078 rad,
# 400 static samples), so 0.08 rad is used with this sensor model.
YAW_MEAS_STD = 0.08


def _skew(vector):
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=float,
    )


def _quat_from_rotvec(rotvec):
    rotvec = np.asarray(rotvec, dtype=float).reshape(3)
    angle = float(np.linalg.norm(rotvec))
    if angle <= 1.0e-16:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = rotvec / angle
    half = 0.5 * angle
    sine = float(np.sin(half))
    return np.array(
        [float(np.cos(half)), axis[0] * sine, axis[1] * sine, axis[2] * sine],
        dtype=float,
    )


def _symmetrize(matrix):
    matrix = 0.5 * (matrix + matrix.T)
    diag = np.diag(matrix)
    np.fill_diagonal(matrix, np.maximum(diag, 1.0e-18))
    return matrix


class ErrorStateNavEKF:
    """9+3 error-state INS with lighthouse scalar updates."""

    backend = "kalman"

    def __init__(
        self,
        quad,
        imu,
        lighthouse,
        dt,
        process_noise=None,
        lighthouse_yaw_update=False,
    ):
        self.imu = imu
        self.lighthouse = lighthouse
        self.dt = float(dt)
        # Firmware Lighthouse heading update (lighthouse_position_est.c,
        # estimateYaw; mm_yaw_error.c): the rays of the four deck sensors are
        # intersected with the deck plane of the estimated pose, the measured
        # sensor diagonals are compared with the expected ones, and the mean
        # angle is fused as a heading measurement.
        self.lighthouse_yaw_update = bool(lighthouse_yaw_update)
        process_noise = {} if process_noise is None else dict(process_noise)
        self.proc_pos = float(process_noise.get("pos", 0.02))
        self.proc_vel = float(process_noise.get("vel", 0.25))
        self.proc_att = float(process_noise.get("att", 0.05))
        self.proc_gyro_bias = float(process_noise.get("gyro_bias", 1.0e-5))
        self.proc_accel = float(process_noise.get("accel", 0.50))
        self.accel_meas_std = float(process_noise.get("accel_meas_std", 0.35))
        self.angle_meas_std = float(
            process_noise.get("angle_meas_std", lighthouse.angle_noise_std)
        )
        if self.angle_meas_std <= 1.0e-9:
            self.angle_meas_std = 1.0e-4
        self.prime(quad)

    @classmethod
    def create(cls, dt, lighthouse, process_noise=None):
        """Build a filter that consumes explicit sensor samples, not a plant."""
        instance = cls.__new__(cls)
        instance.imu = None
        instance.lighthouse = lighthouse
        instance.dt = float(dt)
        instance.lighthouse_yaw_update = False
        process_noise = {} if process_noise is None else dict(process_noise)
        instance.proc_pos = float(process_noise.get("pos", 0.02))
        instance.proc_vel = float(process_noise.get("vel", 0.25))
        instance.proc_att = float(process_noise.get("att", 0.05))
        instance.proc_gyro_bias = float(process_noise.get("gyro_bias", 1.0e-5))
        instance.proc_accel = float(process_noise.get("accel", 0.50))
        instance.accel_meas_std = float(process_noise.get("accel_meas_std", 0.35))
        instance.angle_meas_std = float(
            process_noise.get("angle_meas_std", lighthouse.angle_noise_std)
        )
        if instance.angle_meas_std <= 1.0e-9:
            instance.angle_meas_std = 1.0e-4
        instance.prime_from_state(
            np.zeros(3), np.zeros(3), np.zeros(3), omega=np.zeros(3)
        )
        return instance

    def prime(self, quad):
        """Initialize on the known parked / hover initial state."""
        self.lighthouse.reset()
        omega = (
            self.imu.gyro(quad.omega())
            if self.imu is not None
            else np.asarray(quad.omega(), dtype=float).reshape(3)
        )
        self.prime_from_state(
            quad.position(),
            quad.velocity(),
            quad.attitude(),
            omega=omega,
        )

    def prime_from_state(self, pos, vel, att, omega=None):
        """Initialize from an explicit pose; does not reset the lighthouse."""
        self.pos_est = np.asarray(pos, dtype=float).reshape(3).copy()
        self.vel_est = np.asarray(vel, dtype=float).reshape(3).copy()
        att = np.asarray(att, dtype=float).reshape(3)
        self.q_est = rpy_to_quat(float(att[0]), float(att[1]), float(att[2]))
        self._att_est = att.copy()
        self.gyro_bias_est = np.zeros(3, dtype=float)
        self.P = np.diag(
            np.concatenate(
                [
                    np.full(3, 0.01**2),
                    np.full(3, 0.10**2),
                    np.full(3, 0.05**2),
                    np.full(3, 0.02**2),
                ]
            )
        ).astype(float)
        if omega is None:
            omega = np.zeros(3, dtype=float)
        self._omega_est = np.asarray(omega, dtype=float).reshape(3).copy()
        self._spin_yaw_bias_frozen = False
        self._gyro_saturated = np.zeros(3, dtype=bool)

    @property
    def att_est(self):
        return self._att_est

    @att_est.setter
    def att_est(self, value):
        self._att_est = np.asarray(value, dtype=float).reshape(3).copy()
        self.q_est = rpy_to_quat(
            float(self._att_est[0]),
            float(self._att_est[1]),
            float(self._att_est[2]),
        )

    def _refresh_attitude_view(self):
        self.q_est = normalize_quat(self.q_est)
        self._att_est = np.array(quat_to_rpy(self.q_est), dtype=float)

    def _inject_attitude(self, att_err):
        att_err = np.asarray(att_err, dtype=float).reshape(3)
        if float(np.linalg.norm(att_err)) <= 1.0e-16:
            return
        delta = np.array(
            [1.0, 0.5 * att_err[0], 0.5 * att_err[1], 0.5 * att_err[2]],
            dtype=float,
        )
        self.q_est = normalize_quat(quat_mul(self.q_est, delta))
        self._refresh_attitude_view()

    def _delayed_pose(self):
        delay_s = float(self.lighthouse.delay_s)
        if delay_s <= 1.0e-12:
            return self.pos_est.copy(), quat_to_rot_body_to_world(self.q_est)
        q_delay = quat_mul(self.q_est, _quat_from_rotvec(-self._omega_est * delay_s))
        pos_delay = self.pos_est - self.vel_est * delay_s
        return pos_delay, quat_to_rot_body_to_world(q_delay)

    def _scalar_update(self, observation_row, innovation, meas_std):
        observation_row = np.asarray(observation_row, dtype=float).reshape(_STATE_DIM)
        meas_var = float(meas_std) ** 2
        ph = self.P.dot(observation_row)
        residual_var = float(observation_row.dot(ph)) + meas_var
        if residual_var <= 1.0e-18:
            return False
        if abs(float(innovation)) > _INNOV_GATE * float(np.sqrt(residual_var)):
            return False
        gain = ph / residual_var
        if bool(np.any(self._gyro_saturated)):
            gain[_BIAS] = 0.0
        delta = gain * float(innovation)
        self.pos_est = self.pos_est + delta[_POS]
        self.vel_est = self.vel_est + delta[_VEL]
        self.gyro_bias_est = self.gyro_bias_est + delta[_BIAS]
        self._inject_attitude(delta[_ATT])
        i_kh = np.eye(_STATE_DIM) - np.outer(gain, observation_row)
        self.P = i_kh.dot(self.P).dot(i_kh.T) + meas_var * np.outer(gain, gain)
        self.P = _symmetrize(self.P)
        return True

    def _lighthouse_update(self, sample):
        pos_delay, rot_delay = self._delayed_pose()
        meas_std = max(float(self.angle_meas_std), 1.0e-3)
        # Firmware-style scalar sweep angles on one photodiode.  Large R so
        # they contribute heading/position information without a noisy
        # velocity gain.
        for i_bs, station in enumerate(self.lighthouse.basestations):
            bs_pos = np.asarray(station["position"], dtype=float).reshape(3)
            bs_rot = np.asarray(station["rotation"], dtype=float).reshape(3, 3)
            sensor_body = self.lighthouse.sensors_body[0]
            sensor_world = pos_delay + rot_delay.dot(sensor_body)
            for i_sweep, (rotor_rotation, tilt) in enumerate(self.lighthouse.sweeps):
                if not bool(sample.valid[i_bs, 0, i_sweep]):
                    continue
                predicted = sweep_angle_and_grad(
                    sensor_world, bs_pos, bs_rot, rotor_rotation, tilt
                )
                if predicted is None:
                    continue
                angle_pred, grad_world = predicted
                innovation = (
                    float(sample.angles[i_bs, 0, i_sweep]) - angle_pred + np.pi
                ) % (2.0 * np.pi) - np.pi
                observation = np.zeros(_STATE_DIM, dtype=float)
                observation[_POS] = grad_world
                self._scalar_update(observation, innovation, max(meas_std, 0.02))
                pos_delay, rot_delay = self._delayed_pose()
                sensor_world = pos_delay + rot_delay.dot(sensor_body)
        if self.lighthouse_yaw_update:
            self._lighthouse_yaw_update(sample)

    def _lighthouse_yaw_delta(self, sample):
        """Heading difference (rad) between the deck-sensor pattern seen by the base stations and
        the estimated pose, averaged over base stations with all sensors and sweeps valid."""
        pos, rot = self._delayed_pose()
        normal = rot[:, 2]
        sensors = np.asarray(self.lighthouse.sensors_body, dtype=float)
        if sensors.shape[0] != 4 or len(self.lighthouse.sweeps) != 2:
            return None
        (_, tilt0), (_, tilt1) = self.lighthouse.sweeps
        deltas = []
        for i_bs, station in enumerate(self.lighthouse.basestations):
            if not bool(np.all(sample.valid[i_bs])):
                continue
            bs_pos = np.asarray(station["position"], dtype=float).reshape(3)
            bs_rot = np.asarray(station["rotation"], dtype=float).reshape(3, 3)
            points = []
            for i_s in range(4):
                d_bs = _lh2_direction_bs(sample.angles[i_bs, i_s, 0], sample.angles[i_bs, i_s, 1],
                                         tilt0, tilt1)
                if d_bs is None:
                    break
                d_world = bs_rot.dot(d_bs)
                denom = float(normal.dot(d_world))
                if abs(denom) < 1.0e-6:
                    break
                plane_point = pos + rot.dot(np.array([0.0, 0.0, sensors[i_s, 2]]))
                points.append(bs_pos + float(normal.dot(plane_point - bs_pos)) / denom * d_world)
            if len(points) < 4:
                continue
            points = np.asarray(points)
            for i, j in ((0, 3), (1, 2)):
                measured = points[j] - points[i]
                expected = rot.dot(sensors[j] - sensors[i])
                deltas.append(float(np.arctan2(normal.dot(np.cross(expected, measured)),
                                               expected.dot(measured))))
        if not deltas:
            return None
        return float(np.mean(deltas))

    def _lighthouse_yaw_update(self, sample):
        delta = self._lighthouse_yaw_delta(sample)
        if delta is None:
            return
        observation = np.zeros(_STATE_DIM, dtype=float)
        observation[_ATT] = quat_to_rot_body_to_world(self.q_est)[2, :]
        self._scalar_update(observation, delta, YAW_MEAS_STD)

    def _accel_update(self, accel_meas):
        """Thrust-axis / gravity update, gated to near-hover.

        In flight the accelerometer reads approximately ``T/m`` along body-z,
        not world gravity.  Treating it as ``R^T g`` during a 10–20°
        translation command levels the quaternion and dumps the hover.
        Only fuse when the vehicle is already near-level, slow, and the
        specific-force magnitude is close to ``g``.
        """
        rot_body_to_world = quat_to_rot_body_to_world(self.q_est)
        tilt = float(
            np.arccos(np.clip(float(rot_body_to_world[2, 2]), -1.0, 1.0))
        )
        rate = float(np.linalg.norm(self._omega_est))
        residual_norm = abs(float(np.linalg.norm(accel_meas) - params.g))
        if tilt > np.radians(6.0) or rate > 1.5 or residual_norm > 0.40:
            return
        gravity_world = np.array([0.0, 0.0, params.g], dtype=float)
        predicted = rot_body_to_world.T.dot(gravity_world)
        innovation = np.asarray(accel_meas, dtype=float).reshape(3) - predicted
        h_att = _skew(predicted)
        for axis in range(3):
            observation = np.zeros(_STATE_DIM, dtype=float)
            observation[_ATT] = h_att[axis]
            self._scalar_update(
                observation, float(innovation[axis]), self.accel_meas_std
            )

    def update_from_sensors(
        self,
        gyro_meas,
        accel_meas,
        dt,
        lighthouse_sample=None,
        gyro_saturated=None,
    ):
        """Advance using already-sampled IMU / lighthouse measurements."""
        dt = float(dt)
        gyro_meas = np.asarray(gyro_meas, dtype=float).reshape(3)
        accel_meas = np.asarray(accel_meas, dtype=float).reshape(3)
        if gyro_saturated is None:
            gyro_saturated = np.zeros(3, dtype=bool)
        else:
            gyro_saturated = np.asarray(gyro_saturated, dtype=bool).reshape(3)
        self._gyro_saturated = gyro_saturated
        self._omega_est = gyro_meas - self.gyro_bias_est

        omega = self._omega_est
        # Exact rotation increment.  The first-order step q + dt*qdot followed by
        # normalization under-rotates by about (|omega| dt)^3/12 per step, 2.1 deg/s
        # of heading drift at 48 rad/s with dt = 2 ms.
        rate = float(np.linalg.norm(omega))
        angle = rate * dt
        if rate > 1.0e-12:
            axis = omega / rate
            increment = np.array([np.cos(0.5 * angle), *(np.sin(0.5 * angle) * axis)], dtype=float)
        else:
            increment = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.q_est = normalize_quat(quat_mul(self.q_est, increment))
        rot_body_to_world = quat_to_rot_body_to_world(self.q_est)
        accel_world = rot_body_to_world.dot(accel_meas) - np.array(
            [0.0, 0.0, params.g], dtype=float
        )
        self.vel_est = self.vel_est + dt * accel_world
        self.pos_est = self.pos_est + dt * self.vel_est
        self._refresh_attitude_view()

        transition = np.eye(_STATE_DIM)
        transition[_POS, _VEL] = np.eye(3) * dt
        transition[_VEL, _ATT] = -rot_body_to_world.dot(_skew(accel_meas)) * dt
        transition[_ATT, _ATT] = np.eye(3) - _skew(omega) * dt
        transition[_ATT, _BIAS] = -np.eye(3) * dt
        process = np.zeros((_STATE_DIM, _STATE_DIM), dtype=float)
        process[_POS, _POS] = np.eye(3) * (self.proc_pos**2) * dt
        process[_VEL, _VEL] = np.eye(3) * (self.proc_vel**2) * dt
        att_intensity = self.proc_att**2
        if bool(np.any(gyro_saturated)):
            att_intensity = att_intensity * 100.0
        process[_ATT, _ATT] = np.eye(3) * att_intensity * dt
        process[_BIAS, _BIAS] = np.eye(3) * (self.proc_gyro_bias**2) * dt
        self.P = transition.dot(self.P).dot(transition.T) + process
        self.P = _symmetrize(self.P)

        # Specific force in flight is T/m along body-z, not R^T g.  The
        # gated gravity update is kept for analysis but is not applied on
        # the closed-loop path; tilt comes from gyro quaternion kinematics.
        if lighthouse_sample is not None:
            self._lighthouse_update(lighthouse_sample)
        self._omega_est = gyro_meas - self.gyro_bias_est
        omega_z = abs(float(self._omega_est[2]))
        self._spin_yaw_bias_frozen = omega_z > 20.0

    def update(self, quad, dt, t, external_force_world, wind_vel_world=None):
        if self.imu is None:
            raise RuntimeError("update() requires an IMU; use update_from_sensors")
        gyro_meas = self.imu.gyro(quad.omega())
        accel_meas = self.imu.accel(quad, external_force_world, wind_vel_world)
        sample = self.lighthouse.update(quad, t)
        self.update_from_sensors(
            gyro_meas,
            accel_meas,
            dt,
            lighthouse_sample=sample,
            gyro_saturated=self.imu.last_gyro_saturated,
        )

    def view_rotation(self):
        return quat_to_rot_body_to_world(self.q_est)

    def view(self, quad):
        from crazyflie_sim.estimation.estimator import EstimatedQuadView

        return EstimatedQuadView(
            quad, self.pos_est, self.vel_est, self.att_est, self._omega_est
        )
