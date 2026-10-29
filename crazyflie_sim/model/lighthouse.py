"""Lighthouse sweep-angle sensor: two base stations, four deck photodiodes.

The measurement is the firmware-style scalar sweep angle, not delayed Euler
RPY.  Geometry follows the Crazyflie lighthouse Kalman model:

    https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/lighthouse/kalman_measurement_model/

LH1 uses a horizontal rotor (identity) and a vertical rotor
``R_vert = [[1,0,0],[0,0,1],[0,-1,0]]``.  The closed-loop plant is
Lighthouse V2: two planes in the same rotor frame with firmware tilts
``±π/6``, ~50 Hz positioning, and one sample of pipeline delay (~20 ms).
Those hardware numbers live here, not in experiment YAML.

Pipeline delay is one V2 sample (~20 ms).  White sweep-angle jitter stays
milliradian.  A once-per-run unmodeled angle bias is sized so the navigation
EKF's Euclidean position error versus plant truth sits in the Taffanel et al.
2021 Lighthouse+EKF accuracy class (mean/median about 2--4 cm, flight
outliers about 5 cm).  That bias stands for calibration/geometry error.
The sensor never emits Euler RPY.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from crazyflie_sim.utils.utils import RPYToRot

# Lighthouse V2 hardware.  Not experiment-config knobs.
LH2_PROTOCOL = "lh2"
LH2_TILT_RAD = float(np.pi / 6.0)
LH2_RATE_HZ = 50.0
LH2_DELAY_STEPS = 1
LH2_ANGLE_NOISE_STD = 0.0008
# Unmodeled constant-in-run sweep-angle bias (rad).  White jitter remains
# LH2_ANGLE_NOISE_STD.  0.010 rad at a ~3 m station range is centimetre-class
# pose error after the EKF.
LH2_ANGLE_BIAS_STD = 0.010

# Approximate Crazyflie lighthouse-deck photodiode offsets (meters, body).
DEFAULT_SENSORS_BODY = np.array(
    [
        [0.0165, 0.0165, 0.005],
        [0.0165, -0.0165, 0.005],
        [-0.0165, 0.0165, 0.005],
        [-0.0165, -0.0165, 0.005],
    ],
    dtype=float,
)

# Vertical LH1 drum: maps (x, y, z)_bs -> (x, z, -y).
R_VERT = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=float,
)

_MIN_FRONT_X = 0.05
_MIN_RADIAL_SQ = 1.0e-10


def lookat_basestation_rotation(position, target, up=None):
    """Rotation mapping base-station axes into the world (columns are axes).

    Base-station +x looks at ``target``.  +z is the projection of ``up``
    orthogonal to the look axis; +y completes the right-handed frame.
    """
    position = np.asarray(position, dtype=float).reshape(3)
    target = np.asarray(target, dtype=float).reshape(3)
    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        up = np.asarray(up, dtype=float).reshape(3)
    x_axis = target - position
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= 1.0e-9:
        raise ValueError("basestation look-at target coincides with position")
    x_axis = x_axis / x_norm
    z_axis = up - x_axis * float(np.dot(up, x_axis))
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm <= 1.0e-9:
        raise ValueError("basestation up vector is parallel to the look axis")
    z_axis = z_axis / z_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / float(np.linalg.norm(y_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


def default_basestations():
    """Two opposite-corner stations looking at a 1 m hover workspace."""
    specs = (
        (np.array([-2.0, -2.0, 2.4]), np.array([0.0, 0.0, 1.0])),
        (np.array([2.0, 2.0, 2.4]), np.array([0.0, 0.0, 1.0])),
    )
    return tuple(
        {"position": pos.copy(), "rotation": lookat_basestation_rotation(pos, target)}
        for pos, target in specs
    )


def sweep_frames(protocol, tilt_rad):
    """Return ``((R_r, tilt), ...)`` for one base station."""
    protocol = str(protocol).strip().lower()
    tilt = float(tilt_rad)
    if protocol == "lh1":
        return ((np.eye(3, dtype=float), 0.0), (R_VERT.copy(), 0.0))
    if protocol == "lh2":
        if abs(tilt) <= 1.0e-12:
            raise ValueError(
                "lighthouse protocol 'lh2' requires nonzero tilt_rad "
                "(firmware uses ±π/6)"
            )
        identity = np.eye(3, dtype=float)
        return ((identity, tilt), (identity.copy(), -tilt))
    raise ValueError("lighthouse protocol must be 'lh1' or 'lh2', got {!r}".format(protocol))


def sweep_angle_and_grad(sensor_world, bs_position, bs_rotation, rotor_rotation, tilt_rad):
    """Predicted sweep angle and world-frame position gradient, or None.

    ``α_p = atan2(y, x) + asin(z tan t / r)`` in the rotor frame, matching
    the firmware measurement model.  The gradient is ``dα/ds_world``.
    """
    sensor_world = np.asarray(sensor_world, dtype=float).reshape(3)
    bs_position = np.asarray(bs_position, dtype=float).reshape(3)
    bs_rotation = np.asarray(bs_rotation, dtype=float).reshape(3, 3)
    rotor_rotation = np.asarray(rotor_rotation, dtype=float).reshape(3, 3)
    sensor_bs = bs_rotation.T.dot(sensor_world - bs_position)
    if float(sensor_bs[0]) < _MIN_FRONT_X:
        return None
    sensor_rotor = rotor_rotation.dot(sensor_bs)
    x, y, z = (float(sensor_rotor[0]), float(sensor_rotor[1]), float(sensor_rotor[2]))
    radial_sq = x * x + y * y
    if radial_sq < _MIN_RADIAL_SQ:
        return None
    tan_t = float(np.tan(tilt_rad))
    angle = float(np.arctan2(y, x))
    if abs(tan_t) > 1.0e-12:
        radial = float(np.sqrt(radial_sq))
        asin_arg = z * tan_t / radial
        if abs(asin_arg) >= 1.0:
            return None
        angle = angle + float(np.arcsin(asin_arg))
        disc = radial_sq - (z * tan_t) ** 2
        if disc <= 1.0e-16:
            return None
        q_term = tan_t / float(np.sqrt(disc))
        grad_rotor = np.array(
            [
                (-y - x * z * q_term) / radial_sq,
                (x - y * z * q_term) / radial_sq,
                q_term,
            ],
            dtype=float,
        )
    else:
        grad_rotor = np.array([-y / radial_sq, x / radial_sq, 0.0], dtype=float)
    grad_world = bs_rotation.dot(rotor_rotation.T.dot(grad_rotor))
    return angle, grad_world


def _wrap_pi(angle):
    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _sweeps_are_lh2(sweeps):
    if len(sweeps) != 2:
        return False
    (rotor0, tilt0), (rotor1, tilt1) = sweeps
    if abs(float(tilt0)) <= 1.0e-12 or abs(float(tilt0) + float(tilt1)) > 1.0e-9:
        return False
    identity = np.eye(3, dtype=float)
    return np.allclose(rotor0, identity) and np.allclose(rotor1, identity)


def _lh1_direction_bs(azimuth, elevation):
    azimuth = float(np.clip(azimuth, -1.2, 1.2))
    elevation = float(np.clip(elevation, -1.2, 1.2))
    direction = np.array([1.0, np.tan(azimuth), np.tan(elevation)], dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-9:
        return None
    return direction / norm


def _lh2_direction_bs(alpha0, alpha1, tilt0, tilt1):
    """Ray in the base-station frame from an LH2 ±tilt angle pair.

    ``α± = atan2(y, x) ± asin(z tan t / r)``, so the circular mean is
    azimuth and the half-difference recovers ``z / r``.
    """
    tilt0 = float(tilt0)
    tilt1 = float(tilt1)
    alpha0 = float(alpha0)
    alpha1 = float(alpha1)
    if tilt0 < 0.0:
        alpha0, alpha1 = alpha1, alpha0
        tilt0, tilt1 = tilt1, tilt0
    tan_t = float(np.tan(tilt0))
    if abs(tan_t) <= 1.0e-12:
        return None
    azimuth = _wrap_pi(alpha0 + 0.5 * _wrap_pi(alpha1 - alpha0))
    delta = 0.5 * _wrap_pi(alpha0 - alpha1)
    cos_az = float(np.cos(azimuth))
    if cos_az <= 1.0e-6:
        return None
    direction = np.array(
        [1.0, np.tan(azimuth), float(np.sin(delta)) / (tan_t * cos_az)],
        dtype=float,
    )
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-9:
        return None
    return direction / norm


def triangulate_position(
    angles,
    valid,
    basestations,
    sweeps,
    sensors_body=None,
    rot_body_to_world=None,
    sensor_index=0,
):
    """Closest-point triangulation from two base-station rays.

    LH1 rays use azimuth / elevation.  LH2 rays invert the firmware
    ``±tilt`` plane pair.  Uses one photodiode (not an angle average) so
    the intersected point is that sensor.  Subtracting ``R @ s_body``
    recovers the COM.  Two or more rays return their midpoint; one ray
    returns None.  This is a position-only measurement, not Euler RPY.
    """
    origins = []
    directions = []
    lh2 = _sweeps_are_lh2(sweeps)
    for i_bs, station in enumerate(basestations):
        if len(sweeps) < 2:
            continue
        if not (
            bool(valid[i_bs, sensor_index, 0]) and bool(valid[i_bs, sensor_index, 1])
        ):
            continue
        if lh2:
            direction_bs = _lh2_direction_bs(
                angles[i_bs, sensor_index, 0],
                angles[i_bs, sensor_index, 1],
                sweeps[0][1],
                sweeps[1][1],
            )
        else:
            direction_bs = _lh1_direction_bs(
                angles[i_bs, sensor_index, 0],
                angles[i_bs, sensor_index, 1],
            )
        if direction_bs is None:
            continue
        bs_rot = np.asarray(station["rotation"], dtype=float).reshape(3, 3)
        origins.append(np.asarray(station["position"], dtype=float).reshape(3))
        directions.append(bs_rot.dot(direction_bs))
    if len(origins) < 2:
        return None
    # Midpoint of the closest points on the first two rays.
    p1, p2 = origins[0], origins[1]
    d1, d2 = directions[0], directions[1]
    d1d2 = float(np.dot(d1, d2))
    denom = 1.0 - d1d2 * d1d2
    if abs(denom) <= 1.0e-8:
        return None
    r = p2 - p1
    t1 = float(np.dot(r, d1) - np.dot(r, d2) * d1d2) / denom
    t2 = float(np.dot(r, d1) * d1d2 - np.dot(r, d2)) / denom
    closest1 = p1 + t1 * d1
    closest2 = p2 + t2 * d2
    sensor_world = 0.5 * (closest1 + closest2)
    if sensors_body is None or rot_body_to_world is None:
        return sensor_world
    offset = np.asarray(rot_body_to_world, dtype=float).reshape(3, 3).dot(
        np.asarray(sensors_body, dtype=float).reshape(-1, 3)[sensor_index]
    )
    return sensor_world - offset


def predict_sweep_angles(pos_world, rot_body_to_world, sensors_body, basestations, sweeps):
    """Noise-free ``(n_bs, n_sensors, n_sweeps)`` angles and a validity mask."""
    pos_world = np.asarray(pos_world, dtype=float).reshape(3)
    rot_body_to_world = np.asarray(rot_body_to_world, dtype=float).reshape(3, 3)
    sensors_body = np.asarray(sensors_body, dtype=float).reshape(-1, 3)
    n_bs = len(basestations)
    n_sensors = sensors_body.shape[0]
    n_sweeps = len(sweeps)
    angles = np.full((n_bs, n_sensors, n_sweeps), np.nan, dtype=float)
    valid = np.zeros((n_bs, n_sensors, n_sweeps), dtype=bool)
    for i_bs, station in enumerate(basestations):
        bs_pos = np.asarray(station["position"], dtype=float).reshape(3)
        bs_rot = np.asarray(station["rotation"], dtype=float).reshape(3, 3)
        for i_sensor, sensor_body in enumerate(sensors_body):
            sensor_world = pos_world + rot_body_to_world.dot(sensor_body)
            for i_sweep, (rotor_rotation, tilt) in enumerate(sweeps):
                predicted = sweep_angle_and_grad(
                    sensor_world, bs_pos, bs_rot, rotor_rotation, tilt
                )
                if predicted is None:
                    continue
                angles[i_bs, i_sensor, i_sweep] = predicted[0]
                valid[i_bs, i_sensor, i_sweep] = True
    return angles, valid


@dataclass(frozen=True)
class SweepAngleSample:
    """Delayed lighthouse sweep-angle frame.  No Euler RPY."""

    angles: np.ndarray
    valid: np.ndarray


class Lighthouse:
    """Rate-gated sweep-angle sensor with a mocap-style delay ring."""

    TOL_S = 1e-9

    def __init__(
        self,
        rng,
        rate_hz=None,
        angle_noise_std=None,
        angle_bias_std=None,
        delay_steps=None,
        basestations=None,
        sensors_body=None,
        tilt_rad=None,
        protocol=None,
        bias_shared_across_sensors=False,
    ):
        # True draws the constant-in-run angle bias once per base station and
        # sweep and applies it to all four deck sensors (a base-station
        # calibration error is the same for photodiodes 3 cm apart).  False
        # draws it independently per sensor, under which the sensors' relative
        # positions, and so the heading, cannot be recovered.
        self.bias_shared_across_sensors = bool(bias_shared_across_sensors)
        self.rate_hz = float(LH2_RATE_HZ if rate_hz is None else rate_hz)
        if self.rate_hz <= 0.0:
            raise ValueError("lighthouse rate_hz must be positive")
        self.angle_noise_std = float(
            LH2_ANGLE_NOISE_STD if angle_noise_std is None else angle_noise_std
        )
        if self.angle_noise_std < 0.0:
            raise ValueError("lighthouse angle_noise_std must be non-negative")
        self.angle_bias_std = float(
            LH2_ANGLE_BIAS_STD if angle_bias_std is None else angle_bias_std
        )
        if self.angle_bias_std < 0.0:
            raise ValueError("lighthouse angle_bias_std must be non-negative")
        self.delay_steps = int(
            LH2_DELAY_STEPS if delay_steps is None else delay_steps
        )
        if self.delay_steps < 0:
            raise ValueError("lighthouse delay_steps must be non-negative")
        self.rng = rng
        self.tilt_rad = float(LH2_TILT_RAD if tilt_rad is None else tilt_rad)
        self.protocol = str(
            LH2_PROTOCOL if protocol is None else protocol
        ).strip().lower()
        self.sweeps = sweep_frames(self.protocol, self.tilt_rad)
        self.sensors_body = np.asarray(
            DEFAULT_SENSORS_BODY if sensors_body is None else sensors_body,
            dtype=float,
        ).reshape(-1, 3).copy()
        if basestations is None:
            self.basestations = default_basestations()
        else:
            self.basestations = tuple(basestations)
        if len(self.basestations) < 1:
            raise ValueError("lighthouse needs at least one basestation")
        self._period = 1.0 / self.rate_hz
        self._buffer = deque(maxlen=self.delay_steps + 1)
        self._next_sample_time = 0.0
        self._angle_bias = None
        self._draw_angle_bias()

    @property
    def delay_s(self):
        if self.rate_hz <= 1.0e-9:
            return 0.0
        return float(self.delay_steps) / float(self.rate_hz)

    def _draw_angle_bias(self):
        shape = (
            len(self.basestations),
            int(self.sensors_body.shape[0]),
            len(self.sweeps),
        )
        if self.angle_bias_std <= 0.0:
            self._angle_bias = np.zeros(shape, dtype=float)
            return
        if self.bias_shared_across_sensors:
            per_station = self.rng.normal(0.0, self.angle_bias_std, size=(shape[0], 1, shape[2]))
            self._angle_bias = np.repeat(per_station, shape[1], axis=1)
            return
        self._angle_bias = self.rng.normal(
            0.0, self.angle_bias_std, size=shape
        )

    def reset(self):
        self._buffer.clear()
        self._next_sample_time = 0.0
        self._draw_angle_bias()

    def predict(self, pos_world, rot_body_to_world):
        """Noise-free geometric prediction for the navigation EKF."""
        return predict_sweep_angles(
            pos_world,
            rot_body_to_world,
            self.sensors_body,
            self.basestations,
            self.sweeps,
        )

    def update(self, quad, t):
        """Possibly sample the quad; returns a :class:`SweepAngleSample` or None."""
        t = float(t)
        if t + self.TOL_S < self._next_sample_time:
            return None
        idx = int(np.floor((t + self.TOL_S) / self._period))
        self._next_sample_time = (idx + 1) * self._period

        rot_body_to_world = RPYToRot(*[float(a) for a in quad.attitude()]).T
        angles, valid = self.predict(quad.position(), rot_body_to_world)
        noise = self.rng.normal(0.0, self.angle_noise_std, size=angles.shape)
        measured = np.where(valid, angles + self._angle_bias + noise, np.nan)
        sample = SweepAngleSample(measured, valid.copy())
        self._buffer.append(sample)
        oldest = self._buffer[0]
        return SweepAngleSample(oldest.angles.copy(), oldest.valid.copy())
