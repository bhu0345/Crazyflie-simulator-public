
import numpy as np
from math import sin, cos, asin, atan2


def RotToRPY(R):
    phi = asin(R[1, 2])
    theta = atan2(-R[0, 2] / cos(phi), R[2, 2] / cos(phi))
    psi = atan2(-R[1, 0] / cos(phi), R[1, 1] / cos(phi))
    return phi, theta, psi


def RPYToRot(phi, theta, psi):
    """
    phi, theta, psi = roll, pitch , yaw

    Returns the world-to-body rotation matrix used throughout the simulator.
    """
    return np.array(
        [
            [
                cos(psi) * cos(theta) - sin(phi) * sin(psi) * sin(theta),
                cos(theta) * sin(psi) + cos(psi) * sin(phi) * sin(theta),
                -cos(phi) * sin(theta),
            ],
            [-cos(phi) * sin(psi), cos(phi) * cos(psi), sin(phi)],
            [
                cos(psi) * sin(theta) + cos(theta) * sin(phi) * sin(psi),
                sin(psi) * sin(theta) - cos(psi) * cos(theta) * sin(phi),
                cos(phi) * cos(theta),
            ],
        ]
    )


def normalize_quat(q):
    q = np.asarray(q, dtype=float).reshape(4,)
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-16:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm


def quat_mul(q1, q2):
    """Hamilton product; both quaternions are ``[w, x, y, z]``."""
    w1, x1, y1, z1 = np.asarray(q1, dtype=float).reshape(4,)
    w2, x2, y2, z2 = np.asarray(q2, dtype=float).reshape(4,)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_to_rot_body_to_world(q):
    """Rotation matrix that maps body vectors into the world frame."""
    w, x, y, z = normalize_quat(q)
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )


def rot_body_to_world_to_quat(R):
    """Convert a body-to-world rotation matrix to a unit quaternion."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return normalize_quat(np.array([w, x, y, z], dtype=float))


def rpy_to_quat(phi, theta, psi):
    """Quaternion matching ``RPYToRot(phi, theta, psi).T`` (body-to-world)."""
    return rot_body_to_world_to_quat(RPYToRot(phi, theta, psi).T)


def quat_to_rpy(q):
    """Roll/pitch/yaw consistent with ``RotToRPY(quat_to_rot_body_to_world(q).T)``."""
    return RotToRPY(quat_to_rot_body_to_world(q).T)


def quat_derivative(q, omega_body):
    """``q_dot = 0.5 * q ⊗ [0, ω_body]`` for a body-to-world quaternion."""
    omega_body = np.asarray(omega_body, dtype=float).reshape(3,)
    omega_quat = np.array(
        [0.0, omega_body[0], omega_body[1], omega_body[2]], dtype=float
    )
    return 0.5 * quat_mul(q, omega_quat)
