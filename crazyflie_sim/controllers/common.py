import numpy as np

import crazyflie_sim.model.params as params


def clip_sym(x, lim):
    return float(np.clip(x, -lim, lim))


def wrap_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def wrap_deg(deg):
    return (deg + 180.0) % 360.0 - 180.0


def translational_pid_target_thrust(
    quad,
    des_state,
    i_error_pos,
    dt,
    mass,
    kp_xy,
    kd_xy,
    ki_xy,
    i_range_xy,
    kp_z,
    kd_z,
    ki_z,
    i_range_z,
):
    """Translational PID -> world-frame target thrust vector.

    Position/velocity PID with clipped position integrators, shared by the
    Mellinger and yaw-released-fault backends. The integrator array
    ``i_error_pos`` (shape (3,)) is owned by the caller and updated in place.
    """
    dt = max(float(dt), 1.0e-6)
    pos = np.asarray(quad.position(), dtype=float).reshape(3,)
    vel = np.asarray(quad.velocity(), dtype=float).reshape(3,)

    r_error = np.asarray(des_state.pos, dtype=float).reshape(3,) - pos
    v_error = np.asarray(des_state.vel, dtype=float).reshape(3,) - vel

    i_error_pos[0] = clip_sym(i_error_pos[0] + r_error[0] * dt, i_range_xy)
    i_error_pos[1] = clip_sym(i_error_pos[1] + r_error[1] * dt, i_range_xy)
    i_error_pos[2] = clip_sym(i_error_pos[2] + r_error[2] * dt, i_range_z)

    acc_ff = np.asarray(des_state.acc, dtype=float).reshape(3,)
    target_thrust = np.array(
        [
            mass * acc_ff[0]
            + kp_xy * r_error[0]
            + kd_xy * v_error[0]
            + ki_xy * i_error_pos[0],
            mass * acc_ff[1]
            + kp_xy * r_error[1]
            + kd_xy * v_error[1]
            + ki_xy * i_error_pos[1],
            mass * (acc_ff[2] + params.g)
            + kp_z * r_error[2]
            + kd_z * v_error[2]
            + ki_z * i_error_pos[2],
        ],
        dtype=float,
    )
    return target_thrust
