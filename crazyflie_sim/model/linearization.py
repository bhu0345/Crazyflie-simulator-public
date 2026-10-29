import numpy as np

import crazyflie_sim.model.params as params
from crazyflie_sim.model.actuator_profile import get_actuator_profile
from crazyflie_sim.model.quadcopter import Quadcopter

# Full nonlinear simulator state order.
STATE_ORDER = (
    "x", "y", "z",
    "xdot", "ydot", "zdot",
    "phi", "theta", "psi",
    "p", "q", "r",
)

# Input order for linearization.
INPUT_ORDER = ("F", "Mx", "My", "Mz")


def hover_equilibrium(pos=(0.0, 0.0, 0.0), attitude=(0.0, 0.0, 0.0), actuator_profile=None):
    """Return equilibrium state/input around hover.

    The nominal input is total thrust equal to weight and zero body moments.
    """
    x_eq = np.zeros(12, dtype=float)

    x_eq[0:3] = np.asarray(pos, dtype=float)
    x_eq[6:9] = np.asarray(attitude, dtype=float)

    mass = get_actuator_profile(actuator_profile).mass
    u_eq = np.array([mass * params.g, 0.0, 0.0, 0.0], dtype=float)
    return x_eq, u_eq


def _dynamics(state, control, external_force_world, actuator_profile=None):
    """Continuous-time nonlinear dynamics x_dot = f(x,u)."""
    # state_dot() is stateless; this helper only gives us access to model equations.
    quad = Quadcopter((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), actuator_profile=actuator_profile)
    F = float(control[0])
    M = np.asarray(control[1:4], dtype=float).reshape(3, 1)
    return quad.state_dot(
        np.asarray(state, dtype=float),
        0.0,
        F,
        M,
        np.asarray(external_force_world, dtype=float).reshape(3,),
    )


def linearize(state_eq, input_eq, state_eps=1e-6, input_eps=1e-6, external_force_world=None,
              actuator_profile=None):
    """Numerically linearize x_dot = f(x,u) around (state_eq, input_eq).

    Returns matrices (A, B) of:
        d/dt(delta_x) = A delta_x + B delta_u
    where delta_x = x - state_eq and delta_u = u - input_eq.
    """
    x0 = np.asarray(state_eq, dtype=float).copy()
    u0 = np.asarray(input_eq, dtype=float).copy()
    if x0.shape != (12,):
        raise ValueError("state_eq must have shape (12,).")
    if u0.shape != (4,):
        raise ValueError("input_eq must have shape (4,).")

    if external_force_world is None:
        external_force_world = np.zeros(3, dtype=float)
    else:
        external_force_world = np.asarray(external_force_world, dtype=float).reshape(3,)

    n = x0.size
    m = u0.size
    A = np.zeros((n, n), dtype=float)
    B = np.zeros((n, m), dtype=float)

    state_step = np.full(n, float(state_eps), dtype=float) if np.isscalar(state_eps) else np.asarray(state_eps, dtype=float)
    input_step = np.full(m, float(input_eps), dtype=float) if np.isscalar(input_eps) else np.asarray(input_eps, dtype=float)

    if state_step.shape != (n,):
        raise ValueError("state_eps must be scalar or shape (12,).")
    if input_step.shape != (m,):
        raise ValueError("input_eps must be scalar or shape (4,).")

    # A = df/dx via central difference
    for i in range(n):
        dx = np.zeros(n, dtype=float)
        h = max(abs(state_step[i]), 1e-9)
        dx[i] = h
        f_plus = _dynamics(x0 + dx, u0, external_force_world, actuator_profile)
        f_minus = _dynamics(x0 - dx, u0, external_force_world, actuator_profile)
        A[:, i] = (f_plus - f_minus) / (2.0 * h)

    # B = df/du via central difference
    for j in range(m):
        du = np.zeros(m, dtype=float)
        h = max(abs(input_step[j]), 1e-9)
        du[j] = h
        f_plus = _dynamics(x0, u0 + du, external_force_world, actuator_profile)
        f_minus = _dynamics(x0, u0 - du, external_force_world, actuator_profile)
        B[:, j] = (f_plus - f_minus) / (2.0 * h)

    return A, B


def linearize_hover(
    pos=(0.0, 0.0, 0.0),
    attitude=(0.0, 0.0, 0.0),
    state_eps=1e-6,
    input_eps=1e-6,
    actuator_profile=None,
):
    """Convenience wrapper for hover linearization of ``actuator_profile``."""
    x_eq, u_eq = hover_equilibrium(pos=pos, attitude=attitude, actuator_profile=actuator_profile)
    A, B = linearize(
        state_eq=x_eq,
        input_eq=u_eq,
        state_eps=state_eps,
        input_eps=input_eps,
        external_force_world=np.zeros(3),
        actuator_profile=actuator_profile,
    )
    return A, B, x_eq, u_eq
