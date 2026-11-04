
import numpy as np

from crazyflie_sim.controllers.base import DesiredState  # noqa: F401  (re-exported)


def _wrap_pi(angle):
    """Wrap angle to [-pi, pi].
    avoid discontinuity when interpolating yaw angles across the -pi/pi boundary.
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def format_waypoints(waypoints):
    """Return waypoints as Nx4 [x, y, z, yaw(rad)].

    If Nx3 is provided, yaw defaults to 0.0 for every waypoint.
    """
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] not in (3, 4):
        raise ValueError("Waypoints must be an Nx3 or Nx4 array/list.")
    if waypoints.shape[0] < 2:
        raise ValueError("Need at least 2 waypoints.")

    if waypoints.shape[1] == 3:
        zeros = np.zeros((waypoints.shape[0], 1), dtype=float)
        waypoints = np.hstack((waypoints, zeros))
    return waypoints

def get_helix_waypoints(t, n, yaw=0.0):
    """Generate n helix waypoints with explicit yaw.

    Output shape is [n, 4] with columns [x, y, z, yaw(rad)].
    """
    waypoints_t = np.linspace(t, t + 2*np.pi, n)
    x = 0.5*np.cos(waypoints_t)
    y = 0.5*np.sin(waypoints_t)
    z = waypoints_t
    yaw_col = np.full((n,), float(yaw), dtype=float)

    return np.stack((x, y, z, yaw_col), axis=-1)

def _quintic_hermite(s, T, p0, p1, v0, v1):
    """(pos, vel, acc) at s in [0, 1] of the quintic from (p0, v0) to (p1, v1)
    over T seconds with zero acceleration at both ends."""
    h = (10*s**3 - 15*s**4 + 6*s**5, s - 6*s**3 + 8*s**4 - 3*s**5, -4*s**3 + 7*s**4 - 3*s**5)
    dh = (30*s**2 - 60*s**3 + 30*s**4, 1 - 18*s**2 + 32*s**3 - 15*s**4, -12*s**2 + 28*s**3 - 15*s**4)
    ddh = (60*s - 180*s**2 + 120*s**3, -36*s + 96*s**2 - 60*s**3, -24*s + 84*s**2 - 60*s**3)
    delta = p1 - p0
    pos = p0 + delta * h[0] + T * (v0 * h[1] + v1 * h[2])
    vel = delta * dh[0] / T + v0 * dh[1] + v1 * dh[2]
    acc = delta * ddh[0] / T**2 + (v0 * ddh[1] + v1 * ddh[2]) / T
    return pos, vel, acc

def minimum_jerk(t, duration, start, end):
    """(pos, vel, acc) of the rest-to-rest minimum-jerk move start -> end."""
    start = np.asarray(start, dtype=float)
    zero = np.zeros_like(start)
    s = float(np.clip(t / duration, 0.0, 1.0))
    return _quintic_hermite(s, duration, start, np.asarray(end, dtype=float), zero, zero)

def timed_waypoint_velocities(times, waypoints):
    """Nx4 velocities at the waypoints of a timed trajectory (rows [x, y, z,
    yaw] with yaw unwrapped).  Each axis moves monotonically between two
    waypoints: it stops where it reverses or holds, and otherwise passes at
    the shape-preserving (PCHIP) slope, capped at twice the smaller adjacent
    one so the quintic segments cannot overshoot."""
    h = np.diff(times)[:, None]
    d = np.diff(waypoints, axis=0) / h
    v = np.zeros_like(waypoints)
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    same = d[:-1] * d[1:] > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = (w1 + w2) / (w1 / d[:-1] + w2 / d[1:])
    cap = 2.0 * np.minimum(np.abs(d[:-1]), np.abs(d[1:]))
    v[1:-1] = np.where(same, np.sign(d[1:]) * np.minimum(np.abs(mean), cap), 0.0)
    return v

def generate_timed_trajectory(t, times, waypoints, velocities):
    """Desired state at t of the path that reaches waypoint i (row [x, y, z,
    yaw], yaw unwrapped) at times[i], at rest at both ends; velocities from
    timed_waypoint_velocities.  Before times[0] and after times[-1] it holds."""
    t = float(t)
    if t <= times[0] or t >= times[-1]:
        row = waypoints[0] if t <= times[0] else waypoints[-1]
        return DesiredState(row[:3].copy(), np.zeros(3), np.zeros(3), _wrap_pi(row[3]), 0.0)
    k = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2))
    T = times[k + 1] - times[k]
    pos, vel, acc = _quintic_hermite((t - times[k]) / T, T, waypoints[k], waypoints[k + 1],
                                     velocities[k], velocities[k + 1])
    return DesiredState(pos[:3], vel[:3], acc[:3], _wrap_pi(pos[3]), vel[3])

def get_MST_coefficients(waypoints):
    # Segment durations are proportional to length; a common speed cancels
    # from the continuity constraints in normalized segment coordinates.
    waypoints = format_waypoints(waypoints)
    durations = np.linalg.norm(np.diff(waypoints[:, :3], axis=0), axis=1)
    coeff_x = MST(waypoints[:,0], durations).transpose()[0]
    coeff_y = MST(waypoints[:,1], durations).transpose()[0]
    coeff_z = MST(waypoints[:,2], durations).transpose()[0]
    return (coeff_x, coeff_y, coeff_z)

def generate_trajectory(t, v, waypoints, coeff_x=None, coeff_y=None, coeff_z=None):
    """ The function takes known number of waypoints and time, then generates a
    minimum snap trajectory which goes through each waypoint. The output is
    the desired state associated with the next waypont for the time t.
    waypoints is [N,3] or [N,4] matrix, waypoints = [[x,y,z] or [x,y,z,yaw]].
    v is velocity in m/s
    """
    waypoints = format_waypoints(waypoints)
    if v <= 0.0:
        raise ValueError("Trajectory speed v must be > 0.")

    if coeff_x is None and coeff_y is None and coeff_z is None:
        coeff_x, coeff_y, coeff_z = get_MST_coefficients(waypoints)
    elif coeff_x is None or coeff_y is None or coeff_z is None:
        raise ValueError("Provide all of coeff_x/coeff_y/coeff_z or none of them.")

    yawdot = 0.0
    pos = np.zeros(3)
    acc = np.zeros(3)
    vel = np.zeros(3)

    # distance vector array, represents each segment's distance
    distance = waypoints[0:-1, :3] - waypoints[1:, :3]
    # T is now each segment's travel time
    T = (1.0 / v) * np.sqrt(distance[:,0]**2 + distance[:,1]**2 + distance[:,2]**2)
    T = np.maximum(T, 1e-6)
    # accumulated time
    S = np.zeros(len(T) + 1)
    S[1:] = np.cumsum(T)
    yaw_points = np.unwrap(waypoints[:, 3])
    t = float(t)

    # prepare the next desired state
    if t <= 0.0:
        pos = waypoints[0, :3]
        yaw = _wrap_pi(yaw_points[0])
        yawdot = 0.0
    # stay hover at the last waypoint position
    elif t >= S[-1]:
        pos = waypoints[-1, :3]
        yaw = _wrap_pi(yaw_points[-1])
        yawdot = 0.0
    else:
        # find which segment current t belongs to
        t_index = int(np.searchsorted(S, t, side="right") - 1)
        t_index = int(np.clip(t_index, 0, len(T) - 1))

        # scaled time
        scale = (t - S[t_index]) / T[t_index]
        start = 8 * t_index
        end = 8 * (t_index + 1)

        t0 = get_poly_cc(8, 0, scale)
        pos = np.array([coeff_x[start:end].dot(t0), coeff_y[start:end].dot(t0), coeff_z[start:end].dot(t0)])

        t1 = get_poly_cc(8, 1, scale)
        # chain rule applied
        vel = np.array([coeff_x[start:end].dot(t1), coeff_y[start:end].dot(t1), coeff_z[start:end].dot(t1)]) * (1.0 / T[t_index])

        t2 = get_poly_cc(8, 2, scale)
        # chain rule applied
        acc = np.array([coeff_x[start:end].dot(t2), coeff_y[start:end].dot(t2), coeff_z[start:end].dot(t2)]) * (1.0 / T[t_index]**2)

        # User-defined yaw interpolation along the same segment schedule.
        yaw_start = yaw_points[t_index]
        yaw_end = yaw_points[t_index + 1]
        yaw = _wrap_pi(yaw_start + scale * (yaw_end - yaw_start))
        yawdot = (yaw_end - yaw_start) / T[t_index]
    return DesiredState(pos, vel, acc, yaw, yawdot)

def get_poly_cc(n, k, t):
    """ This is a helper function to get the coeffitient of coefficient for n-th
        order polynomial with k-th derivative at time t.
    """
    assert (n > 0 and k >= 0), "order and derivative must be positive."

    cc = np.ones(n)
    D  = np.linspace(0, n-1, n)

    for i in range(n):
        for j in range(k):
            cc[i] = cc[i] * D[i]
            D[i] = D[i] - 1
            if D[i] == -1:
                D[i] = 0

    for i, c in enumerate(cc):
        cc[i] = c * np.power(t, D[i])

    return cc

# Minimum Snap Trajectory
def MST(waypoints, durations=None):
    """ This function takes a list of desired waypoint i.e. [x0, x1, x2...xN] and
    segment durations, returns a [8N,1] coefficients matrix in normalized
    segment time. Omitted durations mean equal-length time intervals.

    1.The Problem
    Generate a full trajectory across N+1 waypoint made of N polynomial line segment.
    Each segment is defined as 7 order polynomial defined as follow:
    Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5 + ai6*t^6 + ai7*t^7

    Each polynomial has 8 unknown coefficients, thus we will have 8*N unknown to
    solve in total, so we need to come up with 8*N constraints.

    2.The constraints
    In general, the constraints is a set of condition which define the initial
    and final state, continuity between each piecewise function. This includes
    specifying continuity in higher derivatives of the trajectory at the
    intermediate waypoints.

    3.Matrix Design
    Since we have 8*N unknown coefficients to solve, and if we are given 8*N
    equations(constraints), then the problem becomes solving a linear equation.

    A * Coeff = B

    Let's look at B matrix first, B matrix is simple because it is just some constants
    on the right hand side of the equation. There are 8xN constraints,
    so B matrix will be [8N, 1].

    Now, how do we determine the dimension of Coeff matrix? Coeff is the final
    output matrix consists of 8*N elements. Since B matrix is only one column,
    thus Coeff matrix must be [8N, 1].

    Coeff.transpose = [a10 a11..a17...aN0 aN1..aN7]

    A matrix is tricky, we then can think of A matrix as a coeffient-coeffient matrix.
    We are no longer looking at a particular polynomial Pi, but rather P1, P2...PN
    as a whole. Since now our Coeff matrix is [8N, 1], and B is [8N, 8N], thus
    A matrix must have the form [8N, 8N].

    A = [A10 A12 ... A17 ... AN0 AN1 ...AN7
         ...
        ]

    Each element in a row represents the coefficient of coeffient aij under
    a certain constraint, where aij is the jth coeffient of Pi with i = 1...N, j = 0...7.
    """

    n = len(waypoints) - 1
    durations = np.ones(n) if durations is None else np.asarray(durations, dtype=float)
    if durations.shape != (n,) or not np.all(np.isfinite(durations)) or np.any(durations <= 0):
        raise ValueError("segment durations must be finite, positive, and match the segments")

    # initialize A, and B matrix
    A = np.zeros((8*n, 8*n))
    B = np.zeros((8*n, 1))

    # populate B matrix.
    for i in range(n):
        B[i] = waypoints[i]
        B[i + n] = waypoints[i+1]

    # Constraint 1
    for i in range(n):
        A[i][8*i:8*(i+1)] = get_poly_cc(8, 0, 0)

    # Constraint 2
    for i in range(n):
        A[i+n][8*i:8*(i+1)] = get_poly_cc(8, 0, 1)

    # Constraint 3
    for k in range(1, 4):
        A[2*n+k-1][:8] = get_poly_cc(8, k, 0)

    # Constraint 4
    for k in range(1, 4):
        A[2*n+3+k-1][-8:] = get_poly_cc(8, k, 1)

    # Constraint 5: match physical-time derivatives, not normalized ones.
    for i in range(n-1):
        for k in range(1, 7):
            A[2*n+6 + i*6+k-1][i*8 : (i*8+16)] = np.concatenate((
                get_poly_cc(8, k, 1) / durations[i]**k,
                -get_poly_cc(8, k, 0) / durations[i+1]**k,
            ))

    # solve for the coefficients
    Coeff = np.linalg.solve(A, B)
    return Coeff
