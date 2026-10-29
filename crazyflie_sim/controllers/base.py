import abc
from collections import namedtuple

# Reference handed to every controller each tick: position, velocity and
# acceleration (world frame, 3-vectors), yaw (rad) and yaw rate (rad/s).
DesiredState = namedtuple("DesiredState", "pos vel acc yaw yawdot")


class ControllerBase(abc.ABC):
    """Shared interface for all control backends.

    Every backend is constructed with a control timestep, exposes
    set_dt/reset lifecycle hooks, maps (quad, des_state[, sim_time]) to
    per-rotor PWM commands, and reports a diagnostics dict for logging.
    """

    @abc.abstractmethod
    def set_dt(self, dt):
        """Update the control timestep used by the backend."""

    @abc.abstractmethod
    def reset(self):
        """Clear all integrators and cached command state."""

    @abc.abstractmethod
    def run(self, quad, des_state, sim_time=0.0):
        """Compute per-rotor PWM commands, returns np.ndarray of shape (4,)."""

    @abc.abstractmethod
    def diagnostics(self):
        """Return a dict of the backend's latest internal commands/state."""
