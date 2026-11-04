"""What one run flies: initial state, reference, disturbances, plant options.

A :class:`Scenario` is plain data; :class:`sim.simulation.Simulation` turns it
into a plant, a controller and a step loop.  The controller is not part of the
scenario, so the same scenario can be flown by every controller in a
comparison.  YAML configs (``crazyflie_sim/configs/``) load into this dataclass through
:mod:`sim.config`; every field below is also a config key.
"""

from dataclasses import dataclass, field

import numpy as np

from crazyflie_sim.controllers.base import DesiredState
from crazyflie_sim.model.rotor_fault import RotorFaultConfig
from crazyflie_sim.model.wind import WindModel
from crazyflie_sim.sim import trajectory

DT = 1.0 / 500.0  # control period (500 Hz attitude loop)

SIM_MODES = ("trajectory", "hover_wind", "hover_rotor_fault", "hovering")
WAYPOINT_SOURCES = ("helix", "list", "csv", "timed")
HOVER_DURATION_S = 50.0  # default length of the two hover modes

# Named rotor-fault cases -> remaining rotor effectiveness.
FAULT_CASE_EFFECTIVENESS = {
    "power_loss_25pct": 0.75,
    "power_loss_50pct": 0.50,
    "power_loss_75pct": 0.25,
    "self_defined": 0.3,
    "failure": 0.0,
}


def fault_case_text(case):
    """Display name of a fault case: "power_loss_50pct" -> "50 % power loss"."""
    effectiveness = FAULT_CASE_EFFECTIVENESS.get(str(case))
    if effectiveness is None:
        return str(case).replace("_", " ")
    if effectiveness == 0.0:
        return "complete failure"
    return "{:g} % power loss".format(round(100.0 * (1.0 - effectiveness)))

# Controller stress-test course used by sim_mode "trajectory":
#   1) climb to a hover checkpoint
#   2) square circuit with 90-deg corners and a 45-deg yaw change
#   3) diagonal climb across the square
#   4) S-slalom back down with -45-deg yaw
#   5) return to a final hover near the start
DEFAULT_WAYPOINTS = (
    (0.0, 0.0, 0.6, 0.0),
    (0.6, 0.0, 0.6, 0.0),
    (0.6, 0.6, 0.6, 0.0),
    (0.0, 0.6, 0.6, 0.7854),
    (0.0, 0.0, 0.6, 0.7854),
    (0.6, 0.6, 1.1, 0.0),
    (0.9, 0.3, 0.8, -0.7854),
    (0.6, 0.0, 0.5, -0.7854),
    (0.0, 0.0, 0.5, 0.0),
)


def rotor_fault_config(case="failure", rotor=1, start_time_s=3.0, mechanism="motor_loe"):
    """Enabled RotorFaultConfig for a named case (see FAULT_CASE_EFFECTIVENESS)."""
    try:
        effectiveness = float(FAULT_CASE_EFFECTIVENESS[case])
    except KeyError:
        raise ValueError(
            "rotor fault case must be one of {}".format(sorted(FAULT_CASE_EFFECTIVENESS))
        ) from None
    return RotorFaultConfig(
        enabled=True,
        rotor_index=rotor,
        start_time_s=start_time_s,
        effectiveness=effectiveness,
        mechanism=mechanism,
        label=str(case),
    )


@dataclass
class RotorFaultSpec:
    """Config-level rotor fault; active only in sim_mode hover_rotor_fault."""

    case: str = "failure"
    rotor: object = 1  # 1..4, or a list such as [1, 2]
    start_time_s: float = 3.0
    enabled: bool = True
    mechanism: str = "motor_loe"

    def build(self):
        if not self.enabled:
            return RotorFaultConfig()
        return rotor_fault_config(self.case, self.rotor, self.start_time_s, self.mechanism)


@dataclass
class Scenario:
    name: str = "scenario"
    sim_mode: str = "hover_rotor_fault"
    duration_s: float = None  # None: 50 s hover, or trajectory length + 2 s
    initial_pos: tuple = (0.0, 0.0, 0.0)
    initial_attitude: tuple = (0.0, 0.0, 0.0)
    hover_setpoint: tuple = (0.0, 0.0, 1.0)
    hover_yaw: float = 0.0
    # > 0: the hover reference climbs from initial_pos to hover_setpoint along
    # a minimum-jerk profile of this length; 0 steps to the setpoint at t = 0.
    takeoff_duration_s: float = 0.0
    # trajectory reference
    trajectory_speed: float = 0.3  # m/s along the path
    waypoint_source: str = "list"
    user_waypoints: tuple = DEFAULT_WAYPOINTS  # Nx3 or Nx4 [x, y, z(, yaw)]
    # waypoint_source "timed": rows [t, x, y, z, yaw]; the reference passes
    # each waypoint at its time t (s) instead of at trajectory_speed.
    timed_waypoints: tuple = ()
    waypoints_csv_path: str = "waypoints.csv"
    helix_start_t: float = 0.0
    helix_num_waypoints: int = 9
    # disturbances (wind acts only in hover_wind, the fault only in hover_rotor_fault)
    wind: WindModel = field(default_factory=WindModel)
    rotor_fault: RotorFaultSpec = field(default_factory=RotorFaultSpec)
    # sensing / actuation
    estimator: dict = field(default_factory=dict)  # model: estimation.estimator.build_estimator
    seed: int = 0  # estimator noise seed
    control_link: dict = field(default_factory=dict)  # model: model.control_link
    actuator_profile: str = None  # the vehicle; None: crazyflie21_physical
    actuator_plant_backend: str = None  # None: legacy_thrust_euler
    actuator_thrust_map: str = None  # None: the profile's curve
    plant_overrides: dict = field(default_factory=dict)  # see Simulation

    def __post_init__(self):
        if self.sim_mode not in SIM_MODES:
            raise ValueError("sim_mode must be one of {}".format(SIM_MODES))
        if self.waypoint_source not in WAYPOINT_SOURCES:
            raise ValueError("waypoint_source must be one of {}".format(WAYPOINT_SOURCES))
        from crazyflie_sim.model.control_link import validate_control_link_spec

        self.control_link = validate_control_link_spec(dict(self.control_link or {}))

    # ---- reference --------------------------------------------------------
    def timed_waypoint_rows(self):
        """(times (N,), Nx4 [x, y, z, yaw]) for waypoint_source "timed".

        A first waypoint after t = 0 gets the start pose prepended at t = 0;
        one at t = 0 must be the start position.
        """
        rows = np.asarray(self.timed_waypoints, dtype=float)
        if rows.ndim != 2 or rows.shape[1] != 5 or rows.shape[0] < 1:
            raise ValueError("timed_waypoints must be a list of [t, x, y, z, yaw] rows.")
        times, waypoints = rows[:, 0], rows[:, 1:]
        start = np.asarray(self.initial_pos, dtype=float).reshape(3,)
        if times[0] < 0.0:
            raise ValueError("timed_waypoints: times must not be negative.")
        if times[0] == 0.0:
            if np.linalg.norm(waypoints[0, :3] - start) > 1e-6:
                raise ValueError("timed_waypoints: a waypoint at t = 0 must be initial_pos {}.".format(
                    tuple(start)))
        else:
            times = np.concatenate(([0.0], times))
            waypoints = np.vstack((np.append(start, float(self.initial_attitude[2])), waypoints))
        if len(times) < 2:
            raise ValueError("timed_waypoints: need a waypoint after t = 0.")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("timed_waypoints: times must increase strictly.")
        return times, waypoints

    def waypoints(self):
        """Nx4 [x, y, z, yaw] waypoints for sim_mode trajectory."""
        if self.waypoint_source == "timed":
            return self.timed_waypoint_rows()[1]
        if self.waypoint_source == "helix":
            waypoints = trajectory.get_helix_waypoints(
                self.helix_start_t, self.helix_num_waypoints
            )
        elif self.waypoint_source == "csv":
            waypoints = np.atleast_2d(
                np.loadtxt(self.waypoints_csv_path, delimiter=",", dtype=float)
            )
        else:
            waypoints = np.asarray(self.user_waypoints, dtype=float)
            if waypoints.ndim != 2 or waypoints.shape[1] not in (3, 4):
                raise ValueError("user_waypoints must be an Nx3 or Nx4 list.")
            start = np.asarray(self.initial_pos, dtype=float).reshape(3,)
            if np.linalg.norm(waypoints[0, :3] - start) > 1e-6:
                first = start if waypoints.shape[1] == 3 else np.append(start, 0.0)
                waypoints = np.vstack((first, waypoints))
        waypoints = trajectory.format_waypoints(waypoints)
        segments = np.linalg.norm(np.diff(waypoints[:, :3], axis=0), axis=1)
        if np.any(segments < 1e-6):
            raise ValueError("Consecutive waypoints are identical or too close together.")
        return waypoints

    def reference(self):
        """Return (desired_state_fn(t), plot_waypoints, natural_duration_s)."""
        if self.sim_mode == "trajectory" and self.waypoint_source == "timed":
            times, waypoints = self.timed_waypoint_rows()
            waypoints = np.column_stack((waypoints[:, :3], np.unwrap(waypoints[:, 3])))
            velocities = trajectory.timed_waypoint_velocities(times, waypoints)

            def desired(t):
                return trajectory.generate_timed_trajectory(t, times, waypoints, velocities)

            return desired, waypoints[:, :3], float(times[-1]) + 2.0

        if self.sim_mode == "trajectory":
            waypoints = self.waypoints()
            cx, cy, cz = trajectory.get_MST_coefficients(waypoints)
            speed = float(self.trajectory_speed)

            def desired(t):
                return trajectory.generate_trajectory(t, speed, waypoints, cx, cy, cz)

            length = float(np.sum(np.linalg.norm(np.diff(waypoints[:, :3], axis=0), axis=1)))
            return desired, waypoints[:, :3], length / speed + 2.0

        setpoint = np.array(self.hover_setpoint, dtype=float)
        yaw = float(self.hover_yaw)
        start = np.array(self.initial_pos, dtype=float)
        climb = float(self.takeoff_duration_s)
        if climb > 0.0 and np.linalg.norm(setpoint - start) > 1e-9:
            def desired(t):
                pos, vel, acc = trajectory.minimum_jerk(t, climb, start, setpoint)
                return DesiredState(pos=pos, vel=vel, acc=acc, yaw=yaw, yawdot=0.0)

            return desired, np.vstack((start, setpoint)), HOVER_DURATION_S

        def desired(t):
            return DesiredState(
                pos=setpoint.copy(), vel=np.zeros(3), acc=np.zeros(3), yaw=yaw, yawdot=0.0
            )

        return desired, np.vstack((setpoint, setpoint)), HOVER_DURATION_S

    def resolved_duration(self):
        if self.duration_s is not None:
            return float(self.duration_s)
        return self.reference()[2]

    # ---- disturbances -------------------------------------------------------
    def active_wind(self):
        return self.wind if self.sim_mode == "hover_wind" else None

    def active_fault(self):
        if self.sim_mode != "hover_rotor_fault":
            return RotorFaultConfig()
        return self.rotor_fault.build()

    def event(self):
        """(time_s, label) of the disturbance onset, or (None, "")."""
        if self.sim_mode == "hover_wind":
            return float(self.wind.start_time), "wind on"
        fault = self.active_fault()
        if fault.enabled:
            return float(fault.start_time_s), "rotor fault"
        return None, ""
