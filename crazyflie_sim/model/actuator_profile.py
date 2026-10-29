"""The vehicle: shared, immutable descriptions of airframe and rotors.

A run's ``ActuatorProfile`` is the single description of the vehicle.  The
plant flies it, and the registry hands the same profile to every controller,
so a controller's model and the plant agree unless a run asks otherwise
(``plant_overrides`` perturb the plant's copy only).  ``model.params`` holds
no vehicle constants.

The command bounds in :class:`ActuatorProfile` are per-rotor healthy thrust
setpoint bounds in newtons.  They are design/allocation bounds; ``pwm_map`` is
the independent plant-side voltage-to-thrust polynomial used to model the
available PWM authority at the current battery voltage.

Profiles deliberately contain no fault identity or fault-dependent tuning.
Rotor effectiveness remains a time-varying plant quantity supplied through
``RotorFaultConfig`` (or estimated by a controller).
"""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

import crazyflie_sim.model.params as params


def _immutable_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    # Back the returned array with immutable ``bytes``.  Merely clearing the
    # writeable flag on an owning ndarray can be reversed by a caller.
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(shape)


@dataclass(frozen=True, eq=False)
class ActuatorProfile:
    """Validated four-rotor, nonnegative-thrust actuator configuration."""

    name: str
    wrench_matrix: np.ndarray
    command_lower: np.ndarray
    command_upper: np.ndarray
    tau_up: float
    tau_down: float
    tau_uncertainty: float
    mass: float
    inertia: np.ndarray
    angular_damping: np.ndarray
    yaw_drag_coefficient: float
    pwm_map: np.ndarray
    rotor_speed_limit: float
    # Rotor speed-to-thrust curve: "quadratic" (c_T Omega^2) or "graefe_cubic"
    # (the identified curve).  The plant and any controller that models it
    # use it unless told otherwise, so their calibrations agree.
    thrust_map: str = "quadratic"
    # Rotor + motor-bell inertia J_m (kg m^2): gyroscopic and reaction moments
    # of the spinning rotors.  Zero leaves the plant a rigid body.
    rotor_inertia: float = 0.0

    def __post_init__(self):
        name = str(self.name).strip()
        if not name:
            raise ValueError("name must be nonempty")

        wrench_matrix = _immutable_array(
            self.wrench_matrix, (4, 4), "wrench_matrix"
        )
        command_lower = _immutable_array(
            self.command_lower, (4,), "command_lower"
        )
        command_upper = _immutable_array(
            self.command_upper, (4,), "command_upper"
        )
        inertia = _immutable_array(self.inertia, (3, 3), "inertia")
        angular_damping = _immutable_array(
            self.angular_damping, (3,), "angular_damping"
        )
        pwm_map = _immutable_array(self.pwm_map, (4,), "pwm_map")

        if np.any(command_lower < 0.0):
            raise ValueError("command_lower must be nonnegative")
        if np.any(command_upper <= command_lower):
            raise ValueError("command_upper must be greater than command_lower")
        if np.any(wrench_matrix[0] < 0.0):
            raise ValueError("wrench_matrix thrust row must be nonnegative")
        if not np.allclose(inertia, inertia.T, rtol=0.0, atol=1.0e-14):
            raise ValueError("inertia must be symmetric")
        if np.min(np.linalg.eigvalsh(inertia)) <= 0.0:
            raise ValueError("inertia must be positive definite")
        if np.any(angular_damping < 0.0):
            raise ValueError("angular_damping must be nonnegative")

        tau_up = float(self.tau_up)
        tau_down = float(self.tau_down)
        tau_uncertainty = float(self.tau_uncertainty)
        mass = float(self.mass)
        yaw_drag_coefficient = float(self.yaw_drag_coefficient)
        rotor_speed_limit = float(self.rotor_speed_limit)
        scalars = np.array(
            [
                tau_up,
                tau_down,
                tau_uncertainty,
                mass,
                yaw_drag_coefficient,
                rotor_speed_limit,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(scalars)):
            raise ValueError("profile scalar fields must be finite")
        if min(tau_up, tau_down, mass, rotor_speed_limit) <= 0.0:
            raise ValueError(
                "tau_up, tau_down, mass and rotor_speed_limit must be positive"
            )
        if tau_uncertainty < 0.0:
            raise ValueError("tau_uncertainty must be nonnegative")
        if yaw_drag_coefficient < 0.0:
            raise ValueError("yaw_drag_coefficient must be nonnegative")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "wrench_matrix", wrench_matrix)
        object.__setattr__(self, "command_lower", command_lower)
        object.__setattr__(self, "command_upper", command_upper)
        object.__setattr__(self, "tau_up", tau_up)
        object.__setattr__(self, "tau_down", tau_down)
        object.__setattr__(self, "tau_uncertainty", tau_uncertainty)
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "inertia", inertia)
        object.__setattr__(self, "angular_damping", angular_damping)
        object.__setattr__(
            self, "yaw_drag_coefficient", yaw_drag_coefficient
        )
        object.__setattr__(self, "pwm_map", pwm_map)
        object.__setattr__(self, "rotor_speed_limit", rotor_speed_limit)
        if self.thrust_map not in ("quadratic", "graefe_cubic"):
            raise ValueError("thrust_map must be 'quadratic' or 'graefe_cubic'")
        rotor_inertia = float(self.rotor_inertia)
        if not np.isfinite(rotor_inertia) or rotor_inertia < 0.0:
            raise ValueError("rotor_inertia must be finite and nonnegative")
        object.__setattr__(self, "rotor_inertia", rotor_inertia)

    @property
    def mixer_arm(self):
        """Moment arm of each rotor about the roll and pitch axes (m)."""
        return float(np.max(np.abs(self.wrench_matrix[1])))

    @property
    def thrust_to_torque(self):
        """Yaw reaction torque per newton of thrust (m)."""
        return float(np.max(np.abs(self.wrench_matrix[3])))

    @property
    def max_rotor_thrust(self):
        """Smallest per-rotor command ceiling (N)."""
        return float(np.min(self.command_upper))

    @property
    def max_total_thrust(self):
        """Sum of the per-rotor command ceilings (N)."""
        return float(np.sum(self.command_upper))

    @property
    def nominal_thrust_coefficient(self):
        """Quadratic ``c_T`` matching the design upper bound at speed limit."""
        reference_force = float(np.max(self.command_upper))
        return reference_force / (self.rotor_speed_limit * self.rotor_speed_limit)

    def to_config(self):
        """Return a JSON/YAML-serializable copy for run artifacts."""
        return {
            "name": self.name,
            "wrench_matrix": self.wrench_matrix.tolist(),
            "command_lower": self.command_lower.tolist(),
            "command_upper": self.command_upper.tolist(),
            "tau_up": self.tau_up,
            "tau_down": self.tau_down,
            "tau_uncertainty": self.tau_uncertainty,
            "mass": self.mass,
            "inertia": self.inertia.tolist(),
            "angular_damping": self.angular_damping.tolist(),
            "yaw_drag_coefficient": self.yaw_drag_coefficient,
            "pwm_map": self.pwm_map.tolist(),
            "rotor_speed_limit": self.rotor_speed_limit,
            "thrust_map": self.thrust_map,
            "rotor_inertia": self.rotor_inertia,
        }


def _x_wrench_matrix(mixer_arm, thrust_to_torque):
    arm = float(mixer_arm)
    torque = float(thrust_to_torque)
    return np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [-arm, -arm, arm, arm],
            [-arm, arm, arm, -arm],
            [-torque, torque, -torque, torque],
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Rotor speed-to-thrust curves (``ActuatorProfile.thrust_map``)
# ---------------------------------------------------------------------------
# "quadratic" is c_T Omega^2 with c_T = profile.nominal_thrust_coefficient
# (model.actuators.rotor_speed_to_thrust).  "graefe_cubic" is the measured
# curve of Graefe et al. (ICRA 2026), their eq. (9), normalised by
# sigma = 2900 rad/s; F(1) = 0.289 N.  The two agree at sigma and differ by up
# to about 24% below it.
GRAEFE_THRUST_COEFFICIENTS = (0.0, -0.043, 0.562, -0.23)   # in x = Omega/sigma
GRAEFE_SPEED_SCALE = 2900.0


def graefe_thrust(omega):
    """Measured thrust in N at mechanical speed ``omega`` (rad/s), clamped at 0.

    The cubic is negative between 0 and about 229 rad/s, so the physical map
    takes max(F, 0); it is strictly increasing from there to sigma.
    """
    x = np.asarray(omega, dtype=float) / GRAEFE_SPEED_SCALE
    c0, c1, c2, c3 = GRAEFE_THRUST_COEFFICIENTS
    return np.maximum(c0 + x * (c1 + x * (c2 + c3 * x)), 0.0)


_GRAEFE_TABLE_W = None
_GRAEFE_TABLE_F = None


def graefe_speed(thrust):
    """Inverse of :func:`graefe_thrust` on its increasing branch.

    Interpolated on a fixed 2048-point monotone table built once; targets at or
    below zero give zero speed.  The table is the reason this inverse is a
    numerical approximation and the audit records its residual.
    """
    global _GRAEFE_TABLE_W, _GRAEFE_TABLE_F
    if _GRAEFE_TABLE_W is None:
        lo = 0.0791 * GRAEFE_SPEED_SCALE          # the curve's positive root
        _GRAEFE_TABLE_W = np.linspace(lo, GRAEFE_SPEED_SCALE, 2048)
        _GRAEFE_TABLE_F = graefe_thrust(_GRAEFE_TABLE_W)
    t = np.asarray(thrust, dtype=float)
    out = np.interp(t, _GRAEFE_TABLE_F, _GRAEFE_TABLE_W, left=0.0,
                    right=GRAEFE_SPEED_SCALE)
    return np.where(t <= 0.0, 0.0, out)


# ---------------------------------------------------------------------------
# Crazyflie 2.1 Brushless constants shared by the crazyflie21_* profiles
# ---------------------------------------------------------------------------
# Mass and inertia from Graefe, Scherer, Hoenig and Trimpe, "How to Model Your
# Crazyflie Brushless", ICRA 2026 (arXiv:2603.05944), Table II, column without
# propeller guards: 40 g and diag([1.8, 2.4, 3.0]) * 1e-5 kg m^2 (x and y
# differ by 20%; the plant keeps the asymmetry).
_CF_MASS = 0.040
_CF_INERTIA = np.array(
    [
        (1.8e-5, 0, 0),
        (0, 2.4e-5, 0),
        (0, 0, 3.0e-5),
    ]
)
_CF_ARM_LENGTH = 0.05  # m center-to-motor (100 mm motor-to-motor diagonal)
_CF_MIXER_ARM = np.sqrt(0.5) * _CF_ARM_LENGTH
# Graefe et al. (ICRA 2026) measure tau_f(Omega)/F_thrust(Omega) on a force/torque
# testbed: 3.47 mm at hover, 2.99 mm at the post-loss operating point, 2.84 mm at
# full throttle.  The firmware constant THRUST2TORQUE is 0.00569, a factor 1.9
# larger.  The post-loss value is used.
_CF_THRUST_TO_TORQUE = 0.00299
# Bitcraze cf21bl motor-voltage-to-thrust cubic VMOTOR2THRUST{0..3} (per
# motor, force in N), platform_defaults_cf21bl.h.
_CF_VMOTOR_TO_THRUST = np.array(
    [
        -0.014058926705279723,
        0.04265273261724981,
        0.0018327760144017432,
        0.0020576974784587178,
    ],
    dtype=float,
)
# First-order motor/propulsion dynamics (seconds).  Graefe et al. (ICRA 2026)
# Table II identify a single time constant T = 50 ms, fitted to the rotor
# speed from in-flight command/RPM records; they do not split it into a rising
# and a falling value, so both carry it.  Applying 50 ms to thrust dynamics or
# to an unpowered failed rotor is a simulation assumption, not that
# measurement.
_CF_MOTOR_TAU = 0.050
_CF_TAU_UNCERTAINTY = 0.2 * max(_CF_MOTOR_TAU, _CF_MOTOR_TAU)
# Rotational damping torque: tau_damp = -c_omega * omega, N m / (rad/s).
_CF_ANGULAR_DAMPING = np.array([2.0e-6, 2.0e-6, 3.0e-6])
# Quadratic yaw drag tau_z = -Cq r |r| with Cq = 0.5 rho Cd A d^3, a
# geometry-based estimate for the 100 mm x 30 mm envelope (not identified),
# times a scale of 3.  Formulation: https://arxiv.org/abs/2404.06340
_CF_YAW_DRAG_COEFFICIENT = float(3.0) * (
    0.5
    * params.air_density
    * params.drag_coefficient_xy
    * (0.03 * 0.10)            # m^2, 30 mm x 100 mm projected envelope
    * 0.05 ** 3                # m, half of the 100 mm envelope
)

_CRAZYFLIE_WRENCH = _x_wrench_matrix(_CF_MIXER_ARM, _CF_THRUST_TO_TORQUE)
_ZERO = np.zeros(4, dtype=float)


# The default vehicle.  Per-rotor ceiling = the firmware THRUST_MAX 0.2 N.  Do
# not raise it to the measured 0.289 N: the voltage cubic saturates at
# 0.2874 N at 3.8 V, so the thrust-map inversion in model/actuators.py
# overflows.  Runs that want more
# authority select another profile.  Rotor speed limit 38000 RPM; quadratic
# rotor curve; rotor inertia off (the historical rigid-body plant).
CRAZYFLIE21_PHYSICAL = ActuatorProfile(
    name="crazyflie21_physical",
    wrench_matrix=_CRAZYFLIE_WRENCH,
    command_lower=_ZERO,
    command_upper=np.full(4, params.thrust_cmd_max_per_motor, dtype=float),
    tau_up=_CF_MOTOR_TAU,
    tau_down=_CF_MOTOR_TAU,
    tau_uncertainty=_CF_TAU_UNCERTAINTY,
    mass=_CF_MASS,
    inertia=_CF_INERTIA,
    angular_damping=_CF_ANGULAR_DAMPING,
    yaw_drag_coefficient=_CF_YAW_DRAG_COEFFICIENT,
    pwm_map=_CF_VMOTOR_TO_THRUST,
    rotor_speed_limit=float(38000.0) * (2.0 * np.pi / 60.0),
)


# Crazyflie 2.1 Brushless model centered on the unguarded parameters identified
# by Graefe et al. (ICRA 2026).  The identified rotor curve gives 0.289 N at
# its maximum command speed K = 2900 rad/s.  The plant retains the independent
# Bitcraze voltage-to-thrust map, whose 3.8 V endpoint is 0.2874 N.
CRAZYFLIE21_GRAEFE = ActuatorProfile(
    name="crazyflie21_graefe",
    wrench_matrix=_CRAZYFLIE_WRENCH,
    command_lower=_ZERO,
    command_upper=np.full(4, 0.289, dtype=float),
    tau_up=_CF_MOTOR_TAU,
    tau_down=_CF_MOTOR_TAU,
    tau_uncertainty=_CF_TAU_UNCERTAINTY,
    mass=_CF_MASS,
    inertia=_CF_INERTIA,
    angular_damping=_CF_ANGULAR_DAMPING,
    yaw_drag_coefficient=_CF_YAW_DRAG_COEFFICIENT,
    pwm_map=_CF_VMOTOR_TO_THRUST,
    rotor_speed_limit=2900.0,
)


# Crazyflie 2.1 Brushless in its standard configuration (legs, 350 mAh battery,
# no deck).  34 g take-off mass per the Bitcraze datasheet; inertia from a component-mass
# model of that configuration, diag(1.88, 1.93, 3.68)e-5 kg m^2 (about +-4 % per
# axis); yaw torque-to-thrust ratio 5.69 mm from the whole-vehicle ATI Nano43
# bench measurement merged into the firmware (crazyflie-firmware PR #1526).  The
# lag and speed limit are those of CRAZYFLIE21_GRAEFE; the rotor curve is the
# identified cubic; rotor inertia 7.35e-8 kg m^2 from a component model of the
# rotor and motor bell.
CRAZYFLIE21_STANDARD = ActuatorProfile(
    name="crazyflie21_standard",
    wrench_matrix=_x_wrench_matrix(_CF_MIXER_ARM, 0.00569),
    command_lower=_ZERO,
    command_upper=np.full(4, 0.289, dtype=float),
    tau_up=_CF_MOTOR_TAU,
    tau_down=_CF_MOTOR_TAU,
    tau_uncertainty=_CF_TAU_UNCERTAINTY,
    mass=0.034,
    inertia=np.diag([1.88e-5, 1.93e-5, 3.68e-5]),
    angular_damping=_CF_ANGULAR_DAMPING,
    yaw_drag_coefficient=_CF_YAW_DRAG_COEFFICIENT,
    pwm_map=_CF_VMOTOR_TO_THRUST,
    rotor_speed_limit=2900.0,
    thrust_map="graefe_cubic",
    rotor_inertia=7.35e-8,
)


# Assumed 0.50 N airframe with Crazyflie mass, geometry and inertia.
# Calibrate the voltage cubic at the nominal battery voltage, not against
# the separate 0.20 N firmware command cap. The default profile stays physical.
_CRAZYFLIE_MAP_MAX = float(np.polynomial.polynomial.polyval(
    params.battery_voltage_nominal, _CF_VMOTOR_TO_THRUST
))
_CRAZYFLIE_THRUST_SCALE = 0.50 / _CRAZYFLIE_MAP_MAX
CRAZYFLIE21_THRUST_RICH = ActuatorProfile(
    name="crazyflie21_thrust_rich",
    wrench_matrix=_CRAZYFLIE_WRENCH,
    command_lower=_ZERO,
    command_upper=np.full(4, 0.50, dtype=float),
    tau_up=_CF_MOTOR_TAU,
    tau_down=_CF_MOTOR_TAU,
    tau_uncertainty=_CF_TAU_UNCERTAINTY,
    mass=_CF_MASS,
    inertia=_CF_INERTIA,
    angular_damping=_CF_ANGULAR_DAMPING,
    yaw_drag_coefficient=_CF_YAW_DRAG_COEFFICIENT,
    pwm_map=_CRAZYFLIE_THRUST_SCALE * np.asarray(_CF_VMOTOR_TO_THRUST, dtype=float),
    rotor_speed_limit=(
        CRAZYFLIE21_PHYSICAL.rotor_speed_limit * float(np.sqrt(_CRAZYFLIE_THRUST_SCALE))
    ),
)


ACTUATOR_PROFILE_REGISTRY = MappingProxyType(
    {
        profile.name: profile
        for profile in (
            CRAZYFLIE21_PHYSICAL,
            CRAZYFLIE21_GRAEFE,
            CRAZYFLIE21_STANDARD,
            CRAZYFLIE21_THRUST_RICH,
        )
    }
)


def available_actuator_profiles():
    """Return registered profile names in deterministic order."""
    return tuple(ACTUATOR_PROFILE_REGISTRY)


def get_actuator_profile(profile=None):
    """Resolve ``None``, a registry name, or an existing profile instance.

    The default stays ``crazyflie21_physical``.  Runs that want the
    identified unguarded-airframe parameters select ``crazyflie21_graefe``
    explicitly.  The 0.50 N profile remains a separate
    thrust-reserve sensitivity case.
    """
    if profile is None:
        return CRAZYFLIE21_PHYSICAL
    if isinstance(profile, ActuatorProfile):
        return profile
    name = str(profile)
    try:
        return ACTUATOR_PROFILE_REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(available_actuator_profiles())
        raise ValueError(
            f"unknown actuator profile {name!r}; expected one of: {choices}"
        ) from exc


__all__ = [
    "ActuatorProfile",
    "ACTUATOR_PROFILE_REGISTRY",
    "CRAZYFLIE21_PHYSICAL",
    "CRAZYFLIE21_GRAEFE",
    "CRAZYFLIE21_STANDARD",
    "CRAZYFLIE21_THRUST_RICH",
    "available_actuator_profiles",
    "get_actuator_profile",
    "graefe_speed",
    "graefe_thrust",
]
