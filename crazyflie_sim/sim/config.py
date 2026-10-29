"""YAML/JSON run configs -> :class:`RunConfig` (a Scenario plus a controller).

Loading never touches module state.  The one global left is ``model.params``:
``params_overrides`` are applied by the caller with :func:`apply_params_overrides`
(the live GUI) or scoped with :func:`params_overridden` (headless runs).  They
cover simulator-wide settings only; the vehicle is ``actuator_profile``, and
``plant_overrides`` change the plant's copy of it (mass, inertia, lags, ...)
while the controller keeps the profile.

Keys (all optional; ``extends`` merges this file over another one)::

    extends: hover_wind_default.yaml     # relative to this file
    name: my_run
    controller_mode: mellinger           # any name in controllers.registry
    controller_gains:                    # constructor arguments, per mode
      mellinger: {kp_xy: 2.5}
    params_overrides: {battery_sag_enabled: true}   # model.params leaves (not the vehicle)
    metrics: {settling_threshold_m: 0.1}
    monte_carlo: {...}                   # experiments.monte_carlo
    # every other key is a sim.scenario.Scenario field:
    sim_mode, duration_s, initial_pos, initial_attitude, hover_setpoint,
    hover_yaw, trajectory_speed, waypoint_source, user_waypoints,
    waypoints_csv_path, helix_start_t, helix_num_waypoints,
    wind: {...}           # model.wind.WindModel.from_dict
    rotor_fault: {case, rotor, start_time_s, enabled, mechanism}
    estimator: {...}      # estimation.estimator.build_estimator
    seed, control_link: {...}, actuator_profile, actuator_plant_backend,
    actuator_thrust_map, plant_overrides: {...}
"""

import contextlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import yaml

from crazyflie_sim.model.wind import WindModel
from crazyflie_sim.sim.scenario import SIM_MODES, WAYPOINT_SOURCES, RotorFaultSpec, Scenario

# The bundled run configs (package data, so they ship with the wheel).
CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

DEFAULT_CONTROLLER = "baseline"  # a config without controller_mode flies the cascaded PID

METRICS_KEYS = ("settling_threshold_m", "settling_hold_s", "steady_state_window_s")
ROTOR_FAULT_KEYS = tuple(f.name for f in fields(RotorFaultSpec))
PLANT_BACKENDS = ("legacy_thrust_euler", "rotor_speed_exact_zoh", "thrust_exact_zoh")
THRUST_MAPS = ("quadratic", "graefe_cubic")

# model.params leaves a config may override.  Only parameters without a stale
# derived constant are listed.  Vehicle parameters are not model.params any
# more; VEHICLE_KEYS says where each one moved.
VEHICLE_KEYS = {
    "mass": "plant_overrides.mass (plant only) or another actuator_profile",
    "I": "plant_overrides.inertia (plant only) or another actuator_profile",
    "motor_tau_up": "plant_overrides.motor_tau_up (plant only)",
    "motor_tau_down": "plant_overrides.motor_tau_down (plant only)",
    "rotor_speed_max_rpm": "the actuator_profile's rotor_speed_limit",
    "yaw_drag_scale": "plant_overrides.yaw_drag_scale (plant only)",
}
PARAMS_OVERRIDE_KEYS = frozenset(
    [
        "battery_voltage_nominal",
        "battery_sag_enabled",
        "battery_internal_resistance",
        "battery_max_current_draw",
        "battery_sag_tau",
        "rotor_speed_model_enabled",
        "gyro_range_dps",
        "sim_gyro_range_dps",
        "prop_yaw_torque_loss_exponent",
        "aero_drag_factor_xy",
        "aero_drag_factor_z",
        "air_density",
        "drag_coefficient_xy",
        "drag_coefficient_z",
        "prop_radius",
        "rotor_drag_enabled",
        "rotor_drag_coefficient_x",
        "rotor_drag_coefficient_y",
        "body_frame_drag_enabled",
        "ground_effect_enabled",
        "use_quaternion_attitude",
    ]
)


@dataclass
class RunConfig:
    name: str = "run"
    scenario: Scenario = field(default_factory=Scenario)
    controller_mode: str = DEFAULT_CONTROLLER
    controller_gains: dict = field(default_factory=dict)
    params_overrides: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    monte_carlo: dict = field(default_factory=dict)
    source: str = None

    def gains(self, mode=None):
        """Constructor overrides for ``mode`` (default: controller_mode)."""
        return dict(self.controller_gains.get(mode or self.controller_mode, {}))

    def to_dict(self):
        """Plain, YAML-serializable snapshot for run archives; from_dict()
        loads it back (provenance lives in the archive's env.json)."""
        scenario = {
            f.name: getattr(self.scenario, f.name)
            for f in fields(self.scenario)
            if f.name not in ("wind", "rotor_fault")
        }
        scenario["wind"] = self.scenario.wind.to_dict()
        scenario["rotor_fault"] = asdict(self.scenario.rotor_fault)
        return _plain(
            {
                "name": self.name,
                "controller_mode": self.controller_mode,
                "controller_gains": self.controller_gains,
                "params_overrides": self.params_overrides,
                "metrics": self.metrics,
                "monte_carlo": self.monte_carlo,
                **scenario,
            }
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _err(key, message):
    raise ValueError("config key {!r}: {}".format(key, message))


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(key, value):
    if not _is_number(value):
        _err(key, "expected a number, got {!r}".format(value))
    return float(value)


def _int(key, value):
    if not isinstance(value, int) or isinstance(value, bool):
        _err(key, "expected an integer, got {!r}".format(value))
    return int(value)


def _bool(key, value):
    if not isinstance(value, bool):
        _err(key, "expected a boolean, got {!r}".format(value))
    return value


def _str(key, value):
    if not isinstance(value, str):
        _err(key, "expected a string, got {!r}".format(value))
    return value


def _choice(key, value, choices):
    value = _str(key, value)
    if value not in choices:
        _err(key, "must be one of: {}".format(", ".join('"{}"'.format(c) for c in choices)))
    return value


def _seq3(key, value):
    if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(
        _is_number(v) for v in value
    ):
        _err(key, "expected a sequence of 3 numbers, got {!r}".format(value))
    return tuple(float(v) for v in value)


def _mapping(key, value, allowed=None):
    if value is None:
        return {}
    if not isinstance(value, dict):
        _err(key, "expected a mapping, got {!r}".format(value))
    if allowed is not None:
        unknown = sorted(set(value) - set(allowed))
        if unknown:
            _err(key, "unknown sub-key(s) {}; valid: {}".format(unknown, sorted(allowed)))
    return dict(value)


def _waypoints(key, value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        _err(key, "expected a list of at least 2 waypoint rows")
    rows = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) not in (3, 4) or not all(
            _is_number(v) for v in row
        ):
            _err(key, "each waypoint must be 3 [x,y,z] or 4 [x,y,z,yaw] numbers")
        rows.append(tuple(float(v) for v in row))
    return tuple(rows)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read(path):
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        raw = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        raise ValueError("config {!r}: expected .yaml/.yml or .json".format(str(path)))
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise ValueError("config {!r}: top level must be a mapping".format(str(path)))
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    base = _read(path.parent / parent)
    return _merge(base, raw)


def _merge(base, override):
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_config_path(path):
    """``path`` if it exists, else a config bundled in crazyflie_sim/configs/:
    "hover_wind_default", "hover_wind_default.yaml", "benchmark/01_hover_gust"."""
    path = Path(path)
    if path.exists():
        return path
    for candidate in (CONFIGS_DIR / path, CONFIGS_DIR / path.with_suffix(".yaml")):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "config {!r}: no such file, and no bundled config of that name in {}".format(
            str(path), CONFIGS_DIR))


def load_config(path):
    """Load and validate a config file (a path or a bundled config's name);
    raises ValueError naming the bad key."""
    path = resolve_config_path(path)
    raw = _read(path)
    return from_dict(raw, source=str(path))


def from_dict(raw, source=None):
    raw = dict(raw)
    run_keys = {f.name for f in fields(RunConfig)} - {"scenario", "source"}
    scenario_keys = {f.name for f in fields(Scenario)} - {"name"}
    unknown = sorted(set(raw) - run_keys - scenario_keys)
    if unknown:
        raise ValueError(
            "config {}: unknown key(s) {}; valid: {}".format(
                source or "", unknown, sorted(run_keys | scenario_keys)
            )
        )

    from crazyflie_sim.controllers import registry

    cfg = RunConfig(source=source)
    scenario = {}
    for key, value in raw.items():
        if key == "name":
            cfg.name = _str(key, value)
        elif key == "controller_mode":
            cfg.controller_mode = _choice(key, value, registry.available())
        elif key == "controller_gains":
            gains = _mapping(key, value, allowed=registry.available())
            cfg.controller_gains = {m: _mapping(key + "." + m, g) for m, g in gains.items()}
        elif key == "params_overrides":
            _reject_vehicle_keys(value if isinstance(value, dict) else {})
            cfg.params_overrides = _mapping(key, value, allowed=PARAMS_OVERRIDE_KEYS)
        elif key == "metrics":
            metrics = _mapping(key, value, allowed=METRICS_KEYS)
            cfg.metrics = {k: _number("metrics." + k, v) for k, v in metrics.items()}
        elif key == "monte_carlo":
            cfg.monte_carlo = _mapping(key, value)
        else:
            scenario[key] = _scenario_value(key, value)
    cfg.scenario = Scenario(name=cfg.name, **scenario)
    return cfg


def _scenario_value(key, value):
    if key == "sim_mode":
        return _choice(key, value, SIM_MODES)
    if key == "waypoint_source":
        return _choice(key, value, WAYPOINT_SOURCES)
    if key == "duration_s":
        if value is None:
            return None
        value = _number(key, value)
        if value <= 0.0:
            _err(key, "must be positive")
        return value
    if key in ("initial_pos", "initial_attitude", "hover_setpoint"):
        return _seq3(key, value)
    if key in ("hover_yaw", "trajectory_speed", "helix_start_t"):
        return _number(key, value)
    if key == "helix_num_waypoints":
        value = _int(key, value)
        if value < 2:
            _err(key, "must be >= 2")
        return value
    if key == "user_waypoints":
        return _waypoints(key, value)
    if key == "waypoints_csv_path":
        return _str(key, value)
    if key == "wind":
        try:
            return WindModel.from_dict(_mapping(key, value))
        except (TypeError, ValueError) as exc:
            _err(key, str(exc))
    if key == "rotor_fault":
        spec = _mapping(key, value, allowed=ROTOR_FAULT_KEYS)
        if "case" in spec:
            _str("rotor_fault.case", spec["case"])
        if "rotor" in spec:
            rotor = spec["rotor"]
            spec["rotor"] = (
                [_int("rotor_fault.rotor", r) for r in rotor]
                if isinstance(rotor, list)
                else _int("rotor_fault.rotor", rotor)
            )
        if "start_time_s" in spec:
            spec["start_time_s"] = _number("rotor_fault.start_time_s", spec["start_time_s"])
        if "enabled" in spec:
            _bool("rotor_fault.enabled", spec["enabled"])
        if "mechanism" in spec:
            _choice("rotor_fault.mechanism", spec["mechanism"], ("motor_loe", "prop_loe", "combined"))
        return RotorFaultSpec(**spec)
    if key in ("estimator", "plant_overrides"):
        return _mapping(key, value)
    if key == "control_link":
        from crazyflie_sim.model.control_link import validate_control_link_spec

        return validate_control_link_spec(_mapping(key, value))
    if key == "seed":
        return 0 if value is None else _int(key, value)
    if key == "actuator_profile":
        if value is None:
            return None
        from crazyflie_sim.model.actuator_profile import available_actuator_profiles

        return _choice(key, value, available_actuator_profiles())
    if key == "actuator_plant_backend":
        return None if value is None else _choice(key, value, PLANT_BACKENDS)
    if key == "actuator_thrust_map":
        return None if value is None else _choice(key, value, THRUST_MAPS)
    raise AssertionError(key)


# ---------------------------------------------------------------------------
# model.params overrides (the one remaining global)
# ---------------------------------------------------------------------------


def _reject_vehicle_keys(overrides):
    for key in dict(overrides or {}):
        if key in VEHICLE_KEYS:
            raise ValueError(
                "params_overrides key {!r} is a vehicle parameter; use {}".format(
                    key, VEHICLE_KEYS[key]
                )
            )


def apply_params_overrides(overrides):
    """Set whitelisted model.params leaves."""
    import crazyflie_sim.model.params as params

    _reject_vehicle_keys(overrides)
    for key, value in dict(overrides or {}).items():
        if key not in PARAMS_OVERRIDE_KEYS:
            raise ValueError(
                "params_overrides key {!r}: not in {}".format(key, sorted(PARAMS_OVERRIDE_KEYS))
            )
        setattr(params, key, value)


@contextlib.contextmanager
def params_overridden(overrides):
    """Apply params overrides for the duration of a ``with`` block."""
    import crazyflie_sim.model.params as params

    overrides = dict(overrides or {})
    saved = {key: getattr(params, key) for key in overrides if hasattr(params, key)}
    try:
        apply_params_overrides(overrides)
        yield
    finally:
        for key, value in saved.items():
            setattr(params, key, value)


def _plain(value):
    """numpy containers/scalars -> plain JSON/YAML types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value
