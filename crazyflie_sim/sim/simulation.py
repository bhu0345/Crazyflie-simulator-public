"""The one step loop.  The live GUI, headless runs, experiments and the golden
regression all advance the plant through :meth:`Simulation.step`.

Per control period k (t = k*dt):

    reference(t) -> wind(t) -> controller(plant or estimate) -> control link
    -> plant.update(dt) -> estimator.update -> record

Typical use::

    from crazyflie_sim.sim.config import load_config
    from crazyflie_sim.sim.simulation import Simulation

    cfg = load_config("hover_rotor_fault_default.yaml")  # bundled config, or a path
    run = Simulation.from_scenario(cfg.scenario, "mellinger").run()
    run["pos"], run["motor_pwm"], ...

or with an unregistered controller instance, for quick experiments::

    Simulation(MyController(dt=0.002), duration_s=5.0, reference=hover_fn).run()
"""

import time

import numpy as np

from crazyflie_sim.controllers import registry
from crazyflie_sim.controllers.base import ControllerBase, DesiredState
from crazyflie_sim.model.quadcopter import Quadcopter
from crazyflie_sim.sim.scenario import DT

PLANT_OVERRIDE_KEYS = (
    "battery_voltage",
    "battery_sag",
    "use_quaternion",
    "mass",
    "inertia",
    "motor_tau_up",
    "motor_tau_down",
    "thrust_map",
    "rotor_inertia_kgm2",
    "rotor_gyro",
    "rotor_reaction",
    "yaw_drag_scale",
)


def hover_reference(setpoint=(0.0, 0.0, 1.0), yaw=0.0):
    setpoint = np.array(setpoint, dtype=float)

    def desired(t):
        return DesiredState(
            pos=setpoint.copy(), vel=np.zeros(3), acc=np.zeros(3), yaw=float(yaw), yawdot=0.0
        )

    return desired


class FunctionWind:
    """Adapter: any ``f(t) -> (3,)`` as a wind source (force, or velocity)."""

    def __init__(self, fn, as_velocity=False):
        self.fn = fn
        self.as_velocity = bool(as_velocity)
        self.start_time = 0.0

    def value(self, t):
        return np.asarray(self.fn(t), dtype=float).reshape(3,)

    def reset(self):
        pass


def _build_estimator(spec, quad, dt, seed):
    """None, an estimator spec dict, an (spec, seed) tuple, a factory or an instance."""
    from crazyflie_sim.estimation.estimator import build_estimator

    if spec is None or spec is False:
        return None
    if isinstance(spec, dict):
        if not spec or not spec.get("enabled", False):
            return None
        return build_estimator(spec, quad, dt, seed=seed)
    if isinstance(spec, tuple):
        est_cfg, est_seed = spec
        return build_estimator(est_cfg, quad, dt, seed=est_seed)
    if callable(spec):
        return spec(quad)
    spec.prime(quad)
    return spec


def _build_link(spec):
    from crazyflie_sim.model.control_link import build_control_link

    if spec is None or spec is False:
        return None
    if isinstance(spec, dict):
        return build_control_link(spec) if spec else None
    return spec


class Simulation:
    """One plant, one controller, one reference; advance with step() or run().

    controller : a registered mode name (see controllers.registry) or a
                 ControllerBase instance.
    reference  : callable t -> DesiredState (default: hover at (0, 0, 1)).
    wind       : model.wind.WindModel, FunctionWind, or None.  A velocity wind
                 acts through the aero model, a force wind is added directly.
    rotor_fault: model.rotor_fault.RotorFaultConfig (or subclass) or None.
    estimator  : None for perfect state, or a spec accepted by
                 estimation.estimator.build_estimator; the controller then
                 sees the estimate.
    control_link: None, a model.control_link spec dict, or a ControlChannel.
    dynamic_actuator_plant: True/False forces the rotor-speed plant on/off when
                 no actuator_plant_backend is given; None follows model.params.
    plant_overrides: plant-only mass/inertia/lags/battery/... (PLANT_OVERRIDE_KEYS);
                 the controller keeps its nominal parameters.
    plant_event_times: instants inside a control period (fault onsets) at
                 which the plant integration is split, holding the command.
    """

    def __init__(
        self,
        controller,
        *,
        duration_s,
        dt=DT,
        reference=None,
        initial_pos=(0.0, 0.0, 0.0),
        initial_attitude=(0.0, 0.0, 0.0),
        rotor_fault=None,
        wind=None,
        estimator=None,
        seed=0,
        control_link=None,
        actuator_profile=None,
        actuator_plant_backend=None,
        dynamic_actuator_plant=None,
        controller_overrides=None,
        plant_overrides=None,
        plant_event_times=None,
    ):
        self.dt = float(dt)
        self.duration_s = float(duration_s)
        self.n_steps = int(np.ceil(self.duration_s / self.dt))
        self.reference = reference if reference is not None else hover_reference()

        plant_options = dict(plant_overrides or {})
        unknown = sorted(set(plant_options) - set(PLANT_OVERRIDE_KEYS))
        if unknown:
            raise ValueError("unknown plant_overrides key(s): {}".format(unknown))
        # None lets the plant follow params.rotor_speed_model_enabled; an
        # explicit actuator_plant_backend wins over both.
        self.quad = Quadcopter(
            initial_pos,
            initial_attitude,
            rotor_fault=rotor_fault,
            use_rotor_speed_model=None if dynamic_actuator_plant is None else bool(dynamic_actuator_plant),
            actuator_profile=actuator_profile,
            actuator_plant_backend=actuator_plant_backend,
            **plant_options,
        )

        if isinstance(controller, ControllerBase):
            if controller_overrides:
                raise ValueError("controller_overrides need a registered controller name")
            self.mode = type(controller).__name__
            self.controller = controller
        else:
            self.mode = str(controller)
            self.controller = registry.build(
                self.mode, self.dt, controller_overrides, actuator_profile=actuator_profile
            )

        self.wind = wind
        if self.wind is not None:
            self.wind.reset()
        self.estimator = _build_estimator(estimator, self.quad, self.dt, seed)
        self.link = _build_link(control_link)
        self.plant_event_times = tuple(float(e) for e in (plant_event_times or ()))

        self.k = 0
        self._diag_shapes = dict(getattr(self.controller, "DIAGNOSTICS_SHAPES", {}) or {})
        self._log = self._allocate(self.n_steps)
        self.wall_time_s = 0.0

    @classmethod
    def from_scenario(cls, scenario, controller, controller_overrides=None, **kwargs):
        """Build from a sim.scenario.Scenario; kwargs override its fields."""
        reference, _, _ = scenario.reference()
        fault = scenario.active_fault()
        options = dict(
            duration_s=scenario.resolved_duration(),
            reference=reference,
            initial_pos=scenario.initial_pos,
            initial_attitude=scenario.initial_attitude,
            rotor_fault=fault,
            # A fault between two control ticks starts at its exact time.
            plant_event_times=fault.start_times_s if fault.enabled else (),
            wind=scenario.active_wind(),
            estimator=scenario.estimator,
            seed=scenario.seed,
            control_link=scenario.control_link,
            actuator_profile=scenario.actuator_profile,
            actuator_plant_backend=scenario.actuator_plant_backend,
            controller_overrides=controller_overrides,
            plant_overrides=dict(scenario.plant_overrides),
        )
        if scenario.actuator_thrust_map is not None:
            options["plant_overrides"].setdefault("thrust_map", scenario.actuator_thrust_map)
        options.update(kwargs)
        return cls(controller, **options)

    # ---- stepping ---------------------------------------------------------
    @property
    def t(self):
        return float(self.k) * self.dt

    @property
    def done(self):
        return self.k >= self.n_steps

    def step(self):
        """Advance one control period; returns the motor PWM that was applied."""
        if self.done:
            raise RuntimeError("simulation already ran {} steps".format(self.n_steps))
        k, dt, quad = self.k, self.dt, self.quad
        tk = float(k) * dt
        des = self.reference(tk)

        wind_k = np.zeros(3) if self.wind is None else self.wind.value(tk)
        if self.wind is not None and self.wind.as_velocity:
            ext_k, wind_vel_k = np.zeros(3), wind_k
        else:
            ext_k, wind_vel_k = wind_k, None

        ctrl_quad = self.estimator.view(quad) if self.estimator is not None else quad
        motor_pwm = self.controller.run(ctrl_quad, des, sim_time=tk)
        if self.link is not None:
            motor_pwm = self.link.apply(motor_pwm)

        cuts = sorted(
            set([tk, tk + dt] + [e for e in self.plant_event_times if tk < e < tk + dt])
        )
        for left, right in zip(cuts[:-1], cuts[1:]):
            quad.update(
                dt if len(cuts) == 2 else right - left,
                motor_pwm,
                external_force_world=ext_k,
                sim_time=left,
                wind_velocity_world=wind_vel_k,
            )
        if self.estimator is not None:
            self.estimator.update(quad, dt, tk + dt, ext_k, wind_vel_k)

        # The recorded state is at the end of the interval; compare it with
        # the reference at that same instant (as the live view does).
        self._record(k, tk + dt, self.reference(tk + dt), wind_k, motor_pwm)
        self.k += 1
        return motor_pwm

    def run(self, progress=False):
        """Step to the end and return the time series (see results())."""
        print_every = max(1, self.n_steps // 20)
        start = time.perf_counter()
        while not self.done:
            self.step()
            if progress and (self.k % print_every == 0 or self.done):
                print(
                    "    [{:5.1f}%]  t = {:6.2f} s   err = {:.4f} m".format(
                        100.0 * self.k / self.n_steps, self.t, self._log["err"][self.k - 1]
                    ),
                    flush=True,
                )
        self.wall_time_s += time.perf_counter() - start
        return self.results()

    # ---- recording ----------------------------------------------------------
    def _allocate(self, n):
        log = {
            "t": np.empty(n),
            "pos": np.empty((n, 3)),
            "vel": np.empty((n, 3)),
            "att": np.empty((n, 3)),  # roll, pitch, yaw (rad)
            "omega": np.empty((n, 3)),  # body rates (rad/s)
            "err": np.empty(n),  # ||p - p_d|| (m)
            "desired_pos": np.empty((n, 3)),
            "wind": np.empty((n, 3)),  # wind waveform (N, or m/s for a velocity wind)
            "motor_pwm": np.empty((n, 4)),
            "motor_thrusts": np.empty((n, 4)),  # realized per-rotor thrust (N)
            "rotor_speed_truth_rad_s": np.empty((n, 4)),
        }
        if self.estimator is not None:
            log["est_pos"] = np.empty((n, 3))
            log["est_att"] = np.empty((n, 3))
        for key, width in self._diag_shapes.items():
            if key in log:
                raise KeyError("diagnostics key {!r} collides with a log column".format(key))
            log[key] = np.full((n, int(width)), np.nan)
        return log

    def _record(self, k, t, des, wind_k, motor_pwm):
        log, quad = self._log, self.quad
        log["t"][k] = t
        log["pos"][k] = quad.position()
        log["vel"][k] = quad.velocity()
        log["att"][k] = quad.attitude()
        log["omega"][k] = quad.omega()
        log["desired_pos"][k] = np.asarray(des.pos, dtype=float).reshape(3,)
        log["err"][k] = float(np.linalg.norm(log["pos"][k] - log["desired_pos"][k]))
        log["wind"][k] = wind_k
        log["motor_pwm"][k] = motor_pwm
        log["motor_thrusts"][k] = quad.motor_thrusts
        log["rotor_speed_truth_rad_s"][k] = quad.rotor_speed_rad_s()
        if self.estimator is not None:
            log["est_pos"][k] = self.estimator.pos_est
            log["est_att"][k] = self.estimator.att_est
        if self._diag_shapes:
            diag = self.controller.diagnostics()
            for key, width in self._diag_shapes.items():
                if key in diag:
                    log[key][k] = np.asarray(diag[key], dtype=float).reshape(int(width),)

    def results(self):
        """Arrays for the steps taken so far, plus run metadata."""
        n = self.k
        out = {"mode": self.mode}
        for key, values in self._log.items():
            values = values[:n]
            if key in self._diag_shapes:
                if np.all(np.isnan(values)):
                    continue
                if values.shape[1] == 1:
                    values = values.reshape(n)
            out[key] = values
        out.update(
            wall_time_s=self.wall_time_s,
            dynamic_actuator_plant=str(self.quad.actuator_plant_backend) != "legacy_thrust_euler",
            actuator_profile=str(self.quad.actuator_profile.name),
            actuator_plant_backend=str(self.quad.actuator_plant_backend),
        )
        return out


def run(scenario, controller, controller_overrides=None, progress=False, **kwargs):
    """Fly ``scenario`` with ``controller`` headless; returns Simulation.results()."""
    sim = Simulation.from_scenario(scenario, controller, controller_overrides, **kwargs)
    return sim.run(progress=progress)
