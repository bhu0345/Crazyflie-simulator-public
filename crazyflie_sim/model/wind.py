"""Wind / external disturbance, one waveform per world axis.

A :class:`WindModel` is an ordinary object owned by a scenario, so two runs in
the same process never share wind state.  Each axis picks a mode:

- ``off``        always zero
- ``sin``        offset + amplitude*sin(2*pi*f*(t-t0) + phase)
- ``constant``   fixed value
- ``square``     high/low square wave with a duty cycle
- ``impulse``    one pulse of ``impulse_width`` starting ``impulse_time`` after t0
- ``smoothstep`` smooth ramp to ``smoothstep_magnitude`` (s = h^2(3-2h))
- ``dryden``     Ornstein-Uhlenbeck gust around ``offset`` (stateful, seeded)

The waveform is a force in newtons, or, with ``as_velocity=True``, a wind
velocity in m/s that acts through the aerodynamic model (``model/aero.py``)
instead of as a direct force.  It is zero before ``start_time`` and after
``end_time``.

YAML (``wind:`` in a config; every key optional)::

    wind:
      start_time: 2.0
      as_velocity: false
      x: {mode: smoothstep, smoothstep_magnitude: 0.3, smoothstep_rise_time: 0.35}
      y: {mode: "off"}
      z: {mode: dryden, dryden_sigma: 0.1, dryden_tau: 1.0}
"""

from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np

MODE_OFF = "off"
MODE_SIN = "sin"
MODE_CONSTANT = "constant"
MODE_SQUARE = "square"
MODE_IMPULSE = "impulse"
MODE_SMOOTHSTEP = "smoothstep"
MODE_DRYDEN = "dryden"
MODES = (
    MODE_OFF,
    MODE_SIN,
    MODE_CONSTANT,
    MODE_SQUARE,
    MODE_IMPULSE,
    MODE_SMOOTHSTEP,
    MODE_DRYDEN,
)


@dataclass(frozen=True)
class AxisWind:
    """Waveform of one world axis.  Only the fields of ``mode`` matter."""

    mode: str = MODE_OFF
    amplitude: float = 0.0
    frequency_hz: float = 0.0
    phase: float = 0.0
    offset: float = 0.0
    constant: float = 0.0
    square_high: float = 0.0
    square_low: float = 0.0
    square_duty: float = 0.5
    impulse_magnitude: float = 0.0
    impulse_time: float = 1.0
    impulse_width: float = 0.05
    smoothstep_magnitude: float = 0.0
    smoothstep_rise_time: float = 0.35
    dryden_sigma: float = 0.0
    dryden_tau: float = 1.0

    def __post_init__(self):
        if str(self.mode).strip().lower() not in MODES:
            raise ValueError(
                "unknown wind mode {!r}; expected one of {}".format(self.mode, MODES)
            )


# The historical default: three slow sinusoids.
DEFAULT_X = AxisWind(
    mode=MODE_SIN, amplitude=0.4, frequency_hz=0.1, constant=0.2,
    square_high=0.3, square_low=-0.3, square_duty=0.05,
    impulse_magnitude=0.8, impulse_time=0.5,
)
DEFAULT_Y = AxisWind(
    mode=MODE_SIN, amplitude=0.4, frequency_hz=0.05,
    square_high=0.4, square_low=-0.4, impulse_magnitude=0.4,
)
DEFAULT_Z = AxisWind(
    mode=MODE_SIN, amplitude=0.1, frequency_hz=0.5, impulse_magnitude=0.2,
)


def _sinusoid(amplitude, frequency_hz, phase, t_rel):
    return float(amplitude) * np.sin(2.0 * np.pi * float(frequency_hz) * t_rel + float(phase))


def _square_wave(high, low, frequency_hz, phase, duty, t_rel):
    duty = float(np.clip(duty, 0.0, 1.0))
    frequency_hz = float(frequency_hz)
    if frequency_hz <= 0.0:
        return float(high)
    phase_cycles = float(phase) / (2.0 * np.pi)
    phase_in_period = (t_rel * frequency_hz + phase_cycles) % 1.0
    return float(high) if phase_in_period < duty else float(low)


def _impulse(magnitude, impulse_time, impulse_width, t_rel):
    impulse_time = float(impulse_time)
    impulse_width = max(float(impulse_width), 1e-6)
    if impulse_time <= t_rel < (impulse_time + impulse_width):
        return float(magnitude)
    return 0.0


def _smoothstep(magnitude, rise_time, t_rel):
    rise_time = max(float(rise_time), 1e-6)
    h = float(np.clip(t_rel / rise_time, 0.0, 1.0))
    s = h * h * (3.0 - 2.0 * h)
    return float(magnitude) * s


def _analytic(axis, t_rel):
    mode = str(axis.mode).strip().lower()
    if mode == MODE_OFF:
        return 0.0
    if mode == MODE_SIN:
        return float(axis.offset) + _sinusoid(axis.amplitude, axis.frequency_hz, axis.phase, t_rel)
    if mode == MODE_CONSTANT:
        return float(axis.constant)
    if mode == MODE_SQUARE:
        return _square_wave(
            axis.square_high, axis.square_low, axis.frequency_hz, axis.phase,
            axis.square_duty, t_rel,
        )
    if mode == MODE_IMPULSE:
        return _impulse(axis.impulse_magnitude, axis.impulse_time, axis.impulse_width, t_rel)
    if mode == MODE_SMOOTHSTEP:
        return _smoothstep(axis.smoothstep_magnitude, axis.smoothstep_rise_time, t_rel)
    raise ValueError("unknown wind mode {!r}".format(axis.mode))


@dataclass
class WindModel:
    start_time: float = 2.0
    end_time: float = None
    as_velocity: bool = False
    dryden_seed: int = 0
    x: AxisWind = DEFAULT_X
    y: AxisWind = DEFAULT_Y
    z: AxisWind = DEFAULT_Z
    _state: dict = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        self.reset()

    @classmethod
    def from_dict(cls, spec):
        """Build from a config ``wind:`` mapping; unset keys keep the defaults."""
        spec = dict(spec or {})
        axes = {}
        for name, default in (("x", DEFAULT_X), ("y", DEFAULT_Y), ("z", DEFAULT_Z)):
            axis_spec = dict(spec.pop(name, None) or {})
            unknown = sorted(set(axis_spec) - {f.name for f in fields(AxisWind)})
            if unknown:
                raise ValueError("wind.{}: unknown key(s) {}".format(name, unknown))
            axes[name] = replace(default, **axis_spec)
        top = {"start_time", "end_time", "as_velocity", "dryden_seed"}
        unknown = sorted(set(spec) - top)
        if unknown:
            hint = ""
            if any(str(k).isupper() for k in unknown):
                hint = " (the UPPERCASE disturbance.py names are gone; see model/wind.py)"
            raise ValueError("wind: unknown key(s) {}{}".format(unknown, hint))
        return cls(**spec, **axes)

    def to_dict(self):
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "as_velocity": self.as_velocity,
            "dryden_seed": self.dryden_seed,
            "x": asdict(self.x),
            "y": asdict(self.y),
            "z": asdict(self.z),
        }

    def reset(self, seed=None):
        """Restart the Dryden gusts (x = 0, fresh RNG); a new run calls this."""
        self._state = {
            "rng": np.random.default_rng(self.dryden_seed if seed is None else seed),
            "x": np.zeros(3),
            "last_t": None,
            "last_value": None,
        }

    def _dryden_step(self, index, axis, dt):
        """phi = exp(-dt/tau);  x = phi*x + sigma*sqrt(1-phi^2)*xi,  xi ~ N(0,1)."""
        state = self._state
        tau = max(float(axis.dryden_tau), 1e-9)
        phi = np.exp(-dt / tau)
        xi = state["rng"].standard_normal()
        state["x"][index] = (
            phi * state["x"][index] + float(axis.dryden_sigma) * np.sqrt(1.0 - phi * phi) * xi
        )
        return float(axis.offset) + float(state["x"][index])

    def value(self, t):
        """Waveform at time t: force (N), or wind velocity (m/s) if as_velocity.

        Analytic modes are pure functions of t.  Dryden axes advance once per
        new in-window t; a repeated t returns the cached value, gated calls do
        not touch the state, and an earlier t restarts the gusts.
        """
        state = self._state
        t = float(t)
        if state["last_t"] is not None and t < state["last_t"] - 1e-12:
            self.reset()
            state = self._state
        if state["last_t"] is not None and abs(t - state["last_t"]) < 1e-12:
            return state["last_value"]
        if t < self.start_time or (self.end_time is not None and t > self.end_time):
            return np.zeros(3)

        t_rel = t - self.start_time
        dt = 0.0 if state["last_t"] is None else max(0.0, t - state["last_t"])
        out = []
        for index, axis in enumerate((self.x, self.y, self.z)):
            if str(axis.mode).strip().lower() == MODE_DRYDEN:
                out.append(self._dryden_step(index, axis, dt))
            else:
                out.append(_analytic(axis, t_rel))
        value = np.array(out, dtype=float)
        state["last_t"] = t
        state["last_value"] = value
        return value

    def force_world(self, t):
        """External force (N) on the plant; zero when the wind is a velocity."""
        return np.zeros(3) if self.as_velocity else self.value(t)

    def velocity_world(self, t):
        """Wind velocity (m/s) for the aero model, or None for calm air."""
        return self.value(t) if self.as_velocity else None

    def formula_text(self):
        lines = ["Wind VELOCITY field (m/s, acts through the aero model)"] if self.as_velocity else []
        window = "" if self.end_time is None else " and t <= t1, t1={:.2f} s".format(float(self.end_time))
        lines += [
            "Wind model (flexible):",
            "active for t >= t0{} ".format(window),
            "t0={:.2f} s".format(self.start_time),
        ]
        for name, axis in (("Fx", self.x), ("Fy", self.y), ("Fz", self.z)):
            lines.append(_axis_formula(name, axis, self.dryden_seed))
        return "\n".join(lines)


def _axis_formula(name, axis, seed):
    mode = str(axis.mode).strip().lower()
    if mode == MODE_OFF:
        return "{}(t) = 0".format(name)
    if mode == MODE_SIN:
        return "{}(t) = {:.3f} + {:.3f} sin(2pi*{:.3f}(t-t0)+{:.3f})".format(
            name, float(axis.offset), float(axis.amplitude), float(axis.frequency_hz), float(axis.phase)
        )
    if mode == MODE_CONSTANT:
        return "{}(t) = {:.3f} N".format(name, float(axis.constant))
    if mode == MODE_SQUARE:
        return "{}(t) = square(high={:.3f}, low={:.3f}, f={:.3f} Hz, duty={:.2f}, phase={:.3f})".format(
            name, float(axis.square_high), float(axis.square_low), float(axis.frequency_hz),
            float(axis.square_duty), float(axis.phase),
        )
    if mode == MODE_IMPULSE:
        return "{}(t) = impulse(mag={:.3f} N, t={:.3f} s, width={:.3f} s)".format(
            name, float(axis.impulse_magnitude), float(axis.impulse_time), float(axis.impulse_width)
        )
    if mode == MODE_SMOOTHSTEP:
        return "{}(t) = smoothstep(mag={:.3f} N, rise={:.2f} s)".format(
            name, float(axis.smoothstep_magnitude), float(axis.smoothstep_rise_time)
        )
    return "{}(t) = OU(mean={:.3f}, sigma={:.3f} N, tau={:.2f} s, seed={})".format(
        name, float(axis.offset), float(axis.dryden_sigma), float(axis.dryden_tau), int(seed)
    )
