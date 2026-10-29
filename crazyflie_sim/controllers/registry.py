"""The one list of controller modes.

A controller joins the simulator by decorating its class::

    from crazyflie_sim.controllers.registry import register

    @register("my_ctrl", label="My controller", color="#d62728")
    class MyController(ControllerBase):
        def __init__(self, dt, gain=1.0): ...

Every module in this package is imported on first lookup, so dropping the
file into ``controllers/`` is enough: runsim, configs, the experiment scripts
and the plots all read their mode lists, labels and colours from here.

A constructor that takes ``actuator_profile`` receives the run's vehicle, the
same profile the plant flies (None means the default crazyflie21_physical);
read mass, inertia, mixer and limits from it rather than from model.params,
which holds no vehicle constants.  Extra keyword arguments to ``register``
are fixed constructor arguments, which lets one class serve several modes.
"""

import importlib
import inspect
import pkgutil
from dataclasses import dataclass, field

_REGISTRY = {}
_DISCOVERED = False


@dataclass(frozen=True)
class ControllerSpec:
    name: str
    cls: type
    label: str
    color: str
    fixed_kwargs: dict = field(default_factory=dict)

    def parameters(self):
        """Constructor keyword arguments a config may set (``dt`` excluded)."""
        params = inspect.signature(self.cls.__init__).parameters
        return sorted(
            name
            for name, p in params.items()
            if name not in ("self", "dt")
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        )

    def accepts_any_keyword(self):
        params = inspect.signature(self.cls.__init__).parameters.values()
        return any(p.kind is p.VAR_KEYWORD for p in params)


def register(name, *, label, color, **fixed_kwargs):
    """Class decorator that adds a controller mode."""

    def decorate(cls):
        if name in _REGISTRY and _REGISTRY[name].cls is not cls:
            raise ValueError("controller mode {!r} is already registered".format(name))
        _REGISTRY[name] = ControllerSpec(
            name=name,
            cls=cls,
            label=label,
            color=color,
            fixed_kwargs=dict(fixed_kwargs),
        )
        return cls

    return decorate


def _discover():
    global _DISCOVERED
    if _DISCOVERED:
        return
    import crazyflie_sim.controllers as controllers

    for info in sorted(pkgutil.iter_modules(controllers.__path__), key=lambda m: m.name):
        if info.name not in ("base", "common", "registry"):
            importlib.import_module(controllers.__name__ + "." + info.name)
    _DISCOVERED = True


def available():
    """Registered mode names, in registration order."""
    _discover()
    return tuple(_REGISTRY)


def get(name):
    _discover()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            "unknown controller {!r}; available: {}".format(name, ", ".join(_REGISTRY))
        ) from None


def build(name, dt, overrides=None, actuator_profile=None):
    """Construct, set_dt and reset one controller.

    ``overrides`` are constructor keyword arguments (config ``controller_gains``);
    they win over the registration's fixed arguments.  ``actuator_profile`` is
    passed to every controller whose constructor takes it.
    """
    spec = get(name)
    kwargs = dict(spec.fixed_kwargs)
    kwargs.update(dict(overrides or {}))
    if "dt" in kwargs:
        raise ValueError("controller {!r}: 'dt' is set by the simulator".format(name))
    if "actuator_profile" in spec.parameters():
        kwargs.setdefault("actuator_profile", actuator_profile)
    unknown = sorted(set(kwargs) - set(spec.parameters()))
    if unknown and not spec.accepts_any_keyword():
        raise ValueError(
            "controller {!r}: unknown parameter(s) {}; valid: {}".format(
                name, unknown, spec.parameters()
            )
        )
    backend = spec.cls(dt=dt, **kwargs)
    backend.set_dt(dt)
    backend.reset()
    return backend
