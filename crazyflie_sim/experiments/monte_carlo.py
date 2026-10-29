"""Monte Carlo parameter-uncertainty support.

A ``monte_carlo:`` config section describes how model parameters are
resampled per benchmark replicate::

    monte_carlo:
      n_runs: 20            # int >= 2
      seed: 123             # int; drives ONE np.random.default_rng
      perturb:              # PLANT_PERTURB_KEYS or sim.config.PARAMS_OVERRIDE_KEYS
        mass: {dist: uniform, low: 0.038, high: 0.042}
        aero_drag_factor_xy: {dist: normal, mean: 0.00539, std: 0.0005, clip: [0.004, 0.007]}
        motor_tau_up: {dist: uniform, low: 0.02, high: 0.03}
        I: {dist: uniform, low: 0.9, high: 1.1, mode: scale}

Distributions:
  "uniform"  -- required fields low, high (low <= high)
  "normal"   -- required fields mean, std (std >= 0); optional clip [lo, hi]
                applied after the draw

Modes:
  "absolute" (default) -- the sample IS the new parameter value
  "scale"              -- the sample multiplies the parameter's nominal
                          value.  For "I" only "scale" is
                          allowed; it draws 3 independent scale factors applied
                          to the nominal inertia diagonal.

The vehicle keys (PLANT_PERTURB_KEYS: mass, I, motor_tau_up, motor_tau_down)
perturb the PLANT only, through ``plant_overrides``; their nominal value is
the run's actuator profile, and the controllers keep it, so a replicate is a
model mismatch.  The other keys are simulator-wide model.params leaves.

Reproducibility: one np.random.default_rng(seed) draws every replicate's
overrides in order (replicate 0 first, one draw per perturb key in mapping
order, 3 draws for "I"), so the same config yields identical perturbations.
"""

import numpy as np

from crazyflie_sim.model.actuator_profile import get_actuator_profile
from crazyflie_sim.sim.config import PARAMS_OVERRIDE_KEYS, params_overridden

# perturb key -> plant_overrides key
PLANT_PERTURB_KEYS = {"mass": "mass", "I": "inertia",
                "motor_tau_up": "motor_tau_up", "motor_tau_down": "motor_tau_down"}

_DISTS = ("uniform", "normal")
_MODES = ("absolute", "scale")
_COMMON_FIELDS = ("dist", "mode")
_UNIFORM_FIELDS = ("low", "high")
_NORMAL_FIELDS = ("mean", "std", "clip")


def _err(key, message):
    raise ValueError("monte_carlo key {!r}: {}".format(key, message))


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_number(key, value):
    if not _is_number(value):
        _err(key, "expected a number, got {!r}".format(value))
    return float(value)


def _validate_perturb_entry(key, spec):
    """Validate one perturb entry; return its normalized form."""
    full_key = "perturb." + key
    if key not in PARAMS_OVERRIDE_KEYS and key not in PLANT_PERTURB_KEYS:
        _err(
            full_key,
            "not a vehicle key {} nor in the params whitelist {}".format(
                sorted(PLANT_PERTURB_KEYS), sorted(PARAMS_OVERRIDE_KEYS)
            ),
        )
    if not isinstance(spec, dict):
        _err(full_key, "expected a mapping, got {!r}".format(spec))

    dist = spec.get("dist")
    if dist not in _DISTS:
        _err(
            full_key + ".dist",
            "must be one of {}; got {!r}".format(
                ", ".join('"{}"'.format(d) for d in _DISTS), dist
            ),
        )
    mode = spec.get("mode", "absolute")
    if mode not in _MODES:
        _err(
            full_key + ".mode",
            "must be one of {}; got {!r}".format(
                ", ".join('"{}"'.format(m) for m in _MODES), mode
            ),
        )
    if key == "I" and mode != "scale":
        _err(full_key + ".mode", 'only "scale" is allowed for "I"')

    allowed = set(_COMMON_FIELDS) | set(
        _UNIFORM_FIELDS if dist == "uniform" else _NORMAL_FIELDS
    )
    unknown = sorted(set(spec) - allowed)
    if unknown:
        _err(
            full_key,
            "unknown field(s) {} for dist {!r}; valid fields: {}".format(
                unknown, dist, sorted(allowed)
            ),
        )

    out = {"dist": dist, "mode": mode}
    if dist == "uniform":
        for name in _UNIFORM_FIELDS:
            if name not in spec:
                _err(full_key + "." + name, "missing required field for 'uniform'")
            out[name] = _require_number(full_key + "." + name, spec[name])
        if out["low"] > out["high"]:
            _err(full_key, "low must be <= high")
    else:
        for name in ("mean", "std"):
            if name not in spec:
                _err(full_key + "." + name, "missing required field for 'normal'")
            out[name] = _require_number(full_key + "." + name, spec[name])
        if out["std"] < 0.0:
            _err(full_key + ".std", "must be >= 0")
        if "clip" in spec:
            clip = spec["clip"]
            if (
                not isinstance(clip, (list, tuple))
                or len(clip) != 2
                or not all(_is_number(c) for c in clip)
            ):
                _err(full_key + ".clip", "expected [lo, hi] numbers, got {!r}".format(clip))
            lo, hi = float(clip[0]), float(clip[1])
            if lo > hi:
                _err(full_key + ".clip", "lo must be <= hi")
            out["clip"] = [lo, hi]
    return out


def validate_monte_carlo_spec(mc_cfg: dict) -> dict:
    """Validate a ``monte_carlo:`` config section; return its normalized form.

    Raises ValueError naming the offending key on any violation.  The returned
    dict is {"n_runs": int, "seed": int, "perturb": {key: normalized spec}}
    with all numeric fields converted to plain floats.
    """
    if not isinstance(mc_cfg, dict):
        _err("", "expected a mapping, got {!r}".format(mc_cfg))
    unknown = sorted(set(mc_cfg) - {"n_runs", "seed", "perturb"})
    if unknown:
        _err("", "unknown sub-key(s) {}; valid sub-keys: ['n_runs', 'perturb', 'seed']".format(unknown))

    n_runs = mc_cfg.get("n_runs")
    if not isinstance(n_runs, int) or isinstance(n_runs, bool):
        _err("n_runs", "expected an integer, got {!r}".format(n_runs))
    if n_runs < 2:
        _err("n_runs", "must be >= 2, got {}".format(n_runs))

    seed = mc_cfg.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        _err("seed", "expected an integer, got {!r}".format(seed))

    perturb = mc_cfg.get("perturb")
    if not isinstance(perturb, dict):
        _err("perturb", "expected a mapping, got {!r}".format(perturb))

    return {
        "n_runs": int(n_runs),
        "seed": int(seed),
        "perturb": {
            key: _validate_perturb_entry(key, spec) for key, spec in perturb.items()
        },
    }


def _draw_one(spec, rng):
    """Draw one sample from a validated perturb spec."""
    if spec["dist"] == "uniform":
        return rng.uniform(spec["low"], spec["high"])
    value = rng.normal(spec["mean"], spec["std"])
    if "clip" in spec:
        lo, hi = spec["clip"]
        value = min(max(value, lo), hi)
    return value


def _nominal(key, profile):
    import crazyflie_sim.model.params as params

    if key == "mass":
        return profile.mass
    if key == "motor_tau_up":
        return profile.tau_up
    if key == "motor_tau_down":
        return profile.tau_down
    return float(getattr(params, key))


def sample_perturbations(perturb_spec: dict, rng, actuator_profile=None) -> dict:
    """Draw one replicate's overrides from a validated perturb spec.

    ``perturb_spec`` is the "perturb" sub-dict returned by
    validate_monte_carlo_spec; ``rng`` is an np.random.Generator.  Returns
    {key: value} in the spec's keys: "I" maps to a 3-element list of scaled
    diagonal values (3 independent draws), scalars map to floats.  "scale"
    mode multiplies the nominal value: the profile's for vehicle keys, the
    current model.params leaf otherwise.  :func:`split_overrides` routes the
    result to plant_overrides and params_overrides.
    """
    profile = get_actuator_profile(actuator_profile)
    overrides = {}
    for key, spec in perturb_spec.items():
        if key == "I":
            scales = [_draw_one(spec, rng) for _ in range(3)]
            diagonal = np.diag(np.asarray(profile.inertia, dtype=float))
            overrides["I"] = [
                float(s * d) for s, d in zip(scales, diagonal)
            ]
        else:
            value = _draw_one(spec, rng)
            if spec["mode"] == "scale":
                value = value * _nominal(key, profile)
            overrides[key] = float(value)
    return overrides


def split_overrides(overrides: dict):
    """-> (plant_overrides, params_overrides) for one replicate."""
    plant = {PLANT_PERTURB_KEYS[k]: v for k, v in overrides.items() if k in PLANT_PERTURB_KEYS}
    rest = {k: v for k, v in overrides.items() if k not in PLANT_PERTURB_KEYS}
    return plant, rest


# Apply one replicate's overrides inside a ``with`` block; params is restored on exit.
perturbed_params = params_overridden


def aggregate_metrics(per_replicate: list) -> dict:
    """Aggregate per-replicate metrics into mean/std/n/values per mode.

    ``per_replicate`` is one {mode: metrics_dict} per replicate.  Returns
    {mode: {metric: {"mean":…, "std":…, "n":…, "values": [...]}}} where mean
    and std use np.nanmean/np.nanstd (population std, ddof=0) and n is the
    count of finite values; mean/std are None when no finite value exists.
    "wall_time_s" is aggregated like any other metric but is
    machine-dependent -- only the deterministic metrics are reproducible.
    """
    modes = []
    for rep in per_replicate:
        for mode in rep:
            if mode not in modes:
                modes.append(mode)
    out = {}
    for mode in modes:
        metric_names = []
        for rep in per_replicate:
            if mode not in rep:
                continue
            for name in rep[mode]:
                if name not in metric_names:
                    metric_names.append(name)
        mode_out = {}
        for name in metric_names:
            values = [
                float(rep[mode][name])
                for rep in per_replicate
                if mode in rep and name in rep[mode]
            ]
            arr = np.asarray(values, dtype=float)
            n = int(np.count_nonzero(np.isfinite(arr)))
            if n > 0:
                mean = float(np.nanmean(arr))
                std = float(np.nanstd(arr))
            else:
                mean = None
                std = None
            mode_out[name] = {"mean": mean, "std": std, "n": n, "values": values}
        out[mode] = mode_out
    return out
