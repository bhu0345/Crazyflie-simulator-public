"""Golden trajectories: short reference runs that must not change silently.

    python -m crazyflie_sim.tools.golden capture   # after an intended physics/controller change
    python -m crazyflie_sim.tools.golden compare   # what tests/test_golden.py runs

Each scenario flies through sim.simulation, like everything else.  The stored
files carry a fingerprint of model.params and the actuator profiles; a
mismatch says "the plant constants changed, recapture if intended" instead of
reporting a wall of numeric differences.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from crazyflie_sim.sim.scenario import RotorFaultSpec, Scenario
from crazyflie_sim.sim.simulation import run

# The goldens live with the tests of a source checkout, not in the package.
REPO = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO / "tests" / "golden"
# Bit-identical results are not portable across BLAS builds (macOS / Linux);
# these bounds still catch a real controller or plant change.
ATOL = {"state": 1.0e-5, "pwm": 0.05}


def scenarios():
    """name -> (Scenario, controller, gains).  Controlled flights only: a
    tumbling vehicle is chaotic and would not compare across platforms."""
    wind = Scenario(name="wind", sim_mode="hover_wind", duration_s=5.0)
    trajectory = Scenario(name="trajectory", sim_mode="trajectory", duration_s=5.0)
    partial = Scenario(name="partial", sim_mode="hover_rotor_fault", duration_s=5.0,
                       initial_pos=(0.0, 0.0, 1.0), rotor_fault=RotorFaultSpec(case="power_loss_50pct"))
    out = {}
    for mode in ("baseline", "mellinger", "full_state_feedback"):
        out["wind_" + mode] = (wind, mode, {})
    for mode in ("baseline", "full_state_feedback"):
        out["traj_" + mode] = (trajectory, mode, {})
    out["partial_baseline"] = (partial, "baseline", {})
    return out


def params_fingerprint():
    import crazyflie_sim.model.params as params
    from crazyflie_sim.model.actuator_profile import ACTUATOR_PROFILE_REGISTRY

    values = {}
    for name in sorted(dir(params)):
        value = getattr(params, name)
        if name.startswith("_") or callable(value) or type(value).__name__ == "module":
            continue
        values[name] = np.asarray(value).tolist() if isinstance(value, np.ndarray) else repr(value)
    values["profiles"] = {k: p.to_config() for k, p in ACTUATOR_PROFILE_REGISTRY.items()}
    blob = json.dumps(values, sort_keys=True, default=repr).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def fly(name):
    scenario, mode, gains = scenarios()[name]
    out = run(scenario, mode, gains)
    state = np.hstack([out["pos"], out["vel"], out["att"], out["omega"]])
    return {"t": out["t"], "state": state, "pwm": out["motor_pwm"]}


def capture(names=None):
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint = params_fingerprint()
    for name in names or scenarios():
        data = fly(name)
        np.savez_compressed(GOLDEN_DIR / (name + ".npz"), fingerprint=fingerprint, **data)
        print("captured", name)


def compare(name):
    """Return a list of problems (empty when the scenario matches)."""
    path = GOLDEN_DIR / (name + ".npz")
    if not path.exists():
        return ["no golden file; run python -m crazyflie_sim.tools.golden capture"]
    with np.load(path) as ref:
        stored = str(ref["fingerprint"])
        expected = {k: ref[k] for k in ("t", "state", "pwm")}
    if stored != params_fingerprint():
        return ["model.params or an actuator profile changed since capture "
                "(fingerprint {} != {}); recapture if intended".format(stored, params_fingerprint())]
    actual = fly(name)
    problems = []
    if not np.array_equal(expected["t"], actual["t"]):
        problems.append("time grid differs")
    for key, atol in ATOL.items():
        diff = float(np.max(np.abs(expected[key] - actual[key])))
        if not diff <= atol:
            problems.append("{} max|diff| = {:.3e} > {:.0e}".format(key, diff, atol))
    return problems


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["capture"]:
        capture(argv[1:] or None)
        return 0
    if argv[:1] == ["compare"]:
        failed = 0
        for name in scenarios():
            problems = compare(name)
            print("{:<28} {}".format(name, "ok" if not problems else "; ".join(problems)))
            failed += bool(problems)
        return 1 if failed else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
